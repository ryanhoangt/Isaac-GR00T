"""Replay an observed action prefix, then execute a short fixed-image-goal test.

Only standard benchmark reset, RGB, robot proprioception and commands are used.
No object state, simulator snapshots, predicates or online phase matching.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
import imageio
import numpy as np

from libero.libero import benchmark
from examples.Libero.eval.run_libero_eval import GR00TPolicy
from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image
from gr00t.rag.fixed_goal_protocol import (
    REFERENCE_KEYS,
    ROBOT_KEYS,
    action_agreement,
    load_reference,
    observation_identity,
    reference_observation,
    replay_prefix,
    robot_observation,
    target_metrics,
)
from gr00t.rag.oracle_goal_protocol import case_seed


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False) + "\n")
    temporary.replace(path)


def frame(observation, goal, caption):
    current, wrist = get_libero_image(observation)
    target = get_libero_image(goal)[0]
    result = np.pad(np.concatenate((current, wrist, target), axis=1), ((32, 0), (0, 0), (0, 0)))
    cv2.putText(result, caption, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)
    return result


def rollout(config, adapter, task, initial, reference, mode, directory):
    start = time.perf_counter()
    env, _ = get_libero_env(task, resolution=256)
    writer = None
    prefix, horizon = config["prefix"], config["horizon"]
    goal_index = prefix + horizon
    target = reference_observation(reference, goal_index)
    try:
        # Benchmark initial-state vectors are used only through its standard
        # reset API. They are never inspected or passed to the policy/metrics.
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        try:
            obs, checks = replay_prefix(
                env,
                obs,
                reference,
                prefix,
                rgb_tolerance=config["rgb_tolerance"],
                proprio_tolerance=config["proprio_tolerance"],
            )
        except ValueError as error:
            atomic_json(
                directory / "replay_failure.json", dict(error=str(error), reference=config["reference"], mode=mode)
            )
            raise
        atomic_json(directory / "replay_checks.json", checks)
        queue = deque()
        trajectory = [robot_observation(obs)]
        actions, queries, done_flags = [], [], []
        writer = imageio.get_writer(str(directory / "rollout.tmp.mp4"), fps=20, macro_block_size=1)
        with (directory / "queries.jsonl").open("w") as log:
            for offset in range(horizon):
                if not queue:
                    request = dict(
                        observations=adapter._process_observation(obs, task.language),
                        mode=mode,
                        flow_seed=case_seed(
                            config["seed"], config["suite"], config["initial_index"], prefix + offset
                        ),
                    )
                    if mode == "goal":
                        request["goal_observations"] = adapter._process_observation(target, task.language)
                    prediction = adapter.policy.call_endpoint("predict_oracle", request)
                    if prediction["task"].strip() != task.language.strip():
                        raise ValueError("Policy/reference task differs")
                    chunk = adapter._convert_to_libero_action(prediction.pop("action"))
                    if chunk.shape != (16, 7) or not np.isfinite(chunk).all():
                        raise ValueError("Expected finite 16x7 chunk")
                    prediction.pop("normalized_actions")
                    query = dict(
                        control_step=prefix + offset, local_step=offset, fixed_goal_frame=goal_index, **prediction
                    )
                    queries.append(query)
                    log.write(json.dumps(query, default=plain, allow_nan=False) + "\n")
                    log.flush()
                    queue.extend(chunk[: config["execute_steps"]])
                caption = f"{config['suite']} init={config['initial_index']} {mode} t={prefix + offset} fixed goal={goal_index} exec={config['execute_steps']}"
                writer.append_data(frame(obs, target, caption))
                command = np.asarray(queue.popleft()).copy()
                actions.append(command)
                obs, _, done, _ = env.step(command.tolist())
                done_flags.append(bool(done))
                obs = robot_observation(obs)
                trajectory.append(obs)
        writer.append_data(
            frame(obs, target, f"{mode} FINAL t={goal_index} / target reference frame {goal_index}")
        )
        writer.close()
        writer = None
        (directory / "rollout.tmp.mp4").replace(directory / "rollout.mp4")
        metrics, errors = target_metrics(trajectory, reference, goal_index)
        agreement = action_agreement(actions, reference["actions"][prefix:goal_index])
        if mode == "action" and config["execute_steps"] == 16 and agreement["reference_action_max_abs"] > 1e-5:
            raise ValueError(f"Mode-1 continuation does not reproduce its reference: {agreement}")
        data = {key: np.asarray([o[k] for o in trajectory]) for key, k in zip(REFERENCE_KEYS[:-1], ROBOT_KEYS)}
        data.update(
            actions=np.asarray(actions),
            reference_actions=reference["actions"][prefix:goal_index],
            goal_agent=target["agentview_image"],
            goal_wrist=target["robot0_eye_in_hand_image"],
            goal_ee=target["robot0_eef_pos"],
            goal_quat=target["robot0_eef_quat"],
            goal_gripper=target["robot0_gripper_qpos"],
        )
        with (directory / "trace.tmp.npz").open("wb") as handle:
            np.savez_compressed(handle, **data)
        (directory / "trace.tmp.npz").replace(directory / "trace.npz")
        result = dict(
            complete=True,
            suite=config["suite"],
            task_id=config["task_id"],
            task=task.language,
            initial_index=config["initial_index"],
            mode=mode,
            prefix=prefix,
            horizon=horizon,
            fixed_goal_frame=goal_index,
            execute_steps=config["execute_steps"],
            control_steps=len(actions),
            replay_matched=True,
            replay_frames_checked=len(checks),
            max_replay_rgb_error=max(max(c["workspace_max_abs"], c["wrist_max_abs"]) for c in checks),
            max_replay_proprio_error=max(c["proprio_max_abs"] for c in checks),
            start_observation_identity=observation_identity(trajectory[0]),
            goal_observation_identity=observation_identity(target),
            metrics=metrics,
            action_agreement=agreement,
            target_error_curve=errors,
            queries=queries,
            task_success_seen=any(done_flags),
            subgoal_success=None,
            note="Short-window RGB/proprio proxies, not automated semantic sub-goal success.",
            reference=config["reference"],
            wall_seconds=time.perf_counter() - start,
            video="rollout.mp4",
        )
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main(config):
    reference = load_reference(Path(config["reference"]) / "trace.npz", config["prefix"], config["horizon"])
    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    task = suite.get_task(config["task_id"])
    if task.language != config["task"]:
        raise ValueError("Benchmark/reference task mismatch")
    initial = suite.get_task_init_states(config["task_id"])[config["initial_index"]]
    adapter = GR00TPolicy(host="127.0.0.1", port=config["port"], headless=True, inference_timeout_ms=120000)
    try:
        for mode in config["modes"]:
            directory = Path(config["output"]) / mode / Path(config["reference"]).name
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "episode.json"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old["signature"] != config["signature"]:
                    raise ValueError("Resume protocol differs")
                if old["complete"] and all(
                    (directory / p).is_file() for p in ("trace.npz", "rollout.mp4", "replay_checks.json")
                ):
                    print("RESUMED", directory, flush=True)
                    continue
                raise ValueError("Completed case artifacts missing")
            result = rollout(config, adapter, task, initial, reference, mode, directory)
            result["signature"] = config["signature"]
            atomic_json(result_path, result)
            print("COMPLETE", directory, json.dumps(result["metrics"]), flush=True)
    finally:
        adapter.policy.socket.close(linger=0)
        adapter.policy.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
