"""LIBERO worker: Mode-1 reference followed by privileged same-scene goal control."""

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
from examples.Libero.eval.run_dual_diagnosis import atomic_json, predicates
from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image
from gr00t.rag.oracle_goal_protocol import array_hash, case_seed, choose_milestone, phase_schema, phase_vector


def reference_observation(reference, index):
    return dict(
        agentview_image=reference["agent"][index],
        robot0_eye_in_hand_image=reference["wrist"][index],
        robot0_eef_pos=reference["ee"][index],
        robot0_eef_quat=reference["quat"][index],
        robot0_gripper_qpos=reference["gripper"][index],
    )


def rollout(config, adapter, task, initial, initial_index, directory, reference=None):
    mode = "action" if reference is None else "goal"
    env, _ = get_libero_env(task, resolution=256)
    writer = None
    start = time.perf_counter()
    try:
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        initial_hash = array_hash(env.get_sim_state())
        keys = phase_schema(obs)
        if reference is not None:
            if initial_hash != reference["initial_sim_sha256"].item():
                raise ValueError("Goal/reference simulator start differs")
            if keys != reference["phase_keys"].tolist():
                raise ValueError("Goal/reference object schema differs")
        queue = deque()
        anchor, goal, phase_error = 0, 0, None
        arrays = {key: [] for key in ("agent", "wrist", "ee", "quat", "gripper", "phase", "sim_states")}
        actions, distances, queries = [], [], []
        success = False
        writer = imageio.get_writer(str(directory / "rollout.tmp.mp4"), fps=20, macro_block_size=1)
        with (directory / "queries.jsonl").open("w") as log:
            for step in range(config["max_steps"] + 1):
                position = np.asarray(obs["robot0_eef_pos"]).copy()
                arrays["ee"].append(position)
                arrays["quat"].append(np.asarray(obs["robot0_eef_quat"]).copy())
                arrays["gripper"].append(np.asarray(obs["robot0_gripper_qpos"]).copy())
                arrays["phase"].append(phase_vector(obs, keys))
                if reference is None:
                    arrays["agent"].append(obs["agentview_image"].copy())
                    arrays["wrist"].append(obs["robot0_eye_in_hand_image"].copy())
                    arrays["sim_states"].append(env.get_sim_state().copy())
                else:
                    distances.append(float(np.linalg.norm(reference["ee"] - position, axis=1).min()))
                if success or step == config["max_steps"]:
                    break
                if not queue:
                    request = dict(
                        observations=adapter._process_observation(obs, task.language),
                        mode=mode,
                        flow_seed=case_seed(config["seed"], config["suite"], initial_index, step),
                    )
                    if reference is not None:
                        anchor, goal, phase_error = choose_milestone(
                            reference["phase"], arrays["phase"][-1], anchor
                        )
                        goal_obs = reference_observation(reference, goal)
                        request["goal_observations"] = adapter._process_observation(goal_obs, task.language)
                    result = adapter.policy.call_endpoint("predict_oracle", request)
                    if result["task"].strip() != task.language.strip():
                        raise ValueError("Policy task differs from simulator task")
                    chunk = adapter._convert_to_libero_action(result.pop("action"))
                    if chunk.shape != (16, 7) or not np.isfinite(chunk).all():
                        raise ValueError("Expected a finite 16x7 chunk")
                    result.pop("normalized_actions")
                    query = dict(
                        control_step=step,
                        anchor_frame=anchor if reference is not None else None,
                        goal_frame=goal if reference is not None else None,
                        phase_error=phase_error,
                        **result,
                    )
                    queries.append(query)
                    log.write(json.dumps(query, allow_nan=False) + "\n")
                    log.flush()
                    queue.extend(chunk)
                command = np.asarray(queue.popleft()).copy()
                image, wrist = get_libero_image(obs)
                goal_image = (
                    get_libero_image(reference_observation(reference, goal))[0]
                    if reference is not None
                    else np.zeros_like(image)
                )
                frame = np.pad(np.concatenate((image, wrist, goal_image), axis=1), ((32, 0), (0, 0), (0, 0)))
                cv2.putText(
                    frame,
                    f"{config['suite']} init={initial_index} mode={mode} t={step} ref={anchor} goal={goal}",
                    (5, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                )
                writer.append_data(frame)
                actions.append(command)
                obs, _, done, _ = env.step(command.tolist())
                success = bool(done)
        writer.close()
        writer = None
        (directory / "rollout.tmp.mp4").replace(directory / "rollout.mp4")
        data = {key: np.asarray(values) for key, values in arrays.items() if values}
        data.update(
            actions=np.asarray(actions), phase_keys=np.asarray(keys), initial_sim_sha256=np.asarray(initial_hash)
        )
        if reference is not None:
            data["d_ee"] = np.asarray(distances)
        with (directory / "trace.tmp.npz").open("wb") as handle:
            np.savez_compressed(handle, **data)
        (directory / "trace.tmp.npz").replace(directory / "trace.npz")
        status = predicates(env)
        return dict(
            complete=True,
            eligible=True,
            case="action_reference" if reference is None else "goal_oracle",
            suite=config["suite"],
            task_id=config["task_id"],
            task=task.language,
            initial_index=initial_index,
            success=success,
            control_steps=len(actions),
            max_steps=config["max_steps"],
            initial_sha256=array_hash(initial),
            initial_sim_sha256=initial_hash,
            final_predicates=status,
            predicate_completion=sum(status.values()) / len(status) if status else None,
            wall_seconds=time.perf_counter() - start,
            queries=queries,
            video="rollout.mp4",
            privileged_goal=reference is not None,
            phase_selector="monotone local nearest EE/object positions/gripper; +16 future frame"
            if reference is not None
            else None,
            smoke_only=config.get("smoke", False),
        )
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main(config):
    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    task = suite.get_task(config["task_id"])
    if task.language != config["task"]:
        raise ValueError("Task differs from trained goal-control inventory")
    adapter = GR00TPolicy(host="127.0.0.1", port=config["port"], headless=True, inference_timeout_ms=120000)
    try:
        states = suite.get_task_init_states(config["task_id"])
        for initial_index in range(config["trials"]):
            references = (
                Path(config["output"])
                / "action_reference"
                / f"{config['suite']}_task{config['task_id']:02d}_init{initial_index:03d}"
            )
            goals = Path(config["output"]) / "goal_oracle" / references.name
            for directory in (references, goals):
                directory.mkdir(parents=True, exist_ok=True)

            def completed(directory):
                path = directory / "episode.json"
                if not path.exists():
                    return None
                row = json.loads(path.read_text())
                if row["signature"] != config["signature"]:
                    raise ValueError("Resume signature differs")
                if row.get("skipped") or (
                    (directory / "trace.npz").exists() and (directory / "rollout.mp4").exists()
                ):
                    return row
                raise ValueError("Completed episode artifacts missing")

            base = completed(references)
            if base is None:
                base = rollout(config, adapter, task, states[initial_index], initial_index, references)
                base["signature"] = config["signature"]
                atomic_json(references / "episode.json", base)
                print(f"COMPLETE {references} success={base['success']}", flush=True)
            if completed(goals) is not None:
                continue
            if not base["success"] and not config.get("smoke", False):
                row = dict(
                    complete=True,
                    skipped=True,
                    eligible=False,
                    success=False,
                    case="goal_oracle",
                    suite=config["suite"],
                    task_id=config["task_id"],
                    initial_index=initial_index,
                    reason="Mode-1 reference did not complete; not a valid task-trajectory oracle",
                    signature=config["signature"],
                )
            else:
                with np.load(references / "trace.npz", allow_pickle=False) as archive:
                    reference = {key: archive[key] for key in archive.files}
                row = rollout(config, adapter, task, states[initial_index], initial_index, goals, reference)
                row.update(
                    signature=config["signature"], reference=str(references), eligible=bool(base["success"])
                )
            atomic_json(goals / "episode.json", row)
            print(f"COMPLETE {goals} success={row['success']} skipped={row.get('skipped', False)}", flush=True)
    finally:
        adapter.policy.socket.close(linger=0)
        adapter.policy.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
