"""LIBERO-Long support-gate evaluation, with episode-local RPC state (Python 3.8)."""

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

from examples.Libero.eval.run_dual_diagnosis import atomic_json, plain, predicates
from examples.Libero.eval.run_libero_eval import GR00TPolicy
from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image
from gr00t.rag.diagnosis_protocol import array_digest, digest, instruction_key, query_seed


def rollout(env, initial, task, initial_index, adapter, config, directory, save_video):
    case, task_id = config["case"], config["task_id"]
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    queue, state, queries = deque(), {}, []
    actions, modes, positions, distances = [], [], [], []
    reference = np.load(config["reference_ee"], allow_pickle=False)
    writer = (
        imageio.get_writer(str(directory / "rollout.tmp.mp4"), fps=20, macro_block_size=1) if save_video else None
    )
    success, prediction = False, None
    started = time.perf_counter()
    try:
        with (directory / "queries.jsonl").open("w") as log:
            for step in range(config["max_steps"] + 1):
                position = np.asarray(obs["robot0_eef_pos"]).copy()
                positions.append(position)
                distances.append(float(np.linalg.norm(reference - position, axis=1).min()))
                if success or step == config["max_steps"]:
                    break
                if not queue:
                    request = dict(
                        observations=adapter._process_observation(obs, task.language),
                        episode_id=digest(
                            dict(run=config["signature"], case=case, task=task_id, initial=initial_index)
                        ),
                        strategy=case["strategy"],
                        variant=case["variant"],
                        gate_state=state,
                        flow_seed=query_seed(config["seed"], task_id, initial_index, step),
                    )
                    before = time.perf_counter()
                    result = adapter.policy.call_endpoint("get_action_support", request)
                    if instruction_key(result["task"]) != instruction_key(task.language):
                        raise ValueError("Reference and simulator tasks differ")
                    chunk = adapter._convert_to_libero_action(result.pop("action"))
                    if chunk.shape != (16, 7) or not np.isfinite(chunk).all():
                        raise ValueError("Expected finite 16x7 actions")
                    state = result["gate_state"]
                    prediction = dict(control_step=step, rpc_ms=(time.perf_counter() - before) * 1000, **result)
                    queries.append(prediction)
                    log.write(json.dumps(prediction, default=plain, allow_nan=False) + "\n")
                    log.flush()
                    queue.extend(chunk)
                action = np.asarray(queue.popleft()).copy()
                mode = prediction["mode"]
                if writer is not None:
                    image, wrist = get_libero_image(obs)
                    frame = np.pad(np.concatenate((image, wrist), axis=1), ((48, 0), (0, 0), (0, 0)))
                    labels = [
                        f"t={step} mode={mode} event={prediction['gate_event']}",
                        f"d={prediction['support_fp32']:.4f} enter={prediction['threshold_enter']:.4f} exit={prediction['threshold_exit']:.4f}",
                        f"nearest={prediction['nearest_episode']}:{prediction['nearest_frame']} goal={prediction['retrieved_episode']}:{prediction['goal_frame']}",
                    ]
                    for line, text in enumerate(labels):
                        cv2.putText(
                            frame,
                            text,
                            (5, 13 + 15 * line),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            (255, 180, 80) if mode == "goal" else (130, 240, 180),
                            1,
                        )
                    writer.append_data(frame)
                actions.append(action)
                modes.append(int(mode == "goal"))
                obs, _, done, _ = env.step(action.tolist())
                success = bool(done)
    finally:
        if writer is not None:
            writer.close()
    if save_video:
        (directory / "rollout.tmp.mp4").replace(directory / "rollout.mp4")
    with (directory / "trace.npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            ee=np.asarray(positions),
            d_ee=np.asarray(distances),
            executed_actions=np.asarray(actions),
            goal_mode=np.asarray(modes, dtype=np.int8),
        )
    (directory / "trace.npz.tmp").replace(directory / "trace.npz")
    status = predicates(env)
    return dict(
        complete=True,
        case=case,
        task_id=task_id,
        task=task.language,
        initial_index=initial_index,
        initial_sha256=array_digest(initial),
        success=success,
        control_steps=len(actions),
        goal_control_steps=sum(modes),
        action_control_steps=len(modes) - sum(modes),
        smoke_only=config["max_steps"] < 1000,
        wall_seconds=time.perf_counter() - started,
        final_predicates=status,
        predicate_completion=sum(status.values()) / len(status) if status else None,
        queries=queries,
        video="rollout.mp4" if save_video else None,
    )


def main(config):
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(config["task_id"])
    config["reference_ee"] = config["references"][instruction_key(task.language)]
    initials = suite.get_task_init_states(config["task_id"])
    if config["initial_start"] + config["trials"] > len(initials):
        raise ValueError("Initial-state range exceeds benchmark")
    env = adapter = None
    try:
        for initial in range(config["initial_start"], config["initial_start"] + config["trials"]):
            directory = (
                Path(config["output"]) / config["case"]["name"] / f"task{config['task_id']:02d}_init{initial:03d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            signature = digest(
                dict(run=config["signature"], case=config["case"], task=config["task_id"], initial=initial)
            )
            result_path = directory / "episode.json"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get("signature") != signature:
                    raise ValueError("Episode resume signature differs")
                if (
                    old.get("complete")
                    and (directory / "trace.npz").exists()
                    and (not old["video"] or (directory / old["video"]).exists())
                ):
                    print(f"RESUMED {directory}", flush=True)
                    continue
            if env is None:
                env, _ = get_libero_env(task, resolution=256)
                adapter = GR00TPolicy(
                    host="127.0.0.1", port=config["port"], headless=True, inference_timeout_ms=120000
                )
            result = rollout(
                env,
                initials[initial],
                task,
                initial,
                adapter,
                config,
                directory,
                initial - config["initial_start"] < config["video_trials"],
            )
            result["signature"] = signature
            atomic_json(result_path, result)
            print(
                f"COMPLETE {directory} success={result['success']} goal_steps={result['goal_control_steps']}",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()
        if adapter is not None:
            adapter.policy.socket.close(linger=0)
            adapter.policy.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
