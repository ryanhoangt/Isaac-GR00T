"""LIBERO-side diagnosis worker (also compatible with the Python 3.8 simulator)."""

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
from gr00t.rag.diagnosis_protocol import (
    array_digest,
    digest,
    disturbance,
    flush_queue,
    instruction_key,
    perturb_action,
    phase,
    query_seed,
)


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


def predicates(env):
    domain = env.env
    goals = domain.parsed_problem.get("goal_state", [])
    if not goals or not hasattr(domain, "_eval_predicate"):
        return None
    return {json.dumps(goal): bool(domain._eval_predicate(goal)) for goal in goals}


def rollout(env, initial, task, task_id, initial_index, adapter, config, directory, save_video):
    case = config["case"]
    seed = query_seed(config["seed"], task_id, initial_index)
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    queue = deque()
    fixed_context, recovery_context = None, None
    base, target = disturbance(seed, case.get("scale", 0))
    positions, distances, actions, nominal, perturbations, queries = [], [], [], [], [], []
    reference = np.load(config["reference_ee"], allow_pickle=False)
    start = case["trigger"] + 4 if case["kind"] == "recovery" else None
    pre_hash = recovery_hash = None
    success = False
    video_path = directory / "rollout.mp4"
    temporary_video = directory / "rollout.tmp.mp4"
    writer = imageio.get_writer(str(temporary_video), fps=20, macro_block_size=1) if save_video else None
    prediction = None
    started = time.perf_counter()
    try:
        with (directory / "queries.jsonl").open("w") as log:
            for step in range(config["max_steps"] + 1):
                position = np.asarray(obs["robot0_eef_pos"]).copy()
                positions.append(position)
                distances.append(float(np.linalg.norm(reference - position, axis=1).min()))
                if case["kind"] == "recovery" and step == case["trigger"]:
                    pre_hash = array_digest(env.get_sim_state())
                if step == start:
                    recovery_hash = array_digest(env.get_sim_state())
                if success or step == config["max_steps"]:
                    break
                if flush_queue(case, step):
                    queue.clear()
                current_phase = phase(case, step)
                if not queue:
                    mode = "action" if current_phase == "perturb" else current_phase
                    request = {
                        "observations": adapter._process_observation(obs, task.language),
                        "mode": mode,
                        "flow_seed": query_seed(config["seed"], task_id, initial_index, step),
                    }
                    if current_phase == "goal" and case["kind"] == "recovery" and fixed_context is not None:
                        request.update(fixed_context)
                    query_started = time.perf_counter()
                    result = adapter.policy.call_endpoint("get_action_diagnosis", request)
                    if instruction_key(result["task"]) != instruction_key(task.language):
                        raise ValueError("Retrieved context task differs from the simulator task")
                    chunk = adapter._convert_to_libero_action(result.pop("action"))
                    if chunk.shape != (16, 7) or not np.isfinite(chunk).all():
                        raise ValueError("Expected a finite 16x7 action chunk")
                    prediction = {
                        "control_step": step,
                        "phase": current_phase,
                        "rpc_ms": (time.perf_counter() - query_started) * 1000,
                        **result,
                    }
                    queries.append(prediction)
                    log.write(json.dumps(prediction, default=plain, allow_nan=False) + "\n")
                    log.flush()
                    queue.extend(chunk)
                    if step == start:
                        recovery_context = {
                            key: result[key] for key in ("anchor_index", "goal_index", "retrieved_episode")
                        }
                    if current_phase == "goal" and case["kind"] == "recovery":
                        fixed_context = {key: result[key] for key in ("anchor_index", "goal_index")}
                command = np.asarray(queue.popleft()).copy()
                nominal.append(command.copy())
                if current_phase == "perturb":
                    command, delivered = perturb_action(command, target, env.env.robots[0].controller)
                    perturbations.append({"control_step": step, **delivered})
                if writer is not None:
                    image, wrist = get_libero_image(obs)
                    frame = np.pad(np.concatenate((image, wrist), axis=1), ((32, 0), (0, 0), (0, 0)))
                    color = (255, 90, 50) if current_phase == "perturb" else (100, 230, 150)
                    cv2.putText(
                        frame,
                        f"t={step} phase={current_phase} support={prediction['support_fp32']:.4f}",
                        (5, 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        color,
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"ref demo={prediction['retrieved_episode']} frame={prediction['retrieved_frame']} goal={prediction['goal_frame']}",
                        (5, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (240, 240, 240),
                        1,
                    )
                    writer.append_data(frame)
                actions.append(command.copy())
                obs, _, done, _ = env.step(command.tolist())
                success = bool(done)
    finally:
        if writer is not None:
            writer.close()
    if save_video:
        temporary_video.replace(video_path)
    with (directory / "trace.npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            ee=np.asarray(positions),
            d_ee=np.asarray(distances),
            executed_actions=np.asarray(actions),
            nominal_actions=np.asarray(nominal),
        )
    (directory / "trace.npz.tmp").replace(directory / "trace.npz")
    eligible = len(perturbations) == 4
    contraction = (
        {
            str(delta): distances[start + delta] if eligible and start + delta < len(distances) else None
            for delta in (0, 8, 16, 32)
        }
        if start is not None
        else {}
    )
    status = predicates(env)
    return {
        "complete": True,
        "case": case,
        "task_id": task_id,
        "task": task.language,
        "initial_index": initial_index,
        "success": success,
        "control_steps": len(actions),
        "smoke_only": config["max_steps"] < 1000,
        "wall_seconds": time.perf_counter() - started,
        "recovery_eligible": eligible,
        "perturbation_commands": perturbations,
        "base_offset_m": base.tolist(),
        "requested_offset_m": target.tolist(),
        "observed_ee_motion_m": float(np.linalg.norm(positions[start] - positions[case["trigger"]]))
        if eligible
        else None,
        "initial_sha256": array_digest(initial),
        "pre_perturb_sim_sha256": pre_hash,
        "recovery_sim_sha256": recovery_hash,
        "prefix_actions_sha256": array_digest(actions[:start]) if start is not None else None,
        "prefix_ee_sha256": array_digest(positions[: start + 1]) if start is not None else None,
        "recovery_context": recovery_context,
        "contraction": contraction,
        "final_predicates": status,
        "predicate_completion": sum(status.values()) / len(status) if status else None,
        "queries": queries,
        "video": "rollout.mp4" if save_video else None,
    }


def main(config):
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_id = config["task_id"]
    task = suite.get_task(task_id)
    references = config["references"]
    key = instruction_key(task.language)
    if key not in references:
        raise ValueError(f"No unique reference mapping for {task.language}")
    config["reference_ee"] = references[key]
    env, _ = get_libero_env(task, resolution=256)
    adapter = GR00TPolicy(host="127.0.0.1", port=config["port"], headless=True, inference_timeout_ms=120000)
    initial_states = suite.get_task_init_states(task_id)
    try:
        for initial_index in range(config["initial_start"], config["initial_start"] + config["trials"]):
            if initial_index >= len(initial_states):
                raise ValueError("Requested initial-state range exceeds benchmark states")
            directory = (
                Path(config["output"]) / config["case"]["name"] / f"task{task_id:02d}_init{initial_index:03d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            expected = digest(
                {"run": config["signature"], "case": config["case"], "task": task_id, "initial": initial_index}
            )
            result_path = directory / "episode.json"
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get("signature") != expected:
                    raise ValueError(f"Resume signature differs: {directory}")
                if (
                    old.get("complete")
                    and (directory / "trace.npz").exists()
                    and (not old["video"] or (directory / old["video"]).exists())
                ):
                    print(f"RESUMED {directory}", flush=True)
                    continue
            result = rollout(
                env,
                initial_states[initial_index],
                task,
                task_id,
                initial_index,
                adapter,
                config,
                directory,
                initial_index - config["initial_start"] < config["video_trials"],
            )
            result["signature"] = expected
            atomic_json(result_path, result)
            print(f"COMPLETE {directory} success={result['success']} steps={result['control_steps']}", flush=True)
    finally:
        env.close()
        adapter.policy.socket.close(linger=0)
        adapter.policy.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
