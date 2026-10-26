"""Matched LIBERO-Long standard and perturb/recover rollouts."""

import json
import hashlib
import time
from collections import deque
from pathlib import Path

import numpy as np

from .data import LANGUAGE_KEY
from .runtime import RecoverySchedule, action_matrix, disturbed_command, sample_disturbance


def synchronize(policy):
    import torch

    if policy.model.device.type == "cuda":
        torch.cuda.synchronize(policy.model.device)


def predict(policy, observation, seed):
    import torch

    # Reusing a seed gives common initial flow noise in counterfactual probes
    # and identical pre-perturbation policies in A/B/C.
    devices = [policy.model.device] if policy.model.device.type == "cuda" else []
    synchronize(policy)
    start = time.perf_counter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        action = action_matrix(policy.get_action(observation))
    synchronize(policy)
    return action, (time.perf_counter() - start) * 1000


def condition(observation, mode, goal=None):
    result = dict(observation)
    if mode == "vanilla":
        return result
    result["goal_mode_id"] = np.array([int(mode != "language")], dtype=np.int64)
    if mode != "language":
        if goal is None:
            raise ValueError("Goal modes require a reference milestone")
        result["subgoal.image"] = np.asarray(goal)[None]
    if mode == "goal":
        result[LANGUAGE_KEY] = [""]
    return result


def predicate_status(env):
    # These are BDDL task-goal predicates, not labels for the retrieved frame.
    domain = env.env
    goals = domain.parsed_problem.get("goal_state", [])
    if not goals or not hasattr(domain, "_eval_predicate"):
        return None
    return {json.dumps(goal): bool(domain._eval_predicate(goal)) for goal in goals}


def rollout(
    env,
    initial,
    task,
    task_id,
    adapter,
    policy,
    reference,
    mode="language",
    strategy="standard",
    trigger=None,
    seed=0,
    max_steps=1000,
    save_video=None,
    goal_ee_threshold=0.03,
):
    from examples.Libero.eval.utils import get_libero_image, normalize_gripper_action

    schedule = RecoverySchedule(strategy, trigger)
    env.reset()
    observation = env.set_init_state(initial)
    for _ in range(10):
        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    queue = deque()
    perturbation = sample_disturbance(np.random.default_rng(seed)) if trigger is not None else None
    fixed_goal = None
    positions, distances, predictions, predicates, goals, video = [], [], [], [], [], []
    success, perturb_applied = False, 0
    started = time.perf_counter()
    current_goal = None
    for step in range(max_steps + 1):
        position = np.asarray(observation["robot0_eef_pos"]).copy()
        positions.append(position)
        distances.append(reference.distance(position, task_id) if reference is not None else np.nan)
        predicates.append(predicate_status(env))
        if current_goal is not None:
            current_goal["min_ee_distance"] = min(
                current_goal["min_ee_distance"], float(np.linalg.norm(position - current_goal["position"]))
            )
        if success or step == max_steps:
            break
        if save_video is not None:
            image, wrist = get_libero_image(observation)
            video.append(np.concatenate((image, wrist), axis=1))
        if schedule.invalidate_queue(step):
            queue.clear()
            if step == schedule.recovery_start:
                fixed_goal = None
            else:
                current_goal = None
        phase = schedule.phase(step)
        effective_mode = mode if strategy == "standard" else ("both" if phase == "both" else "language")
        if not queue:
            retrieval = None
            retrieval_ms = 0.0
            if effective_mode in ("goal", "both") or phase == "copy":
                query_image, _ = get_libero_image(observation)
                start = time.perf_counter()
                retrieval = fixed_goal if fixed_goal is not None else reference.retrieve(query_image, task_id)
                if strategy == "C":
                    fixed_goal = retrieval
                retrieval_ms = (time.perf_counter() - start) * 1000
            if phase == "copy":
                chunk, latency = retrieval["actions"].copy(), None
                # Raw reference actions use the same gripper convention as the
                # model's denormalized outputs, so conversion is shared below.
            else:
                raw = adapter._process_observation(observation, task.language)
                chunk, latency = predict(
                    policy,
                    condition(raw, effective_mode, None if retrieval is None else retrieval["goal_image"]),
                    seed + step,
                )
            queue.extend(normalize_gripper_action(chunk.copy(), binarize=True))
            row = {
                "step": step,
                "mode": effective_mode,
                "phase": phase,
                "latency_ms": latency,
                "retrieval_ms": retrieval_ms,
            }
            if retrieval is not None:
                row.update({key: retrieval[key] for key in ("episode", "frame", "goal_frame", "similarity")})
                current_goal = {
                    "step": step,
                    "episode": retrieval["episode"],
                    "frame": retrieval["goal_frame"],
                    "position": retrieval["goal_position"].tolist(),
                    "min_ee_distance": float("inf"),
                }
                goals.append(current_goal)
            predictions.append(row)
        action = np.asarray(queue.popleft()).copy()
        if phase == "perturb":
            action = disturbed_command(action, perturbation, env.robots[0].controller)
            perturb_applied += 1
        observation, _, done, _ = env.step(action.tolist())
        success = bool(done)
    count = len(positions) - 1
    contraction = {}
    if schedule.recovery_start is not None and perturb_applied == 4:
        # Delta=0 is the state immediately AFTER the four perturbation actions.
        for delta in (0, 8, 16, 32):
            idx = schedule.recovery_start + delta
            contraction[str(delta)] = distances[idx] if idx <= count else None
    measured = None
    if trigger is not None and perturb_applied == 4:
        measured = (positions[schedule.recovery_start] - positions[trigger]).tolist()
    final_predicates = predicates[-1]
    prefix_hash = None
    if trigger is not None and perturb_applied == 4:
        prefix_hash = hashlib.sha256(
            np.asarray(positions[: schedule.recovery_start + 1], dtype=np.float64).tobytes()
        ).hexdigest()
    result = {
        "task_id": task_id,
        "mode": mode,
        "strategy": strategy,
        "trigger": trigger,
        "seed": seed,
        "success": success,
        "action_steps": count,
        "complete": bool(success or count == max_steps),
        "smoke_only": max_steps < 1000,
        "perturbation_applied_steps": perturb_applied,
        "recovery_eligible": perturb_applied == 4,
        "pre_recovery_ee_sha256": prefix_hash,
        "perturbation_target_m": None if perturbation is None else perturbation.tolist(),
        "perturbation_measured_displacement_m": measured,
        "contraction_m": contraction,
        "final_predicates": final_predicates,
        "predicate_history": predicates,
        "subtask_predicate_fraction": None
        if final_predicates is None
        else float(np.mean(list(final_predicates.values()))),
        "goal_ee_proxy_threshold_m": goal_ee_threshold,
        "goal_ee_proxy_rate": None
        if not goals
        else float(np.mean([item["min_ee_distance"] <= goal_ee_threshold for item in goals])),
        "goals": goals,
        "predictions": predictions,
        "wall_seconds": time.perf_counter() - started,
    }
    if save_video is not None and video:
        import imageio.v2 as imageio

        imageio.mimsave(str(save_video), video, fps=20)
    return result, {"ee_positions": np.asarray(positions), "reference_distance_m": np.asarray(distances)}


def evaluate(args):
    import os
    import sys
    import yaml
    from .training import load_policy
    from .runtime import ReferenceBuffer, VisionFeatures

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    libero = Path(args.libero_repo).resolve()
    if not (libero / "libero/libero/bddl_files").is_dir():
        raise ValueError(f"LIBERO checkout not found: {libero}")
    sys.path.insert(0, str(libero))
    if "LIBERO_CONFIG_PATH" not in os.environ:
        config_dir = output / "libero_config"
        config_dir.mkdir()
        root = libero / "libero/libero"
        config = {
            "benchmark_root": str(root),
            "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"),
            "assets": str(root / "assets"),
            "datasets": str(libero / "libero/datasets"),
        }
        (config_dir / "config.yaml").write_text(yaml.safe_dump(config))
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    elif not (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").is_file():
        raise ValueError("Explicit LIBERO_CONFIG_PATH must contain config.yaml")
    os.environ.setdefault("MUJOCO_GL", "egl" if args.device.startswith("cuda") else "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    from libero.libero import benchmark
    from examples.Libero.eval.run_libero_eval import GR00TPolicy
    from examples.Libero.eval.utils import get_libero_env

    split = json.loads(Path(args.split).read_text())
    policy = load_policy(args.checkpoint, split["split_id"], args.device)
    goal_enabled = getattr(policy.model.action_head.config, "goal_conditioning", False)
    if args.strategy != "standard" and not goal_enabled:
        raise ValueError("A/B/C use the dual model with a common language-only prefix")
    if (args.mode == "vanilla") == goal_enabled:
        raise ValueError("vanilla requires a baseline checkpoint; other modes require a dual checkpoint")
    requires_reference = args.mode in ("both", "goal") or args.strategy != "standard"
    if requires_reference and not args.index:
        raise ValueError("Goal/perturbation evaluation requires --index")
    reference = ReferenceBuffer(args.index, split["split_id"], VisionFeatures(policy)) if args.index else None
    if reference is not None and reference.signature["encoder"] != policy.pilot2_recipe["base_model"]:
        raise ValueError("Reference index was built using another base visual encoder")
    adapter = GR00TPolicy.__new__(GR00TPolicy)
    adapter.headless = True
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    manifest = {
        "status": "running",
        "split_id": split["split_id"],
        "training_recipe": policy.pilot2_recipe,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "control": {"horizon": 16, "actions_per_query": 16, "denoising_steps": 4},
        "initial_states": "official LIBERO initial states; exclusion from demos is not verified",
        "subtask_metric": "final fraction of BDDL task predicates; not retrieved-frame success",
        "episodes": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            initials = suite.get_task_init_states(task_id)
            if args.initial_start < 0 or args.initial_start + args.trials > len(initials):
                raise ValueError("Requested initial-state range is unavailable")
            # The underlying robosuite budget must include the stabilization steps.
            env, _ = get_libero_env(task, horizon=args.max_steps + 10)
            try:
                for initial_id in range(args.initial_start, args.initial_start + args.trials):
                    env.seed(args.seed)
                    name = f"task{task_id:02d}_init{initial_id:03d}"
                    result, arrays = rollout(
                        env,
                        initials[initial_id],
                        task,
                        task_id,
                        adapter,
                        policy,
                        reference,
                        args.mode,
                        args.strategy,
                        args.trigger,
                        seed=args.seed + task_id * 10000 + initial_id * 1000,
                        max_steps=args.max_steps,
                        save_video=output / f"{name}.mp4" if args.save_video else None,
                    )
                    result["initial_state_index"] = initial_id
                    (output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    np.savez_compressed(output / f"{name}.npz", **arrays)
                    manifest["episodes"].append(name)
                    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    print(f"{name}: success={result['success']}; steps={result['action_steps']}", flush=True)
            finally:
                env.close()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
