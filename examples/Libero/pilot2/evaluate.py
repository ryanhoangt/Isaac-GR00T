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


def array_sha256(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def evaluation_settings(args):
    scale = getattr(args, "perturb_scale", 1.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("--perturb-scale must be finite and positive")
    settings = {
        key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    settings.pop("resume", None)
    # Keep the settings schema of existing 1x runs so an in-flight launcher
    # can start its remaining subprocesses or resume without changing protocol.
    if scale == 1.0:
        settings.pop("perturb_scale", None)
    settings["checkpoint"] = str(Path(args.checkpoint).resolve())
    settings["split"] = str(Path(args.split).resolve())
    if args.index:
        settings["index"] = str(Path(args.index).resolve())
    return settings


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
    reference_task_id=None,
    perturb_scale=1.0,
):
    from examples.Libero.eval.utils import get_libero_image, normalize_gripper_action

    schedule = RecoverySchedule(strategy, trigger)
    reference_task_id = task_id if reference_task_id is None else reference_task_id
    env.reset()
    observation = env.set_init_state(initial)
    for _ in range(10):
        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    queue = deque()
    perturbation = sample_disturbance(np.random.default_rng(seed), perturb_scale) if trigger is not None else None
    fixed_goal = None
    positions, distances, predictions, predicates, goals, video = [], [], [], [], [], []
    success, perturb_applied = False, 0
    executed, nominal, perturb_log = [], [], []
    recovery_sim_hash = None
    pre_perturb_sim_hash = None
    initial_hash = None if initial is None else array_sha256(initial)
    started = time.perf_counter()
    current_goal = None
    for step in range(max_steps + 1):
        position = np.asarray(observation["robot0_eef_pos"]).copy()
        positions.append(position)
        distances.append(reference.distance(position, reference_task_id) if reference is not None else np.nan)
        predicates.append(predicate_status(env))
        if step == trigger and hasattr(env, "get_sim_state"):
            pre_perturb_sim_hash = array_sha256(env.get_sim_state())
        if step == schedule.recovery_start and hasattr(env, "get_sim_state"):
            recovery_sim_hash = array_sha256(env.get_sim_state())
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
                retrieval = (
                    fixed_goal if fixed_goal is not None else reference.retrieve(query_image, reference_task_id)
                )
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
                "retrieval_cached": bool(strategy == "C" and phase == "both" and step != schedule.recovery_start),
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
        nominal.append(action.copy())
        if phase == "perturb":
            before = action.copy()
            action = disturbed_command(action, perturbation, env.robots[0].controller)
            controller = env.robots[0].controller
            gain = (np.asarray(controller.output_max)[:3] - np.asarray(controller.output_min)[:3]) / (
                np.asarray(controller.input_max)[:3] - np.asarray(controller.input_min)[:3]
            )
            # Compare with the nominal command after the controller's input clipping.
            baseline_command = np.clip(
                before[:3], np.asarray(controller.input_min)[:3], np.asarray(controller.input_max)[:3]
            )
            added = action[:3] - baseline_command
            perturb_log.append(
                {
                    "step": step,
                    "added_command": added.tolist(),
                    "added_delta_target_m": (added * gain).tolist(),
                    "clipped": bool(not np.allclose(added * gain, perturbation / 4, atol=1e-8)),
                }
            )
            perturb_applied += 1
        executed.append(action.copy())
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
        "task_language": task.language,
        "reference_task_id": reference_task_id,
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
        "pre_perturb_sim_sha256": pre_perturb_sim_hash,
        "pre_perturb_ee_sha256": array_sha256(positions[: trigger + 1])
        if trigger is not None and count >= trigger
        else None,
        "pre_perturb_actions_sha256": array_sha256(executed[:trigger])
        if trigger is not None and count >= trigger
        else None,
        "initial_state_sha256": initial_hash,
        "pre_recovery_sim_sha256": recovery_sim_hash,
        "pre_recovery_actions_sha256": array_sha256(executed[: schedule.recovery_start])
        if perturb_applied == 4
        else None,
        "perturbation_target_m": None if perturbation is None else perturbation.tolist(),
        "perturbation_scale": perturb_scale,
        "perturbation_base_target_m": None if perturbation is None else (perturbation / perturb_scale).tolist(),
        "perturbation_measured_displacement_m": measured,
        "perturbation_commands": perturb_log,
        "perturbation_command_offset_m": np.sum([p["added_delta_target_m"] for p in perturb_log], axis=0).tolist()
        if perturb_log
        else None,
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
    return result, {
        "ee_positions": np.asarray(positions),
        "reference_distance_m": np.asarray(distances),
        "executed_actions": np.asarray(executed),
        "nominal_actions": np.asarray(nominal),
    }


def evaluate(args):
    import os
    import sys
    import yaml
    from .training import load_policy
    from .runtime import ReferenceBuffer, VisionFeatures
    from .task_mapping import build_task_mapping, validate_retrieval_provenance

    output = Path(args.output)
    settings = evaluation_settings(args)
    perturb_scale = getattr(args, "perturb_scale", 1.0)
    manifest_path = output / "manifest.json"
    previous = None
    if output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError(f"Evaluation output exists; pass --resume for matching runs: {output}")
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            raise ValueError("Cannot resume evaluation with different settings or checkpoint")
    else:
        output.mkdir(parents=True)
    libero = Path(args.libero_repo).resolve()
    if not (libero / "libero/libero/bddl_files").is_dir():
        raise ValueError(f"LIBERO checkout not found: {libero}")
    sys.path.insert(0, str(libero))
    if "LIBERO_CONFIG_PATH" not in os.environ:
        config_dir = output / "libero_config"
        config_dir.mkdir(exist_ok=True)
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
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_mapping = build_task_mapping(suite, Path(args.split).resolve().parent / "reference", args.task_ids)
    print("Instruction-based task mapping: " + json.dumps(task_mapping, sort_keys=True), flush=True)
    if previous is not None and previous.get("task_mapping") != task_mapping:
        raise ValueError("Task mapping changed or is missing in the old evaluation; use a new output directory")
    if previous is not None and args.strategy != "standard" and previous.get("recovery_protocol_version") != 1:
        raise ValueError("Recovery protocol changed; use a new output directory")
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
    if reference is not None and reference.root.resolve() != (Path(args.split).resolve().parent / "reference"):
        raise ValueError("Reference index points to a different dataset than the instruction-based task mapping")
    if reference is not None and reference.signature["encoder"] != policy.pilot2_recipe["base_model"]:
        raise ValueError("Reference index was built using another base visual encoder")
    adapter = GR00TPolicy.__new__(GR00TPolicy)
    adapter.headless = True
    manifest = {
        "status": "running",
        "split_id": split["split_id"],
        "training_recipe": policy.pilot2_recipe,
        "task_mapping_version": 1,
        "task_mapping": task_mapping,
        "reference_episode_tasks": {str(item["episode_index"]): item["task_id"] for item in split["reference"]},
        "settings": settings,
        "control": {"horizon": 16, "actions_per_query": 16, "denoising_steps": 4},
        "recovery_protocol_version": 1,
        "perturbation_protocol": {
            "trigger_clock": "executed control actions after 10 stabilization actions",
            "duration_steps": 4,
            "target_norm_m": [0.05 * perturb_scale, 0.08 * perturb_scale],
            "component_bounds_m": [-0.07 * perturb_scale, 0.07 * perturb_scale],
            "B_copy_steps": 16,
            "C_goal_steps": 32,
            "goal": "one same-task milestone held fixed for two chunks",
            "observed_displacement": "EE motion during perturbation, includes nominal policy motion; not a causal disturbance displacement",
        },
        "initial_states": "official LIBERO initial states; exclusion from demos is not verified",
        "subtask_metric": "final fraction of BDDL task predicates; not retrieved-frame success",
        "episodes": [] if previous is None else previous["episodes"],
    }
    if perturb_scale != 1.0:
        manifest["perturbation_protocol"]["scale"] = perturb_scale
    if previous is not None and (
        previous["split_id"] != split["split_id"] or previous["training_recipe"] != policy.pilot2_recipe
    ):
        raise ValueError("Cannot resume evaluation with a different split or training recipe")

    def save_manifest():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
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
                    video_requested = args.save_video and (
                        args.video_trials is None or initial_id < args.initial_start + args.video_trials
                    )
                    if name in manifest["episodes"]:
                        result_path = output / f"{name}.json"
                        array_path = output / f"{name}.npz"
                        if not result_path.is_file() or not array_path.is_file():
                            raise ValueError(f"Manifest contains incomplete episode: {name}")
                        completed = json.loads(result_path.read_text())
                        if (
                            not completed["complete"]
                            or completed["task_id"] != task_id
                            or completed["initial_state_index"] != initial_id
                            or completed["seed"] != args.seed + task_id * 10000 + initial_id * 1000
                            or (video_requested and not (output / f"{name}.mp4").is_file())
                        ):
                            raise ValueError(f"Manifest contains mismatched episode: {name}")
                        print(f"{name}: already complete; skipping", flush=True)
                        continue
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
                        save_video=output / f"{name}.mp4" if video_requested else None,
                        reference_task_id=task_mapping[str(task_id)]["reference_task_id"],
                        perturb_scale=perturb_scale,
                    )
                    result["initial_state_index"] = initial_id
                    validate_retrieval_provenance(manifest, result)
                    (output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    np.savez_compressed(output / f"{name}.npz", **arrays)
                    manifest["episodes"].append(name)
                    save_manifest()
                    print(f"{name}: success={result['success']}; steps={result['action_steps']}", flush=True)
            finally:
                env.close()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        save_manifest()
