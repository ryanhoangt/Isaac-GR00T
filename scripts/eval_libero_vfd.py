"""Record GR00T velocity disagreement, calibrate thresholds, and plot rollouts.

See examples/Libero/VFD.md. Commands are intentionally separate: calibration
must use different official initial states from the test recordings.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

REPO = Path(os.environ.get("GR00T_REPO", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(REPO))

from scripts.libero_vfd import (  # noqa: E402
    SnapshotVFDPolicy,
    calibrate,
    detection_metrics,
    flow_grid,
    get_threshold,
    load_run,
)

SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "long": "libero_10"}
LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 600, "libero_10": 1000}
DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_libero(output, renderer):
    os.environ["MUJOCO_GL"] = renderer
    os.environ["PYOPENGL_PLATFORM"] = renderer
    for key, value in {
        "USE_TF": "0",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
    }.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("HF_MODULES_CACHE", str(REPO / ".cache/huggingface/modules"))
    libero = Path(os.environ.get("LIBERO_REPO", REPO.parent / "LIBERO")).resolve()
    if not (libero / "libero/libero/assets").is_dir():
        raise FileNotFoundError("LIBERO assets missing; set LIBERO_REPO to a complete checkout")
    sys.path.insert(0, str(libero))
    # Never rewrite ~/.libero or trigger asset downloads during monitoring.
    if "LIBERO_CONFIG_PATH" not in os.environ:
        base = libero / "libero/libero"
        config_dir = output / ".libero"
        config_dir.mkdir()
        save_json(
            config_dir / "config.yaml",
            {
                "benchmark_root": str(base),
                "assets": str(base / "assets"),
                "bddl_files": str(base / "bddl_files"),
                "init_states": str(base / "init_files"),
                "datasets": str(base.parent / "datasets"),
            },
        )
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def record(args):
    actor = args.actor.expanduser().resolve(strict=True)
    scorer = args.scorer.expanduser().resolve(strict=True)
    manifest_path = actor.parent / "run_manifest.json"
    training_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    suite = SUITES.get(args.suite, args.suite)
    if suite == "auto":
        suite = SUITES.get(training_manifest.get("held_out_suite"))
    if suite not in LIMITS:
        raise ValueError("pass --suite spatial|object|goal|long, or use an actor with a training manifest")
    if actor == scorer:
        raise ValueError("actor and scorer must differ")
    if min(args.trials, args.num_action_samples, args.snapshot_stride) < 1 or args.initial_state_start < 0:
        raise ValueError("trials/sample count/snapshot stride must be positive; initial state must be nonnegative")
    max_steps = args.max_steps or LIMITS[suite]
    if not 1 <= max_steps <= LIMITS[suite] or args.wait_steps < 0:
        raise ValueError("invalid action horizon/wait steps")
    times = [float(t) for t in args.flow_times.split(",")] if args.flow_times else None
    flow_grid(args.denoising_steps, times)
    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids != "all" else list(range(10))
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(t not in range(10) for t in task_ids):
        raise ValueError("task IDs must be distinct values from 0 through 9")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    settings = {
        "actor": str(actor),
        "scorer": str(scorer),
        "suite": suite,
        "task_ids": task_ids,
        "trials": args.trials,
        "initial_state_start": args.initial_state_start,
        "seed": args.seed,
        "uncertainty_seed": args.uncertainty_seed,
        "denoising_steps": args.denoising_steps,
        "num_action_samples": args.num_action_samples,
        "flow_times": times,
        "max_action_steps": max_steps,
        "wait_steps": args.wait_steps,
        "renderer": args.renderer,
        "data_config": args.data_config,
        "calibration": str(args.calibration.resolve()) if args.calibration else None,
        "gpu_memory_limit_gb": args.gpu_memory_limit_gb,
        "verify_actor": args.verify_actor,
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return
    output.mkdir(parents=True)
    run = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "smoke_only": max_steps < LIMITS[suite],
        "ensemble_kind": "checkpoint_pair; same-run epochs are snapshot disagreement, not independent seeds",
    }
    save_json(output / "manifest.json", run)
    prepare_libero(output, args.renderer)
    import torch
    from transformers import set_seed
    from libero.libero import benchmark
    from examples.Libero.eval.run_libero_eval import GR00TPolicy
    from examples.Libero.eval.utils import get_libero_env, get_libero_dummy_action, get_libero_image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run record in a GPU allocation")
    if args.gpu_memory_limit_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.gpu_memory_limit_gb * 1024**3 / total
        if not 0 < fraction <= 1:
            raise ValueError("GPU memory limit must be positive and no larger than total VRAM")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    suite_object = benchmark.get_benchmark_dict()[suite]()
    states = {task: suite_object.get_task_init_states(task) for task in task_ids}
    for task in task_ids:
        if args.initial_state_start + args.trials > len(states[task]):
            raise ValueError(f"task {task} only has {len(states[task])} official initial states")
    monitor = SnapshotVFDPolicy(
        actor,
        scorer,
        args.data_config,
        args.denoising_steps,
        args.num_action_samples,
        times,
        args.uncertainty_seed,
    )
    run["score_signature"] = monitor.signature
    run["status"] = "running"
    save_json(output / "manifest.json", run)
    # Model construction must not shift the actor's inference RNG between runs.
    set_seed(args.seed)

    class LocalAdapter(GR00TPolicy):
        def __init__(self):
            self.headless = True
            self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            self.verified = False

        def act(self, observation, language):
            processed = self._process_observation(observation, language)
            reference = None
            if args.verify_actor and not self.verified:
                # Reference inference must not consume the monitored actor's RNG.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    reference = monitor.actor.get_action(copy.deepcopy(processed))
            chunk, score, sample_scores = monitor.get_action_with_vfd(processed)
            if reference is not None:
                error = max(float(np.max(np.abs(chunk[key] - reference[key]))) for key in reference)
                for key in reference:
                    np.testing.assert_allclose(
                        chunk[key],
                        reference[key],
                        rtol=1e-3,
                        atol=1e-4,
                        err_msg=f"monitor changed actor output {key}",
                    )
                run["actor_parity_max_abs"] = error
                self.verified = True
                print(f"Actor parity PASS: max absolute action difference={error:.6g}", flush=True)
            return self._convert_to_libero_action(chunk, 0), score, sample_scores

    adapter = LocalAdapter()
    # Validate every planned threshold and split before beginning simulation.
    if calibration:
        for task in task_ids:
            for index in range(args.initial_state_start, args.initial_state_start + args.trials):
                get_threshold(
                    calibration, monitor.signature, {"suite": suite, "task_id": task, "initial_state_index": index}
                )
    try:
        for task_id in task_ids:
            task = suite_object.get_task(task_id)
            task_dir = output / f"task{task_id:02d}"
            task_dir.mkdir()
            env, _ = get_libero_env(task, resolution=256, horizon=max_steps + args.wait_steps)
            try:
                env.seed(args.seed)
                for index in range(args.initial_state_start, args.initial_state_start + args.trials):
                    prefix = task_dir / f"episode{index:03d}"
                    metadata = {
                        "suite": suite,
                        "task_id": task_id,
                        "language": task.language,
                        "initial_state_index": index,
                        "seed": args.seed,
                        "success": False,
                        "complete": False,
                    }
                    threshold = get_threshold(calibration, monitor.signature, metadata) if calibration else None
                    metadata["threshold"] = threshold
                    scores, actions, sample_values, frames, frame_steps = [], [], [], [], []
                    video = None
                    last_frame = None
                    try:
                        if args.save_video:
                            import imageio

                            video = imageio.get_writer(str(prefix.with_suffix(".mp4")), fps=20)
                        env.reset()
                        obs = env.set_init_state(states[task_id][index])
                        for _ in range(args.wait_steps):
                            obs, _, _, _ = env.step(get_libero_dummy_action())
                        with prefix.with_suffix(".jsonl").open("w") as stream:
                            for step in range(max_steps):
                                image, wrist = get_libero_image(obs)
                                last_frame = np.hstack((image, wrist))
                                if step % args.snapshot_stride == 0:
                                    frames.append(last_frame.copy())
                                    frame_steps.append(step)
                                if video:
                                    video.append_data(last_frame)
                                started = time.monotonic()
                                action, score, samples = adapter.act(obs, task.language)
                                elapsed = time.monotonic() - started
                                scores.append(score)
                                actions.append(action)
                                sample_values.append(samples)
                                flagged = threshold is not None and score > threshold
                                event = {
                                    "step": step,
                                    "uncertainty": score,
                                    "threshold": threshold,
                                    "alarm": flagged,
                                    "inference_seconds": elapsed,
                                }
                                stream.write(json.dumps(event) + "\n")
                                stream.flush()
                                if flagged and "first_alarm_step" not in metadata:
                                    metadata["first_alarm_step"] = step
                                    print(
                                        f"ALARM task={task_id} init={index} step={step} score={score:.6g} threshold={threshold:.6g}",
                                        flush=True,
                                    )
                                obs, _, done, _ = env.step(action.tolist())
                                if done:
                                    metadata["success"] = True
                                    break
                        metadata["complete"] = True
                    finally:
                        if video:
                            video.close()
                        if scores and (not frame_steps or frame_steps[-1] != len(scores) - 1):
                            frames.append(last_frame)
                            frame_steps.append(len(scores) - 1)
                        metadata["action_steps"] = len(scores)
                        np.savez_compressed(
                            prefix.with_suffix(".npz"),
                            uncertainty=np.asarray(scores),
                            actions=np.asarray(actions),
                            sample_scores=np.asarray(sample_values),
                            frames=np.asarray(frames, dtype=np.uint8),
                            frame_steps=np.asarray(frame_steps),
                        )
                        save_json(prefix.with_suffix(".json"), metadata)
                    print(
                        f"task={task_id} init={index} success={metadata['success']} steps={len(scores)} peak_vfd={max(scores):.6g}",
                        flush=True,
                    )
                    plot_run(output, calibration, skip_existing=True)
            finally:
                env.close()
        run["status"] = "completed"
    except BaseException as exc:
        run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        run["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        save_json(output / "manifest.json", run)
        plot_run(output, calibration, skip_existing=True)


def plot_run(run_path, calibration=None, normalize=False, skip_existing=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_path = Path(run_path)
    manifest, episodes = load_run(run_path)
    destination = run_path / "plots"
    destination.mkdir(exist_ok=True)
    by_task, descriptions = {}, []
    for path, metadata, scores in episodes:
        threshold = (
            get_threshold(calibration, manifest["score_signature"], metadata)
            if calibration
            else metadata.get("threshold")
        )
        alarms = np.flatnonzero(scores > threshold) if threshold is not None else []
        first = int(alarms[0]) if len(alarms) else None
        label = "success" if metadata["success"] else "failure"
        if manifest.get("smoke_only") and not metadata["success"]:
            label = "not successful within shortened smoke horizon"
        stem = destination / f"{path.parent.name}_{path.stem}"
        if not skip_existing or not (stem.with_suffix(".png").exists() and stem.with_suffix(".pdf").exists()):
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                frames = arrays["frames"] if "frames" in arrays else []
                frame_steps = arrays["frame_steps"] if "frame_steps" in arrays else []
                selected = (
                    np.unique(np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int))
                    if len(frames)
                    else []
                )
                figure = plt.figure(figsize=(13, 6 if len(selected) else 4))
                grid = figure.add_gridspec(
                    2 if len(selected) else 1,
                    max(len(selected), 1),
                    height_ratios=[2, 1] if len(selected) else [1],
                )
                axis = figure.add_subplot(grid[0, :])
                divisor = threshold if normalize and threshold is not None and threshold > 0 else 1
                axis.plot(np.arange(len(scores)), scores / divisor, label="VFD", color="tab:blue")
                if threshold is not None:
                    axis.axhline(threshold / divisor, linestyle="--", color="black", label="calibrated threshold")
                if first is not None:
                    axis.axvline(first, linestyle=":", color="tab:red", label=f"first alarm: {first}")
                axis.set(
                    xlabel="Action-generation step",
                    ylabel="VFD / threshold" if divisor != 1 else "VFD score",
                    title=f"{metadata['suite']} task {metadata['task_id']}, initial state {metadata['initial_state_index']}: {label}",
                )
                axis.grid(alpha=0.2)
                axis.legend()
                for column, index in enumerate(selected):
                    image_axis = figure.add_subplot(grid[1, column])
                    image_axis.imshow(frames[index])
                    image_axis.set_title(f"step {int(frame_steps[index])}", fontsize=9)
                    image_axis.axis("off")
                figure.tight_layout()
                figure.savefig(stem.with_suffix(".png"), dpi=150)
                figure.savefig(stem.with_suffix(".pdf"))
                plt.close(figure)
        descriptions.append(
            {
                **metadata,
                "first_alarm_step": first,
                "peak_vfd": float(max(scores)),
                "threshold": threshold,
                "plot": str(stem.with_suffix(".png").relative_to(run_path)),
            }
        )
        if threshold is not None:
            by_task.setdefault(f"{metadata['suite']}/task{metadata['task_id']:02d}", []).append(
                (metadata["success"], scores, threshold)
            )
    summary = {
        "status": manifest["status"],
        "smoke_only": manifest.get("smoke_only", False),
        "episodes": descriptions,
        "metrics_by_task": {key: detection_metrics(values) for key, values in by_task.items()}
        if not manifest.get("smoke_only")
        else {},
        "note": "No deployment metrics for truncated smoke horizons. Null metrics mean one outcome class is absent.",
    }
    save_json(run_path / "summary.json", summary)
    print(f"Plots: {destination}; results: {run_path / 'summary.json'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("record", help="record online VFD without stopping on alarms")
    command.add_argument("--actor", type=Path, required=True)
    command.add_argument("--scorer", type=Path, required=True)
    command.add_argument("--suite", default="auto", choices=["auto", *SUITES, *LIMITS])
    command.add_argument("--task-ids", default="all", help="comma-separated task IDs, or all")
    command.add_argument("--trials", type=int, default=5)
    command.add_argument("--initial-state-start", type=int, default=0)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--uncertainty-seed", type=int, default=1234)
    command.add_argument("--denoising-steps", type=int, default=8)
    command.add_argument("--num-action-samples", type=int, default=1)
    command.add_argument(
        "--flow-times", default=None, help="comma-separated times on the inference grid; default all"
    )
    command.add_argument("--data-config", default=DATA_CONFIG)
    command.add_argument(
        "--max-steps", type=int, default=None, help="shorter horizons are smoke tests, excluded from calibration"
    )
    command.add_argument("--wait-steps", type=int, default=10)
    command.add_argument("--renderer", choices=["egl", "osmesa"], default=os.environ.get("MUJOCO_GL", "egl"))
    command.add_argument(
        "--gpu-memory-limit-gb",
        type=float,
        default=0,
        help="PyTorch allocator cap in GiB, not a complete GPU-memory reservation",
    )
    command.add_argument("--snapshot-stride", type=int, default=20)
    command.add_argument("--save-video", action="store_true")
    command.add_argument(
        "--verify-actor",
        action="store_true",
        help="compare the first action chunk against original inference with the same RNG",
    )
    command.add_argument("--calibration", type=Path)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--dry-run", action="store_true")
    command = commands.add_parser("calibrate", help="fit threshold from complete successful calibration rollouts")
    command.add_argument("--inputs", nargs="+", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--quantile", type=float, default=0.9)
    command.add_argument("--min-successes", type=int, default=20)
    command.add_argument("--scope", choices=["task", "global"], default="task")
    command = commands.add_parser("plot", help="plot saved rollout scores; no model/GPU required")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--calibration", type=Path)
    command.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "record":
        try:
            record(args)
        except BaseException as exc:
            path = args.output_dir / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                save_json(path, manifest)
            raise
    elif args.command == "calibrate":
        result = calibrate(args.inputs, args.output, args.quantile, args.min_successes, args.scope)
        print(json.dumps(result["thresholds"], indent=2))
    else:
        calibration = json.loads(args.calibration.read_text()) if args.calibration else None
        plot_run(args.run_dir, calibration, args.normalize)


if __name__ == "__main__":
    main()
