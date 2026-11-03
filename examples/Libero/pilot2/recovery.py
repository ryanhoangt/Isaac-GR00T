"""Recovery-only runner: reuse weights/index, audit paired starts, resume per episode."""

import csv
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .recovery_report import audit_pairs
from .screen import _json, _run, _write_json
from .task_mapping import validate_retrieval_provenance
from .tracking import PROJECT


def preflight(args):
    args.perturb_scale = getattr(args, "perturb_scale", 1.0)
    if not np.isfinite(args.perturb_scale) or args.perturb_scale <= 0:
        raise ValueError("--perturb-scale must be finite and positive")
    if args.smoke:
        args.task_ids, args.trials, args.initial_start, args.max_steps = [0, 4], 1, 5, 110
        args.video_trials = min(args.video_trials, 1)
    for key in ("checkpoint", "index", "split", "output", "libero_repo"):
        setattr(args, key, getattr(args, key).resolve())
    if (
        not args.task_ids
        or len(set(args.task_ids)) != len(args.task_ids)
        or any(t not in range(10) for t in args.task_ids)
    ):
        raise ValueError("--task-ids must be unique values in 0..9")
    if not args.triggers or len(set(args.triggers)) != len(args.triggers):
        raise ValueError("--triggers must be unique values from 25 and 60")
    if args.initial_start < 0 or args.initial_start + args.trials > 50:
        raise ValueError("Official initial-state range must be within 0..49")
    if not 0 <= args.video_trials <= args.trials:
        raise ValueError("--video-trials must be between zero and --trials")
    if args.max_steps < max(args.triggers) + 4 + 32:
        raise ValueError("Action budget must include the full 32-action recovery window")
    if not args.smoke and args.max_steps < 1000:
        raise ValueError("Scientific recovery runs require at least 1000 actions; use --smoke for short debugging")
    split = _json(args.split)
    recipe = _json(args.checkpoint / "experiment_cfg/pilot2.json")
    state = _json(args.checkpoint / "trainer_state.json")
    if recipe["variant"] != "dual" or recipe["split_id"] != split["split_id"]:
        raise ValueError("Recovery needs the dual checkpoint from this split")
    if recipe["action_horizon"] != 16 or recipe["denoising_steps_eval"] != 4:
        raise ValueError("Expected horizon 16 and four Euler steps")
    weights = _json(args.checkpoint / "model.safetensors.index.json")
    if any(not (args.checkpoint / name).is_file() for name in set(weights["weight_map"].values())):
        raise ValueError("Checkpoint weight shards are missing")
    metadata = _json(args.index.with_suffix(".json"))
    if (
        metadata["split_id"] != split["split_id"]
        or metadata["encoder"] != recipe["base_model"]
        or Path(metadata["reference"]).resolve() != args.split.parent / "reference"
    ):
        raise ValueError("Reference index split, visual encoder or dataset root differs")
    if metadata["stride"] != 1 and not args.smoke:
        raise ValueError("Scientific recovery uses the complete stride-1 reference index")
    episode_tasks = {item["episode_index"]: item["task_id"] for item in split["reference"]}
    if any(sum(task == t for task in episode_tasks.values()) != 3 for t in range(10)):
        raise ValueError("Expected three held-out reference demonstrations per task")
    with np.load(args.index) as index:
        episodes, tasks = index["episodes"], index["tasks"]
        if len(episodes) != metadata["frames"] or len(tasks) != len(episodes):
            raise ValueError("Index array lengths disagree with its metadata")
        if set(map(int, episodes)) != set(episode_tasks):
            raise ValueError("Index is not exactly the held-out reference episode pool")
        if any(episode_tasks[int(e)] != int(t) for e, t in zip(episodes, tasks)):
            raise ValueError("Index task labels disagree with the split")
    if not (args.libero_repo / "libero/libero/bddl_files").is_dir():
        raise ValueError("LIBERO checkout not found")
    digest = hashlib.sha256()
    with args.index.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    settings = {
        key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
        for key in (
            "checkpoint",
            "index",
            "split",
            "device",
            "task_ids",
            "triggers",
            "trials",
            "initial_start",
            "max_steps",
            "seed",
            "video_trials",
            "libero_repo",
            "smoke",
        )
    }
    settings.update(
        protocol_version=1,
        split_id=split["split_id"],
        checkpoint_step=state["global_step"],
        training_recipe=recipe,
        index_metadata=metadata,
        index_sha256=digest.hexdigest(),
    )
    if args.perturb_scale != 1.0:
        settings["perturb_scale"] = args.perturb_scale
    return settings


def build_stages(args):
    cli = [sys.executable, "-u", "-m", "examples.Libero.pilot2"]
    stages = []
    for trigger in args.triggers:
        for strategy in "ABC":
            label = f"strategy_{strategy}_t{trigger}"
            output = args.output / label
            expected = {
                "checkpoint": str(args.checkpoint),
                "split": str(args.split),
                "index": str(args.index),
                "output": str(output),
                "device": args.device,
                "mode": "language",
                "strategy": strategy,
                "trigger": trigger,
                "task_ids": args.task_ids,
                "trials": args.trials,
                "initial_start": args.initial_start,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "save_video": bool(args.video_trials),
                "video_trials": args.video_trials if args.video_trials else None,
                "libero_repo": str(args.libero_repo),
            }
            if getattr(args, "perturb_scale", 1.0) != 1.0:
                expected["perturb_scale"] = args.perturb_scale
            command = cli + ["evaluate"]
            for key, value in expected.items():
                if key in ("save_video", "video_trials"):
                    continue
                command += ["--" + key.replace("_", "-")]
                command += list(map(str, value)) if isinstance(value, list) else [str(value)]
            if args.video_trials:
                command += ["--save-video", "--video-trials", str(args.video_trials)]
            stages.append((label, output, command, expected))
    report = args.output / "report"
    command = cli + [
        "recovery-report",
        "--inputs",
        *[str(path) for _, path, _, _ in stages],
        "--output",
        str(report),
    ]
    if args.smoke:
        command.append("--allow-smoke")
    return stages, command


def completed(output, expected):
    path = output / "manifest.json"
    if not path.is_file():
        if output.exists():
            raise ValueError(f"Output has no recovery manifest: {output}")
        return False
    manifest = _json(path)
    if manifest.get("recovery_protocol_version") != 1 or any(
        manifest["settings"].get(k) != v for k, v in expected.items()
    ):
        raise ValueError(f"Existing recovery run has different settings/protocol: {output}")
    if manifest["status"] != "completed":
        return False
    names = {
        f"task{task:02d}_init{initial:03d}"
        for task in expected["task_ids"]
        for initial in range(expected["initial_start"], expected["initial_start"] + expected["trials"])
    }
    if set(manifest["episodes"]) != names or len(manifest["episodes"]) != len(names):
        raise ValueError(f"Recovery manifest is missing requested episodes: {output}")
    for name in names:
        row = _json(output / f"{name}.json")
        validate_retrieval_provenance(manifest, row)
        if (
            row["strategy"] != expected["strategy"]
            or row["trigger"] != expected["trigger"]
            or row["seed"] != expected["seed"] + row["task_id"] * 10000 + row["initial_state_index"] * 1000
        ):
            raise ValueError(f"Recovery episode settings differ from the plan: {name}")
        if not row["complete"] or not (output / f"{name}.npz").is_file():
            raise ValueError(f"Recovery episode is incomplete: {name}")
        if (
            expected["save_video"]
            and row["initial_state_index"] < expected["initial_start"] + expected["video_trials"]
        ):
            if not (output / f"{name}.mp4").is_file():
                raise ValueError(f"Requested recovery video is missing: {name}")
    return True


def track_stage(run, label, output):
    if run is None:
        return
    manifest = _json(output / "manifest.json")
    rows = [_json(output / f"{name}.json") for name in manifest["episodes"]]
    eligible = [row for row in rows if row["recovery_eligible"]]
    metrics = {
        f"{label}/episodes": len(rows),
        f"{label}/success_pct": np.mean([r["success"] for r in rows]) * 100,
        f"{label}/fully_perturbed": len(eligible),
        f"{label}/excluded": len(rows) - len(eligible),
    }
    measured = [r for r in eligible if r.get("perturbation_command_offset_m") is not None]
    if measured:
        metrics[f"{label}/clipped_trial_pct"] = 100 * np.mean(
            [any(p["clipped"] for p in r["perturbation_commands"]) for r in measured]
        )
        metrics[f"{label}/requested_offset_cm"] = 100 * np.mean(
            [np.linalg.norm(r["perturbation_target_m"]) for r in measured]
        )
        metrics[f"{label}/delivered_command_offset_cm"] = 100 * np.mean(
            [np.linalg.norm(r["perturbation_command_offset_m"]) for r in measured]
        )
    if eligible:
        metrics[f"{label}/recovery_pct"] = np.mean([r["success"] for r in eligible]) * 100
    run.log(metrics)
    run.summary.update(metrics)


def recover(args):
    settings = preflight(args)
    stages, report_command = build_stages(args)
    total = len(stages) * len(args.task_ids) * args.trials
    print(
        json.dumps(
            {
                "settings": settings,
                "rollouts": total,
                "stages": [s[0] for s in stages],
                "wandb_project": PROJECT,
                "new_training": False,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for _, _, command, _ in stages:
            print("$ " + " ".join(command))
        print("$ " + " ".join(report_command))
        return
    plan_path = args.output / "recovery_plan.json"
    if args.output.exists():
        if not args.resume or not plan_path.is_file():
            raise FileExistsError(f"Output exists; use --resume for a matching recovery run: {args.output}")
        plan = _json(plan_path)
        if plan["settings"] != settings:
            raise ValueError("Recovery settings/checkpoint/index changed; use a new output directory")
    else:
        args.output.mkdir(parents=True)
        plan = {"settings": settings, "status": "running", "stages": {}}
        _write_json(plan_path, plan)
    with (args.output / ".recovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another recovery launcher is using this output directory") from exc
        _execute(args, settings, stages, report_command, plan, plan_path, total)


def _execute(args, settings, stages, report_command, plan, plan_path, total):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("USE_TF", "0")
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        os.environ.update(WANDB_PROJECT=PROJECT, WANDB_DISABLED="false")
        run = wandb.init(
            project=PROJECT,
            entity=args.wandb_entity,
            group=args.output.name,
            name=f"pilot2_recovery_step{settings['checkpoint_step']}"
            + (f"_scale{args.perturb_scale:g}x" if args.perturb_scale != 1 else ""),
            id=wandb.util.generate_id(),
            resume="never",
            mode=args.wandb_mode,
            dir=str(args.output),
            config=settings,
            save_code=False,
            tags=["pilot2", "libero-long", "recovery", f"perturb-scale-{args.perturb_scale:g}x"]
            + (["smoke"] if args.smoke else []),
        )
        _write_json(
            args.output / "wandb_run.json", {"id": run.id, "url": run.url, "project": PROJECT, "status": "running"}
        )
        print(f"W&B: {run.url}", flush=True)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise KeyboardInterrupt("Recovery launcher received SIGTERM")

    signal.signal(signal.SIGTERM, stop)
    plan["status"] = "running"
    plan.pop("error", None)
    exit_code = 1
    current = None
    completed_rollouts = 0
    try:
        for label, output, command, expected in stages:
            current = label
            if completed(output, expected):
                print(f"{label}: already completed; skipping", flush=True)
            else:
                if output.exists():
                    command = command + ["--resume"]
                plan["stages"][label] = {"status": "running", "started_utc": time.time(), "command": command}
                _write_json(plan_path, plan)

                def progress(line):
                    if run is not None and line.startswith("task") and ": success=" in line:
                        count = len(_json(output / "manifest.json")["episodes"])
                        run.log(
                            {
                                "progress/completed_rollouts": completed_rollouts + count,
                                "progress/total_rollouts": total,
                                "progress/current_stage": label,
                            }
                        )

                _run(command, args.output / "logs" / f"{label}.log", on_line=progress)
                if not completed(output, expected):
                    raise RuntimeError(f"Recovery stage {label} exited without complete output")
                plan["stages"][label].update(status="completed", finished_utc=time.time())
                _write_json(plan_path, plan)
            track_stage(run, label, output)
            completed_rollouts += len(args.task_ids) * args.trials
            if expected["strategy"] == "C":
                trigger_inputs = [p for _, p, _, e in stages if e["trigger"] == expected["trigger"]]
                _, audit = audit_pairs(trigger_inputs, args.smoke)
                _write_json(args.output / f"pairing_t{expected['trigger']}.json", audit)
        current = "report"
        report = args.output / "report"
        required_report = (
            "summary.md",
            "summary.csv",
            "pairing_audit.json",
            "paired_trials.csv",
            "paired_contraction.png",
            "paired_comparisons.csv",
            "paired_contraction.csv",
            "recovery_success.png",
            "latency_by_phase.csv",
            "disturbance_delivery.csv",
            "disturbance_metadata.json",
        )
        report_complete = all((report / name).is_file() for name in required_report)
        if not report_complete:
            if report.exists():
                report_command.append("--resume")
            _run(report_command, args.output / "logs/report.log")
            if not all((report / name).is_file() for name in required_report):
                raise RuntimeError("Recovery report exited without all required artifacts")
        else:
            print("report: already completed; skipping", flush=True)
        plan["stages"]["report"] = {"status": "completed", "finished_utc": time.time()}
        if run is not None:
            import wandb

            for name in ("recovery_success", "paired_contraction"):
                run.log({f"recovery/{name}": wandb.Image(str(report / f"{name}.png"))})
            with (report / "paired_comparisons.csv").open() as stream:
                for row in csv.DictReader(stream):
                    if row["net_gain_pp"]:
                        run.summary[f"trigger{row['trigger']}/{row['comparison']}/net_gain_pp"] = float(
                            row["net_gain_pp"]
                        )
        plan["status"] = "completed"
        exit_code = 0
        print(f"Recovery complete: {report / 'summary.md'}", flush=True)
    except BaseException as exc:
        plan.update(status="failed", error=repr(exc))
        if current in plan["stages"]:
            plan["stages"][current]["status"] = "failed"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        _write_json(plan_path, plan)
        if run is not None:
            run.finish(exit_code=exit_code)
            record = _json(args.output / "wandb_run.json")
            record["status"] = "finished" if exit_code == 0 else "failed"
            _write_json(args.output / "wandb_run.json", record)
