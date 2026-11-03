"""Reproducible, resumable first-pass evaluation of a matched Pilot 2 pair."""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .tracking import PROJECT
from .task_mapping import validate_retrieval_provenance


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _preflight(args):
    split_path = args.split.resolve()
    split = _json(split_path)
    root = args.training_root.resolve()
    checkpoints = {name: root / name / f"checkpoint-{args.checkpoint_step}" for name in ("baseline", "dual")}
    recipes = {}
    for name, checkpoint in checkpoints.items():
        recipe_path = checkpoint / "experiment_cfg/pilot2.json"
        if not recipe_path.is_file() or not (checkpoint / "model.safetensors.index.json").is_file():
            raise ValueError(f"Incomplete {name} checkpoint: {checkpoint}")
        recipe = _json(recipe_path)
        state = _json(checkpoint / "trainer_state.json")
        if state["global_step"] != args.checkpoint_step:
            raise ValueError(f"{name} checkpoint step does not match --checkpoint-step")
        if recipe["variant"] != name or recipe["split_id"] != split["split_id"]:
            raise ValueError(f"{name} checkpoint variant or split does not match")
        recipes[name] = recipe
    comparable = (
        "base_model",
        "seed",
        "batch_size",
        "num_gpus",
        "max_steps",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "scheduler",
        "normalization",
        "action_horizon",
        "denoising_steps_eval",
        "frozen_vision",
        "frozen_llm",
    )
    for key in comparable:
        if recipes["baseline"][key] != recipes["dual"][key]:
            raise ValueError(f"Checkpoints differ on {key}")
    if recipes["dual"]["action_horizon"] != 16 or recipes["dual"]["denoising_steps_eval"] != 4:
        raise ValueError("Pilot 2 evaluation requires 16 actions and four Euler steps")
    if len(split["reference"]) != 30 or split["reference_per_task"] != 3:
        raise ValueError("Expected three held-out reference demos per each of ten tasks")
    if {item["task_id"] for item in split["reference"]} != set(range(10)):
        raise ValueError("Reference split does not cover all ten tasks")
    train_ids = {item["original_episode_index"] for item in split["train"]}
    reference_ids = {item["original_episode_index"] for item in split["reference"]}
    if train_ids & reference_ids:
        raise ValueError("Train and reference demonstrations overlap")
    if args.initial_start < 0 or args.initial_start + args.trials > 50:
        raise ValueError("Requested official initial-state range must be within 0..49")
    if (
        not args.task_ids
        or len(set(args.task_ids)) != len(args.task_ids)
        or any(t not in range(10) for t in args.task_ids)
    ):
        raise ValueError("--task-ids must be unique values in 0..9")
    if args.video_trials < 0 or args.video_trials > args.trials:
        raise ValueError("--video-trials must be between zero and --trials")
    if not (args.libero_repo.resolve() / "libero/libero/bddl_files").is_dir():
        raise ValueError("LIBERO checkout not found")
    return split, checkpoints, recipes


def _run(command, log, on_line=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(map(str, command)), flush=True)
    with log.open("a") as stream:
        stream.write("\n$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
                if on_line is not None:
                    on_line(line)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Evaluation stage failed ({code}); see {log}")


def _completed(stage, output, split, recipe, args):
    if stage == "index":
        sidecar = output.with_suffix(".json")
        if not output.is_file() or not sidecar.is_file():
            return False
        metadata = _json(sidecar)
        if (metadata["split_id"], metadata["encoder"], metadata["stride"]) != (
            split["split_id"],
            recipe["base_model"],
            args.stride,
        ):
            raise ValueError("Existing index has different split, encoder or stride")
        return True
    if stage == "sensitivity":
        sidecar = output.with_suffix(".json")
        if not output.is_file() or not sidecar.is_file():
            return False
        metadata = _json(sidecar)
        if metadata["split_id"] != split["split_id"] or metadata["requested_states"] != args.states:
            raise ValueError("Existing sensitivity file has different split or sample count")
        return True
    if stage == "report":
        return (output / "summary.csv").is_file() and (output / "summary.md").is_file()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _json(manifest_path)
    if manifest["split_id"] != split["split_id"]:
        raise ValueError(f"Existing {stage} run uses another split")
    settings = manifest["settings"]
    expected_checkpoint = (
        args.training_root.resolve()
        / ("baseline" if stage == "vanilla" else "dual")
        / f"checkpoint-{args.checkpoint_step}"
    )
    recovery = stage.startswith("strategy_")
    expected_mode = "language" if recovery else stage
    expected_strategy = stage.split("_")[1] if recovery else "standard"
    expected_trigger = int(stage.rsplit("t", 1)[1]) if recovery else None
    if (
        settings["checkpoint"] != str(expected_checkpoint)
        or settings["mode"] != expected_mode
        or settings["strategy"] != expected_strategy
        or settings["trigger"] != expected_trigger
        or settings["trials"] != args.trials
        or settings["initial_start"] != args.initial_start
        or settings["max_steps"] != args.max_steps
        or settings["seed"] != args.seed
        or settings["task_ids"] != args.task_ids
    ):
        raise ValueError(f"Existing {stage} run has different evaluation settings")
    if manifest["status"] != "completed":
        return False
    expected = {
        f"task{task:02d}_init{initial:03d}"
        for task in args.task_ids
        for initial in range(args.initial_start, args.initial_start + args.trials)
    }
    if set(manifest["episodes"]) != expected or len(manifest["episodes"]) != len(expected):
        raise ValueError(f"Existing {stage} run claims completion without all requested episodes")
    if any(not (output / f"{name}.json").is_file() or not (output / f"{name}.npz").is_file() for name in expected):
        raise ValueError(f"Existing {stage} run is missing episode files")
    for name in expected:
        validate_retrieval_provenance(manifest, _json(output / f"{name}.json"))
    return True


def _reuse_sources(source, split, recipe, args):
    source = source.resolve()
    for stage, name in (
        ("index", "reference.npz"),
        ("sensitivity", "sensitivity.csv"),
        ("vanilla", "vanilla"),
        ("language", "language"),
    ):
        if not _completed(stage, source / name, split, recipe, args):
            raise ValueError(f"Cannot reuse incomplete {stage} from {source}")
    return [
        source / name
        for name in (
            "reference.npz",
            "reference.json",
            "sensitivity.csv",
            "sensitivity.json",
            "vanilla",
            "language",
        )
    ]


def _log_stage(run, stage, output):
    if run is None:
        return
    if stage == "report":
        with (output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["task_id"] == "all":
                label = row["condition"]
                for key in (
                    "success_pct",
                    "subtask_predicate_pct",
                    "goal_ee_proxy_pct",
                    "recovery_pct",
                    "latency_ms_mean",
                ):
                    if row[key]:
                        run.summary[f"{label}/{key}"] = float(row[key])
        run.log({"eval/success_plot": __import__("wandb").Image(str(output / "success.png"))})
        run.log({"eval/sensitivity_plot": __import__("wandb").Image(str(output / "sensitivity_latency.png"))})
    elif stage == "sensitivity":
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        for key in ("goal_vs_null", "goal_vs_random", "random_vs_null", "language_latency_ms", "goal_latency_ms"):
            run.summary[f"sensitivity/{key}"] = sum(float(row[key]) for row in rows) / len(rows)
    elif stage in ("vanilla", "language", "goal", "both") or stage.startswith("strategy_"):
        manifest = _json(output / "manifest.json")
        successes = [int(_json(output / f"{name}.json")["success"]) for name in manifest["episodes"]]
        run.summary[f"{stage}/episodes"] = len(successes)
        run.summary[f"{stage}/success_pct"] = 100 * sum(successes) / len(successes)


def screen(args):
    if args.smoke:
        args.stride, args.states, args.task_ids = 64, 4, [0]
        args.trials, args.initial_start, args.max_steps = 1, 0, 110 if args.recovery else 80
        args.video_trials = min(args.video_trials, 1)
    split, checkpoints, recipes = _preflight(args)
    reused = _reuse_sources(args.reuse_screen, split, recipes["dual"], args) if args.reuse_screen else []
    output = args.output.resolve()
    settings = {
        "split": str(args.split.resolve()),
        "split_id": split["split_id"],
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "checkpoint_step": args.checkpoint_step,
        "stride": args.stride,
        "states": args.states,
        "task_ids": args.task_ids,
        "trials": args.trials,
        "initial_start": args.initial_start,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "video_trials": args.video_trials,
        "libero_repo": str(args.libero_repo.resolve()),
        "smoke": args.smoke,
        "recovery": args.recovery,
        "reuse_screen": str(args.reuse_screen.resolve()) if args.reuse_screen else None,
    }
    python = sys.executable
    cli = [python, "-u", "-m", "examples.Libero.pilot2"]
    index = output / "reference.npz"
    sensitivity = output / "sensitivity.csv"
    common = ["--split", str(args.split.resolve())]
    stages = [
        (
            "index",
            index,
            cli
            + [
                "index",
                "--checkpoint",
                str(checkpoints["dual"]),
                *common,
                "--output",
                str(index),
                "--stride",
                str(args.stride),
            ],
        ),
        (
            "sensitivity",
            sensitivity,
            cli
            + [
                "sensitivity",
                "--checkpoint",
                str(checkpoints["dual"]),
                *common,
                "--index",
                str(index),
                "--output",
                str(sensitivity),
                "--states",
                str(args.states),
                "--seed",
                str(args.seed),
            ],
        ),
    ]
    for mode in ("vanilla", "language", "goal", "both"):
        stage_output = output / mode
        command = cli + [
            "evaluate",
            "--checkpoint",
            str(checkpoints["baseline" if mode == "vanilla" else "dual"]),
            *common,
            "--output",
            str(stage_output),
            "--mode",
            mode,
            "--task-ids",
            *map(str, args.task_ids),
            "--trials",
            str(args.trials),
            "--initial-start",
            str(args.initial_start),
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--libero-repo",
            str(args.libero_repo.resolve()),
        ]
        if mode in ("goal", "both"):
            command += ["--index", str(index)]
        if args.video_trials:
            command += ["--save-video", "--video-trials", str(args.video_trials)]
        stages.append((mode, stage_output, command))
    if args.recovery:
        for trigger in (25, 60):
            for strategy in "ABC":
                label = f"strategy_{strategy}_t{trigger}"
                stage_output = output / label
                command = cli + [
                    "evaluate",
                    "--checkpoint",
                    str(checkpoints["dual"]),
                    *common,
                    "--index",
                    str(index),
                    "--output",
                    str(stage_output),
                    "--mode",
                    "language",
                    "--strategy",
                    strategy,
                    "--trigger",
                    str(trigger),
                    "--task-ids",
                    *map(str, args.task_ids),
                    "--trials",
                    str(args.trials),
                    "--initial-start",
                    str(args.initial_start),
                    "--max-steps",
                    str(args.max_steps),
                    "--seed",
                    str(args.seed),
                    "--libero-repo",
                    str(args.libero_repo.resolve()),
                ]
                if args.video_trials:
                    command += ["--save-video", "--video-trials", str(args.video_trials)]
                stages.append((label, stage_output, command))
    report = output / "report"
    command = cli + [
        "report",
        "--inputs",
        *[str(stage_output) for stage, stage_output, _ in stages if stage not in ("index", "sensitivity")],
        "--sensitivity",
        str(sensitivity),
        "--output",
        str(report),
    ]
    if args.smoke or args.max_steps < 1000:
        command.append("--allow-smoke")
    stages.append(("report", report, command))
    print(
        json.dumps(
            {
                "settings": settings,
                "stages": [name for name, _, _ in stages],
                "reused_stages": ["index", "sensitivity", "vanilla", "language"] if reused else [],
                "wandb_project": PROJECT if args.wandb_mode != "disabled" else None,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for _, stage_output, command in stages:
            if stage_output.name in {source.name for source in reused}:
                print(f"Reuse existing stage: {stage_output.name}")
            else:
                print("$ " + " ".join(command))
        return
    plan_path = output / "screen_plan.json"
    if output.exists():
        if not args.resume or not plan_path.is_file():
            raise FileExistsError(f"Output exists; use --resume for this screen: {output}")
        plan = _json(plan_path)
        plan["settings"].setdefault("recovery", False)
        plan["settings"].setdefault("reuse_screen", None)
        if plan["settings"] != settings:
            raise ValueError("Screen settings differ from saved plan; choose a new output")
    else:
        output.mkdir(parents=True)
        plan = {"settings": settings, "status": "running", "stages": {}}
        _write_json(plan_path, plan)
        for source in reused:
            (output / source.name).symlink_to(source, target_is_directory=source.is_dir())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("USE_TF", "0")
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        os.environ["WANDB_PROJECT"] = PROJECT
        os.environ["WANDB_DISABLED"] = "false"
        run = wandb.init(
            project=PROJECT,
            entity=args.wandb_entity,
            group=output.name,
            name=f"pilot2_screen_step{args.checkpoint_step}",
            mode=args.wandb_mode,
            dir=str(output),
            config=settings,
            save_code=False,
            tags=["pilot2", "libero-long", "evaluation"] + (["smoke"] if args.smoke else []),
        )
        _write_json(
            output / "wandb_run.json", {"project": PROJECT, "url": run.url, "id": run.id, "status": "running"}
        )
    exit_code = 1
    try:
        for stage, stage_output, command in stages:
            if _completed(stage, stage_output, split, recipes["dual"], args):
                print(f"{stage}: already completed; skipping", flush=True)
            else:
                if (
                    stage in ("vanilla", "language", "goal", "both") or stage.startswith("strategy_")
                ) and stage_output.exists():
                    command.append("--resume")
                if stage == "report" and stage_output.exists():
                    command.append("--resume")
                if stage in ("index", "sensitivity") and stage_output.exists():
                    raise ValueError(f"Incomplete stage output {stage_output}; choose a fresh screen output")
                started = time.time()
                plan["stages"][stage] = {"status": "running", "started_utc": started, "command": command}
                _write_json(plan_path, plan)
                _run(command, output / "logs" / f"{stage}.log")
                if not _completed(stage, stage_output, split, recipes["dual"], args):
                    raise RuntimeError(f"Stage {stage} exited without complete output")
                plan["stages"][stage] = {
                    "status": "completed",
                    "started_utc": started,
                    "finished_utc": time.time(),
                    "command": command,
                }
                _write_json(plan_path, plan)
            _log_stage(run, stage, stage_output)
        plan["status"] = "completed"
        exit_code = 0
        print(f"Screen complete: {report / 'summary.md'}", flush=True)
    finally:
        _write_json(plan_path, plan)
        if run is not None:
            run.finish(exit_code=exit_code)
            record_path = output / "wandb_run.json"
            record = _json(record_path)
            record["status"] = "finished" if exit_code == 0 else "failed"
            _write_json(record_path, record)
