"""Matched training recipes and checkpoint provenance validation."""

import json
from pathlib import Path


def recipe(args):
    split = json.loads(Path(args.split).read_text())
    return {
        "pilot": 2,
        "variant": args.variant,
        "split_id": split["split_id"],
        "split": str(Path(args.split).resolve()),
        "counts_per_task": split["counts_per_task"],
        "base_model": args.base_model,
        "seed": args.seed,
        "action_horizon": 16,
        "batch_size": args.batch_size,
        "num_gpus": args.num_gpus,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * args.num_gpus,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": args.report_to,
        "wandb_project": args.wandb_project if args.report_to == "wandb" else None,
        "wandb_entity": args.wandb_entity if args.report_to == "wandb" else None,
        "wandb_group": args.wandb_group or Path(args.output).parent.name,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "frozen_vision": True,
        "frozen_llm": True,
        "masking": {"language": 0.4, "both": 0.4, "goal": 0.2} if args.variant == "dual" else {"language": 1.0},
        "goal_pairing": "50% hindsight; 50% same-task cross-demo, relative-progress alignment"
        if args.variant == "dual"
        else None,
        "cross_demo_state_noise_std_raw_units": 0.02 if args.variant == "dual" else 0.0,
        "normalization": "train_only_min_max",
        "denoising_steps_eval": 4,
    }


def train(args):
    import os
    import subprocess
    import sys
    import torch
    import shutil
    import time

    settings = recipe(args)
    if args.report_to == "wandb":
        os.environ["WANDB_DISABLED"] = "false"
    print(json.dumps(settings, indent=2), flush=True)
    if args.dry_run:
        return settings
    if not torch.cuda.is_available():
        raise RuntimeError("Full GR00T fine-tuning needs the GPU environment; use debug on CPU first")
    if args.num_gpus > 1 and "LOCAL_RANK" not in os.environ:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
                "--module",
                "examples.Libero.pilot2",
                *sys.argv[1:],
            ],
            check=True,
        )
        return settings
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if int(os.environ["WORLD_SIZE"]) != args.num_gpus:
            raise ValueError("torchrun world size does not match --num-gpus")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    from scripts.gr00t_finetune import ArgsConfig, main

    split_root = Path(args.split).resolve().parent
    output = Path(args.output)
    exp = output / "experiment_cfg"
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=False)
        exp.mkdir()
        (exp / "pilot2.json").write_text(json.dumps(settings, indent=2) + "\n")
        shutil.copy2(args.split, exp / "pilot2_split.json")
    if distributed:
        torch.distributed.barrier()
    dual = args.variant == "dual"
    config = ArgsConfig(
        dataset_path=[str(split_root / "train")],
        output_dir=str(output),
        pilot2_goals=dual,
        data_config="examples.Libero.pilot2.transforms:GoalDataConfig"
        if dual
        else "examples.Libero.pilot2.transforms:BaselineDataConfig",
        base_model_path=args.base_model,
        seed=args.seed,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        video_backend="torchvision_av",
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.workers,
        dataloader_prefetch_factor=2 if args.workers else None,
        save_final_model=False,
    )
    from .tracking import start_tracking, finish_tracking

    run = start_tracking(args, settings, output)
    exit_code = 1
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        main(config)
        if int(os.environ.get("RANK", "0")) == 0:
            runtime = {
                "wall_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            (output / "training_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
            print(json.dumps(runtime, indent=2), flush=True)
            if run is not None:
                run.summary.update(runtime)
                check = output / "goal_conditioning_check.json"
                if check.is_file():
                    for name, value in json.loads(check.read_text())["parameter_change_l2"].items():
                        run.summary[f"goal_parameter_change/{name}"] = value
        exit_code = 0
    finally:
        finish_tracking(run, output, exit_code)
    return settings


def load_policy(checkpoint, split_id, device="cuda"):
    import torch
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from .transforms import BaselineDataConfig, GoalDataConfig

    checkpoint = Path(checkpoint).resolve()
    metadata = checkpoint / "experiment_cfg/pilot2.json"
    if not metadata.is_file():
        raise ValueError(
            "Checkpoint lacks Pilot 2 split/training provenance; an LOSO checkpoint is not a fair baseline"
        )
    settings = json.loads(metadata.read_text())
    if settings["split_id"] != split_id:
        raise ValueError("Checkpoint was trained on a different split")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Switch to the GPU environment for full GR00T inference; debug uses a tiny CPU head")
    config = GoalDataConfig() if settings["variant"] == "dual" else BaselineDataConfig()
    policy = Gr00tPolicy(
        str(checkpoint),
        EmbodimentTag.NEW_EMBODIMENT,
        config.modality_config(),
        config.transform(),
        denoising_steps=4,
        device=device,
    )
    policy.pilot2_recipe = settings
    if bool(getattr(policy.model.action_head.config, "goal_conditioning", False)) != (
        settings["variant"] == "dual"
    ):
        raise ValueError("Checkpoint architecture disagrees with saved training recipe")
    return policy
