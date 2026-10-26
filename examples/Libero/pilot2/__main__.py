"""python -m examples.Libero.pilot2 --help"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pilot 2: goal-conditioned GR00T on LIBERO-Long")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Stage disjoint train/reference datasets and train-only normalization"
    )
    prepare.add_argument("--source", type=Path, default=Path("demo_data/libero_loso_clean/long"))
    prepare.add_argument("--output", type=Path, default=Path("demo_data/pilot2_long"))
    prepare.add_argument("--reference-per-task", type=int, default=3)
    prepare.add_argument("--strict-50", action="store_true", help="Optional exact-spec data count check")
    debug = commands.add_parser(
        "debug", help="CPU forward/backward with the real head, small DiT and synthetic features"
    )
    debug.add_argument("--output", type=Path, default=Path("outputs/pilot2/cpu_debug.json"))
    debug.add_argument("--steps", type=int, default=3)
    data_debug = commands.add_parser(
        "debug-data", help="CPU decode, transforms and collation on the real prepared dataset"
    )
    data_debug.add_argument("--split", type=Path, default=Path("demo_data/pilot2_long/split.json"))
    data_debug.add_argument("--output", type=Path, default=Path("outputs/pilot2/data_debug.json"))
    train = commands.add_parser("train", help="Matched baseline/dual fine-tuning; --dry-run needs no GPU")
    train.add_argument("--variant", choices=["baseline", "dual"], required=True)
    train.add_argument("--split", type=Path, default=Path("demo_data/pilot2_long/split.json"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--base-model", default="nvidia/GR00T-N1.5-3B")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-gpus", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--max-steps", type=int, default=2000)
    train.add_argument("--save-steps", type=int, default=1000)
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--report-to", choices=["wandb", "tensorboard"], default="wandb")
    train.add_argument("--wandb-project", choices=["icl-vla-uncertainty"], default="icl-vla-uncertainty")
    train.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    train.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    train.add_argument("--run-name", help="Default: <variant>_seed<seed>")
    train.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--dry-run", action="store_true")
    index = commands.add_parser("index", help="Encode only the reference demos with frozen Eagle visual features")
    sensitivity = commands.add_parser(
        "sensitivity", help="Common-noise goal/null/random probes on reference validation states"
    )
    evaluate = commands.add_parser("evaluate", help="Standard or controlled perturb/recover rollouts")
    for command in (index, sensitivity, evaluate):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--split", type=Path, default=Path("demo_data/pilot2_long/split.json"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
    index.add_argument("--stride", type=int, default=1)
    sensitivity.add_argument("--index", type=Path, required=True)
    sensitivity.add_argument("--states", type=int, default=1000)
    sensitivity.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--index", type=Path)
    evaluate.add_argument("--mode", choices=["vanilla", "language", "goal", "both"], default="language")
    evaluate.add_argument("--strategy", choices=["standard", "A", "B", "C"], default="standard")
    evaluate.add_argument("--trigger", type=int, choices=[25, 60])
    evaluate.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    evaluate.add_argument("--trials", type=int, default=20)
    evaluate.add_argument("--initial-start", type=int, default=0)
    evaluate.add_argument("--max-steps", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--save-video", action="store_true")
    evaluate.add_argument("--libero-repo", type=Path, default=Path(__file__).resolve().parents[4] / "LIBERO")
    report = commands.add_parser("report", help="CSV/Markdown tables and diagnostic PNG plots")
    report.add_argument("--inputs", type=Path, nargs="+", default=[])
    report.add_argument("--sensitivity", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    for key in (
        "steps",
        "states",
        "trials",
        "max_steps",
        "stride",
        "batch_size",
        "num_gpus",
        "gradient_accumulation_steps",
        "save_steps",
        "logging_steps",
        "reference_per_task",
    ):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.source, args.output, args.reference_per_task, args.strict_50)
        print(
            json.dumps(
                {
                    "split_id": result["split_id"],
                    "counts_per_task": result["counts_per_task"],
                    "train": len(result["train"]),
                    "reference": len(result["reference"]),
                },
                indent=2,
            )
        )
    elif args.command == "debug":
        from .debug import debug

        print(json.dumps(debug(args.output, args.steps), indent=2))
    elif args.command == "train":
        from .training import train

        train(args)
    elif args.command == "debug-data":
        from .debug import debug_data

        print(json.dumps(debug_data(args.split, args.output), indent=2))
    elif args.command == "index":
        from .training import load_policy
        from .runtime import build_index, VisionFeatures

        split = json.loads(args.split.read_text())
        policy = load_policy(args.checkpoint, split["split_id"], args.device)
        settings = json.loads((args.checkpoint / "experiment_cfg/pilot2.json").read_text())
        result = build_index(
            args.split.resolve().parent / "reference",
            args.split,
            VisionFeatures(policy),
            args.output,
            args.stride,
            encoder_id=settings["base_model"],
        )
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        if args.strategy != "standard" and args.trigger is None:
            parser.error("A/B/C require --trigger 25 or 60")
        if args.strategy == "standard" and args.trigger is not None:
            parser.error("Standard rollouts do not take --trigger")
        if (
            not args.task_ids
            or len(set(args.task_ids)) != len(args.task_ids)
            or any(task not in range(10) for task in args.task_ids)
        ):
            parser.error("--task-ids must be unique values in 0..9")
        from .evaluate import evaluate

        evaluate(args)
    elif args.command == "sensitivity":
        from .diagnostics import sensitivity

        sensitivity(args)
    elif args.command == "report":
        from .diagnostics import summarize

        summarize(args.inputs, args.output, args.sensitivity, args.allow_smoke)


if __name__ == "__main__":
    main()
