"""Explicit per-variant W&B lifecycle; HF Trainer supplies the training curves."""

import json
import os
from pathlib import Path

PROJECT = "icl-vla-uncertainty"


def start_tracking(args, settings, output):
    if args.report_to != "wandb" or int(os.environ.get("RANK", "0")) != 0:
        return None
    if args.wandb_project != PROJECT:
        raise ValueError(f"Pilot 2 logs to the fixed project {PROJECT}")
    import wandb

    output = Path(output).resolve()
    group = args.wandb_group or output.parent.name
    name = args.run_name or f"{args.variant}_seed{args.seed}"
    # Do not reuse an inherited run ID from another experiment or variant.
    run_id = wandb.util.generate_id()
    os.environ.update(
        WANDB_PROJECT=args.wandb_project,
        WANDB_MODE=args.wandb_mode,
        WANDB_DISABLED="false",
        WANDB_LOG_MODEL="false",
        WANDB_WATCH="false",
        WANDB_DIR=str(output),
        WANDB_RUN_GROUP=group,
        WANDB_NAME=name,
        WANDB_RUN_ID=run_id,
        WANDB_RESUME="never",
    )
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=name,
        id=run_id,
        resume="never",
        mode=args.wandb_mode,
        dir=str(output),
        config={"pilot2": settings},
        save_code=False,
        tags=["pilot2", "libero-long", args.variant] + (["smoke"] if args.max_steps <= 2 else []),
    )
    record = {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": run.entity,
        "group": group,
        "url": run.url,
        "mode": args.wandb_mode,
        "status": "running",
    }
    (output / "wandb_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"W&B {args.wandb_mode}: {run.url}", flush=True)
    return run


def finish_tracking(run, output, exit_code):
    if run is None:
        return
    output = Path(output)
    try:
        run.finish(exit_code=exit_code)
    finally:
        path = output / "wandb_run.json"
        record = json.loads(path.read_text())
        record["status"] = "finished" if exit_code == 0 else "failed"
        path.write_text(json.dumps(record, indent=2) + "\n")
