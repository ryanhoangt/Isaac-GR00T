"""Run the matched 50-rollout control evaluation once checkpoint-15000 is complete."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run the watcher on a Slurm worker")
    state = args.checkpoint.parent / "eval_watch.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.rag.diagnose_dual_mode",
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--stage",
        "standard",
        "--conditions",
        "dual_action",
        "--trials",
        "5",
        "--initial-start",
        "0",
        "--workers",
        "2",
        "--video-trials",
        "10",
        "--port",
        str(args.port),
        "--wandb-name",
        "debug_action_matched_15k_50",
    ]
    print("WAIT_FOR_CHECKPOINT", args.checkpoint, flush=True)
    while not (args.checkpoint / "config.json").is_file():
        state.write_text(
            json.dumps(
                dict(status="waiting", checkpoint=str(args.checkpoint), output=str(args.output), command=command),
                indent=2,
            )
            + "\n"
        )
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for training checkpoint")
        if args.training_log.is_file():
            with args.training_log.open("rb") as handle:
                handle.seek(max(0, args.training_log.stat().st_size - 20000))
                tail = handle.read().decode(errors="replace")
            if "Traceback (most recent call last)" in tail:
                raise RuntimeError("Training log contains a failure; checkpoint evaluation was not started")
        time.sleep(30)
    state.write_text(
        json.dumps(
            dict(status="evaluating", checkpoint=str(args.checkpoint), output=str(args.output), command=command),
            indent=2,
        )
        + "\n"
    )
    print("START_MATCHED_CONTROL_EVAL", flush=True)
    result = subprocess.run(command, check=False)
    state.write_text(
        json.dumps(
            dict(
                status="completed" if result.returncode == 0 else "failed",
                exit_code=result.returncode,
                output=str(args.output),
            ),
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--timeout-hours", type=float, default=48)
    main(parser.parse_args())
