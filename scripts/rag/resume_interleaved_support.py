"""Resume a support eval with alternating remaining condition/task groups.

The original launcher still validates every experiment/source fingerprint.
Only its pending-queue factory is replaced; inference and episode seeds are unchanged.
Scheduling provenance is saved separately in the existing output directory.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

from gr00t.rag.diagnosis_protocol import digest
from scripts.rag import eval_support_gate as launcher
from scripts.rag.eval_libero_parallel import atomic_json


def remaining_queue(items, output, manifest):
    queues = {}
    skipped = []
    settings = manifest["settings"]
    for case, task in items:
        complete = True
        for initial in range(settings["initial_start"], settings["initial_start"] + settings["trials"]):
            directory = output / case["name"] / f"task{task:02d}_init{initial:03d}"
            path = directory / "episode.json"
            if not path.is_file():
                complete = False
                break
            row = json.loads(path.read_text())
            expected = digest(dict(run=manifest["signature"], case=case, task=task, initial=initial))
            if (
                row.get("signature") != expected
                or row["case"] != case
                or row["task_id"] != task
                or row["initial_index"] != initial
            ):
                raise ValueError(f"Episode identity/signature differs: {path}")
            if (
                not row.get("complete")
                or not (directory / "trace.npz").exists()
                or (row["video"] and not (directory / row["video"]).exists())
            ):
                complete = False
                break
        if complete:
            skipped.append(f"{case['name']}_task{task:02d}")
        else:
            queues.setdefault(case["name"], deque()).append((case, task))
    result = deque()
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                result.append(queue.popleft())
    return result, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    settings = manifest["settings"]
    if args.workers < 2:
        raise ValueError("At least two workers are required to start both conditions together")

    def pending_factory(items):
        pending, skipped = remaining_queue(items, output, manifest)
        metadata = dict(
            updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scheduler="alternate_remaining_condition_task_groups",
            script_sha256=launcher.file_sha(Path(__file__)),
            experiment_signature=manifest["signature"],
            original_source_guard="enforced by unchanged eval_support_gate.run",
            skipped_completed_task_groups=skipped,
            pending=[f"{case['name']}_task{task:02d}" for case, task in pending],
            workers=args.workers,
        )
        if args.plan_only:
            print(json.dumps(metadata, indent=2))
        else:
            atomic_json(output / "scheduler_resume.json", metadata)
            print("INTERLEAVED PENDING: " + ", ".join(metadata["pending"]), flush=True)
        return pending

    if args.plan_only:
        pending_factory((case, task) for case in manifest["cases"] for task in settings["task_ids"])
        return
    command = [
        "eval_support_gate",
        "--output",
        str(output),
        "--resume",
        "--workers",
        str(args.workers),
        "--port",
        str(args.port),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-name",
        output.name + "_interleaved",
    ]
    for key in (
        "checkpoint",
        "calibration",
        "memory",
        "retriever",
        "trials",
        "initial_start",
        "max_steps",
        "seed",
        "video_trials",
    ):
        command.extend(["--" + key.replace("_", "-"), str(settings[key])])
    for key in ("task_ids", "variants"):
        command.extend(["--" + key.replace("_", "-"), *map(str, settings[key])])
    for key, value in settings["gate"].items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if settings["smoke"]:
        command.append("--smoke")
    # execute() uses this factory only for the case/task pending queue. Keep
    # its scientific source unchanged so the original strict resume guard runs.
    original_factory, original_argv = launcher.deque, sys.argv
    launcher.deque, sys.argv = pending_factory, command
    try:
        launcher.main()
    finally:
        launcher.deque, sys.argv = original_factory, original_argv


if __name__ == "__main__":
    main()
