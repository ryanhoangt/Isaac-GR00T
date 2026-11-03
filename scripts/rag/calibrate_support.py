"""Compute frozen, per-task LODO support thresholds from the existing reference cache."""

import argparse
import csv
import fcntl
from pathlib import Path

from gr00t.rag.support_calibration import build
from scripts.rag.eval_libero_parallel import atomic_json
from scripts.rag.train_dual_mode import check_worker


def main(args):
    check_worker()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".calibration.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "calibration.json").exists():
            raise FileExistsError("Calibration exists; use a fresh output")
        document, rows = build(args.memory, args.query_stride, args.exit_quantile)
        temporary = args.output / "scores.csv.tmp"
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(args.output / "scores.csv")
        lines = [
            "# Offline retrieval-support calibration",
            "",
            document["limitation"],
            "",
            "| Task | Samples | Exit | q90 | q95 |",
            "|---|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {task} | {v['samples']} | {v['exit']:.5f} | {v['q90']:.5f} | {v['q95']:.5f} |"
            for task, v in document["tasks"].items()
        )
        (args.output / "summary.md").write_text("\n".join(lines) + "\n")
        atomic_json(args.output / "calibration.json", document)
        print(f"CALIBRATED {len(document['tasks'])} tasks, {len(rows)} expert queries: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=Path("outputs/dual_mode/long_reference_diagnosis"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-stride", type=int, default=16)
    parser.add_argument("--exit-quantile", type=float, default=0.8)
    main(parser.parse_args())
