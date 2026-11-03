"""Audit and plot a completed same-scene goal-controller pilot on a CPU worker."""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gr00t.rag.oracle_goal_protocol import array_hash


def main(output):
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run report generation on a Slurm worker")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Wait for the pilot to complete")
    paths = sorted(output.glob("*/*/episode.json"))
    if len(paths) != manifest["expected_cases"]:
        raise ValueError("Incomplete cohort")
    rows = [json.loads(path.read_text()) for path in paths]
    for path, row in zip(paths, rows):
        assert row["complete"] and row["signature"] == manifest["signature"]
        if row.get("skipped"):
            continue
        assert (path.parent / "rollout.mp4").is_file()
        assert (path.parent / "trace.npz").is_file()
        assert row["control_steps"] <= row["max_steps"]
        if row["case"] != "goal_oracle":
            continue
        reference = json.loads((Path(row["reference"]) / "episode.json").read_text())
        assert reference["success"]
        assert row["initial_sha256"] == reference["initial_sha256"]
        assert row["initial_sim_sha256"] == reference["initial_sim_sha256"]
        with np.load(Path(row["reference"]) / "trace.npz", allow_pickle=False) as data:
            images, wrists = data["agent"], data["wrist"]
            count = len(data["ee"])
        previous = 0
        for query in row["queries"]:
            anchor, goal = query["anchor_frame"], query["goal_frame"]
            assert previous <= anchor <= min(previous + 32, count - 1)
            assert goal == min(anchor + 16, count - 1)
            assert query["goal_image_sha256"] == array_hash(images[goal][::-1, ::-1][None])
            assert query["goal_wrist_sha256"] == array_hash(wrists[goal][::-1, ::-1][None])
            previous = anchor
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    goals = [(path, row) for path, row in zip(paths, rows) if row["case"] == "goal_oracle" and row.get("eligible")]
    if goals:
        figure, axes = plt.subplots(
            math.ceil(len(goals) / 2), 2, figsize=(13, 3.2 * math.ceil(len(goals) / 2)), squeeze=False
        )
        for ax, (path, row) in zip(axes.flat, goals):
            with np.load(path.parent / "trace.npz", allow_pickle=False) as data:
                distance = data["d_ee"] * 100
            ax.plot(
                np.arange(len(distance)), distance, color="tab:blue", label="Distance to reference EE manifold"
            )
            ax.set_xlabel("Control step")
            ax.set_ylabel("EE distance (cm)", color="tab:blue")
            twin = ax.twinx()
            twin.step(
                [q["control_step"] for q in row["queries"]],
                [q["goal_frame"] for q in row["queries"]],
                where="post",
                color="tab:orange",
                alpha=0.65,
            )
            twin.set_ylabel("Selected reference goal frame", color="tab:orange")
            ax.set_title(
                f"{row['suite']} / init {row['initial_index']} / {'success' if row['success'] else 'failure'}"
            )
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(goals) :]:
            ax.set_visible(False)
        figure.suptitle("Privileged same-scene goal control — clean rollouts (no injected perturbation)")
        figure.tight_layout()
        figure.savefig(analysis / "goal_traces.png", dpi=160)
        plt.close(figure)
    lines = [
        "# Same-scene goal-controller pilot — results",
        "",
        "These are trained GOS tasks, not held-out Long evaluation.",
        "",
        "| Suite / task 0 | Mode-1 reference successes | Goal successes / eligible references |",
        "|---|---:|---:|",
    ]
    for suite in manifest["settings"]["suites"]:
        baseline = [r for r in rows if r["suite"] == suite and r["case"] == "action_reference"]
        eligible = [r for r in rows if r["suite"] == suite and r["case"] == "goal_oracle" and r.get("eligible")]
        lines.append(
            f"| {suite} | {sum(r['success'] for r in baseline)}/{len(baseline)} | {sum(r['success'] for r in eligible)}/{len(eligible)} |"
        )
    lines += [
        "",
        "Goal success is conditional on successful same-scene reference rollouts; it is not directly comparable to unconditional benchmark SR.",
        "",
        f"Audit passed for {len(goals)} executed goal rollouts: matching initial and post-warm-up simulator state hashes; every goal image hash matches its logged reference frame; monotone bounded phase selection; all executed episodes have trace/video files.",
        "",
        "The selector uses privileged physical-state matching, not guaranteed semantic phase labels. EE proximity also does not imply correct object/contact state. There was no injected perturbation and no recovery claim.",
        "",
        "![Clean goal-control traces](goal_traces.png)",
        "",
    ]
    (analysis / "RESULTS.md").write_text("\n".join(lines))
    (analysis / "audit.json").write_text(
        json.dumps(
            dict(
                status="passed",
                complete_cases=len(rows),
                goal_rollouts_audited=len(goals),
                checks=[
                    "initial states",
                    "simulator starts",
                    "goal image identity",
                    "phase pointer bounds",
                    "artifacts",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines[:12]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
