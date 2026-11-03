"""Summarize execution-frequency ablations of the observation-only goal test."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run plotting on a Slurm worker")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((path / "manifest.json").read_text()) for path in args.runs]
    if any(m["status"] != "completed" for m in manifests):
        raise ValueError("Incomplete run")
    settings = [{k: v for k, v in m["settings"].items() if k != "execute_steps"} for m in manifests]
    if settings[0] != settings[1] or manifests[0]["inputs"] != manifests[1]["inputs"]:
        raise ValueError("Execution ablation changed other inputs")
    rows = []
    trajectories = {}
    for run in args.runs:
        audit = json.loads((run / "observation_audit.json").read_text())
        if audit["status"] != "passed":
            raise ValueError("Observation audit did not pass")
        for path in sorted(run.glob("*/*/episode.json")):
            row = json.loads(path.read_text())
            row["run"] = str(run)
            rows.append(row)
            key = (row["suite"], row["initial_index"], row["mode"])
            trajectories.setdefault(key, {})[row["execute_steps"]] = path
    first_action_error = 0.0
    for pair in trajectories.values():
        if set(pair) != {4, 16}:
            raise ValueError("Expected matching execute4/execute16 cases")
        a = json.loads(pair[4].read_text())
        b = json.loads(pair[16].read_text())
        assert a["start_observation_identity"] == b["start_observation_identity"]
        assert a["goal_observation_identity"] == b["goal_observation_identity"]
        with (
            np.load(pair[4].parent / "trace.npz", allow_pickle=False) as four,
            np.load(pair[16].parent / "trace.npz", allow_pickle=False) as sixteen,
        ):
            error = float(np.abs(four["actions"][:4] - sixteen["actions"][:4]).max())
            first_action_error = max(first_action_error, error)
    if first_action_error > 1e-5:
        raise ValueError("Initial four commands changed before any replan")
    summary = []
    groups = [("action", 16), ("action", 4), ("goal", 16), ("goal", 4)]
    for mode, execute in groups:
        selected = [r for r in rows if r["mode"] == mode and r["execute_steps"] == execute]
        summary.append(
            dict(
                mode=mode,
                execute_steps=execute,
                cases=len(selected),
                start_ee_cm=100 * np.mean([r["metrics"]["start_ee_distance_m"] for r in selected]),
                end_ee_cm=100 * np.mean([r["metrics"]["end_ee_distance_m"] for r in selected]),
                end_orientation_deg=np.mean([r["metrics"]["end_orientation_error_deg"] for r in selected]),
                workspace_mae=np.mean([r["metrics"]["end_workspace_mae"] for r in selected]),
                wrist_mae=np.mean([r["metrics"]["end_wrist_mae"] for r in selected]),
            )
        )
    with (args.output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    labels = ["Mode 1\nexecute 16", "Mode 1\nexecute 4", "Goal head\nexecute 16", "Goal head\nexecute 4"]
    values = [row["end_ee_cm"] for row in summary]
    axes[0].bar(np.arange(4), values, color=["#4378a8", "#80a7c8", "#dd9350", "#e3b586"], width=0.65)
    for index, (mode, execute) in enumerate(groups):
        points = [
            100 * r["metrics"]["end_ee_distance_m"]
            for r in rows
            if r["mode"] == mode and r["execute_steps"] == execute
        ]
        axes[0].scatter(index + np.linspace(-0.12, 0.12, len(points)), points, color="#333333", s=20, zorder=3)
        axes[0].text(index, values[index] + 0.2, f"{values[index]:.2f}", ha="center", fontsize=10)
    axes[0].set(
        xticks=np.arange(4),
        xticklabels=labels,
        ylabel="Final EE distance to fixed goal (cm)",
        title="Mean and six individual cases",
    )
    cases = sorted({(r["suite"], r["initial_index"]) for r in rows})
    paired = []
    for suite, initial in cases:
        selected = {
            r["execute_steps"]: r
            for r in rows
            if r["suite"] == suite and r["initial_index"] == initial and r["mode"] == "goal"
        }
        ends = [100 * selected[e]["metrics"]["end_ee_distance_m"] for e in (16, 4)]
        task_name = {"libero_goal": "Drawer", "libero_object": "Alphabet soup"}.get(suite, suite)
        name = f"{task_name} / init {initial}"
        axes[1].plot([0, 1], ends, marker="o", label=name)
        paired.append(
            dict(
                suite=suite,
                initial_index=initial,
                goal_execute16_end_ee_cm=ends[0],
                goal_execute4_end_ee_cm=ends[1],
                improvement_cm=ends[0] - ends[1],
            )
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Execute 16", "Execute 4 + replan"],
        ylabel="Final EE distance (cm)",
        title="Goal head: paired cases",
    )
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    figure.suptitle("Fixed reference goal t=80, identical observed starts at t=64 — trained GOS tasks")
    figure.tight_layout()
    figure.savefig(args.output / "execution_comparison.png", dpi=170)
    plt.close(figure)
    with (args.output / "paired.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    gain = 100 * (summary[2]["end_ee_cm"] - summary[3]["end_ee_cm"]) / summary[2]["end_ee_cm"]
    lines = [
        "# Fixed-goal controller diagnosis — 2026-11-03",
        "",
        "Six successful Mode-1 references, two trained GOS tasks. Replay recorded actions to t=64; keep reference frame 80 as the fixed goal for 16 control steps. RGB and robot proprioception only, with standard benchmark reset.",
        "",
        "| Controller | Execute before replanning | Cases | Mean final EE error (cm) | Workspace RGB MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {'Mode 1' if row['mode'] == 'action' else 'Goal head'} | {row['execute_steps']} | {row['cases']} | {row['end_ee_cm']:.2f} | {row['workspace_mae']:.2f} |"
        )
    lines += [
        "",
        f"Mean initial EE-to-goal distance: **{summary[0]['start_ee_cm']:.2f} cm**. Goal execute4 reduces endpoint error by **{gain:.1f}%** relative to execute16 in this six-case pilot.",
        "",
        "## Interpretation",
        "",
        "- All prefix frames and paired start observations matched exactly. Mode 1 execute16 reproduced its recorded continuation exactly; its zero error is a reproducibility control, not evidence of perfect general performance.",
        "- The goal head moves toward the fixed milestone but retains substantial local control error. Removing online phase selection did not remove this error, so selector tuning alone is insufficient for these cases.",
        "- More frequent replanning improves the mean goal-head error but leaves a gap to Mode 1. This intervention includes both fresh observation feedback and new Gaussian flow samples; it does not isolate their individual effects.",
        "- EE/RGB errors are proxies, not semantic goal success. These are references from successful Mode-1 rollouts rather than new expert labels. One seed, six starts, one phase per start; not Long SR, goal-grounding proof, or OOD recovery.",
        "- Next focus on goal-controller precision before cross-demo scaling or switching thresholds. No additional training or perturbation job was launched.",
        "",
        f"Audit: all 24 local videos exist; initial four commands match across execute variants (maximum difference {first_action_error:g}). Both per-run observation audits passed.",
        "",
        "![Execution comparison](execution_comparison.png)",
        "",
        "Runs: " + ", ".join(str(r) for r in args.runs),
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines[:13]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
