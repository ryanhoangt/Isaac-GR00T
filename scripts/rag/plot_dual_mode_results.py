"""Reproduce slide figures from completed dual-mode RA-VLA experiment artifacts.

Run rendering on a Slurm worker; no GPU is needed. All metrics are read from
saved summaries and checked against per-episode records. No new rollouts run.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import textwrap

import matplotlib
from matplotlib.patches import Patch
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- select the headless backend first


PALETTE = ["#214F78", "#5B82A5", "#BD812A", "#D8AB65", "#A94F50", "#CF8585"]
METHODS = [
    ("RA-VLA (local)", "RA-VLA\n(local)", "ravla_local_newbuffer_100_20261031", "baseline"),
    ("Dual: action only", "Dual model\naction only", "support_gate_10trials_15k_20261031", "dual_action"),
    (
        "Adaptive q90, 2-chunk cap",
        "Adaptive q90\n2-chunk cap",
        "support_gate_10trials_15k_20261031",
        "adaptive_q90",
    ),
    (
        "Adaptive q95, 2-chunk cap",
        "Adaptive q95\n2-chunk cap",
        "support_gate_10trials_15k_20261031",
        "adaptive_q95",
    ),
    (
        "Adaptive q90, until q80",
        "Adaptive q90\nuntil q80",
        "support_gate_untilq80_10trials_15k_20261031",
        "adaptive_q90",
    ),
    (
        "Adaptive q95, until q80",
        "Adaptive q95\nuntil q80",
        "support_gate_untilq80_10trials_15k_20261031",
        "adaptive_q95",
    ),
]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


class Figures:
    def __init__(self, repo, output):
        self.repo, self.output = repo.resolve(), output.resolve()
        self.eval = self.repo / "outputs/ravla_eval"
        self.output.mkdir(parents=True, exist_ok=True)
        self.sources, self.cache, self.exports = {}, {}, []
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "axes.edgecolor": "#CBD2D9",
                "text.color": "#20303E",
                "axes.labelcolor": "#20303E",
                "xtick.color": "#4D5964",
                "ytick.color": "#4D5964",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

    def record(self, path):
        self.sources[str(path.relative_to(self.repo))] = sha(path)

    def json(self, path):
        self.record(path)
        return json.loads(path.read_text())

    def csv(self, path):
        self.record(path)
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def episodes(self, run, condition):
        key = (run, condition)
        if key not in self.cache:
            directory = self.eval / run
            manifest = self.json(directory / "manifest.json")
            if manifest["status"] != "completed":
                raise ValueError(f"Run is not complete: {run}")
            rows = []
            for path in sorted((directory / condition).glob("task*_init*/episode.json")):
                row = self.json(path)
                if not row.get("complete") or row["smoke_only"]:
                    raise ValueError(f"Incomplete/truncated episode: {path}")
                for query in row["queries"]:
                    query.pop("normalized_actions", None)
                row["directory"] = path.parent
                rows.append(row)
            identities = {(r["task_id"], r["initial_index"]) for r in rows}
            if len(rows) != 100 or identities != {(t, i) for t in range(10) for i in range(10)}:
                raise ValueError(f"Expected a complete 100-rollout cohort: {run}/{condition}")
            self.cache[key] = rows
        return self.cache[key]

    def save(self, fig, stem):
        for extension in ("png", "pdf", "svg"):
            path = self.output / f"{stem}.{extension}"
            fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.12)
            self.exports.append(path.name)
        plt.close(fig)
        print(f"SAVED {stem} (PNG/PDF/SVG)", flush=True)

    def table(self, stem, rows):
        with (self.output / f"{stem}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def main_results(self):
        results = []
        reference = {(e["task_id"], e["initial_index"]): e for e in self.episodes(METHODS[0][2], METHODS[0][3])}
        for index, (label, _, run, condition) in enumerate(METHODS):
            episodes = self.episodes(run, condition)
            source = next(r for r in self.csv(self.eval / run / "summary.csv") if r["condition"] == condition)
            successes = sum(e["success"] for e in episodes)
            if int(source["episodes"]) != 100 or int(source["successes"]) != successes:
                raise ValueError("Summary and episode counts differ")
            for episode in episodes:
                original = reference[(episode["task_id"], episode["initial_index"])]
                if original["initial_sha256"] != episode["initial_sha256"]:
                    raise ValueError("Initial states are not matched")
            steps = sum(e["control_steps"] for e in episodes)
            goal_steps = sum(e.get("goal_control_steps", 0) for e in episodes)
            queries = [q for e in episodes for q in e["queries"]]
            entries = sum(q.get("gate_event") == "enter_goal" for q in queries)
            caps = sum(q.get("gate_event") == "exit_cap" for q in queries)
            support = sum(q.get("gate_event") == "exit_support" for q in queries)
            terminal = sum(e["queries"][-1]["mode"] == "goal" for e in episodes)
            if entries != caps + support + terminal:
                raise ValueError("Goal-burst accounting differs")
            results.append(
                dict(
                    method=label,
                    index=index,
                    episodes=100,
                    successes=successes,
                    success_pct=successes,
                    control_steps=steps,
                    goal_control_steps=goal_steps,
                    goal_step_pct=100 * goal_steps / steps,
                    goal_entries=entries,
                    exit_cap=caps,
                    exit_support=support,
                    episode_end_in_goal=terminal,
                )
            )
        self.table("main_results_data", results)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.3), sharey=True)
        positions = np.arange(len(results))
        left, right = axes
        left.barh(positions, [r["success_pct"] for r in results], color=PALETTE, height=0.58)
        left.set_yticks(positions, [m[1] for m in METHODS])
        left.invert_yaxis()
        left.set(xlim=(0, 17), xticks=[0, 5, 10, 15], xlabel="Task success (%)", title="Task completion")
        for y, row in enumerate(results):
            left.text(
                row["success_pct"] + 0.3,
                y,
                f"{row['successes']}/100 ({row['success_pct']}%)",
                va="center",
                fontsize=12,
            )
        right.barh(positions[1:], [r["goal_step_pct"] for r in results[1:]], color=PALETTE[1:], height=0.58)
        right.text(1, 0, "Single-mode baseline", color="#697581", va="center", fontsize=11)
        for y, row in enumerate(results[1:], 1):
            right.text(row["goal_step_pct"] + 1.8, y, f"{row['goal_step_pct']:.1f}%", va="center", fontsize=12)
        right.set(
            xlim=(0, 110),
            xticks=[0, 25, 50, 75, 100],
            xlabel="Executed control steps in goal mode (%)",
            title="Goal-mode usage",
        )
        right.tick_params(axis="y", labelleft=False, left=False)
        for axis in axes:
            axis.set_axisbelow(True)
            axis.xaxis.grid(True, color="#E6EAF0")
            axis.tick_params(axis="y", length=0)
            for boundary in (1.5, 3.5):
                axis.axhline(boundary, color="#D9DFE5", lw=0.8)
        fig.suptitle("LIBERO-Long: matched evaluation", fontsize=22, y=0.99)
        fig.text(
            0.5,
            0.93,
            "100 rollouts per method · same initial states and reference buffer · dual checkpoint 15k",
            ha="center",
            color="#586575",
        )
        fig.text(
            0.5,
            0.01,
            "Exact observed counts; one training seed. Goal-mode usage is a fraction of control steps, not a recovery rate.",
            ha="center",
            fontsize=10.5,
            color="#586575",
        )
        fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.12, wspace=0.19)
        self.save(fig, "01_main_results")
        return results

    def checkpoint_sweep(self):
        source = self.csv(self.eval / "ckpt_sweep_comparison_50_20261101.csv")
        expected = [
            ("RA-VLA local", "ravla_local_newbuffer_100_20261031", "baseline"),
            ("dual 5k", "dual_action_ckpt5k_5trials_20261101", "dual_action"),
            ("dual 10k", "dual_action_ckpt10k_5trials_20261101", "dual_action"),
            ("dual 15k", "support_gate_10trials_15k_20261031", "dual_action"),
        ]
        baseline = None
        rows = []
        for label, run, condition in expected:
            directory = self.eval / run
            manifest = self.json(directory / "manifest.json")
            if manifest["status"] != "completed":
                raise ValueError("Checkpoint sweep contains incomplete runs")
            episodes = []
            for path in (directory / condition).glob("task*_init*/episode.json"):
                episode = self.json(path)
                if episode["initial_index"] < 5:
                    episodes.append(episode)
            identities = {(e["task_id"], e["initial_index"]): e["initial_sha256"] for e in episodes}
            if len(episodes) != 50 or set(identities) != {(t, i) for t in range(10) for i in range(5)}:
                raise ValueError("Checkpoint cohort differs")
            baseline = identities if baseline is None else baseline
            if identities != baseline:
                raise ValueError("Checkpoint initial states differ")
            successes = sum(e["success"] for e in episodes)
            saved = next(r for r in source if r["model"] == label)
            if int(saved["successes"]) != successes:
                raise ValueError("Checkpoint summary differs")
            rows.append(dict(model=label, episodes=50, successes=successes, success_pct=2 * successes))
        self.table("checkpoint_sweep_data", rows)
        fig, axis = plt.subplots(figsize=(10.5, 6.3))
        xs = [0, 1.5, 2.5, 3.5]
        axis.bar(
            xs, [r["success_pct"] for r in rows], width=0.62, color=[PALETTE[0], "#4B8F8B", "#86ABA8", "#AFC7C4"]
        )
        for x, row in zip(xs, rows):
            axis.text(
                x,
                row["success_pct"] + 0.5,
                f"{row['successes']}/50\n{row['success_pct']}%",
                ha="center",
                fontsize=14,
            )
        axis.set_xticks(
            xs,
            ["RA-VLA\nlocal baseline", "Dual 5k\n~0.93 epoch", "Dual 10k\n~1.86 epochs", "Dual 15k\n~2.79 epochs"],
        )
        axis.set(ylim=(0, 19), yticks=[0, 5, 10, 15], ylabel="Task success (%)")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#E6EAF0")
        fig.suptitle("Action-mode checkpoint sweep", fontsize=22, y=0.98)
        fig.text(
            0.5,
            0.91,
            "50 matched rollouts per model · 10 tasks × initial states 0–4",
            ha="center",
            color="#586575",
        )
        fig.text(
            0.5,
            0.015,
            "A smaller cohort than the main evaluation. Few successes limit conclusions about training trends.",
            ha="center",
            fontsize=10.5,
            color="#586575",
        )
        fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.82)
        self.save(fig, "02_checkpoint_sweep")

    def burst_outcomes(self, results):
        rows = results[2:]
        categories = [
            ("exit_support", "Support exit (d ≤ q80)", "#398D8A"),
            ("exit_cap", "Forced exit: 2-chunk cap", "#D1A056"),
            ("episode_end_in_goal", "Episode ended in goal mode", "#B75D61"),
        ]
        fig, axis = plt.subplots(figsize=(13, 5.8))
        y = np.arange(4)
        offset = np.zeros(4)
        for key, label, color in categories:
            values = np.asarray([100 * row[key] / row["goal_entries"] for row in rows])
            axis.barh(y, values, left=offset, height=0.58, color=color, label=label)
            for index, value in enumerate(values):
                if value >= 6:
                    axis.text(
                        offset[index] + value / 2,
                        index,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=13,
                        fontweight="bold",
                    )
            offset += values
        for index, row in enumerate(rows):
            axis.text(102, index, f"n={row['goal_entries']:,} bursts", va="center", fontsize=11)
        axis.set_yticks(y, [m[1] for m in METHODS[2:]])
        axis.invert_yaxis()
        axis.set(xlim=(0, 123), xticks=[0, 25, 50, 75, 100], xlabel="Share of all goal-mode bursts (%)")
        axis.tick_params(axis="y", length=0)
        axis.legend(loc="upper center", bbox_to_anchor=(0.46, -0.2), ncol=3, frameon=False, fontsize=10.5)
        fig.suptitle("How goal-mode bursts end", fontsize=22, y=0.99)
        fig.text(
            0.5,
            0.92,
            "Denominator: every goal-mode entry, including bursts unfinished at episode termination",
            ha="center",
            fontsize=11.5,
            color="#586575",
        )
        fig.text(
            0.5,
            0.01,
            "Support exit is a threshold event, not verified recovery. Episode termination may be task success or timeout.",
            ha="center",
            fontsize=10.5,
            color="#586575",
        )
        fig.subplots_adjust(left=0.19, right=0.98, top=0.8, bottom=0.27)
        self.save(fig, "03_goal_burst_outcomes")

    def sensitivity(self):
        directory = self.eval / "dual_diag_probes_15k_20261031/probes"
        rows = self.csv(directory / "probes.csv")
        summary = self.json(directory / "summary.json")
        if len(rows) != 200 or summary["states"] != 200:
            raise ValueError("Expected 200 goal probes")
        if any(r["shared_noise"].lower() != "true" or r["query_episode"] == r["goal_episode"] for r in rows):
            raise ValueError("Goal probe noise/demo exclusion differs")
        names = ["wrong_relative_all", "random_relative_all"]
        values = [np.asarray([float(r[name]) for r in rows]) for name in names]
        stats = []
        for name, samples in zip(names, values):
            q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
            if not np.isclose(median, summary[name]["median"]):
                raise ValueError("Probe aggregate differs")
            stats.append(
                dict(
                    comparison=name,
                    states=len(samples),
                    median=median,
                    q25=q1,
                    q75=q3,
                    mean=float(samples.mean()),
                    maximum=float(samples.max()),
                )
            )
        self.table("goal_sensitivity_data", stats)
        fig, axis = plt.subplots(figsize=(9.5, 6.2))
        medians = np.asarray([r["median"] for r in stats])
        lower = medians - np.asarray([r["q25"] for r in stats])
        upper = np.asarray([r["q75"] for r in stats]) - medians
        axis.bar([0, 1], medians, width=0.55, color=["#4C7EAA", "#9BB6CF"])
        axis.errorbar([0, 1], medians, yerr=[lower, upper], fmt="none", capsize=8, color="#263B4D", lw=1.8)
        maximum = max(r["q75"] for r in stats)
        for index, row in enumerate(stats):
            axis.text(index, row["q75"] + maximum * 0.07, f"Median {row['median']:.3f}", ha="center", fontsize=14)
        axis.set_xticks([0, 1], ["Distant-time goal\n≥32-step temporal gap", "Random goal\nsame task"])
        axis.set(ylabel="Relative action change (L2 ratio)", ylim=(0, maximum * 1.35))
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#E6EAF0")
        fig.suptitle("Goal sensitivity on reference states", fontsize=22, y=0.98)
        fig.text(
            0.5,
            0.91,
            "200 states · identical Gaussian source per comparison · all seven action channels",
            ha="center",
            fontsize=11.5,
            color="#586575",
        )
        fig.text(
            0.5,
            0.015,
            "Bars: medians. Whiskers: 25th–75th percentiles across states (not confidence intervals).",
            ha="center",
            fontsize=10.5,
            color="#586575",
        )
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.82)
        self.save(fig, "04_goal_sensitivity")

    def switching_example(self, task=0, initial=0):
        methods = [METHODS[2], METHODS[4]]
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True, sharey=True)
        example_rows = []
        for axis, method in zip(axes, methods):
            label, _, run, condition = method
            episode = next(
                e for e in self.episodes(run, condition) if e["task_id"] == task and e["initial_index"] == initial
            )
            queries = episode["queries"]
            x = [q["control_step"] for q in queries]
            axis.plot(x, [q["support_fp32"] for q in queries], color="#214F78", lw=2, label="Support distance")
            axis.axhline(queries[0]["threshold_enter"], color="#B65750", ls="--", lw=1.5, label="Entry q90")
            axis.axhline(queries[0]["threshold_exit"], color="#398D8A", ls=":", lw=1.7, label="Exit q80")
            for query in queries:
                start = query["control_step"]
                if query["mode"] == "goal":
                    axis.axvspan(
                        start, min(start + 16, episode["control_steps"]), color="#E7B45D", alpha=0.27, lw=0
                    )
                example_rows.append(
                    dict(
                        policy=label,
                        task=task,
                        initial=initial,
                        step=start,
                        mode=query["mode"],
                        support=query["support_fp32"],
                        enter=query["threshold_enter"],
                        exit=query["threshold_exit"],
                        event=query["gate_event"],
                        goal_frame=query["goal_frame"],
                    )
                )
            axis.axvline(episode["control_steps"], color="#85919D", lw=1, ls="-.")
            outcome = "success" if episode["success"] else "timeout"
            axis.set(title=f"{label} — {outcome}, {episode['control_steps']} steps", ylabel="1 − cosine")
            axis.grid(axis="y", color="#E6EAF0")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Patch(facecolor="#E7B45D", alpha=0.45))
        labels.append("Goal-mode control steps")
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.93), ncol=4, frameon=False, fontsize=11
        )
        axes[-1].set(xlim=(0, 1000), xlabel="Control step")
        fig.suptitle(f"Illustrative matched pair: task {task}, initial state {initial}", fontsize=21, y=0.99)
        fig.text(
            0.5,
            0.01,
            "Selected example, not an aggregate. This is the capped q90 run's only successful initial state.",
            ha="center",
            fontsize=10.5,
            color="#586575",
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.81, bottom=0.1, hspace=0.43)
        self.table("switching_example_data", example_rows)
        self.save(fig, "05_switching_example")

    def heatmap(self):
        counts = np.zeros((6, 10), dtype=int)
        task_names = {}
        table = []
        for index, (name, _, run, condition) in enumerate(METHODS):
            episodes = self.episodes(run, condition)
            for task in range(10):
                rows = [e for e in episodes if e["task_id"] == task]
                if len(rows) != 10:
                    raise ValueError("Incomplete per-task cohort")
                counts[index, task] = sum(e["success"] for e in rows)
                task_names[task] = rows[0]["task"]
                table.append(
                    dict(
                        method=name,
                        task=task,
                        task_description=rows[0]["task"],
                        successes=int(counts[index, task]),
                        episodes=10,
                    )
                )
        self.table("per_task_success_data", table)
        self.table("task_key", [dict(task=t, description=task_names[t]) for t in range(10)])
        fig, axis = plt.subplots(figsize=(14, 6))
        cmap = plt.get_cmap("Blues")
        image = axis.imshow(counts * 10, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for row in range(6):
            for column in range(10):
                axis.text(
                    column,
                    row,
                    f"{counts[row, column]}/10",
                    ha="center",
                    va="center",
                    color="white" if counts[row, column] >= 6 else "#20303E",
                    fontsize=12,
                )
        axis.set_xticks(range(10), [f"Task {t}" for t in range(10)])
        axis.set_yticks(range(6), [m[0] for m in METHODS])
        axis.tick_params(length=0)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
        colorbar.set_label("Task success (%)")
        fig.suptitle("Per-task LIBERO-Long results", fontsize=22, y=0.99)
        fig.text(
            0.5,
            0.92,
            "Each cell: successes out of 10 matched initial states · task names in task_key.csv",
            ha="center",
            fontsize=12,
            color="#586575",
        )
        fig.subplots_adjust(left=0.25, right=0.94, top=0.83, bottom=0.1)
        self.save(fig, "06_per_task_success")

    def rollout_frames(self, task=0, initial=0):
        import imageio.v2 as imageio

        methods = [METHODS[0], METHODS[1], METHODS[2], METHODS[4]]
        steps = [0, 128, 256, 320]
        fig, axes = plt.subplots(4, 4, figsize=(12, 11))
        for row, method in enumerate(methods):
            label, _, run, condition = method
            episode = next(
                e for e in self.episodes(run, condition) if e["task_id"] == task and e["initial_index"] == initial
            )
            path = episode["directory"] / episode["video"]
            if not path.is_file():
                raise FileNotFoundError(path)
            self.record(path)
            reader = imageio.get_reader(str(path))
            try:
                for column, step in enumerate(steps):
                    if step >= episode["control_steps"]:
                        raise ValueError("Requested frame is after episode termination")
                    frame = reader.get_data(step)
                    # Both evaluators record two 256px views below a 32/48px banner.
                    workspace = frame[-256:, :256, :3]
                    axis = axes[row, column]
                    axis.imshow(workspace)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if row == 0:
                        axis.set_title(f"Step {step}", fontsize=15)
                    if column == 0:
                        outcome = "Success" if episode["success"] else "Timeout"
                        text = "\n".join(textwrap.wrap(label, 22)) + f"\n{outcome}"
                        axis.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=14, fontsize=11)
            finally:
                reader.close()
        fig.suptitle(f"Recorded rollout frames: task {task}, initial state {initial}", fontsize=20, y=0.98)
        fig.text(
            0.53,
            0.925,
            "Same initial state and control-step timestamps · workspace camera",
            ha="center",
            color="#586575",
            fontsize=11.5,
        )
        fig.text(
            0.53,
            0.015,
            "Illustrative example; capped q90 succeeds here in 1/100 cases overall. No new rollouts were generated.",
            ha="center",
            fontsize=10,
            color="#586575",
        )
        fig.subplots_adjust(left=0.24, right=0.99, top=0.88, bottom=0.05, hspace=0.08, wspace=0.03)
        self.save(fig, "07_rollout_frames")

    def finish(self):
        manifest = dict(
            script_sha256=sha(Path(__file__)),
            input_sha256=self.sources,
            exports=self.exports,
            main_cohort=100,
            checkpoint_cohort=50,
            probe_states=200,
            png_dpi=320,
            formats=["png", "pdf", "svg"],
            notes=[
                "All input run summaries and episode counts validated.",
                "No confidence intervals: exact observed counts from one training seed.",
                "Goal-burst outcomes use all entries, including episode endings.",
                "Sensitivity whiskers are empirical IQR, not uncertainty intervals.",
                "Switching traces and rollout frames are explicitly selected examples.",
            ],
        )
        (self.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        lines = [
            "# Dual-mode RA-VLA result figures",
            "",
            "Use PNG for PowerPoint; SVG/PDF retain vector text and graphics (video frames remain raster).",
            "All figures are generated from saved artifacts, with source hashes in figure_manifest.json.",
            "",
            "| File stem | Content |",
            "|---|---|",
        ]
        descriptions = [
            ("01_main_results", "100-rollout success and goal-mode usage"),
            ("02_checkpoint_sweep", "50-rollout checkpoint comparison; separate cohort"),
            ("03_goal_burst_outcomes", "Support exits, cap exits and episode endings; all-burst denominator"),
            ("04_goal_sensitivity", "200-state median/IQR; common Gaussian source"),
            ("05_switching_example", "Selected task 0/init 0: capped success vs until-q80 timeout"),
            ("06_per_task_success", "Per-task success counts; task labels in task_key.csv"),
            ("07_rollout_frames", "Existing videos at matched control-step timestamps"),
        ]
        for stem, description in descriptions:
            if f"{stem}.png" in self.exports:
                lines.append(f"| [{stem}]({stem}.png) | {description} |")
        lines.extend(
            [
                "",
                "Cohorts must not be pooled across the main evaluation and checkpoint sweep.",
                "Support threshold exits are not verified semantic recovery events.",
                "The selected successful capped example is not representative of its aggregate success rate.",
            ]
        )
        (self.output / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("reports/dual_mode_ravla/figures"))
    parser.add_argument("--skip-rollout-frames", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Render on a Slurm worker; no GPU is required")
    output = args.output if args.output.is_absolute() else args.repo / args.output
    figures = Figures(args.repo, output)
    results = figures.main_results()
    figures.checkpoint_sweep()
    figures.burst_outcomes(results)
    figures.sensitivity()
    figures.switching_example()
    figures.heatmap()
    if not args.skip_rollout_frames:
        figures.rollout_frames()
    figures.finish()
    print(f"COMPLETED {figures.output}", flush=True)


if __name__ == "__main__":
    main()
