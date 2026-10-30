"""Plot a small paired-context pilot from recorded RA-VLA rollouts (CPU only)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from examples.Libero.eval.uncertainty_recording import atomic_json


SOURCES = ["retrieved", "random_same_task", "low_similarity_same_task"]
LABELS = ["Retrieved", "Random (same task)", "Low similarity (same task)"]
COLORS = ["#2463a6", "#d28c22", "#a04462"]


def load_episodes(root):
    result = []
    for path in sorted(Path(root).glob("**/uncertainty/*/episode.json")):
        meta = json.loads(path.read_text())
        if not meta["complete"]:
            continue
        rows = [
            json.loads(line)
            for line in path.with_name("queries.jsonl").read_text().splitlines()
        ]
        expected = list(
            range(0, meta["control_steps"], meta["score_signature"]["action_horizon"])
        )
        if [row["control_step"] for row in rows] != expected or len(rows) != meta[
            "num_queries"
        ]:
            raise ValueError(f"Missing/duplicate query steps: {path}")
        traces = {"retrieved": rows}
        for source in SOURCES[1:]:
            if all(source in row.get("context_diagnostics", {}) for row in rows):
                traces[source] = [row["context_diagnostics"][source] for row in rows]
        if len(traces) not in (1, len(SOURCES)):
            raise ValueError("inconsistent context diagnostics")
        arrays = {}
        for source, values in traces.items():
            scores = np.asarray([r["u_ctx"] for r in values])
            dimensions = np.asarray([r["u_ctx_per_dimension"] for r in values])
            if not np.isfinite(scores).all() or not np.isfinite(dimensions).all():
                raise ValueError(f"non-finite diagnostic in {path}")
            np.testing.assert_allclose(dimensions.sum(1), scores, rtol=2e-5, atol=1e-7)
            if any(
                r["score_id"] != meta["score_signature"]["score_id"] for r in values
            ):
                raise ValueError("mixed score signatures")
            arrays[source] = dict(
                scores=scores,
                per_dimension=dimensions,
                similarity=np.asarray(
                    [np.mean(r["context"]["similarities"]) for r in values]
                ),
            )
        result.append(
            dict(
                directory=path.parent,
                metadata=meta,
                rows=rows,
                traces=traces,
                arrays=arrays,
                steps=np.asarray(expected),
            )
        )
    if not result:
        raise ValueError("no complete recorded episodes")
    if len({e["metadata"]["score_signature"]["score_id"] for e in result}) != 1:
        raise ValueError("episodes use different scorer configurations")
    if len({tuple(e["arrays"]) for e in result}) != 1:
        raise ValueError("episodes use different diagnostics")
    return result


def rank(values):
    unique, inverse, count = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    return (np.cumsum(count) - (count - 1) / 2)[inverse]


def correlation(left, right, spearman=False):
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if spearman:
        left, right = rank(left), rank(right)
    return float(np.corrcoef(left, right)[0, 1])


def describe(episodes):
    per_episode, per_source = [], {}
    for episode in episodes:
        meta = episode["metadata"]
        source_stats = {}
        baseline = episode["arrays"]["retrieved"]["scores"]
        for source, arrays in episode["arrays"].items():
            scores = arrays["scores"]
            item = dict(
                mean=float(scores.mean()),
                median=float(np.median(scores)),
                maximum=float(scores.max()),
                minimum=float(scores.min()),
                coefficient_of_variation=float(scores.std() / scores.mean())
                if scores.mean()
                else None,
                spearman_vs_context_similarity=correlation(
                    scores, arrays["similarity"], True
                ),
                per_dimension_mean=arrays["per_dimension"].mean(0).tolist(),
            )
            if source != "retrieved":
                item.update(
                    mean_paired_delta=float((scores - baseline).mean()),
                    fraction_above_retrieved=float(np.mean(scores > baseline)),
                    mean_normalized_action_distance=float(
                        np.mean(
                            [
                                r["normalized_action_distance_to_executed"]
                                for r in episode["traces"][source]
                            ]
                        )
                    ),
                )
            source_stats[source] = item
        per_episode.append(
            dict(
                task_id=meta["task_id"],
                episode_index=meta["episode_index"],
                success=meta["success"],
                control_steps=meta["control_steps"],
                num_queries=len(baseline),
                sources=source_stats,
            )
        )
    for source in episodes[0]["arrays"]:
        stats = [e["sources"][source] for e in per_episode]
        per_source[source] = {
            "episode_balanced_mean": float(np.mean([s["mean"] for s in stats])),
            "per_dimension_episode_balanced_mean": np.mean(
                [s["per_dimension_mean"] for s in stats], axis=0
            ).tolist(),
        }
        if source != "retrieved":
            per_source[source].update(
                episode_balanced_mean_paired_delta=float(
                    np.mean([s["mean_paired_delta"] for s in stats])
                ),
                episode_balanced_fraction_above_retrieved=float(
                    np.mean([s["fraction_above_retrieved"] for s in stats])
                ),
                episode_balanced_action_distance=float(
                    np.mean([s["mean_normalized_action_distance"] for s in stats])
                ),
            )
    return dict(
        episodes=per_episode,
        context_sources=per_source,
        num_episodes=len(episodes),
        num_queries=sum(len(e["rows"]) for e in episodes),
        limitations=[
            "Episodes selected by previous outcomes; not an unbiased SR or failure-detection estimate.",
            "Retrieved/random/low-similarity are selection methods, not phase-alignment ground truth.",
            "No phase AUROC, calibrated threshold, failure-onset annotation or coverage claim.",
            "Each context has its own retrieved-action source and actor flow; this is a full-policy context comparison.",
            "Adjacent queries are dependent; descriptive aggregates weight episodes equally.",
        ],
    )


def plot(episodes, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    output = Path(output)
    columns, rows = 2, (len(episodes) + 1) // 2
    sources = list(episodes[0]["arrays"])
    peak = max(float(e["arrays"][s]["scores"].max()) for e in episodes for s in sources)
    for comparison in [False, True]:
        fig, axes = plt.subplots(rows, columns, figsize=(13, 3.7 * rows), squeeze=False)
        for axis, episode in zip(axes.flat, episodes):
            meta = episode["metadata"]
            for source in sources if comparison else ["retrieved"]:
                index = SOURCES.index(source)
                axis.plot(
                    episode["steps"],
                    episode["arrays"][source]["scores"],
                    label=LABELS[index],
                    color=COLORS[index],
                    linewidth=1.6,
                    alpha=0.9,
                )
            axis.set_title(
                f"Task {meta['task_id']} / init {meta['episode_index']} — "
                + ("success" if meta["success"] else "failure (time limit)")
            )
            axis.set(
                xlabel="Control step (score every 16 steps)",
                ylabel=r"$U_{\mathrm{ctx}}$",
                ylim=(0, max(peak * 1.1, 1e-8)),
            )
            axis.grid(alpha=0.2)
            axis.axvline(
                meta["control_steps"], color="gray", linestyle=":", linewidth=1
            )
            if not comparison:
                twin = axis.twinx()
                twin.plot(
                    episode["steps"],
                    1 - episode["arrays"]["retrieved"]["similarity"],
                    color="#6c9158",
                    alpha=0.5,
                    linestyle="--",
                    label="1 − retrieval cosine",
                )
                twin.set_ylabel("1 − retrieval cosine", color="#527b3d")
        for axis in list(axes.flat)[len(episodes) :]:
            axis.set_visible(False)
        axes.flat[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(
            "Paired context diagnostics on the same rollout states"
            if comparison
            else "RA-VLA context uncertainty — selected pilot episodes",
            fontsize=13,
        )
        fig.tight_layout()
        name = "context_comparison_traces" if comparison else "uctx_traces"
        for suffix in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = [
        np.concatenate([e["arrays"][s]["scores"] for e in episodes]) for s in sources
    ]
    axes[0].boxplot(data, showfliers=False)
    axes[0].set_xticks(np.arange(1, len(sources) + 1))
    axes[0].set_xticklabels(
        [LABELS[SOURCES.index(s)].replace(" (", "\n(") for s in sources]
    )
    for e in episodes:
        axes[0].scatter(
            np.arange(1, len(sources) + 1),
            [e["arrays"][s]["scores"].mean() for s in sources],
            marker="o" if e["metadata"]["success"] else "x",
            color="black",
            alpha=0.65,
        )
    axes[0].set(
        ylabel=r"$U_{\mathrm{ctx}}$",
        title="Query distributions; points = episode means",
        ylim=(0, peak * 1.1),
    )
    dimensions = np.mean(
        [e["arrays"]["retrieved"]["per_dimension"].mean(0) for e in episodes], axis=0
    )
    axes[1].bar(
        ["x", "y", "z", "rot x", "rot y", "rot z", "gripper"],
        dimensions,
        color=COLORS[0],
    )
    axes[1].set(
        ylabel="Mean contribution to score",
        title="Retrieved context: contribution by action dimension",
    )
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            output / f"context_distributions.{suffix}", dpi=170, bbox_inches="tight"
        )
    plt.close(fig)
    if len(sources) > 1:
        fig, axes = plt.subplots(rows, columns, figsize=(13, 3.7 * rows), squeeze=False)
        peak_delta = 0.0
        for axis, episode in zip(axes.flat, episodes):
            baseline = episode["arrays"]["retrieved"]["scores"]
            valid = baseline > 0
            for source in sources[1:]:
                values = np.full(len(baseline), np.nan)
                values[valid] = 100 * (
                    episode["arrays"][source]["scores"][valid] / baseline[valid] - 1
                )
                if valid.any():
                    peak_delta = max(peak_delta, float(np.max(np.abs(values[valid]))))
                axis.plot(
                    episode["steps"],
                    values,
                    color=COLORS[SOURCES.index(source)],
                    label=LABELS[SOURCES.index(source)],
                    linewidth=1.4,
                )
            m = episode["metadata"]
            axis.set_title(
                f"Task {m['task_id']} / init {m['episode_index']} — "
                + ("success" if m["success"] else "failure")
            )
            axis.axhline(0, color="gray", linestyle="--", linewidth=1)
            axis.set(xlabel="Control step", ylabel="Score change vs retrieved (%)")
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[: len(episodes)]:
            axis.set_ylim(-max(peak_delta * 1.1, 1.0), max(peak_delta * 1.1, 1.0))
        for axis in list(axes.flat)[len(episodes) :]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(
            "Paired context changes — positive means higher disagreement for the distractor",
            fontsize=12,
        )
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(
                output / f"paired_context_changes.{suffix}",
                dpi=170,
                bbox_inches="tight",
            )
        plt.close(fig)


def baseline_comparison(episodes, baseline_run):
    import av

    baseline_run = Path(baseline_run)
    data = json.loads((baseline_run / "rollouts/summary.json").read_text())
    index = {
        (e["task_id"], e["episode_index"]): (ordinal, e)
        for ordinal, e in enumerate(data["episodes"], start=1)
    }
    results = []

    def selected_frames(path, selected):
        frames = {}
        with av.open(str(path)) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i in selected:
                    frames[i] = frame.to_ndarray(format="rgb24")
                if i >= max(selected):
                    break
        return frames

    for episode in episodes:
        meta = episode["metadata"]
        ordinal, old = index[(meta["task_id"], meta["episode_index"])]
        current = next(
            episode["directory"].parent.parent.glob(
                f"episode={meta['episode_index'] + 1}--*.mp4"
            )
        )
        previous = next((baseline_run / "rollouts").glob(f"episode={ordinal}--*.mp4"))
        selected = sorted(
            {
                0,
                16,
                64,
                min(meta["control_steps"], old["control_steps"]) // 2,
                min(meta["control_steps"], old["control_steps"]) - 1,
            }
        )
        left, right = (
            selected_frames(current, selected),
            selected_frames(previous, selected),
        )
        matches = [
            i
            for i in selected
            if i in left and i in right and np.array_equal(left[i], right[i])
        ]
        results.append(
            dict(
                task_id=meta["task_id"],
                episode_index=meta["episode_index"],
                previous_success=old["success"],
                current_success=meta["success"],
                previous_control_steps=old["control_steps"],
                current_control_steps=meta["control_steps"],
                sampled_frames=selected,
                exact_pixel_matches=matches,
                all_sampled_frames_match=len(matches) == len(selected),
            )
        )
    return results


def filmstrips(episodes, output):
    import torch
    import matplotlib.pyplot as plt

    repo = Path(__file__).resolve().parents[2]
    processor = json.loads(
        (
            repo / "gr00t/model/backbone/eagle2_hg_model/preprocessor_config.json"
        ).read_text()
    )
    mean = np.asarray(processor["image_mean"])[None, None, :]
    std = np.asarray(processor["image_std"])[None, None, :]
    memory_path = episodes[0]["metadata"]["score_signature"]["memory"]["path"]
    memory = torch.load(memory_path, map_location="cpu", mmap=True, weights_only=True)
    for episode in episodes:
        selected = [0, len(episode["rows"]) // 2, len(episode["rows"]) - 1]
        fig, axes = plt.subplots(len(selected), 4, figsize=(18, 10.5), squeeze=False)
        for row_index, query_index in enumerate(selected):
            row = episode["rows"][query_index]
            if row["observation_path"] is None:
                raise ValueError("filmstrips require saved raw observations")
            with np.load(
                episode["directory"] / row["observation_path"], allow_pickle=False
            ) as obs:
                image = np.concatenate(
                    [obs["video.image"][0], obs["video.wrist_image"][0]], axis=1
                )
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(f"Observation: step {row['control_step']}")
            for col, source in enumerate(SOURCES, start=1):
                context = episode["traces"][source][query_index]["context"]
                pixels = (
                    memory[context["task"]]["pixel_values"][context["indices"][0]]
                    .float()
                    .numpy()
                )
                views = [
                    np.clip(
                        (view.transpose(1, 2, 0) * std + mean) * 255, 0, 255
                    ).astype(np.uint8)
                    for view in pixels
                ]
                axes[row_index, col].imshow(np.concatenate(views, axis=1))
                axes[row_index, col].set_title(
                    f"{LABELS[col - 1]}: row {context['indices'][0]}\ncos={context['similarities'][0]:.3f}"
                )
            for axis in axes[row_index]:
                axis.axis("off")
        meta = episode["metadata"]
        fig.suptitle(
            f"Task {meta['task_id']} init {meta['episode_index']}: context selection audit (first of K)"
        )
        fig.subplots_adjust(top=0.91, bottom=0.025, hspace=0.55, wspace=0.05)
        fig.savefig(
            Path(output)
            / f"contexts_task{meta['task_id']:02d}_init{meta['episode_index']:03d}.png",
            dpi=160,
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--filmstrips", action="store_true")
    args = parser.parse_args()
    output = args.output or args.run / "analysis"
    output.mkdir(parents=True, exist_ok=False)
    episodes = load_episodes(args.run)
    metrics = describe(episodes)
    if args.baseline_run:
        metrics["baseline_comparison"] = baseline_comparison(
            episodes, args.baseline_run
        )
    plot(episodes, output)
    if args.filmstrips:
        filmstrips(episodes, output)
    atomic_json(output / "metrics.json", metrics)
    with (output / "queries.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "task_id",
                "episode_index",
                "success",
                "control_step",
                "context_source",
                "u_ctx",
                "context_cosine",
            ]
        )
        for e in episodes:
            for source, arrays in e["arrays"].items():
                for step, score, sim in zip(
                    e["steps"], arrays["scores"], arrays["similarity"]
                ):
                    m = e["metadata"]
                    writer.writerow(
                        [
                            m["task_id"],
                            m["episode_index"],
                            m["success"],
                            step,
                            source,
                            score,
                            sim,
                        ]
                    )
    lines = [
        "# Context uncertainty pilot",
        "",
        "Selected episodes; descriptive results only.",
        "",
        "| Task / init | Outcome | Steps | Queries | Mean U_ctx | Max U_ctx |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for e in metrics["episodes"]:
        s = e["sources"]["retrieved"]
        lines.append(
            f"| {e['task_id']} / {e['episode_index']} | {'success' if e['success'] else 'failure'} | {e['control_steps']} | {e['num_queries']} | {s['mean']:.6g} | {s['maximum']:.6g} |"
        )
    lines += [
        "",
        "## Context comparison",
        "",
        "Episode-balanced aggregates; positive delta means larger disagreement than retrieved context.",
        "",
        "| Source | Mean U_ctx | Mean paired delta | Fraction above retrieved |",
        "|---|---:|---:|---:|",
    ]
    for name, value in metrics["context_sources"].items():
        delta, fraction = (
            value.get("episode_balanced_mean_paired_delta"),
            value.get("episode_balanced_fraction_above_retrieved"),
        )
        lines.append(
            f"| {name} | {value['episode_balanced_mean']:.6g} | {delta if delta is not None else '—'} | {fraction if fraction is not None else '—'} |"
        )
    lines += ["", "## Limits", ""] + ["- " + item for item in metrics["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
