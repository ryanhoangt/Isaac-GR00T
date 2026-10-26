"""Counterfactual goal probes, summaries and standalone diagnostic plots."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sensitivity(args):
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from examples.Libero.custom_data_config import LiberoDataConfig
    from .evaluate import condition, predict
    from .runtime import ReferenceBuffer, VisionFeatures
    from .training import load_policy

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(Path(args.split).read_text())
    policy = load_policy(args.checkpoint, split["split_id"], args.device)
    if not getattr(policy.model.action_head.config, "goal_conditioning", False):
        raise ValueError("Goal sensitivity requires a dual checkpoint")
    reference = ReferenceBuffer(args.index, split["split_id"], VisionFeatures(policy))
    if reference.signature["encoder"] != policy.pilot2_recipe["base_model"]:
        raise ValueError("Reference encoder does not match the checkpoint")
    config = LiberoDataConfig()
    root = Path(args.split).resolve().parent / "reference"
    dataset = LeRobotSingleDataset(
        root,
        config.modality_config(),
        EmbodimentTag.NEW_EMBODIMENT,
        video_backend="torchvision_av",
        video_backend_kwargs={"num_threads": 1},
    )
    task_map = {item["episode_index"]: item["task_id"] for item in split["reference"]}
    rng = np.random.default_rng(args.seed)
    count = min(args.states, len(dataset))
    indices = rng.choice(len(dataset), count, replace=False)
    rows = []
    for sample_id, idx in enumerate(indices):
        episode, step = dataset.all_steps[idx]
        raw = dataset.get_step_data(episode, step)
        task = task_map[episode]
        # Do not retrieve the query itself from the held-out reference episodes:
        # probes use a direct future hindsight goal and a different-task distractor.
        goal_step = min(step + 16, len(reference.data[episode]) - 1)
        goal = reference.image(episode, goal_step)
        distractors = np.flatnonzero(reference.tasks != task)
        selected = int(rng.choice(distractors))
        random_goal = reference.image(int(reference.episodes[selected]), int(reference.frames[selected]))
        # A raw dataset observation includes actions; remove them for inference.
        raw = {key: value for key, value in raw.items() if not key.startswith("action.")}
        if sample_id == 0:
            predict(policy, condition(raw, "both", goal), args.seed)  # unmeasured warmup
            predict(policy, condition(raw, "language"), args.seed)
        predictions, latencies = {}, {}
        for mode in ("both", "language", "random"):
            obs = condition(
                raw, "language" if mode == "language" else "both", goal if mode == "both" else random_goal
            )
            predictions[mode], latencies[mode] = predict(policy, obs, args.seed + sample_id)
        denominator = max(float(np.linalg.norm(predictions["both"])), 1e-8)
        rows.append(
            {
                "sample": sample_id,
                "episode": int(episode),
                "frame": int(step),
                "task_id": task,
                "goal_vs_null": float(np.linalg.norm(predictions["both"] - predictions["language"]) / denominator),
                "goal_vs_random": float(np.linalg.norm(predictions["both"] - predictions["random"]) / denominator),
                "random_vs_null": float(
                    np.linalg.norm(predictions["random"] - predictions["language"]) / denominator
                ),
                "goal_chunk_norm": denominator,
                "language_latency_ms": latencies["language"],
                "goal_latency_ms": latencies["both"],
            }
        )
        if sample_id % 20 == 0:
            print(f"Sensitivity: {sample_id + 1}/{count}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "split_id": split["split_id"],
        "states": count,
        "requested_states": args.states,
        "validation": "reference-only states, not used for training; reserved for this diagnostic",
        "noise": "shared initial flow noise for all three counterfactuals",
        "action_space": "denormalized 16x7, excludes padded channels",
        "latency": "synchronized end-to-end get_action, excludes retrieval; one warmup per mode",
        "interpretation": "sensitivity is evidence of dependence, not proof that the goal is correct",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def summarize(inputs, output, sensitivity_csv=None, allow_smoke=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups, split_ids, training_signatures, all_episodes = defaultdict(list), set(), set(), []
    for root in map(Path, inputs):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"Evaluation is not completed: {root}")
        split_ids.add(manifest["split_id"])
        if "training_recipe" in manifest:
            training = manifest["training_recipe"]
            comparable = (
                "base_model",
                "seed",
                "batch_size",
                "num_gpus",
                "max_steps",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "scheduler",
                "normalization",
            )
            training_signatures.add(json.dumps({key: training[key] for key in comparable}, sort_keys=True))
        settings = manifest["settings"]
        label = (
            settings["mode"]
            if settings["strategy"] == "standard"
            else f"strategy_{settings['strategy']}_t{settings['trigger']}"
        )
        for name in manifest["episodes"]:
            row = json.loads((root / f"{name}.json").read_text())
            if row["smoke_only"] and not allow_smoke:
                raise ValueError("Smoke rollouts cannot be included in scientific summaries")
            groups[label].append(row)
            all_episodes.append((label, row))
    if len(split_ids) > 1:
        raise ValueError("Cannot compare different dataset splits")
    if len(training_signatures) > 1:
        raise ValueError("Training budgets/seeds/normalization differ; report these runs separately")
    prefixes = defaultdict(set)
    for _, row in all_episodes:
        prefix = row.get("pre_recovery_ee_sha256")
        if prefix is not None:
            prefixes[row["task_id"], row["initial_state_index"], row["trigger"], row["seed"]].add(prefix)
    if any(len(values) > 1 for values in prefixes.values()):
        raise ValueError("A/B/C have different pre-recovery EE trajectories; investigate before comparing")
    records = []
    for label, episodes in sorted(groups.items()):
        ids = [(item["task_id"], item["initial_state_index"]) for item in episodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate task/initial-state trials in {label}")
        for task in ["all"] + sorted({item["task_id"] for item in episodes}):
            selected = episodes if task == "all" else [item for item in episodes if item["task_id"] == task]
            eligible = [item for item in selected if item["recovery_eligible"]]
            subtasks = [
                item["subtask_predicate_fraction"]
                for item in selected
                if item["subtask_predicate_fraction"] is not None
            ]
            proxy = [item["goal_ee_proxy_rate"] for item in selected if item["goal_ee_proxy_rate"] is not None]
            latencies = [
                p["latency_ms"] for item in selected for p in item["predictions"] if p["latency_ms"] is not None
            ]
            by_task = defaultdict(list)
            for item in selected:
                by_task[item["task_id"]].append(item["success"])
            row = {
                "condition": label,
                "task_id": task,
                "episodes": len(selected),
                "tasks_covered": len(by_task),
                "success_pct": float(np.mean([item["success"] for item in selected]) * 100),
                "macro_task_success_pct": float(np.mean([np.mean(values) for values in by_task.values()]) * 100),
                "subtask_predicate_pct": float(np.mean(subtasks) * 100) if subtasks else None,
                "goal_ee_proxy_pct": float(np.mean(proxy) * 100) if proxy else None,
                "recovery_eligible": len(eligible),
                "recovery_pct": float(np.mean([item["success"] for item in eligible]) * 100) if eligible else None,
                "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
                "latency_ms_p95": float(np.quantile(latencies, 0.95)) if latencies else None,
            }
            for delta in (0, 8, 16, 32):
                values = [item["contraction_m"].get(str(delta)) for item in selected]
                values = [value for value in values if value is not None]
                row[f"d_ee_{delta}_m"] = float(np.mean(values)) if values else None
                row[f"d_ee_{delta}_n"] = len(values)
            records.append(row)
    if records:
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    overall = [row for row in records if row["task_id"] == "all"]
    latency_groups = defaultdict(list)
    for label, episode in all_episodes:
        for prediction in episode["predictions"]:
            if prediction["latency_ms"] is not None:
                latency_groups[label, prediction["mode"]].append(prediction["latency_ms"])
    if latency_groups:
        with (output / "latency_by_mode.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["condition", "mode", "chunks", "mean_ms", "p95_ms"])
            writer.writeheader()
            for (label, mode), values in sorted(latency_groups.items()):
                writer.writerow(
                    {
                        "condition": label,
                        "mode": mode,
                        "chunks": len(values),
                        "mean_ms": np.mean(values),
                        "p95_ms": np.quantile(values, 0.95),
                    }
                )
    lines = [
        "# Pilot 2 summary",
        "",
        "Debug smoke results only; not a full scientific evaluation."
        if allow_smoke
        else "Full-horizon rollout results.",
        "",
        "Subtask = final fraction of simulator BDDL task predicates; goal EE distance is a separate proxy.",
        "Contraction is descriptive; global reference EE distance does not establish a Lyapunov guarantee.",
        "",
        "| Condition | N | Tasks | Task success % | Recovery % | Subtask predicates % | ms/chunk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def format_value(value):
        return "N/A" if value is None else f"{value:.2f}"

    for row in overall:
        lines.append(
            f"| {row['condition']} | {row['episodes']} | {row['tasks_covered']} | {row['success_pct']:.2f} | "
            f"{format_value(row['recovery_pct'])} | {format_value(row['subtask_predicate_pct'])} | "
            f"{format_value(row['latency_ms_mean'])} |"
        )
    if overall:
        fig, ax = plt.subplots(figsize=(max(7, len(overall) * 1.2), 4))
        ax.bar([row["condition"] for row in overall], [row["success_pct"] for row in overall])
        ax.set(ylabel="Task success (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output / "success.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for row in overall:
            values = [row[f"d_ee_{delta}_m"] for delta in (0, 8, 16, 32)]
            if any(value is not None for value in values):
                ax.plot(
                    [0, 8, 16, 32],
                    [np.nan if value is None else value for value in values],
                    marker="o",
                    label=row["condition"],
                )
        ax.set(xlabel="Steps after perturbation ends", ylabel="Nearest reference EE distance (m)")
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "contraction.png", dpi=160)
        plt.close(fig)
    if sensitivity_csv:
        with Path(sensitivity_csv).open() as stream:
            probes = list(csv.DictReader(stream))
        if not probes:
            raise ValueError("Empty sensitivity CSV")
        diagnostics = {
            key: float(np.mean([float(row[key]) for row in probes]))
            for key in (
                "goal_vs_null",
                "goal_vs_random",
                "random_vs_null",
                "language_latency_ms",
                "goal_latency_ms",
            )
        }
        (output / "sensitivity_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        lines.extend(
            [
                "",
                f"Sensitivity states: {len(probes)}.",
                "",
                *[f"- {key}: {value:.4f}" for key, value in diagnostics.items()],
            ]
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].boxplot(
            [[float(row[key]) for row in probes] for key in ("goal_vs_null", "goal_vs_random", "random_vs_null")],
            labels=["goal/null", "goal/random", "random/null"],
            showfliers=False,
        )
        axes[0].set_ylabel("Relative action-chunk difference")
        axes[1].bar(
            ["language", "language+goal"], [diagnostics["language_latency_ms"], diagnostics["goal_latency_ms"]]
        )
        axes[1].set_ylabel("ms / chunk (4 Euler steps)")
        fig.tight_layout()
        fig.savefig(output / "sensitivity_latency.png", dpi=160)
        plt.close(fig)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return records
