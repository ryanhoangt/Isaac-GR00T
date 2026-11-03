"""Audit paired A/B/C interventions and report recovery on identical trials."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .task_mapping import validate_retrieval_provenance


def validate_intervention(row):
    start = row["trigger"] + 4
    applied = row["perturbation_applied_steps"]
    if applied not in range(5) or [p["step"] for p in row["perturbation_commands"]] != list(
        range(row["trigger"], row["trigger"] + applied)
    ):
        raise ValueError("Perturbation command log does not match the executed four-step window")
    if row["recovery_eligible"] != (row["perturbation_applied_steps"] == 4):
        raise ValueError("Recovery eligibility disagrees with the four-step disturbance")
    for prediction in row["predictions"]:
        step = prediction["step"]
        strategy = row["strategy"]
        expected = "both" if strategy == "C" and start <= step < start + 32 else "language"
        phase = "copy" if strategy == "B" and start <= step < start + 16 else expected
        if prediction["mode"] != expected or (phase in ("copy", "both") and prediction["phase"] != phase):
            raise ValueError("Recovery query used an incorrect mode or action-copy phase")
        if strategy == "B" and step >= start + 16 and prediction["phase"] == "copy":
            raise ValueError("B copied expert actions beyond its one-chunk window")
    if not row["recovery_eligible"] or row["action_steps"] <= start:
        return
    intervention = [p for p in row["predictions"] if p["step"] >= start]
    if not intervention or intervention[0]["step"] != start:
        raise ValueError("Pending actions were not flushed immediately after perturbation")
    if row["strategy"] == "C":
        goals = [p for p in intervention if p["step"] < start + 32]
        if row["action_steps"] > start + 16 and [p["step"] for p in goals] != [start, start + 16]:
            raise ValueError("C must predict exactly two consecutive goal-conditioned chunks")
        if len({(p["episode"], p["goal_frame"]) for p in goals}) != 1:
            raise ValueError("C changed its fixed milestone during recovery")


def audit_pairs(inputs, allow_smoke=False):
    groups = defaultdict(dict)
    signatures = set()
    for root in map(Path, inputs):
        manifest = json.loads((root / "manifest.json").read_text())
        settings = manifest["settings"]
        scale = settings.get("perturb_scale", 1.0)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Recovery perturbation scale must be finite and positive")
        if manifest["status"] != "completed" or manifest.get("recovery_protocol_version") != 1:
            raise ValueError(f"Incomplete or legacy recovery run: {root}")
        strategy, trigger = settings["strategy"], settings["trigger"]
        if strategy not in ("A", "B", "C") or trigger not in (25, 60) or settings["mode"] != "language":
            raise ValueError("Expected A/B/C with a language-only prefix and trigger 25/60")
        if strategy in groups[trigger]:
            raise ValueError(f"Duplicate strategy {strategy} at trigger {trigger}")
        signatures.add(
            json.dumps(
                {
                    "settings": {
                        key: settings[key]
                        for key in (
                            "checkpoint",
                            "split",
                            "index",
                            "task_ids",
                            "trials",
                            "initial_start",
                            "max_steps",
                            "seed",
                        )
                    },
                    "split_id": manifest["split_id"],
                    "control": manifest["control"],
                    "recipe": manifest["training_recipe"],
                    "protocol": manifest["perturbation_protocol"],
                    "perturb_scale": scale,
                },
                sort_keys=True,
            )
        )
        rows = {}
        for name in manifest["episodes"]:
            row = json.loads((root / f"{name}.json").read_text())
            validate_retrieval_provenance(manifest, row)
            validate_intervention(row)
            if not row["complete"] or (row["smoke_only"] and not allow_smoke):
                raise ValueError("Incomplete/smoke rollout cannot enter this recovery comparison")
            if row["strategy"] != strategy or row["trigger"] != trigger:
                raise ValueError("Episode strategy/trigger disagrees with its manifest")
            key = (row["task_id"], row["initial_state_index"])
            if row["seed"] != settings["seed"] + key[0] * 10000 + key[1] * 1000:
                raise ValueError("Episode flow/disturbance seed disagrees with its task and initial state")
            target = np.asarray(row["perturbation_target_m"])
            if row.get("perturbation_scale", 1.0) != scale:
                raise ValueError("Episode perturbation scale disagrees with its manifest")
            base = target / scale
            if (
                base.shape != (3,)
                or not np.isfinite(base).all()
                or np.any(np.abs(base) > 0.07 + 1e-12)
                or not 0.05 - 1e-12 <= np.linalg.norm(base) <= 0.08 + 1e-12
            ):
                raise ValueError("Scaled disturbance target is outside its declared severity bounds")
            if "perturbation_base_target_m" in row and not np.allclose(
                row["perturbation_base_target_m"], base, atol=1e-12, rtol=0
            ):
                raise ValueError("Stored base disturbance disagrees with the scaled target")
            if key in rows:
                raise ValueError("Duplicate recovery task/initial-state pair")
            rows[key] = row
        expected = {
            (task, initial)
            for task in settings["task_ids"]
            for initial in range(settings["initial_start"], settings["initial_start"] + settings["trials"])
        }
        if set(rows) != expected:
            raise ValueError(f"Missing requested recovery episodes: {root}")
        groups[trigger][strategy] = rows
    if len(signatures) != 1:
        raise ValueError("Recovery checkpoints, initial states, budgets, seeds or protocols differ")
    counts = {}
    for trigger, strategies in groups.items():
        if set(strategies) != set("ABC"):
            raise ValueError(f"Trigger {trigger} requires all three strategies A/B/C")
        keys = set(strategies["A"])
        if any(set(rows) != keys for rows in strategies.values()):
            raise ValueError("A/B/C must cover identical task/initial-state pairs")
        eligible = common32 = 0
        for key in sorted(keys):
            rows = [strategies[s][key] for s in "ABC"]
            for field in (
                "seed",
                "reference_task_id",
                "initial_state_sha256",
                "perturbation_applied_steps",
                "recovery_eligible",
            ):
                if rows[0].get(field) is None or any(r.get(field) != rows[0][field] for r in rows[1:]):
                    raise ValueError(f"A/B/C differ in {field} at trigger={trigger}, trial={key}")
            if any(r["perturbation_target_m"] != rows[0]["perturbation_target_m"] for r in rows[1:]):
                raise ValueError("A/B/C have different perturbation targets")
            if not rows[0]["recovery_eligible"]:
                continue
            eligible += 1
            for field in ("pre_recovery_ee_sha256", "pre_recovery_sim_sha256", "pre_recovery_actions_sha256"):
                if not rows[0].get(field) or any(r.get(field) != rows[0][field] for r in rows[1:]):
                    raise ValueError(f"A/B/C have different {field} at trigger={trigger}, trial={key}")
            if any(r["perturbation_commands"] != rows[0]["perturbation_commands"] for r in rows[1:]):
                raise ValueError("A/B/C executed different perturbation commands")
            if not np.allclose(
                [r["contraction_m"]["0"] for r in rows], rows[0]["contraction_m"]["0"], atol=1e-12, rtol=0
            ):
                raise ValueError("A/B/C have different distances at the recovery starting state")
            # Same query state, feature encoder and reference pool must select the same milestone.
            first = [[p for p in r["predictions"] if p["step"] == trigger + 4] for r in rows[1:]]
            if all(first):
                selected = [tuple(p[0][k] for k in ("episode", "frame", "goal_frame")) for p in first]
                if selected[0] != selected[1]:
                    raise ValueError("B and C retrieved different milestones from the same recovery state")
            elif any(r["action_steps"] > trigger + 4 for r in rows[1:]):
                raise ValueError("B/C did not start their intervention immediately after perturbation")
            common32 += int(all(r["contraction_m"].get("32") is not None for r in rows))
        counts[str(trigger)] = {
            "paired_trials": len(keys),
            "fully_perturbed": eligible,
            "excluded_before_full_perturbation": len(keys) - eligible,
            "common_observed_32_step_cohort": common32,
        }
    return groups, counts


def _csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize_recovery(inputs, output, allow_smoke=False, resume=False):
    from .diagnostics import summarize
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, counts = audit_pairs(inputs, allow_smoke)
    output = Path(output)
    records = summarize(inputs, output, allow_smoke=allow_smoke, resume=resume)
    (output / "pairing_audit.json").write_text(json.dumps(counts, indent=2) + "\n")
    paired, comparisons, contraction, latency, delivery = [], [], [], [], []
    scales = set()
    for trigger, strategies in sorted(groups.items()):
        eligible_keys = [key for key, row in strategies["A"].items() if row["recovery_eligible"]]
        common_keys = [
            key
            for key in eligible_keys
            if all(strategies[s][key]["contraction_m"].get("32") is not None for s in "ABC")
        ]
        for key in sorted(strategies["A"]):
            row = {
                "trigger": trigger,
                "task_id": key[0],
                "initial_state_index": key[1],
                "recovery_eligible": strategies["A"][key]["recovery_eligible"],
            }
            for strategy in "ABC":
                episode = strategies[strategy][key]
                scale = episode.get("perturbation_scale", 1.0)
                scales.add(scale)
                row[f"{strategy}_success"] = episode["success"]
                target = np.asarray(episode["perturbation_target_m"])
                command = episode.get("perturbation_command_offset_m")
                observed = episode.get("perturbation_measured_displacement_m")
                delivery.append(
                    {
                        "trigger": trigger,
                        "strategy": strategy,
                        "task_id": key[0],
                        "initial_state_index": key[1],
                        "perturb_scale": scale,
                        "applied_steps": episode["perturbation_applied_steps"],
                        "requested_norm_cm": float(np.linalg.norm(target) * 100),
                        "command_norm_cm": None if command is None else float(np.linalg.norm(command) * 100),
                        "delivery_projection_fraction": None
                        if command is None
                        else float(np.dot(command, target) / np.dot(target, target)),
                        "observed_EE_motion_cm": None
                        if observed is None
                        else float(np.linalg.norm(observed) * 100),
                        "clipped_actions": sum(p.get("clipped", False) for p in episode["perturbation_commands"]),
                    }
                )
                for delta in (0, 8, 16, 32):
                    row[f"{strategy}_d_ee_{delta}_m"] = episode["contraction_m"].get(str(delta))
            paired.append(row)
        for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
            wins = sum(
                not strategies[first][k]["success"] and strategies[second][k]["success"] for k in eligible_keys
            )
            losses = sum(
                strategies[first][k]["success"] and not strategies[second][k]["success"] for k in eligible_keys
            )
            comparisons.append(
                {
                    "trigger": trigger,
                    "comparison": f"{second}_vs_{first}",
                    "paired_eligible": len(eligible_keys),
                    "second_only_success": wins,
                    "first_only_success": losses,
                    "net_gain_pp": 100 * (wins - losses) / len(eligible_keys) if eligible_keys else None,
                }
            )
        for strategy in "ABC":
            for delta in (0, 8, 16, 32):
                values = [strategies[strategy][k]["contraction_m"][str(delta)] for k in common_keys]
                contraction.append(
                    {
                        "trigger": trigger,
                        "strategy": strategy,
                        "delta_steps": delta,
                        "paired_cohort_n": len(values),
                        "mean_distance_m": float(np.mean(values)) if values else None,
                    }
                )
            by_phase = defaultdict(list)
            for episode in strategies[strategy].values():
                for prediction in episode["predictions"]:
                    by_phase[prediction["phase"]].append(prediction)
            for phase, predictions in by_phase.items():
                model = [p["latency_ms"] for p in predictions if p["latency_ms"] is not None]
                latency.append(
                    {
                        "trigger": trigger,
                        "strategy": strategy,
                        "phase": phase,
                        "chunks": len(predictions),
                        "model_ms": float(np.mean(model)) if model else None,
                        "retrieval_ms": float(np.mean([p["retrieval_ms"] for p in predictions])),
                        "query_ms": float(
                            np.mean([(p["latency_ms"] or 0) + p["retrieval_ms"] for p in predictions])
                        ),
                    }
                )
    _csv(output / "paired_trials.csv", paired)
    _csv(output / "paired_comparisons.csv", comparisons)
    _csv(output / "paired_contraction.csv", contraction)
    _csv(output / "latency_by_phase.csv", latency)
    _csv(output / "disturbance_delivery.csv", delivery)
    scale = next(iter(scales))
    (output / "disturbance_metadata.json").write_text(
        json.dumps({"scale": scale, "requested_norm_cm": [5 * scale, 8 * scale], "duration_steps": 4}, indent=2)
        + "\n"
    )
    triggers = sorted(groups)
    fig, axes = plt.subplots(1, len(triggers), figsize=(6 * len(triggers), 4), squeeze=False)
    for ax, trigger in zip(axes[0], triggers):
        for strategy in "ABC":
            rows = [r for r in contraction if r["trigger"] == trigger and r["strategy"] == strategy]
            if rows[0]["paired_cohort_n"]:
                ax.plot(
                    [r["delta_steps"] for r in rows],
                    [r["mean_distance_m"] for r in rows],
                    marker="o",
                    label=strategy,
                )
        ax.set(
            title=f"{scale:g}x / trigger {trigger}; common cohort N={counts[str(trigger)]['common_observed_32_step_cohort']}",
            xlabel="Actions after perturbation ends",
            ylabel="Nearest reference EE distance (m)",
        )
        if ax.lines:
            ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(output / "paired_contraction.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(triggers))
    for offset, strategy in zip((-0.25, 0, 0.25), "ABC"):
        values = []
        for trigger in triggers:
            selected = [row for row in groups[trigger][strategy].values() if row["recovery_eligible"]]
            values.append(float(np.mean([row["success"] for row in selected]) * 100) if selected else np.nan)
        ax.bar(x + offset, values, width=0.25, label=strategy)
    ax.set(
        xticks=x,
        xticklabels=[f"Trigger {t}" for t in triggers],
        ylabel="Task completion after full perturbation (%)",
        ylim=(0, 100),
    )
    ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(output / "recovery_success.png", dpi=160)
    plt.close(fig)
    lines = [
        "",
        "## Paired recovery audit",
        "",
        f"Perturbation scale: {scale:g}x; requested command-offset norm {5 * scale:g}..{8 * scale:g} cm over four actions.",
        "disturbance_delivery.csv records requested and delivered command-equivalent offsets, clipping and observed EE motion. A 2x/4x request need not produce a 2x/4x physical disturbance.",
        "A/B/C share official initial states, flow seeds, disturbance commands and exact simulator state at recovery start.",
        "Recovery denominator includes trials receiving all four perturbation actions; earlier terminations are excluded.",
        "Observed EE displacement during perturbation includes nominal policy motion and is not a causal perturbation-only displacement.",
        "",
        "| Trigger | Paired trials | Fully perturbed | Common observed 32-step cohort |",
        "|---|---:|---:|---:|",
    ]
    for trigger, row in sorted(counts.items()):
        lines.append(
            f"| {trigger} | {row['paired_trials']} | {row['fully_perturbed']} | {row['common_observed_32_step_cohort']} |"
        )
    lines += [
        "",
        "The paired contraction plot uses the same trials at all four time points and across strategies. It excludes trials ending before +32, including early successes. Raw available-case distances and counts remain in summary.csv. Neither curve establishes a Lyapunov guarantee.",
        "",
        "| Comparison | Trigger | N | Second only succeeds | First only succeeds | Net gain (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        gain = "N/A" if row["net_gain_pp"] is None else f"{row['net_gain_pp']:.2f}"
        lines.append(
            f"| {row['comparison']} | {row['trigger']} | {row['paired_eligible']} | {row['second_only_success']} | {row['first_only_success']} | {gain} |"
        )
    with (output / "summary.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    return records
