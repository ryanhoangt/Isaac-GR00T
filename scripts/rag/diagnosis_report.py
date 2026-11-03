"""Summarize diagnostic rollouts without treating training/probes as task success."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from gr00t.rag.diagnosis_protocol import audit_recovery, digest


def read_episodes(output, manifest):
    episodes = []
    for path in sorted(Path(output).glob("*/task*_init*/episode.json")):
        episode = json.loads(path.read_text())
        expected = digest(
            {
                "run": manifest["signature"],
                "case": episode["case"],
                "task": episode["task_id"],
                "initial": episode["initial_index"],
            }
        )
        if episode.get("signature") != expected or not episode.get("complete"):
            raise ValueError(f"Invalid episode result: {path}")
        episode["directory"] = str(path.parent)
        episodes.append(episode)
    return episodes


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    episodes = read_episodes(output, manifest)
    audit = audit_recovery(episodes)
    (output / "pairing_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["case"]["name"]].append(episode)
    rows, task_rows, comparisons, phase_rows = [], [], [], []
    for name, values in groups.items():
        case = values[0]["case"]
        queries = [q for episode in values for q in episode["queries"]]
        eligible = [episode for episode in values if episode["recovery_eligible"]]
        predicate_values = [e["predicate_completion"] for e in values if e.get("predicate_completion") is not None]
        row = {
            "condition": name,
            "kind": case["kind"],
            "strategy": case["strategy"],
            "trigger": case.get("trigger"),
            "scale": case.get("scale"),
            "episodes": len(values),
            "successes": sum(e["success"] for e in values),
            "success_pct": 100 * np.mean([e["success"] for e in values]),
            "final_predicate_completion_pct": 100 * np.mean(predicate_values) if predicate_values else None,
            "recovery_eligible": len(eligible),
            "recovery_success_pct": 100 * np.mean([e["success"] for e in eligible]) if eligible else None,
            "query_ms": float(np.mean([q["query_ms"] for q in queries])) if queries else None,
            "rpc_ms": float(np.mean([q["rpc_ms"] for q in queries])) if queries else None,
            "support_fp32": float(
                np.mean([np.mean([q["support_fp32"] for q in e["queries"]]) for e in values if e["queries"]])
            ),
            "smoke_only": any(e["smoke_only"] for e in values),
        }
        rows.append(row)
        for task in sorted({e["task_id"] for e in values}):
            selected = [e for e in values if e["task_id"] == task]
            task_rows.append(
                {
                    "condition": name,
                    "task_id": task,
                    "episodes": len(selected),
                    "successes": sum(e["success"] for e in selected),
                    "success_pct": 100 * np.mean([e["success"] for e in selected]),
                }
            )
        for phase in sorted({q["phase"] for q in queries}):
            selected = [q for q in queries if q["phase"] == phase]
            phase_rows.append(
                {
                    "condition": name,
                    "phase": phase,
                    "queries": len(selected),
                    "query_ms": float(np.mean([q["query_ms"] for q in selected])),
                    "rpc_ms": float(np.mean([q["rpc_ms"] for q in selected])),
                }
            )
    recovery = defaultdict(dict)
    for episode in episodes:
        case = episode["case"]
        if case["kind"] == "recovery":
            recovery[(case["trigger"], case["scale"], episode["task_id"], episode["initial_index"])][
                case["strategy"]
            ] = episode
    cohorts = defaultdict(list)
    paired = defaultdict(list)
    for (trigger, scale, task, initial), group in recovery.items():
        if set(group) != set("ABC"):
            continue
        for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
            if group[first]["recovery_eligible"] and group[second]["recovery_eligible"]:
                paired[(trigger, scale, first, second)].append((group[first]["success"], group[second]["success"]))
        if all(
            e["recovery_eligible"] and all(v is not None for v in e["contraction"].values())
            for e in group.values()
        ):
            cohorts[(trigger, scale)].append(group)
    for (trigger, scale, first, second), values in paired.items():
        wins = sum(not a and b for a, b in values)
        losses = sum(a and not b for a, b in values)
        comparisons.append(
            {
                "trigger": trigger,
                "scale": scale,
                "first": first,
                "second": second,
                "pairs": len(values),
                "second_wins": wins,
                "second_losses": losses,
                "net_pp": 100 * (wins - losses) / len(values),
            }
        )
    contraction = []
    for (trigger, scale), groups_at_level in cohorts.items():
        for strategy in "ABC":
            for delta in (0, 8, 16, 32):
                contraction.append(
                    {
                        "trigger": trigger,
                        "scale": scale,
                        "strategy": strategy,
                        "delta": delta,
                        "common_pairs": len(groups_at_level),
                        "d_ee_cm": 100
                        * np.mean([g[strategy]["contraction"][str(delta)] for g in groups_at_level]),
                    }
                )
    write_csv(output / "summary.csv", rows)
    write_csv(output / "per_task.csv", task_rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "contraction.csv", contraction)
    write_csv(output / "latency_by_phase.csv", phase_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rows:
        fig, axis = plt.subplots(figsize=(max(7, len(rows)), 4))
        axis.bar([r["condition"] for r in rows], [r["success_pct"] for r in rows])
        axis.set(ylabel="Task success (%)", ylim=(0, 100))
        axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output / "success.png", dpi=160)
        plt.close(fig)
    if contraction:
        levels = list(cohorts)
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), squeeze=False)
        for axis, (trigger, scale) in zip(axes[0], levels):
            for strategy in "ABC":
                selected = [
                    r
                    for r in contraction
                    if r["trigger"] == trigger and r["scale"] == scale and r["strategy"] == strategy
                ]
                axis.plot(
                    [r["delta"] for r in selected], [r["d_ee_cm"] for r in selected], marker="o", label=strategy
                )
            axis.set(
                title=f"t={trigger}, scale={scale:g}, common n={len(cohorts[(trigger, scale)])}",
                xlabel="Steps after perturbation ends",
                ylabel="Global reference EE distance (cm)",
            )
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / "contraction.png", dpi=160)
        plt.close(fig)
    if episodes:
        representatives = [values[0] for values in groups.values()]
        fig, axes = plt.subplots(len(representatives), 1, figsize=(9, 2.2 * len(representatives)), squeeze=False)
        for axis, episode in zip(axes[:, 0], representatives):
            queries = episode["queries"]
            axis.plot(
                [q["control_step"] for q in queries],
                [q["support_fp32"] for q in queries],
                label="FP32 support distance",
            )
            axis.plot(
                [q["control_step"] for q in queries],
                [q["support_bf16"] for q in queries],
                alpha=0.5,
                label="Legacy BF16",
            )
            if episode["case"]["kind"] == "recovery":
                axis.axvspan(episode["case"]["trigger"], episode["case"]["trigger"] + 4, color="red", alpha=0.15)
            axis.set(
                title=f"{episode['case']['name']} task={episode['task_id']} init={episode['initial_index']}",
                ylabel="1 - cosine",
            )
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("Control step")
        fig.tight_layout()
        fig.savefig(output / "support_traces.png", dpi=140)
        plt.close(fig)
    expected = manifest["expected_episodes"]
    lines = ["# Dual-mode RA-VLA diagnosis", ""]
    if expected or episodes:
        lines.extend(
            [
                f"Completed episodes: {len(episodes)}/{expected}.",
                "",
                "| Condition | Success | Eligible recovery |",
                "|---|---:|---:|",
            ]
        )
        lines.extend(
            f"| {r['condition']} | {r['successes']}/{r['episodes']} ({r['success_pct']:.1f}%) | {r['recovery_eligible']} |"
            for r in rows
        )
    else:
        lines.append("Goal probes only; no simulator rollouts requested.")
    probe_path = output / "probes" / "summary.json"
    if probe_path.exists():
        probes = json.loads(probe_path.read_text())
        lines.extend(
            [
                "",
                f"Goal probes: {probes['states']} held-out reference states; own-demo retrieval excluded.",
                "",
                "| Goal replacement | Mean relative action change | Median |",
                "|---|---:|---:|",
            ]
        )
        for title, key in (
            ("Wrong-time, same demo", "wrong_relative_all"),
            ("Random, same task", "random_relative_all"),
        ):
            value = probes[key]
            lines.append(f"| {title} | {value['mean']:.3f} | {value['median']:.3f} |")
        lines.append("")
        lines.append(
            "Both predictions share Gaussian noise; all seven denormalized action channels enter the L2 ratio."
        )
    lines.extend(
        [
            "",
            "Support is logged only; no uncertainty gate or calibrated threshold is used.",
            "EE distance is a global geometric proxy; observed EE motion includes nominal control and contact response.",
            "Goal probes use reference states and temporal-negative proxies, not semantic phase labels or recovery outcomes.",
        ]
    )
    if any(r["smoke_only"] for r in rows):
        lines.append("SMOKE ONLY: truncated episode budgets do not estimate benchmark success rates.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return rows, audit
