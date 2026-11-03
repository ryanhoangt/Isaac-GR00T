"""Audit gate decisions against frozen thresholds and summarize actual executed modes."""

import json
from pathlib import Path

import numpy as np

from gr00t.rag.diagnosis_protocol import digest, instruction_key
from gr00t.rag.support_gate import GateConfig, decide
from scripts.rag.diagnosis_report import read_episodes, write_csv


def audit_episode(episode, manifest):
    case = episode["case"]
    config = GateConfig(**manifest["settings"]["gate"])
    records = manifest["calibration"]["tasks"]
    matches = [t for t in records if instruction_key(t) == instruction_key(episode["task"])]
    if len(matches) != 1:
        raise ValueError("Unknown episode task")
    thresholds = records[matches[0]]
    high, low = thresholds[case["variant"]], thresholds["exit"]
    state, previous_binding = {}, None
    trace = np.load(Path(episode["directory"]) / "trace.npz", allow_pickle=False)
    expected_modes = np.zeros(episode["control_steps"], dtype=np.int8)
    if [q["control_step"] for q in episode["queries"]] != list(range(0, episode["control_steps"], 16)):
        raise ValueError("Chunk schedule differs")
    for query in episode["queries"]:
        if query["threshold_enter"] != high or query["threshold_exit"] != low:
            raise ValueError("Runtime threshold changed")
        if case["strategy"] == "adaptive":
            state, event = decide(
                query["support_fp32"],
                high,
                low,
                state,
                config,
                query["nearest_index"],
                query["nearest_goal_index"],
            )
        else:
            state = dict(mode="action", goal_chunks=0, cooldown=0, above=0, anchor_index=None, goal_index=None)
            event = "control_action"
        if state != query["gate_state"]["state"] or event != query["gate_event"] or state["mode"] != query["mode"]:
            raise ValueError("Gate replay differs")
        binding = query["gate_state"]["binding"]
        expected_binding = digest(
            dict(
                task=query["task"],
                episode_id=episode["signature"],
                strategy=case["strategy"],
                variant=case["variant"],
                gate=manifest["settings"]["gate"],
                calibration=manifest["calibration"]["signature"],
            )
        )
        if binding != expected_binding or previous_binding not in (None, binding):
            raise ValueError("Episode gate binding differs")
        previous_binding = binding
        if state["mode"] == "goal":
            if query["anchor_index"] != state["anchor_index"] or query["goal_index"] != state["goal_index"]:
                raise ValueError("Fixed goal anchor changed")
        step = query["control_step"]
        expected_modes[step : step + 16] = int(state["mode"] == "goal")
    if (
        len(trace["executed_actions"]) != episode["control_steps"]
        or len(trace["ee"]) != episode["control_steps"] + 1
    ):
        raise ValueError("Trajectory lengths differ")
    if (
        not np.array_equal(trace["goal_mode"], expected_modes)
        or int(expected_modes.sum()) != episode["goal_control_steps"]
    ):
        raise ValueError("Executed mode accounting differs")


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    episodes = read_episodes(output, manifest)
    planned = {c["name"]: c for c in manifest["cases"]}
    for episode in episodes:
        if episode["case"] != planned.get(episode["case"]["name"]):
            raise ValueError("Unexpected episode condition")
        audit_episode(episode, manifest)
    rows, per_task, events = [], [], []
    for name in planned:
        group = [e for e in episodes if e["case"]["name"] == name]
        if not group:
            continue
        queries = [q for e in group for q in e["queries"]]
        steps = sum(e["control_steps"] for e in group)
        enters = sum(q["gate_event"] == "enter_goal" for q in queries)
        caps = sum(q["gate_event"] == "exit_cap" for q in queries)
        predicate_values = [e["predicate_completion"] for e in group if e["predicate_completion"] is not None]
        row = dict(
            condition=name,
            episodes=len(group),
            successes=sum(e["success"] for e in group),
            success_pct=100 * np.mean([e["success"] for e in group]),
            final_predicate_completion_pct=100 * np.mean(predicate_values) if predicate_values else None,
            goal_chunk_pct=100 * np.mean([q["mode"] == "goal" for q in queries]),
            goal_step_pct=100 * sum(e["goal_control_steps"] for e in group) / max(steps, 1),
            goal_entries=enters,
            exit_cap=caps,
            exit_support=sum(q["gate_event"] == "exit_support" for q in queries),
            entries_per_1000_steps=1000 * enters / max(steps, 1),
            query_ms_median=float(np.median([q["query_ms"] for q in queries])),
            rpc_ms_median=float(np.median([q["rpc_ms"] for q in queries])),
            rpc_ms_p90=float(np.quantile([q["rpc_ms"] for q in queries], 0.9)),
            smoke_only=any(e["smoke_only"] for e in group),
        )
        rows.append(row)
        for task in sorted({e["task_id"] for e in group}):
            selected = [e for e in group if e["task_id"] == task]
            per_task.append(
                dict(
                    condition=name,
                    task_id=task,
                    episodes=len(selected),
                    successes=sum(e["success"] for e in selected),
                )
            )
        for episode in group:
            events.extend(
                dict(
                    condition=name,
                    task_id=episode["task_id"],
                    initial_index=episode["initial_index"],
                    **{
                        k: q[k]
                        for k in (
                            "control_step",
                            "mode",
                            "gate_event",
                            "support_fp32",
                            "threshold_enter",
                            "threshold_exit",
                            "nearest_index",
                            "anchor_index",
                            "goal_index",
                        )
                    },
                )
                for q in episode["queries"]
            )
    write_csv(output / "summary.csv", rows)
    write_csv(output / "per_task.csv", per_task)
    write_csv(output / "gate_events.csv", events)
    (output / "gate_audit.json").write_text(
        json.dumps(dict(episodes_checked=len(episodes), passed=True), indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, key, label in zip(
            axes, ("success_pct", "goal_step_pct"), ("Task success (%)", "Executed goal-mode steps (%)")
        ):
            axis.bar([r["condition"] for r in rows], [r[key] for r in rows])
            axis.set(ylabel=label, ylim=(0, 100))
        fig.tight_layout()
        fig.savefig(output / "success_modes.png", dpi=160)
        plt.close(fig)
        representatives = [
            max(
                (e for e in episodes if e["case"]["name"] == r["condition"]), key=lambda e: e["goal_control_steps"]
            )
            for r in rows
        ]
        fig, axes = plt.subplots(len(representatives), 1, figsize=(10, 3 * len(representatives)), squeeze=False)
        for axis, episode in zip(axes[:, 0], representatives):
            queries = episode["queries"]
            axis.plot(
                [q["control_step"] for q in queries], [q["support_fp32"] for q in queries], label="FP32 distance"
            )
            axis.axhline(queries[0]["threshold_enter"], color="red", linestyle="--", label="Enter")
            axis.axhline(queries[0]["threshold_exit"], color="green", linestyle=":", label="Exit")
            for q in queries:
                if q["mode"] == "goal":
                    axis.axvspan(
                        q["control_step"],
                        min(q["control_step"] + 16, episode["control_steps"]),
                        alpha=0.15,
                        color="orange",
                    )
            axis.set(
                title=f"{episode['case']['name']} task={episode['task_id']} init={episode['initial_index']} (example: most goal steps)",
                ylabel="1 - cosine",
            )
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("Control step; orange = executed goal mode")
        fig.tight_layout()
        fig.savefig(output / "support_switches.png", dpi=160)
        plt.close(fig)
    lines = [
        "# Frozen retrieval-support gate evaluation",
        "",
        f"Completed: {len(episodes)}/{manifest['expected_episodes']}; gate replay and executed-mode audits passed.",
        "",
        "| Condition | Success | Goal steps | Entries | Cap exits |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['condition']} | {r['successes']}/{r['episodes']} | {r['goal_step_pct']:.1f}% | {r['goal_entries']} | {r['exit_cap']} |"
        for r in rows
    )
    lines.extend(
        [
            "",
            manifest["calibration"]["limitation"],
            "Thresholds are frozen; no rollout data updates calibration. Low support is not a verified failure label.",
            "The action-only control uses the same dual-mode checkpoint and reference pool.",
            "Latency includes shared-server contention; raw per-query timing is saved.",
        ]
    )
    if any(r["smoke_only"] for r in rows):
        lines.append("SMOKE ONLY: truncated episodes do not estimate benchmark success rates.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return rows
