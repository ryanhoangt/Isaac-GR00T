"""Read-only live monitor for per-episode LIBERO diagnosis/support-gate outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def latest_query(path):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = handle.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except ValueError:
                    pass
    except OSError:
        pass
    return {}


def render(output):
    manifest = read_json(output / "manifest.json")
    progress = read_json(output / "progress.json")
    settings = manifest.get("settings", {})
    groups = defaultdict(list)
    for path in output.glob("*/task*_init*/episode.json"):
        episode = read_json(path)
        if episode.get("complete"):
            groups[episode["case"]["name"]].append(episode)
    names = [case["name"] for case in manifest.get("cases", [])]
    names.extend(sorted(set(groups) - set(names)))
    expected = progress.get("expected_episodes", manifest.get("expected_episodes", "?"))
    per_condition = len(settings.get("task_ids", [])) * settings.get("trials", 0)
    completed = sum(len(values) for values in groups.values())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"{now} | {output.name}",
        f"Status: {progress.get('status', manifest.get('status', 'initializing'))} | completed {completed}/{expected}",
        f"Latest progress update: {progress.get('updated_utc', '-')}",
        "",
        f"{'Condition':<24} {'Completed':>11} {'Success':>10} {'SR':>8} {'Goal steps':>12}",
        "-" * 71,
    ]
    for name in names:
        episodes = groups[name]
        count = len(episodes)
        successes = sum(e["success"] for e in episodes)
        rate = f"{100 * successes / count:.1f}%" if count else "-"
        goal = "-"
        if episodes and all("goal_control_steps" in e for e in episodes):
            steps = sum(e["control_steps"] for e in episodes)
            goal = f"{100 * sum(e['goal_control_steps'] for e in episodes) / max(steps, 1):.1f}%"
        lines.append(
            f"{name:<24} {str(count) + '/' + str(per_condition or '?'):>11} {successes:>10} {rate:>8} {goal:>12}"
        )
        coverage = Counter(e["task_id"] for e in episodes)
        if coverage:
            lines.append(
                "  completed per task: " + ", ".join(f"{task}:{n}" for task, n in sorted(coverage.items()))
            )
        wins = sorted((e["task_id"], e["initial_index"], e["control_steps"]) for e in episodes if e["success"])
        if wins:
            lines.append("  successes (task/init/steps): " + ", ".join(f"{t}/{i}/{s}" for t, i, s in wins))
    lines.extend(["", "Active workers (last query; up to 15 following actions may already be executed):"])
    for worker in progress.get("active", []):
        if "_task" not in worker:
            lines.append(f"  {worker}")
            continue
        condition, task = worker.rsplit("_task", 1)
        candidates = [
            p
            for p in (output / condition).glob(f"task{task}_init*/queries.jsonl")
            if not (p.parent / "episode.json").exists()
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            lines.append(f"  {worker}: initializing / between trials")
            continue
        path = candidates[0]
        query = latest_query(path)
        if not query:
            lines.append(f"  {worker}/{path.parent.name}: awaiting first action")
            continue
        age = max(0, time.time() - path.stat().st_mtime)
        lines.append(
            f"  {worker}/{path.parent.name}: t={query['control_step']} mode={query['mode']} query age={age:.0f}s"
        )
        score = query.get("support_fp32")
        if score is not None:
            detail = f"    distance={score:.5f}"
            if "threshold_enter" in query:
                state = query["gate_state"]["state"]
                detail += (
                    f" enter={query['threshold_enter']:.5f} exit={query['threshold_exit']:.5f}"
                    f" event={query['gate_event']} goal_chunks={state['goal_chunks']}"
                )
            lines.append(detail)
    if not progress.get("active"):
        lines.append("  none")
    lines.extend(
        [
            "",
            "Counts include completed episodes only; partial task coverage can bias SR.",
            "Ctrl+C closes this read-only monitor; evaluation jobs keep running.",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.output.is_dir() or args.interval < 1:
        parser.error("output must exist and interval must be at least one second")
    try:
        while True:
            snapshot = render(args.output)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(snapshot, flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
