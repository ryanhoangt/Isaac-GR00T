"""Shared-Gaussian goal probes on held-out reference states (not rollout success)."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from gr00t.rag.dual_mode_diagnosis import DiagnosticPolicy
from gr00t.rag.dual_mode_memory import milestone_index
from scripts.rag.serve_dual_mode import Args, load_policy


def relative(left, right, section):
    return float(
        np.linalg.norm(left[:, section] - right[:, section]) / max(np.linalg.norm(left[:, section]), 1e-6)
    )


def main(args):
    policy = load_policy(Args(checkpoint=args.checkpoint, memory=args.memory, retriever=args.retriever))
    diagnosis = DiagnosticPolicy(policy)
    if not getattr(policy.vla.action_head.config, "dual_mode", False):
        raise ValueError("Goal probes require a dual-mode checkpoint")
    rng = np.random.default_rng(args.seed)
    tasks = list(policy.memory.manifest["tasks"])
    rows, predictions = [], []
    for task_number, task in enumerate(tasks):
        entry = policy.memory[task]
        count = args.states // len(tasks) + int(task_number < args.states % len(tasks))
        if count > len(entry["keys"]):
            raise ValueError("Probe count exceeds available distinct reference states")
        for current in rng.permutation(len(entry["keys"]))[:count]:
            episode = int(entry["episode_id"][current])
            scores = entry["keys"][current] @ entry["keys"].T
            eligible = entry["episode_id"] != episode
            anchor = int(scores.masked_fill(~eligible, -torch.inf).argmax())
            goal = milestone_index(entry, anchor, 16)
            wrong_candidates = torch.where(
                (entry["episode_id"] == entry["episode_id"][goal])
                & ((entry["frame_index"] - entry["frame_index"][goal]).abs() >= 32)
            )[0].numpy()
            if not len(wrong_candidates):
                raise ValueError("No same-demo temporal negative with gap >=32")
            wrong = int(rng.choice(wrong_candidates))
            random_candidates = torch.where(eligible & (torch.arange(len(scores)) != goal))[0].numpy()
            distractor = int(rng.choice(random_candidates))
            seed = args.seed + len(rows)
            torch.cuda.synchronize(policy.vla.device)
            started = time.perf_counter()
            actions = [
                diagnosis.probe_cached(task, int(current), index, seed) for index in (goal, wrong, distractor)
            ]
            torch.cuda.synchronize(policy.vla.device)
            row = {
                "task": task,
                "query_episode": episode,
                "query_frame": int(entry["frame_index"][current]),
                "goal_episode": int(entry["episode_id"][goal]),
                "goal_frame": int(entry["frame_index"][goal]),
                "wrong_frame": int(entry["frame_index"][wrong]),
                "flow_seed": seed,
                "shared_noise": True,
                "head_ms_per_variant": (time.perf_counter() - started) * 1000 / 3,
                "wrong_relative_all": relative(actions[0], actions[1], slice(None)),
                "wrong_relative_xyz": relative(actions[0], actions[1], slice(0, 3)),
                "wrong_relative_rotation": relative(actions[0], actions[1], slice(3, 6)),
                "random_relative_all": relative(actions[0], actions[2], slice(None)),
                "random_relative_xyz": relative(actions[0], actions[2], slice(0, 3)),
            }
            rows.append(row)
            predictions.append(actions)
        print(f"PROBES {task}: {count}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "probes.csv.tmp").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "probes.csv.tmp").replace(args.output / "probes.csv")
    with (args.output / "actions.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, actions=np.asarray(predictions))
    (args.output / "actions.npz.tmp").replace(args.output / "actions.npz")
    summary = {
        "states": len(rows),
        "own_demo_excluded": True,
        "source": "heldout_reference_states",
        "semantic_phase_labels": False,
    }
    for key in (
        "wrong_relative_all",
        "wrong_relative_xyz",
        "wrong_relative_rotation",
        "random_relative_all",
        "random_relative_xyz",
        "head_ms_per_variant",
    ):
        values = np.asarray([row[key] for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    temporary = args.output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--retriever", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parsed = parser.parse_args()
    if parsed.states < 1:
        parser.error("states must be positive")
    main(parsed)
