"""Episode-separated, empirical reliability selection for chunk dynamics."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


def episode_partition(recipe, seed=20261103):
    """Reserve six calibration and six test demos/task from original train IDs.

    The original validation demos remain an additional audit. No demo used for
    dynamics fitting occurs in either new reliability holdout.
    """
    result = deepcopy(recipe)
    generator = np.random.default_rng(seed)
    for item in result["split"]:
        ids = np.asarray(item["train_episode_ids"], dtype=int)
        if len(ids) < 24 or len(set(ids.tolist())) != len(ids):
            raise ValueError("Need >=24 unique training demos/task")
        shuffled = generator.permutation(ids).tolist()
        item["audit_episode_ids"] = item["validation_episode_ids"]
        item["train_episode_ids"] = sorted(shuffled[12:])
        item["calibration_episode_ids"] = sorted(shuffled[:6])
        item["test_episode_ids"] = sorted(shuffled[6:12])
        item["validation_episode_ids"] = item["calibration_episode_ids"]
        groups = [set(item[k + "_episode_ids"]) for k in ("train", "calibration", "test", "audit")]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise ValueError("Overlapping episode groups")
    return result


@torch.no_grad()
def nearest_rms(queries, reference, query_batch=128, reference_batch=2048):
    """Exact nearest normalized-input RMS distance, with bounded GPU memory."""
    if len(reference) == 0 or queries.shape[-1] != reference.shape[-1]:
        raise ValueError("Invalid support reference")
    parts = []
    for query in queries.split(query_batch):
        best = torch.full((len(query),), float("inf"), device=query.device)
        for points in reference.split(reference_batch):
            # Direct distance avoids cancellation for almost identical inputs.
            distances = torch.cdist(query.float(), points.float(), compute_mode="donot_use_mm_for_euclid_dist")
            best = torch.minimum(best, distances.min(-1).values)
        parts.append(best / queries.shape[-1] ** 0.5)
    return torch.cat(parts) if parts else queries.new_empty(0)


class ReliabilityFilter(nn.Module):
    def __init__(self, reference, mean, scale, std_limit, distance_limit):
        super().__init__()
        self.register_buffer("reference", reference.float())
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float())
        self.std_limit = float(std_limit)
        self.distance_limit = float(distance_limit)

    @torch.no_grad()
    def distances(self, inputs):
        return nearest_rms((inputs.float() - self.mean) / self.scale, self.reference)

    @torch.no_grad()
    def forward(self, inputs, spread):
        distance = self.distances(inputs)
        return (spread <= self.std_limit) & (distance <= self.distance_limit)


def acceptance_summary(
    errors, accepted, provenance, max_mean=0.005, max_p95=0.01, min_coverage=0.1, min_windows=32, min_episodes=2
):
    errors, accepted = np.asarray(errors), np.asarray(accepted, dtype=bool)
    rows = []
    for dataset in sorted({p[0] for p in provenance}):
        task = np.asarray([p[0] == dataset for p in provenance])
        keep = task & accepted
        values = errors[keep]
        episodes = len({p[1] for p, flag in zip(provenance, keep) if flag})
        row = dict(
            dataset=dataset,
            total=int(task.sum()),
            accepted=int(keep.sum()),
            coverage=float(keep.sum() / task.sum()),
            episodes=episodes,
            mean_m=float(values.mean()) if len(values) else None,
            p95_m=float(np.quantile(values, 0.95)) if len(values) else None,
        )
        row["passed"] = bool(
            len(values) >= min_windows
            and episodes >= min_episodes
            and row["coverage"] >= min_coverage
            and row["mean_m"] <= max_mean
            and row["p95_m"] <= max_p95
        )
        rows.append(row)
    selected = errors[accepted]
    return dict(
        passed=all(r["passed"] for r in rows),
        per_task=rows,
        coverage=float(accepted.mean()),
        accepted=int(accepted.sum()),
        mean_m=float(selected.mean()) if len(selected) else None,
        p95_m=float(np.quantile(selected, 0.95)) if len(selected) else None,
    )


def select_threshold(errors, spread, distance, provenance, **criteria):
    """Select maximum coverage among a fixed grid, using calibration only."""
    spread, distance = np.asarray(spread), np.asarray(distance)
    candidates = []
    for sq in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        for dq in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
            sl, dl = float(np.quantile(spread, sq)), float(np.quantile(distance, dq))
            selected = (spread <= sl) & (distance <= dl)
            summary = acceptance_summary(errors, selected, provenance, **criteria)
            candidates.append(
                dict(std_quantile=sq, distance_quantile=dq, std_limit_m=sl, distance_limit=dl, **summary)
            )
    feasible = [row for row in candidates if row["passed"]]
    winner = max(feasible, key=lambda row: (row["coverage"], -row["p95_m"])) if feasible else None
    return winner, candidates
