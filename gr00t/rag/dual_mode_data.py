"""Cross-demo DTW goals, same-scene temporal negatives, explicit context batches."""

from collections import OrderedDict
import random

from dtaidistance import dtw_ndim
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rag.ravla import RAVLADataCollator


def dense_alignment(left, right):
    path = dtw_ndim.warping_path(
        np.ascontiguousarray(left, dtype=np.double),
        np.ascontiguousarray(right, dtype=np.double),
    )
    # Median correspondence when DTW maps a source frame more than once.
    groups = [[] for _ in left]
    for i, j in path:
        groups[i].append(j)
    if any(not values for values in groups):
        raise ValueError("DTW path does not cover the current trajectory")
    return np.asarray([int(np.median(values)) for values in groups], dtype=np.int64)


def temporal_negative(frame, length, gap, rng):
    candidates = [i for i in range(length) if abs(i - frame) >= gap]
    if not candidates:
        return frame, False
    return rng.choice(candidates), True


class DualModeDataset(Dataset):
    def __init__(self, mixture, *, seed=42, goal_probability=0.5, lookahead=16, negative_gap=32):
        if not 0 <= goal_probability <= 1 or lookahead < 1 or negative_gap < 1:
            raise ValueError("Invalid goal probability, lookahead, or negative gap")
        self.mixture = mixture
        self.seed, self.goal_probability = seed, goal_probability
        self.lookahead, self.negative_gap = lookahead, negative_gap
        self.epoch = 0
        self.alignments = OrderedDict()
        self.actions = OrderedDict()
        for dataset in mixture.datasets:
            if any(len(ids) < 2 for ids in dataset.task_groups.values()):
                raise ValueError("Every training task requires at least two demos")

    def __len__(self):
        return len(self.mixture)

    def _actions(self, dataset_id, episode):
        key = (dataset_id, episode)
        value = self.actions.pop(key, None)
        if value is None:
            table = self.mixture.datasets[dataset_id].get_trajectory_data(episode)
            value = np.stack(table["action"].tolist()).astype(np.double)
        self.actions[key] = value
        if len(self.actions) > 64:
            self.actions.popitem(last=False)
        return value

    def __getitem__(self, index):
        dataset_id, episode, frame = self.mixture.all_frame_paths[index]
        dataset = self.mixture.datasets[dataset_id]
        task = dataset.trajectory_tasks[dataset.get_trajectory_index(episode)]
        rng = random.Random(self.seed + self.epoch * 1000003 + int(index) * 9176)
        mode = int(rng.random() < self.goal_probability)
        current = dataset.transforms(dataset.get_step_data(episode, frame))
        provenance = {"dataset": dataset_id, "episode": int(episode), "frame": int(frame), "task": task}
        result = {"current": current, "mode": mode, "provenance": provenance}
        if not mode:
            return result
        other = rng.choice([j for j in dataset.task_groups[task] if j != episode])
        key = (dataset_id, episode, other)
        alignment = self.alignments.pop(key, None)
        if alignment is None:
            alignment = dense_alignment(self._actions(dataset_id, episode), self._actions(dataset_id, other))
        self.alignments[key] = alignment
        if len(self.alignments) > 256:
            self.alignments.popitem(last=False)
        length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(other)])
        aligned = int(alignment[frame])
        goal_frame = min(aligned + self.lookahead, length - 1)
        negative_frame, valid = temporal_negative(goal_frame, length, self.negative_gap, rng)
        result["positive"] = dataset.transforms(dataset.get_step_data(other, goal_frame))
        result["negative"] = dataset.transforms(dataset.get_step_data(other, negative_frame))
        result["negative_valid"] = valid
        provenance.update(
            goal_episode=int(other),
            aligned_frame=aligned,
            goal_frame=goal_frame,
            negative_frame=negative_frame,
            goal_length=length,
            negative_valid=valid,
        )
        return result


class DualModeCollator:
    def __init__(self):
        self.base = RAVLADataCollator()

    def __call__(self, samples):
        indices = [i for i, sample in enumerate(samples) if sample["mode"]]
        result = {
            "current": self.base([s["current"] for s in samples]),
            "modes": torch.tensor([s["mode"] for s in samples]),
            "goal_indices": indices,
            "provenance": [s["provenance"] for s in samples],
            "negative_valid": torch.tensor([s.get("negative_valid", True) for s in samples]),
        }
        if indices:
            result["positive"] = self.base([samples[i]["positive"] for i in indices])
            result["negative"] = self.base([samples[i]["negative"] for i in indices])
        return result


def pack_context(rows, device, dtype):
    features = pad_sequence(
        [r[0].to(device=device, dtype=dtype) for r in rows], batch_first=True, padding_side="left"
    )
    masks = pad_sequence([r[1].to(device=device) for r in rows], batch_first=True, padding_side="left")
    actions = torch.stack([r[2].to(device=device, dtype=dtype) for r in rows])
    return BatchFeature(data={"backbone_features": features, "backbone_attention_mask": masks}), actions


@torch.no_grad()
def encode_context(model, batch):
    values = {key: value.to(model.device) for key, value in batch.items() if key.startswith("eagle_")}
    values = {
        key: value.to(dtype=model.action_head.dtype) if value.is_floating_point() else value
        for key, value in values.items()
    }
    return model.backbone(BatchFeature(data=values))


@torch.no_grad()
def prepare_training_batch(model, retriever, memory, batch, *, negative_seed=0):
    current = dict(batch["current"])
    tasks = current.pop("tasks")
    modes = batch["modes"]
    positives, negatives = [None] * len(tasks), [None] * len(tasks)
    rng = random.Random(negative_seed)
    # Keep the existing learned top-1 selection for action-mode samples.
    # The legacy memory has no episode provenance; no cross-demo claim is made
    # for those rows. Cross-demo exclusion is explicit for DTW goal samples.
    queries = retriever(model, current)
    for i, task in enumerate(tasks):
        if modes[i]:
            continue
        entry = memory[task]
        scores = queries[i] @ entry["keys"].to(queries).T
        best = int(scores.argmax())
        options = [j for j in range(len(scores)) if j != best]
        if not options:
            raise ValueError("Action adherence requires a distinct negative memory row")
        negative = rng.choice(options)

        def row(j):
            return entry["backbone_features"][j], entry["backbone_attention_mask"][j], entry["action"][j]

        positives[i], negatives[i] = row(best), row(negative)
    if batch["goal_indices"]:
        positive = encode_context(model, batch["positive"])
        negative = encode_context(model, batch["negative"])
        for j, i in enumerate(batch["goal_indices"]):
            # No expert actions are even carried for the goal branch.
            dummy = torch.zeros(model.action_head.action_horizon, model.action_head.action_dim)
            positives[i] = positive.backbone_features[j], positive.backbone_attention_mask[j], dummy
            negatives[i] = negative.backbone_features[j], negative.backbone_attention_mask[j], dummy
    pos, actions = pack_context(positives, model.device, model.action_head.dtype)
    neg, neg_actions = pack_context(negatives, model.device, model.action_head.dtype)
    current.update(
        context_mode=modes,
        ret_actions=actions,
        ret_backbone_output=pos,
        neg_ret_actions=neg_actions,
        neg_backbone_output=neg,
        negative_valid=batch["negative_valid"],
    )
    return current
