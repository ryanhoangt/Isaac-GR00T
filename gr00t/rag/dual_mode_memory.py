"""Provenance-aware milestone memory and fixed-mode inference, K=1."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.rag.dual_mode_data import pack_context


def metadata_digest(metadata):
    value = {key: item.model_dump(mode="json") for key, item in metadata.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(data))
    return digest.hexdigest()


def milestone_index(entry, index, lookahead):
    if lookahead < 1:
        raise ValueError("lookahead must be positive")
    episode = int(entry["episode_id"][index])
    rows = torch.where(entry["episode_id"] == episode)[0]
    frames = entry["frame_index"][rows]
    if not bool(torch.all(frames[1:] > frames[:-1])):
        raise ValueError("Memory frames must be strictly increasing within an episode")
    target = min(int(entry["frame_index"][index]) + lookahead, int(entry["trajectory_length"][index]) - 1)
    offset = int(torch.searchsorted(frames, target))
    if offset >= len(rows):
        raise ValueError("Memory must include each demonstration's terminal frame")
    return int(rows[offset])


class MilestoneMemory:
    def __init__(self, directory, normalization=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("status") != "completed" or self.manifest.get("schema_version") != 1:
            raise ValueError("Expected a completed provenance-aware memory")
        if normalization is not None and self.manifest["normalization_sha256"] != metadata_digest(normalization):
            raise ValueError("Memory/checkpoint normalization differs")
        self.entries = OrderedDict()

    def __getitem__(self, task):
        if task not in self.manifest["tasks"]:
            raise KeyError(f"Task absent from milestone memory: {task!r}")
        entry = self.entries.pop(task, None)
        if entry is None:
            path = self.directory / self.manifest["tasks"][task]["file"]
            entry = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            n = len(entry["keys"])
            required = (
                "action",
                "backbone_features",
                "backbone_attention_mask",
                "episode_id",
                "frame_index",
                "trajectory_length",
            )
            if any(len(entry[key]) != n for key in required):
                raise ValueError("Memory column lengths differ")
        self.entries[task] = entry
        if len(self.entries) > 2:
            self.entries.popitem(last=False)
        return entry


class DualModePolicy:
    def __init__(self, vla, retriever, memory, transforms, collator, modality_configs, *, mode=0, lookahead=16):
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (action) or 1 (goal)")
        self.vla, self.retriever, self.memory = vla, retriever, memory
        self.transforms, self.collator = transforms, collator
        self.modality_configs, self.mode, self.lookahead = modality_configs, mode, lookahead

    def get_modality_config(self):
        return self.modality_configs

    @torch.inference_mode()
    def get_action_with_context(self, observations):
        batch = self.collator([self.transforms(observations.copy())])
        task = batch.pop("tasks")[0]
        query = self.retriever(self.vla, batch)[0]
        entry = self.memory[task]
        scores = query @ entry["keys"].to(query).T
        index = int(scores.argmax())
        goal_index = milestone_index(entry, index, self.lookahead) if self.mode else index
        row = (
            entry["backbone_features"][goal_index],
            entry["backbone_attention_mask"][goal_index],
            entry["action"][goal_index],
        )
        context, actions = pack_context([row], self.vla.device, self.vla.action_head.dtype)
        batch.update(context_mode=torch.tensor([self.mode]), ret_actions=actions, ret_backbone_output=context)
        # Context was selected above. Bypass RAVLA.augment_inputs for compatibility
        # with the original one-mode checkpoint and the exact same reference pool.
        normalized = GR00T_N1_5.get_action(self.vla, batch).action_pred
        result = self.transforms.unapply({"action": normalized[0].float().cpu()})
        return {
            "action": result,
            "context_mode": self.mode,
            "retrieved_episode": int(entry["episode_id"][index]),
            "retrieved_frame": int(entry["frame_index"][index]),
            "goal_frame": int(entry["frame_index"][goal_index]),
            "similarity": float(scores[index]),
        }

    def get_action(self, observations):
        return self.get_action_with_context(observations)["action"]
