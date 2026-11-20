"""Strict within-episode transitions and frozen-Eagle cache for endpoint training."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from gr00t.rag.dual_mode_data import encode_context
from gr00t.rag.ravla import RAVLADataCollator


def windows(states, actions, horizon=16):
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 8 or actions.shape != (len(states), 7):
        raise ValueError("Invalid proprio/action episode shape")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite trajectory")
    n = len(states) - horizon
    if n < 1:
        raise ValueError("Episode is too short for a full endpoint")
    chunks = np.stack([actions[t : t + horizon] for t in range(n)])
    return states[:n], chunks, states[horizon:, :3] - states[:n, :3]


def read_episode(dataset_path, episode):
    root = Path(dataset_path)
    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["data_path"].format(
        episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode
    )
    table = pd.read_parquet(path, columns=["observation.state", "action"])
    return np.stack(table["observation.state"]), np.stack(table["action"])


def collect_transitions(training_config, split):
    states, actions, deltas, provenance = [], [], [], []
    paths = training_config["args"]["dataset_paths"]
    for item in training_config["split"]:
        for episode in item[split + "_episode_ids"]:
            s, a = read_episode(paths[item["dataset"]], episode)
            ss, aa, dd = windows(s, a)
            states.append(ss)
            actions.append(aa)
            deltas.append(dd)
            provenance.extend((item["dataset"], episode, t) for t in range(len(ss)))
    return tuple(torch.from_numpy(np.concatenate(values)) for values in (states, actions, deltas)), provenance


@torch.inference_mode()
def build_cache(model, mixture, inventory, directory, batch_size=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    collator = RAVLADataCollator()
    records = []
    for item in inventory:
        dataset = mixture.datasets[item["dataset"]]
        dataset.transforms.eval()
        dataset.transforms.transforms[-1].train()
        for split in ("train", "validation"):
            for episode in item[split + "_episode_ids"]:
                raw, _ = read_episode(dataset.dataset_path, episode)
                length = len(raw)
                collected = {
                    k: []
                    for k in (
                        "backbone_features",
                        "backbone_attention_mask",
                        "state",
                        "embodiment_id",
                        "action",
                        "action_mask",
                    )
                }
                for start in range(0, length, batch_size):
                    samples = [
                        dataset.transforms(dataset.get_step_data(episode, t))
                        for t in range(start, min(start + batch_size, length))
                    ]
                    batch = collator(samples)
                    batch.pop("tasks")
                    features = encode_context(model, batch)
                    for key in collected:
                        collected[key].append((features[key] if key.startswith("backbone_") else batch[key]).cpu())
                values = {key: torch.cat(parts) for key, parts in collected.items()}
                values["raw_state"] = torch.from_numpy(raw.astype(np.float32))
                filename = f"dataset{item['dataset']}_episode{episode:06d}.pt"
                torch.save(values, directory / filename)
                records.append(
                    dict(dataset=item["dataset"], episode=episode, length=length, split=split, file=filename)
                )
                dataset._video_cache.clear()
                print(
                    "ENDPOINT_CACHE", split, filename, length, list(values["backbone_features"].shape), flush=True
                )
    (directory / "manifest.json").write_text(
        json.dumps(dict(status="completed", records=records), indent=2) + "\n"
    )
    return records


class EndpointCache(Dataset):
    def __init__(self, directory, split):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("Incomplete frozen-feature cache")
        self.paths = [
            (r["file"], t) for r in manifest["records"] if r["split"] == split for t in range(r["length"] - 16)
        ]
        self.open = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        filename, t = self.paths[index]
        shard = self.open.pop(filename, None)
        if shard is None:
            shard = torch.load(self.directory / filename, map_location="cpu", mmap=True, weights_only=True)
        self.open[filename] = shard
        if len(self.open) > 3:
            self.open.popitem(last=False)
        result = {key: shard[key][t] for key in ("state", "embodiment_id", "action", "action_mask")}
        result.update(
            current={k: shard[k][t] for k in ("backbone_features", "backbone_attention_mask")},
            goal={k: shard[k][t + 16] for k in ("backbone_features", "backbone_attention_mask")},
            raw_state=shard["raw_state"][t],
            delta=shard["raw_state"][t + 16, :3] - shard["raw_state"][t, :3],
        )
        return result


def collate_cached(samples):
    result = {
        key: torch.stack([s[key] for s in samples])
        for key in ("state", "embodiment_id", "action", "action_mask", "raw_state", "delta")
    }
    for kind in ("current", "goal"):
        result[kind] = {
            key: pad_sequence([s[kind][key] for s in samples], batch_first=True, padding_side="left")
            for key in ("backbone_features", "backbone_attention_mask")
        }
    return result
