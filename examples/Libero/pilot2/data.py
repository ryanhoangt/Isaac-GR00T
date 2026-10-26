"""Split provenance and goal sampling, usable without loading a VLA checkpoint."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LANGUAGE_KEY = "annotation.human.action.task_description"
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def episode_path(root, info, episode, video_key=None):
    values = {"episode_index": int(episode), "episode_chunk": int(episode) // info["chunks_size"]}
    if video_key is not None:
        values["video_key"] = video_key
    return Path(root) / info["video_path" if video_key is not None else "data_path"].format(**values)


def catalog(root):
    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {item["task"]: int(item["task_index"]) for item in read_jsonl(root / "meta/tasks.jsonl")}
    episodes = []
    for item in read_jsonl(root / "meta/episodes.jsonl"):
        task_ids = {tasks[name] for name in item["tasks"]}
        if len(task_ids) != 1:
            raise ValueError(f"Episode must belong to exactly one task: {item}")
        task = task_ids.pop()
        frame = pd.read_parquet(episode_path(root, info, item["episode_index"]), columns=["task_index"])
        if set(frame.task_index.astype(int)) != {task} or len(frame) != item["length"]:
            raise ValueError(f"Inconsistent task/length in episode {item['episode_index']}")
        episodes.append({**item, "task_id": task})
    return info, episodes


def make_split(episodes, reference_per_task=3):
    grouped = defaultdict(list)
    for item in episodes:
        grouped[item["task_id"]].append(item)
    train, reference = [], []
    for task, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["episode_index"])
        if len(items) < reference_per_task + 2:
            raise ValueError(f"Task {task} needs reference episodes and at least two training demos")
        reference.extend(items[:reference_per_task])
        train.extend(items[reference_per_task:])
    return train, reference


def stage_partition(source, destination, info, selected):
    """Copy only small parquet/metadata files; link videos; recompute split stats."""
    import shutil
    import pandas as pd

    destination = Path(destination)
    (destination / "meta").mkdir(parents=True, exist_ok=False)
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copy2(Path(source) / "meta" / name, destination / "meta" / name)
    video_keys = [key for key, item in info["features"].items() if item["dtype"] == "video"]
    mapping, output_episodes, arrays = [], [], defaultdict(list)
    offset = 0
    for new_id, item in enumerate(selected):
        original_id = item["episode_index"]
        frame = pd.read_parquet(episode_path(source, info, original_id))
        frame["episode_index"] = new_id
        frame["index"] = np.arange(offset, offset + len(frame), dtype=np.int64)
        output = episode_path(destination, info, new_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        for key in ("observation.state", "action"):
            arrays[key].append(np.stack(frame[key]).astype(np.float32))
        for key in video_keys:
            original = episode_path(source, info, original_id, key).resolve(strict=True)
            link = episode_path(destination, info, new_id, key)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(original)
        output_episodes.append({k: v for k, v in {**item, "episode_index": new_id}.items() if k != "task_id"})
        mapping.append(
            {"episode_index": new_id, "original_episode_index": original_id, "task_id": item["task_id"]}
        )
        offset += len(frame)
    stats = {}
    for key, pieces in arrays.items():
        values = np.concatenate(pieces)
        stats[key] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    output_info = {
        **info,
        "total_episodes": len(selected),
        "total_frames": offset,
        "total_videos": len(video_keys) * len(selected),
        "total_chunks": (len(selected) + info["chunks_size"] - 1) // info["chunks_size"],
        "splits": {"train": f"0:{len(selected)}"},
    }
    for filename, value in (("info.json", output_info), ("stats.json", stats), ("episode_mapping.json", mapping)):
        (destination / "meta" / filename).write_text(json.dumps(value, indent=2) + "\n")
    (destination / "meta/episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in output_episodes))
    return mapping


def prepare(source, destination, reference_per_task=3, strict_50=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    info, episodes = catalog(source)
    counts = {task: sum(item["task_id"] == task for item in episodes) for task in range(10)}
    if set(item["task_id"] for item in episodes) != set(range(10)):
        raise ValueError("Expected all ten LIBERO-Long tasks")
    if strict_50 and set(counts.values()) != {50}:
        raise ValueError(f"Expected 50 demonstrations/task, found {counts}")
    train, reference = make_split(episodes, reference_per_task)
    identity = {"source": str(source), "episodes": episodes, "reference_per_task": reference_per_task}
    split_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "suite": "libero_10",
        "split_id": split_id,
        "source": str(source),
        "counts_per_task": counts,
        "reference_per_task": reference_per_task,
        "matches_50_per_task": set(counts.values()) == {50},
        "normalization": "train_only",
        "train": stage_partition(source, destination / "train", info, train),
        "reference": stage_partition(source, destination / "reference", info, reference),
    }
    (destination / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pair_goal(episodes, current_id, step, rng, horizon=16, cross_probability=0.5):
    current = episodes[current_id]
    candidates = [
        key for key, item in episodes.items() if key != current_id and item["task_id"] == current["task_id"]
    ]
    cross = bool(rng.random() < cross_probability)
    if cross and not candidates:
        raise ValueError("Cross-demo sampling requires another training demo from the same task")
    target_id = int(rng.choice(candidates)) if cross else current_id
    target = episodes[target_id]
    aligned = int(step / current["length"] * target["length"]) if cross else step
    offset = int(rng.integers(horizon, 2 * horizon + 1))
    return target_id, min(aligned + offset, target["length"] - 1), cross


def sample_mode(rng):
    value = rng.random()
    return "language" if value < 0.4 else "both" if value < 0.8 else "goal"


def perturb_state(raw, rng, sigma=0.02):
    result = dict(raw)
    for name in STATE_KEYS:
        key = f"state.{name}"
        value = np.asarray(result[key])
        result[key] = (value + rng.normal(0, sigma, value.shape)).astype(value.dtype)
    return result
