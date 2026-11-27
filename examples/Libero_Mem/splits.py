"""Copy complementary LIBERO-Mem training folds into standalone LeRobot v2.1.

Videos are copied byte-for-byte, never linked or re-encoded. Each fold gets
compact indices and normalization recomputed only from its own training rows.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import av
import numpy as np
import pandas as pd

from examples.Libero_Mem.conversion import CAMERAS, FPS, MODALITY_PATH, file_sha256
from examples.Libero_Mem.tasks import TASKS, TASK_BY_NAME, get_fold
from gr00t.utils.lerobot_conversion import DATA_PATH, VIDEO_PATH, statistics, write_json, write_jsonl


SPLIT_VERSION = 1
META_FILES = (
    "manifest.json",
    "meta/info.json",
    "meta/modality.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episode_mapping.json",
    "meta/stats.json",
    "meta/episodes_stats.jsonl",
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _index(records, key, label):
    result = {}
    for record in records:
        value = record[key]
        if type(value) is not int or value < 0 or value in result:
            raise ValueError(f"Invalid or duplicate {label}: {value}")
        result[value] = record
    return result


def _identity(path):
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _unchanged(path, identity):
    if _identity(path) != identity:
        raise ValueError(f"Source changed during fold extraction: {path}")


def _asset(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset asset path must stay relative: {relative}")
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty dataset asset: {path}")
    return path


def _source_inventory(source):
    root = Path(source).expanduser().resolve(strict=True)
    identities = {name: _identity(root / name) for name in META_FILES}
    hashes = {name: file_sha256(root / name) for name in META_FILES}
    manifest = json.loads((root / "manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("converter") != "examples.Libero_Mem.conversion"
        or info.get("codebase_version") != "v2.1"
        or info.get("fps") != FPS
    ):
        raise ValueError("Expected a completed canonical LIBERO-Mem LeRobot v2.1 conversion at 20 FPS")
    if modality != json.loads(MODALITY_PATH.read_text()):
        raise ValueError("Source modality mapping differs from the existing LIBERO schema")
    if type(info.get("chunks_size")) is not int or info["chunks_size"] < 1:
        raise ValueError("Invalid source chunk size")
    video_keys = {key for key, value in info["features"].items() if value["dtype"] == "video"}
    if video_keys != set(CAMERAS):
        raise ValueError("Expected the two canonical LIBERO camera videos")
    for key, width in (("observation.state", 8), ("action", 7)):
        if info["features"][key]["shape"] != [width] or info["features"][key]["dtype"] != "float32":
            raise ValueError(f"Invalid canonical vector feature: {key}")
    episodes = _index(_read_jsonl(root / "meta/episodes.jsonl"), "episode_index", "episode ID")
    mappings = _index(json.loads((root / "meta/episode_mapping.json").read_text()), "episode_index", "mapping ID")
    vocabulary = _index(_read_jsonl(root / "meta/tasks.jsonl"), "task_index", "instruction ID")
    if (
        set(episodes) != set(range(len(episodes)))
        or set(episodes) != set(mappings)
        or len(episodes) != info["total_episodes"]
        or manifest.get("episodes") != len(episodes)
    ):
        raise ValueError("Canonical episode/mapping counts or IDs disagree")
    expected_names = {task.name for task in TASKS}
    names = {mapping["task"] for mapping in mappings.values()}
    if names != expected_names or info["total_tasks"] != 10 or manifest.get("tasks") != 10:
        raise ValueError("Fold extraction requires all ten canonical task families")
    if len(vocabulary) != 10 or {item["task"] for item in vocabulary.values()} != {
        task.instruction for task in TASKS
    }:
        raise ValueError("Canonical instruction vocabulary differs from registry")
    task_indices = {value["task"]: index for index, value in vocabulary.items()}
    entries, keys, offset = [], set(), 0
    for episode_id in sorted(episodes):
        episode, mapping = episodes[episode_id], mappings[episode_id]
        task = TASK_BY_NAME[mapping["task"]]
        demo_id = mapping["source_demo_id"]
        source_key = f"{task.name}/{demo_id}"
        length = episode["length"]
        if (
            type(length) is not int
            or length < 1
            or mapping["source_frames"] != length
            or mapping["source_task_id"] != task.number
            or re.fullmatch(r"demo_(0|[1-9][0-9]*)", demo_id) is None
            or mapping["source_episode_key"] != source_key
            or source_key in keys
            or episode["tasks"] != [task.instruction]
        ):
            raise ValueError(f"Invalid canonical episode provenance: {episode_id}")
        keys.add(source_key)
        values = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"]}
        parquet = _asset(root, info["data_path"].format(**values))
        videos = {key: _asset(root, info["video_path"].format(**values, video_key=key)) for key in CAMERAS}
        entries.append(
            {
                "episode": episode,
                "mapping": mapping,
                "task_index": task_indices[task.instruction],
                "frame_offset": offset,
                "parquet": parquet,
                "videos": videos,
                "identities": {str(path): _identity(path) for path in (parquet, *videos.values())},
            }
        )
        offset += length
    if (
        offset != info["total_frames"]
        or offset != manifest.get("frames")
        or info["total_videos"] != len(entries) * 2
    ):
        raise ValueError("Canonical frame/video counts disagree")
    for name, identity in identities.items():
        _unchanged(root / name, identity)
    return {
        "root": root,
        "info": info,
        "manifest": manifest,
        "entries": entries,
        "metadata_identities": identities,
        "metadata_sha256": hashes,
    }


def _validated_rows(entry):
    data = pd.read_parquet(entry["parquet"], use_threads=False)
    episode, mapping = entry["episode"], entry["mapping"]
    length, episode_id = episode["length"], episode["episode_index"]
    expected = {
        "episode_index": np.full(length, episode_id),
        "frame_index": np.arange(length),
        "index": np.arange(entry["frame_offset"], entry["frame_offset"] + length),
        "task_index": np.full(length, entry["task_index"]),
        "source_task_id": np.full(length, mapping["source_task_id"]),
        "source_frame_index": np.arange(length),
    }
    if len(data) != length:
        raise ValueError(f"Parquet/episode length mismatch: {entry['parquet']}")
    for key, value in expected.items():
        if key not in data or data[key].dtype.kind not in "iu" or not np.array_equal(data[key], value):
            raise ValueError(f"Invalid {key}: {entry['parquet']}")
    for key in ("source_demo_id", "source_episode_key"):
        if key not in data or not (data[key] == mapping[key]).all():
            raise ValueError(f"Row provenance differs from mapping ({key}): {entry['parquet']}")
    for key, width in (("observation.state", 8), ("action", 7)):
        vectors = np.stack(data[key])
        if vectors.shape != (length, width) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise ValueError(f"Expected finite float32 [{length},{width}] {key}: {entry['parquet']}")
    if not np.allclose(data["timestamp"], np.arange(length) / FPS, rtol=0, atol=1e-12):
        raise ValueError(f"Invalid source timestamps: {entry['parquet']}")
    if (
        not pd.api.types.is_bool_dtype(data["next.done"])
        or not data["next.done"].iloc[-1]
        or data["next.done"].iloc[:-1].any()
    ):
        raise ValueError(f"Invalid final done flag: {entry['parquet']}")
    original_gripper = data["source_action_gripper"].to_numpy()
    if not np.isin(original_gripper, [-1, 1]).all() or not np.array_equal(
        np.stack(data["action"])[:, -1], (1 - original_gripper) / 2
    ):
        raise ValueError(f"Gripper provenance mismatch: {entry['parquet']}")
    _unchanged(entry["parquet"], entry["identities"][str(entry["parquet"])])
    return data


def _video_header(path, length):
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        if stream.frames != length or stream.average_rate != FPS or (stream.width, stream.height) != (256, 256):
            raise ValueError(f"Video header differs from source episode: {path}")


def _copy_video(source, target, identity):
    """Create independent bytes/inode and verify the destination checksum."""
    _unchanged(source, identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            dst.write(block)
            digest.update(block)
    shutil.copystat(source, target)
    _unchanged(source, identity)
    if file_sha256(target) != digest.hexdigest():
        raise ValueError(f"Copied video checksum mismatch: {target}")
    return digest.hexdigest()


def extract_fold(source, output, *, training_fold, chunks_size=1000, dry_run=False):
    fold = get_fold(training_fold)
    if type(chunks_size) is not int or chunks_size < 1:
        raise ValueError("chunks_size must be positive")
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("Fold output must be outside the canonical source")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    inventory = _source_inventory(source)
    selected = [entry for entry in inventory["entries"] if entry["mapping"]["task"] in fold["train_tasks"]]
    total_frames = sum(entry["episode"]["length"] for entry in selected)
    video_bytes = sum(path.stat().st_size for entry in selected for path in entry["videos"].values())
    if dry_run:
        return {
            "status": "dry_run",
            "training_fold": training_fold,
            **fold,
            "episodes": len(selected),
            "frames": total_frames,
            "videos": len(selected) * 2,
            "video_bytes_to_copy": video_bytes,
            "output": str(output),
            "storage": "independent_copies",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    task_map = {name: i for i, name in enumerate(fold["train_tasks"])}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        (stage / "meta").mkdir(parents=True)
        episodes, mappings, episode_stats, assets, states, actions = [], [], [], [], [], []
        offset = 0
        for index, entry in enumerate(selected):
            data = _validated_rows(entry)
            task = entry["mapping"]["task"]
            data["canonical_episode_index"] = data["episode_index"]
            data["canonical_task_index"] = data["task_index"]
            data["canonical_index"] = data["index"]
            data["episode_index"] = np.full(len(data), index, dtype=np.int64)
            data["task_index"] = np.full(len(data), task_map[task], dtype=np.int64)
            data["index"] = np.arange(offset, offset + len(data), dtype=np.int64)
            values = {"episode_index": index, "episode_chunk": index // chunks_size}
            parquet = stage / DATA_PATH.format(**values)
            parquet.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(parquet, index=False)
            video_hashes = {}
            for key, original in entry["videos"].items():
                _video_header(original, len(data))
                target = stage / VIDEO_PATH.format(**values, video_key=key)
                video_hashes[key] = _copy_video(original, target, entry["identities"][str(original)])
            state, action = np.stack(data["observation.state"]), np.stack(data["action"])
            states.append(state)
            actions.append(action)
            episode_stats.append(
                {
                    "episode_index": index,
                    "stats": {"observation.state": statistics(state), "action": statistics(action)},
                }
            )
            episodes.append({**entry["episode"], "episode_index": index})
            mappings.append(
                {
                    **entry["mapping"],
                    "episode_index": index,
                    "canonical_episode_index": entry["episode"]["episode_index"],
                    "canonical_task_index": entry["task_index"],
                }
            )
            assets.append(
                {"episode_index": index, "parquet_sha256": file_sha256(parquet), "video_sha256": video_hashes}
            )
            offset += len(data)
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(selected):
                print(f"Fold {training_fold}: copied {index + 1}/{len(selected)} episodes", flush=True)
        info = copy.deepcopy(inventory["info"])
        info.update(
            total_episodes=len(selected),
            total_frames=offset,
            total_tasks=5,
            total_videos=len(selected) * 2,
            total_chunks=(len(selected) + chunks_size - 1) // chunks_size,
            chunks_size=chunks_size,
            splits={"train": f"0:{len(selected)}"},
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
        )
        for key in ("canonical_episode_index", "canonical_task_index", "canonical_index"):
            info["features"][key] = {"dtype": "int64", "shape": [1], "names": None}
        write_json(stage / "meta/info.json", info)
        shutil.copy2(source / "meta/modality.json", stage / "meta/modality.json")
        write_json(
            stage / "meta/stats.json",
            {
                "observation.state": statistics(np.concatenate(states)),
                "action": statistics(np.concatenate(actions)),
            },
        )
        write_json(stage / "meta/episode_mapping.json", mappings)
        write_jsonl(stage / "meta/episodes.jsonl", episodes)
        write_jsonl(stage / "meta/episodes_stats.jsonl", episode_stats)
        write_jsonl(
            stage / "meta/tasks.jsonl",
            [
                {"task_index": task_map[name], "task": TASK_BY_NAME[name].instruction}
                for name in fold["train_tasks"]
            ],
        )
        write_json(stage / "meta/asset_checksums.json", assets)
        manifest = {
            "extractor": "examples.Libero_Mem.splits",
            "split_version": SPLIT_VERSION,
            "status": "complete",
            "format": "gr00t_lerobot_v2.1",
            "training_fold": training_fold,
            **fold,
            "source": str(source),
            "source_metadata_sha256": inventory["metadata_sha256"],
            "episodes": len(selected),
            "frames": offset,
            "tasks": 5,
            "videos": len(selected) * 2,
            "episodes_per_task": dict(sorted(Counter(mapping["task"] for mapping in mappings).items())),
            "normalization_scope": "selected_training_rows_only",
            "storage": "independent_copies",
            "copied_video_bytes": video_bytes,
            "resampled": False,
            "filtered_frames": 0,
            "data_config": "examples.Libero.custom_data_config:LiberoDataConfig",
            "source_conversion": inventory["manifest"],
        }
        write_json(stage / "manifest.json", manifest)
        for name, identity in inventory["metadata_identities"].items():
            _unchanged(source / name, identity)
        for entry in inventory["entries"]:
            for path, identity in entry["identities"].items():
                _unchanged(Path(path), identity)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {output}")
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Canonical ten-task LeRobot v2.1 dataset")
    parser.add_argument("--output", type=Path, required=True, help="New independent training-fold dataset")
    parser.add_argument(
        "--fold", choices=("A", "B"), required=True, help="Training fold; the other fold is held out"
    )
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = extract_fold(
        args.source, args.output, training_fold=args.fold, chunks_size=args.chunks_size, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
