"""Index RoboMME task families and COPY selected tasks into independent v2.1 datasets.

Instruction IDs are not environment task IDs. The index uses original HDF5 task
metadata and resolves shared instructions with sampled state/action signatures.
Extraction needs only that validated index and the converted video dataset.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


SUITES = {
    "counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "permanence": ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"),
    "reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
TASK_SUITE = {task: suite for suite, tasks in SUITES.items() for task in tasks}
PRIMARY_HELDOUT = ("PickXtimes", "ButtonUnmaskSwap", "PickHighlight", "PatternLock")
SPLITS = {"primary": PRIMARY_HELDOUT, **{f"loso_{suite}": tasks for suite, tasks in SUITES.items()}}
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))


def identity(root):
    return {name: hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest() for name in META_FILES}


def load_source(source):
    root = Path(source).expanduser().resolve(strict=True)
    info = read_json(root / "meta/info.json")
    if not str(info["codebase_version"]).startswith("v2."):
        raise ValueError("Expected a converted LeRobot v2.x dataset")
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if not cameras or any(feature["dtype"] == "image" for feature in info["features"].values()):
        raise ValueError("Use the converted video dataset, not the original embedded-image dataset")
    episodes = sorted(read_jsonl(root / "meta/episodes.jsonl"), key=lambda row: row["episode_index"])
    if not episodes or len({row["episode_index"] for row in episodes}) != len(episodes):
        raise ValueError("Empty or duplicate episode list")
    return root, info, episodes, cameras


def episode_paths(root, info, episode_id, cameras):
    values = dict(episode_index=episode_id, episode_chunk=episode_id // info["chunks_size"])
    parquet = root / info["data_path"].format(**values)
    videos = {key: root / info["video_path"].format(**values, video_key=key) for key in cameras}
    return parquet, videos


def controls(data):
    result = [np.stack(data[key]).astype(np.float32) for key in ("state", "actions")]
    if any(value.shape != (len(data), 8) or not np.isfinite(value).all() for value in result):
        raise ValueError("Expected finite (frames, 8) state/action vectors")
    return result


def control_hash(state, action):
    digest = hashlib.sha256(str(len(state)).encode())
    for value in (state, action):
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def point_signature(state, action, total_frames, demo_frames):
    positions = sorted({0, len(state) // 2, len(state) - 1})
    return f"{total_frames}:{demo_frames}:" + control_hash(state[positions], action[positions])


def hdf5_signatures(path):
    import h5py

    result = defaultdict(list)
    with h5py.File(path, "r") as handle:
        for name in sorted(handle):
            if not name.startswith("episode_"):
                continue
            episode = handle[name]
            indices = sorted(int(key.removeprefix("timestep_")) for key in episode if key.startswith("timestep_"))
            if indices != list(range(len(indices))) or not indices:
                raise ValueError(f"Non-contiguous timesteps: {path}:{name}")
            # Demo rows form a prefix in the source; locate its boundary without reading images.
            low, high = 0, len(indices)
            while low < high:
                middle = (low + high) // 2
                if bool(episode[f"timestep_{middle}/info/is_video_demo"][()]):
                    low = middle + 1
                else:
                    high = middle
            start, length = low, len(indices) - low
            if not length:
                continue
            positions = sorted({0, length // 2, length - 1})
            states, actions = [], []
            for position in positions:
                step = episode[f"timestep_{start + position}"]
                states.append(np.concatenate([step["obs/joint_state"][()], step["obs/gripper_state"][()][:1]]))
                actions.append(step["action/joint_action"][()])
            signature = f"{len(indices)}:{start}:" + control_hash(np.asarray(states), np.asarray(actions))
            result[signature].append(name)
    return result


def build_task_index(source, hdf5_root, output):
    import h5py

    root, info, episodes, cameras = load_source(source)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    goals, raw_files = defaultdict(set), {}
    for path in sorted(Path(hdf5_root).expanduser().glob("record_dataset_*.h5")):
        task = path.stem.removeprefix("record_dataset_")
        if task not in TASK_SUITE:
            continue
        raw_files[task] = path.resolve()
        with h5py.File(path, "r") as handle:
            for name in handle:
                if name.startswith("episode_"):
                    for value in np.asarray(handle[f"{name}/setup/task_goal"][()]).reshape(-1):
                        text = value.decode() if isinstance(value, bytes) else str(value)
                        goals[text.strip().lower()].add(task)
    if not raw_files:
        raise ValueError("No canonical RoboMME HDF5 files found")
    mappings = {row["episode_index"]: row for row in read_json(root / "meta/episode_mapping.json")}
    entries, signatures = [], {}
    for number, episode in enumerate(episodes, 1):
        sets = [goals[text.strip().lower()] for text in episode["tasks"]]
        candidates = set.intersection(*sets) if sets else set()
        if not candidates:
            raise ValueError(f"No HDF5 task matches episode {episode['episode_index']} instructions")
        parquet, _ = episode_paths(root, info, episode["episode_index"], cameras)
        data = pd.read_parquet(parquet, columns=["state", "actions", "task_index"], use_threads=False)
        state, action = controls(data)
        if len(data) != episode["length"]:
            raise ValueError(f"Wrong episode length: {parquet}")
        method = "unique_instruction_in_hdf5_metadata"
        matches = {}
        if len(candidates) > 1:
            mapping = mappings[episode["episode_index"]]
            if mapping["source_frames"] - mapping["excluded_demo_frames"] != len(data):
                raise ValueError("Source demo boundary disagrees with converted length")
            signature = point_signature(state, action, mapping["source_frames"], mapping["excluded_demo_frames"])
            for task in sorted(candidates):
                if task not in signatures:
                    signatures[task] = hdf5_signatures(raw_files[task])
                if signature in signatures[task]:
                    matches[task] = signatures[task][signature]
            candidates = set(matches)
            method = "hdf5_execution_state_action_signature"
        if len(candidates) != 1:
            raise ValueError(
                f"Task identity unresolved for episode {episode['episode_index']}: {sorted(candidates)}"
            )
        task = next(iter(candidates))
        entries.append(
            {
                "episode_index": episode["episode_index"],
                "task": task,
                "suite": TASK_SUITE[task],
                "length": len(data),
                "execution_sha256": control_hash(state, action),
                "instruction_indices": sorted(int(x) for x in data["task_index"].unique()),
                "identification": method,
                "hdf5_file": str(raw_files[task]),
                "matching_hdf5_episodes": matches.get(task),
            }
        )
        if number % 100 == 0:
            print(f"Indexed {number}/{len(episodes)} episodes", flush=True)
    index = {
        "version": 1,
        "source_metadata_sha256": identity(root),
        "episodes": entries,
        "episodes_per_task": dict(sorted(Counter(row["task"] for row in entries).items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects a previously validated mapping from accidental replacement.
    with output.open("x") as handle:
        handle.write(json.dumps(index, indent=2, allow_nan=False) + "\n")
    return index


def selected_task_names(available, *, split=None, tasks=None, exclude_tasks=None):
    if sum(value is not None for value in (split, tasks, exclude_tasks)) != 1:
        raise ValueError("Choose exactly one of split, tasks or exclude_tasks")
    if split is not None:
        if set(available) != set(TASK_SUITE):
            raise ValueError("Named splits require a source containing all 16 task families")
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        selected = set(available) - set(SPLITS[split])
    else:
        requested = set(tasks if tasks is not None else exclude_tasks)
        if requested - set(available):
            raise ValueError(f"Unknown or unavailable tasks: {sorted(requested - set(available))}")
        selected = requested if tasks is not None else set(available) - requested
    if not selected:
        raise ValueError("Selection contains no tasks")
    return sorted(selected)


def statistics(values):
    values = np.asarray(values, np.float64)
    return {
        key: value.tolist()
        for key, value in {
            "min": values.min(0),
            "max": values.max(0),
            "mean": values.mean(0),
            "std": values.std(0),
            "q01": np.quantile(values, 0.01, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
        }.items()
    }


def extract_tasks(
    source, task_map, output, *, split=None, tasks=None, exclude_tasks=None, workers=4, dry_run=False
):
    root, info, episodes, cameras = load_source(source)
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output).expanduser().absolute()
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must be outside the source dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    index = read_json(Path(task_map))
    if index["version"] != 1 or index["source_metadata_sha256"] != identity(root):
        raise ValueError("Task index does not match source metadata")
    entries = {row["episode_index"]: row for row in index["episodes"]}
    if len(entries) != len(index["episodes"]) or set(entries) != {row["episode_index"] for row in episodes}:
        raise ValueError("Task index must cover every source episode exactly once")
    if any(row["task"] not in TASK_SUITE or row["suite"] != TASK_SUITE[row["task"]] for row in entries.values()):
        raise ValueError("Invalid task/suite in task index")
    chosen = selected_task_names(
        {row["task"] for row in entries.values()}, split=split, tasks=tasks, exclude_tasks=exclude_tasks
    )
    kept = [episode for episode in episodes if entries[episode["episode_index"]]["task"] in chosen]
    vocabulary = {row["task_index"]: row["task"] for row in read_jsonl(root / "meta/tasks.jsonl")}
    used = sorted({old for episode in kept for old in entries[episode["episode_index"]]["instruction_indices"]})
    if set(used) - vocabulary.keys():
        raise ValueError("Task index references unknown language instruction IDs")
    remap = {old: new for new, old in enumerate(used)}
    jobs, offset, video_bytes = [], 0, 0
    for new_id, episode in enumerate(kept):
        old_id = episode["episode_index"]
        parquet, videos = episode_paths(root, info, old_id, cameras)
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        for path in videos.values():
            video_bytes += path.stat().st_size
        jobs.append((new_id, offset, episode, parquet, videos))
        offset += episode["length"]
    plan = {
        "source": str(root),
        "output": str(output),
        "split": split,
        "selected_tasks": chosen,
        "excluded_tasks": sorted({row["task"] for row in entries.values()} - set(chosen)),
        "episodes": len(kept),
        "frames": offset,
        "instructions": len(used),
        "videos": len(kept) * len(cameras),
        "video_bytes_to_copy": video_bytes,
        "copy_mode": "independent_files",
        "workers": workers,
        "episodes_per_task": dict(sorted(Counter(entries[ep["episode_index"]]["task"] for ep in kept).items())),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    parent_mappings = {row["episode_index"]: row for row in read_json(root / "meta/episode_mapping.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        (stage / "meta").mkdir(parents=True)

        def copy_episode(job):
            new_id, frame_offset, episode, parquet, videos = job
            old_id = episode["episode_index"]
            entry = entries[old_id]
            data = pd.read_parquet(parquet, use_threads=False)
            state, action = controls(data)
            if len(data) != episode["length"] or control_hash(state, action) != entry["execution_sha256"]:
                raise ValueError(f"State/action changed since task indexing: {parquet}")
            if "is_demo" in data and data["is_demo"].any():
                raise ValueError("Expected execution-only converted input")
            if sorted(int(x) for x in data["task_index"].unique()) != entry["instruction_indices"]:
                raise ValueError(f"Language task IDs changed: {parquet}")
            if not (data["episode_index"] == old_id).all() or not np.array_equal(
                data["frame_index"], np.arange(len(data))
            ):
                raise ValueError(f"Invalid source episode/frame indices: {parquet}")
            data["episode_index"] = np.full(len(data), new_id, dtype=data["episode_index"].dtype)
            if "epis_idx" in data:
                data["epis_idx"] = np.full(len(data), new_id, dtype=data["epis_idx"].dtype)
            data["index"] = np.arange(frame_offset, frame_offset + len(data), dtype=np.int64)
            data["task_index"] = data["task_index"].map(remap).astype(np.int64)
            values = dict(episode_index=new_id, episode_chunk=new_id // info["chunks_size"])
            destination = stage / info["data_path"].format(**values)
            destination.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(destination, index=False)
            for key, path in videos.items():
                destination = stage / info["video_path"].format(**values, video_key=key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)  # Real independent files; no links or re-encoding.
            mapping = {
                **parent_mappings[old_id],
                "episode_index": new_id,
                "parent_episode_index": old_id,
                "task": entry["task"],
                "suite": entry["suite"],
            }
            episode_stats = {"state": statistics(state), "actions": statistics(action)}
            output_entry = {
                **entry,
                "episode_index": new_id,
                "instruction_indices": [remap[old] for old in entry["instruction_indices"]],
            }
            return (
                {
                    "episode_index": new_id,
                    "length": len(data),
                    "tasks": [vocabulary[x] for x in entry["instruction_indices"]],
                },
                mapping,
                output_entry,
                episode_stats,
                state,
                action,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = []
            # Bound pending work so an invalid episode does not trigger copying the entire dataset.
            for start in range(0, len(jobs), workers):
                for result in pool.map(copy_episode, jobs[start : start + workers]):
                    results.append(result)
                    if len(results) % 100 == 0 or len(results) == len(jobs):
                        print(f"Copied {len(results)}/{len(jobs)} episodes", flush=True)
        output_info = copy.deepcopy(info)
        output_info.update(
            total_episodes=len(kept),
            total_frames=offset,
            total_tasks=len(used),
            total_videos=len(kept) * len(cameras),
            total_chunks=(len(kept) + info["chunks_size"] - 1) // info["chunks_size"],
            splits={"train": f"0:{len(kept)}"},
        )
        write_json(stage / "meta/info.json", output_info)
        shutil.copy2(root / "meta/modality.json", stage / "meta/modality.json")
        write_jsonl(stage / "meta/episodes.jsonl", [result[0] for result in results])
        write_jsonl(
            stage / "meta/tasks.jsonl", [{"task_index": remap[old], "task": vocabulary[old]} for old in used]
        )
        write_json(stage / "meta/episode_mapping.json", [result[1] for result in results])
        write_jsonl(
            stage / "meta/episodes_stats.jsonl",
            [{"episode_index": result[0]["episode_index"], "stats": result[3]} for result in results],
        )
        write_json(
            stage / "meta/stats.json",
            {
                "state": statistics(np.concatenate([result[4] for result in results])),
                "actions": statistics(np.concatenate([result[5] for result in results])),
            },
        )
        write_json(
            stage / "meta/episode_tasks.json",
            {
                "version": 1,
                "source_metadata_sha256": identity(stage),
                "episodes": [result[2] for result in results],
            },
        )
        manifest = {
            **plan,
            "status": "complete",
            "extractor": "scripts/extract_robomme_tasks.py",
            "normalization_scope": "selected_execution_frames_only",
            "reencoded_video": False,
            "task_index_metadata_sha256": index["source_metadata_sha256"],
        }
        write_json(stage / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {output}")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Build a validated episode-to-task index once")
    index_parser.add_argument("--source", type=Path, required=True)
    index_parser.add_argument("--hdf5-root", type=Path, required=True)
    index_parser.add_argument("--output", type=Path, required=True)
    extract_parser = subparsers.add_parser("extract", help="Copy selected tasks into one standalone dataset")
    extract_parser.add_argument("--source", type=Path, required=True)
    extract_parser.add_argument("--task-map", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    selection = extract_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split", choices=tuple(SPLITS))
    selection.add_argument("--tasks", nargs="+", choices=tuple(TASK_SUITE))
    selection.add_argument("--exclude-tasks", nargs="+", choices=tuple(TASK_SUITE))
    extract_parser.add_argument("--workers", type=int, default=4)
    extract_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "index":
        result = build_task_index(args.source, args.hdf5_root, args.output)
        print(json.dumps(result["episodes_per_task"], indent=2))
    else:
        extract_tasks(
            args.source,
            args.task_map,
            args.output,
            split=args.split,
            tasks=args.tasks,
            exclude_tasks=args.exclude_tasks,
            workers=args.workers,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
