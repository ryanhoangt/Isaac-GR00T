"""Convert a LOCAL RoboMME LeRobot v2.x image dataset to GR00T's video layout.

No downloads. Demo rows are excluded; state/action pairing and annotations are
preserved. FPS comes from source metadata, with no temporal resampling.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import io
import json
from multiprocessing import get_context
import os
from pathlib import Path
import tempfile

import av
import numpy as np
import pandas as pd
from PIL import Image

from gr00t.utils.lerobot_conversion import DATA_PATH, VIDEO_PATH, statistics, write_json, write_jsonl

CAMERAS = ("image", "wrist_image")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def modality_metadata():
    result = {}
    for modality, original in (("state", "state"), ("action", "actions")):
        result[modality] = {
            name: {"original_key": original, "start": start, "end": end, "absolute": True, "dtype": "float32"}
            for name, start, end in (("joint_position", 0, 7), ("gripper", 7, 8))
        }
    result["video"] = {key: {"original_key": key} for key in CAMERAS}
    result["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    return result


def decode_image(value, source, parquet):
    """Decode Hugging Face Image structs using only local bytes or paths."""
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = value["path"]
        else:
            raise ValueError("Image struct has neither bytes nor a local path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        resource = io.BytesIO(value)
    elif isinstance(value, str):
        if "://" in value:
            raise ValueError("Remote image URLs are unsupported; download the dataset first")
        path = Path(value)
        candidates = [path] if path.is_absolute() else [source / path, parquet.parent / path]
        resource = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resource is None:
            raise FileNotFoundError(f"Missing local image: {value}")
    else:
        raise ValueError(f"Unsupported parquet image value: {type(value).__name__}")
    with Image.open(resource) as image:
        return np.asarray(image.convert("RGB"))


def load_source(source, episode_indices=None):
    source = Path(source).expanduser().resolve(strict=True)
    info = json.loads((source / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2."):
        raise ValueError("Expected local LeRobot v2.x, such as Yinpei/robomme_data_lerobot")
    for key in CAMERAS:
        feature = info["features"][key]
        if feature["dtype"] != "image":
            raise ValueError(f"Expected {key} to be an image feature, got {feature['dtype']}")
        shape = feature["shape"]
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 2 or any(n % 2 for n in shape[:2]):
            raise ValueError(f"Expected even-sized RGB metadata for {key}, got {shape}")
    fps = float(info["fps"])
    if not np.isfinite(fps) or not fps.is_integer() or fps <= 0:
        raise ValueError("Source FPS must be a positive integer")
    tasks = read_jsonl(source / "meta/tasks.jsonl")
    if len({task["task_index"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate source task indices")
    episodes = sorted(read_jsonl(source / "meta/episodes.jsonl"), key=lambda ep: ep["episode_index"])
    ids = {ep["episode_index"] for ep in episodes}
    if len(ids) != len(episodes):
        raise ValueError("Duplicate source episode indices")
    if episode_indices is not None:
        selected = set(episode_indices)
        if selected - ids:
            raise ValueError(f"Unknown source episode IDs: {sorted(selected - ids)}")
        episodes = [ep for ep in episodes if ep["episode_index"] in selected]
    if not episodes:
        raise ValueError("No source episodes selected")
    jobs = []
    for output_id, episode in enumerate(episodes):
        source_id = episode["episode_index"]
        parquet = source / info["data_path"].format(
            episode_chunk=source_id // info["chunks_size"], episode_index=source_id
        )
        stat = parquet.stat()
        jobs.append(
            {
                "source": str(source),
                "parquet": str(parquet),
                "source_episode_index": source_id,
                "episode_index": output_id,
                "source_length": episode["length"],
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "fps": int(fps),
                "shapes": {key: info["features"][key]["shape"] for key in CAMERAS},
            }
        )
    return source, info, tasks, jobs


def convert_one(job, cache):
    """Read one episode parquet per worker; publish each cache atomically."""
    cache = Path(cache)
    if cache.exists():
        record = json.loads((cache / "record.json").read_text())
        if record["job"] != job:
            raise ValueError(f"Cached episode settings differ: {cache}")
        for name in ("data.parquet", "image.mp4", "wrist_image.mp4"):
            if not (cache / name).is_file():
                raise ValueError(f"Incomplete cached episode: {cache / name}")
        return record
    source, parquet = Path(job["source"]), Path(job["parquet"])
    data = pd.read_parquet(parquet, engine="pyarrow", use_threads=False)
    if len(data) != job["source_length"]:
        raise ValueError(f"Episode length differs from metadata: {parquet}")
    if not (data["episode_index"] == job["source_episode_index"]).all():
        raise ValueError(f"Wrong episode IDs in {parquet}")
    if not np.array_equal(data["frame_index"], np.arange(len(data))):
        raise ValueError(f"Non-contiguous source frame indices: {parquet}")
    if not pd.api.types.is_bool_dtype(data["is_demo"]):
        raise ValueError(f"is_demo must be a boolean column: {parquet}")
    demo = data["is_demo"].to_numpy()
    execution = np.flatnonzero(~demo)
    if not len(execution):
        raise ValueError(f"Episode has no execution frames: {parquet}")
    start = int(execution[0])
    if demo[start:].any():
        raise ValueError(f"Found demo frame after execution began: {parquet}")
    if "exec_start_idx" in data and not (data["exec_start_idx"] == start).all():
        raise ValueError(f"exec_start_idx disagrees with is_demo: {parquet}")
    data = data.iloc[start:].copy().reset_index(drop=True)
    for key in ("state", "actions"):
        values = np.stack(data[key]).astype(np.float32)
        if values.shape != (len(data), 8) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite (frames, 8) vectors in {key}: {parquet}")
        data[key] = list(values)
    if not np.isin(np.stack(data["actions"])[:, -1], [-1, 1]).all():
        raise ValueError(f"Gripper actions must be -1 or +1: {parquet}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{cache.name}-", dir=cache.parent) as temporary:
        stage = Path(temporary) / "episode"
        stage.mkdir()
        with ExitStack() as stack:
            writers = {}
            for key in CAMERAS:
                container = stack.enter_context(av.open(str(stage / f"{key}.mp4"), mode="w"))
                stream = container.add_stream("libx264", rate=job["fps"])
                stream.height, stream.width, _ = job["shapes"][key]
                stream.pix_fmt = "yuv420p"
                stream.codec_context.thread_count = 1
                stream.options = {"crf": "18", "preset": "fast"}
                writers[key] = (container, stream)
            for index in range(len(data)):
                for key, (container, stream) in writers.items():
                    image = decode_image(data[key].iloc[index], source, parquet)
                    if list(image.shape) != job["shapes"][key]:
                        raise ValueError(f"Wrong image shape: {parquet}, {key}, source row {index + start}")
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                        container.mux(packet)
            for container, stream in writers.values():
                for packet in stream.encode():
                    container.mux(packet)
        data = data.drop(columns=list(CAMERAS))
        data["source_episode_index"] = job["source_episode_index"]
        data["source_frame_index"] = data["frame_index"]
        data["source_index"] = data["index"]
        data["source_timestamp"] = data["timestamp"]
        data["source_exec_start_idx"] = start
        data["source_task_index"] = data["task_index"]
        data["frame_index"] = np.arange(len(data), dtype=np.int64)
        data["timestamp"] = np.arange(len(data), dtype=np.float64) / job["fps"]
        data["episode_index"] = job["episode_index"]
        if "epis_idx" in data:
            data["epis_idx"] = np.full(len(data), job["episode_index"], dtype=data["epis_idx"].dtype)
        data["exec_start_idx"] = np.zeros(len(data), dtype=np.int32)
        data.to_parquet(stage / "data.parquet", index=False)
        record = {
            "job": job,
            "length": len(data),
            "excluded_demo_frames": start,
            "source_task_indices": sorted(int(x) for x in data["task_index"].unique()),
        }
        write_json(stage / "record.json", record)
        stage.rename(cache)
    return record


def finalize(stage, work, info, tasks, jobs, settings):
    records = [
        json.loads((work / "episodes" / f"{job['episode_index']:06d}" / "record.json").read_text()) for job in jobs
    ]
    used_tasks = sorted({index for record in records for index in record["source_task_indices"]})
    vocabulary = {task["task_index"]: task["task"] for task in tasks}
    if set(used_tasks) - vocabulary.keys():
        raise ValueError("Parquet references task IDs missing from meta/tasks.jsonl")
    task_map = {old: new for new, old in enumerate(used_tasks)}
    (stage / "meta").mkdir(parents=True)
    episodes, mappings, episode_stats, states, actions = [], [], [], [], []
    offset = 0
    for job, record in zip(jobs, records):
        index = job["episode_index"]
        cache = work / "episodes" / f"{index:06d}"
        data = pd.read_parquet(cache / "data.parquet", use_threads=False)
        data["index"] = np.arange(offset, offset + len(data), dtype=np.int64)
        data["task_index"] = data["source_task_index"].map(task_map).astype(np.int64)
        values = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        path = stage / DATA_PATH.format(**values)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        for key in CAMERAS:
            video = stage / VIDEO_PATH.format(**values, video_key=key)
            video.parent.mkdir(parents=True, exist_ok=True)
            os.link(cache / f"{key}.mp4", video)
        state, action = np.stack(data["state"]), np.stack(data["actions"])
        states.append(state)
        actions.append(action)
        episode_stats.append(
            {"episode_index": index, "stats": {"state": statistics(state), "actions": statistics(action)}}
        )
        episodes.append(
            {
                "episode_index": index,
                "length": len(data),
                "tasks": [vocabulary[x] for x in record["source_task_indices"]],
            }
        )
        mappings.append(
            {
                "episode_index": index,
                "source_episode_index": job["source_episode_index"],
                "source_parquet": job["parquet"],
                "source_frames": job["source_length"],
                "excluded_demo_frames": record["excluded_demo_frames"],
                "exported_frames": len(data),
            }
        )
        offset += len(data)
    output_info = copy.deepcopy(info)
    output_info.update(
        codebase_version="v2.1",
        total_episodes=len(jobs),
        total_frames=offset,
        total_tasks=len(used_tasks),
        total_videos=2 * len(jobs),
        total_chunks=(len(jobs) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(jobs)}"},
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
    )
    for key in CAMERAS:
        output_info["features"][key] = {
            "dtype": "video",
            "shape": info["features"][key]["shape"],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": info["fps"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for key in (
        "source_episode_index",
        "source_frame_index",
        "source_index",
        "source_exec_start_idx",
        "source_task_index",
    ):
        output_info["features"][key] = {"dtype": "int64", "shape": [1], "names": None}
    output_info["features"]["source_timestamp"] = copy.deepcopy(info["features"]["timestamp"])
    output_info["features"]["timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    output_info["features"]["exec_start_idx"]["dtype"] = "int32"
    for key in ("state", "actions"):
        output_info["features"][key]["dtype"] = "float32"
    write_json(stage / "meta/info.json", output_info)
    write_json(stage / "meta/modality.json", modality_metadata())
    write_json(
        stage / "meta/stats.json",
        {"state": statistics(np.concatenate(states)), "actions": statistics(np.concatenate(actions))},
    )
    write_json(stage / "meta/episode_mapping.json", mappings)
    write_jsonl(stage / "meta/episodes.jsonl", episodes)
    write_jsonl(stage / "meta/episodes_stats.jsonl", episode_stats)
    write_jsonl(
        stage / "meta/tasks.jsonl", [{"task_index": task_map[old], "task": vocabulary[old]} for old in used_tasks]
    )
    manifest = {
        "converter": "scripts/convert_robomme_lerobot.py",
        "status": "complete",
        "source": settings["source"],
        "format": "gr00t_lerobot_v2.1",
        "episodes": len(jobs),
        "frames": offset,
        "fps": info["fps"],
        "excluded_demo_frames": sum(record["excluded_demo_frames"] for record in records),
        "alignment": "same_source_row",
        "action_representation": "absolute_joint_targets",
        "normalization_scope": "exported_execution_frames_only",
        "resampled": False,
        "video_encoding": {"codec": "libx264", "crf": 18, "pixel_format": "yuv420p"},
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


def convert_dataset(source, output, *, workers=4, episode_indices=None, resume=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    source, info, tasks, jobs = load_source(source, episode_indices)
    output = Path(output).expanduser().absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source dataset directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    work = output.with_name(f".{output.name}.work")
    settings = {
        "version": 1,
        "source": str(source),
        "episodes": jobs,
        "metadata_sha256": {
            name: hashlib.sha256((source / "meta" / name).read_bytes()).hexdigest()
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
        },
    }
    if work.exists() and not resume:
        raise FileExistsError(f"Work directory exists; use --resume: {work}")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = work / "configuration.json"
        if config.exists():
            if json.loads(config.read_text()) != settings:
                raise ValueError("Resume source or settings differ from the saved conversion")
        else:
            write_json(config, settings)
        print(f"Converting {len(jobs)} episodes with {workers} workers; cache: {work}", flush=True)
        if workers == 1:
            for done, job in enumerate(jobs, 1):
                convert_one(job, work / "episodes" / f"{job['episode_index']:06d}")
                print(f"[{done}/{len(jobs)}] source episode {job['source_episode_index']} ready", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = {
                    pool.submit(convert_one, job, work / "episodes" / f"{job['episode_index']:06d}"): job
                    for job in jobs
                }
                for done, future in enumerate(as_completed(futures), 1):
                    future.result()
                    print(
                        f"[{done}/{len(jobs)}] source episode {futures[future]['source_episode_index']} ready",
                        flush=True,
                    )
        print("Writing merged metadata and execution-only statistics...", flush=True)
        with tempfile.TemporaryDirectory(prefix="publish-", dir=work) as temporary:
            stage = Path(temporary) / "dataset"
            manifest = finalize(stage, work, info, tasks, jobs, settings)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {output}")
            stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local dataset root containing meta/ and data/")
    parser.add_argument("--output", type=Path, required=True, help="New GR00T dataset root; never overwritten")
    parser.add_argument("--workers", type=int, default=4, help="Parallel episode workers (default: 4)")
    parser.add_argument(
        "--episode-indices", nargs="+", type=int, help="Optional source episode IDs; defaults to ALL"
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed episodes after interruption")
    args = parser.parse_args()
    manifest = convert_dataset(
        args.source, args.output, workers=args.workers, episode_indices=args.episode_indices, resume=args.resume
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
