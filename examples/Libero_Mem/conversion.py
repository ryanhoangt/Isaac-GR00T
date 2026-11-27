"""Convert audited LIBERO-Mem Full HDF5 to the existing GR00T LIBERO schema.

Every source row is retained. Images are streamed one frame at a time; numeric
arrays are bounded by one episode per worker. No simulator or GPU is required.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import tempfile

import av
import h5py
import numpy as np
import pandas as pd

from examples.Libero_Mem.source import AUDIT_VERSION, vector_fingerprint
from examples.Libero_Mem.tasks import TASK_BY_NUMBER
from gr00t.utils.lerobot_conversion import DATA_PATH, VIDEO_PATH, statistics, write_json, write_jsonl


CONVERTER_VERSION = 1
FPS = 20  # Upstream ControlEnv's control_freq. No frame resampling.
CAMERAS = {
    "observation.images.image": "obs/agentview_rgb",
    "observation.images.wrist_image": "obs/eye_in_hand_rgb",
}
MODALITY_PATH = Path(__file__).resolve().parents[1] / "Libero/modality.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rgb(image):
    """Undo refinement's extra horizontal flip to match get_libero_image.

    Refine stores rot180(raw[:, ::-1]); GR00T LIBERO uses rot180(raw).
    Thus stored[:, ::-1] has exactly the existing evaluator's orientation.
    """
    image = np.asarray(image)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB [256,256,3]")
    return np.ascontiguousarray(image[:, ::-1])


def _check_source(path, identity):
    stat = path.stat()
    if stat.st_size != identity["size_bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
        raise ValueError(f"Source file differs from audit: {path}")


def conversion_plan(source, audit, *, task_ids=None, demo_ids=None, chunks_size=1000):
    """Read the small unit-1 report; hash metadata without parsing its annotations."""
    source = Path(source).expanduser().resolve(strict=True)
    audit_path = Path(audit).expanduser().resolve(strict=True)
    report = json.loads(audit_path.read_text())
    if report.get("audit_version") != AUDIT_VERSION or report.get("status") != "complete":
        raise ValueError("Expected a completed unit-1 source audit")
    if type(chunks_size) is not int or chunks_size < 1:
        raise ValueError("chunks_size must be positive")
    ids = list(range(1, 11)) if task_ids is None else list(task_ids)
    if not ids or any(type(i) is not int or i not in TASK_BY_NUMBER for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("task_ids must be distinct integers in 1..10")
    if demo_ids is not None:
        if (
            not demo_ids
            or any(type(i) is not int or i < 0 for i in demo_ids)
            or len(set(demo_ids)) != len(demo_ids)
        ):
            raise ValueError("demo_ids must be distinct nonnegative source demo numbers")
    records = {record["number"]: record for record in report["tasks"]}
    if len(records) != len(report["tasks"]) or set(ids) - records.keys():
        raise ValueError("Audit does not cover the selected tasks, or has duplicate task records")
    meta_path = source / "metainfo.json"
    _check_source(meta_path, report["metadata"])
    if file_sha256(meta_path) != report["metadata"]["sha256"]:
        raise ValueError("Source metadata differs from audit")
    _check_source(meta_path, report["metadata"])
    jobs, offset = [], 0
    for task_index, number in enumerate(sorted(ids)):
        task = TASK_BY_NUMBER[number]
        record = records[number]
        if record["name"] != task.name or record["instruction"] != task.instruction:
            raise ValueError(f"Audit task identity differs from registry: T{number}")
        path = source / task.filename
        _check_source(path, record["full_file"])
        demos = record["full_demos"]
        names = [demo["demo_id"] for demo in demos]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate audited demos: {task.name}")
        if demo_ids is not None:
            selected = {f"demo_{number}" for number in demo_ids}
            if selected - set(names):
                raise ValueError(f"Requested demos missing from T{task.number}: {sorted(selected - set(names))}")
            demos = [demo for demo in demos if demo["demo_id"] in selected]
        for demo in sorted(demos, key=lambda entry: int(entry["demo_id"].removeprefix("demo_"))):
            if type(demo["frames"]) is not int or demo["frames"] < 1:
                raise ValueError(f"Invalid audited episode length: {task.name}/{demo['demo_id']}")
            jobs.append(
                {
                    "source_file": str(path),
                    "source_identity": record["full_file"],
                    "task_number": task.number,
                    "task_name": task.name,
                    "instruction": task.instruction,
                    "task_index": task_index,
                    "episode_index": len(jobs),
                    "frame_offset": offset,
                    "source_episode_key": f"{task.name}/{demo['demo_id']}",
                    "demo": demo,
                }
            )
            offset += demo["frames"]
    if not jobs:
        raise ValueError("No audited Full episodes selected")
    return {
        "converter_version": CONVERTER_VERSION,
        "source": str(source),
        "audit_sha256": file_sha256(audit_path),
        "metadata": report["metadata"],
        "modality_sha256": file_sha256(MODALITY_PATH),
        "fps": FPS,
        "chunks_size": chunks_size,
        "rgb_conversion": "refined_rgb_horizontal_flip",
        "gripper_conversion": "(1-source_gripper)/2",
        "video_encoding": {"codec": "libx264", "crf": 18, "preset": "fast", "pixel_format": "yuv420p"},
        "jobs": jobs,
    }


def _numeric(group, key, shape):
    if key not in group or not isinstance(group[key], h5py.Dataset) or group[key].shape != shape:
        raise ValueError(f"Missing or incorrectly shaped dataset: {group.name}/{key}, expected {shape}")
    value = group[key][()]
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Expected finite numeric values: {group.name}/{key}")
    return value


def _encode(dataset, path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.height = stream.width = 256
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "18", "preset": "fast"}
        for i in range(len(dataset)):
            frame = av.VideoFrame.from_ndarray(canonical_rgb(dataset[i]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def convert_one(job, cache):
    """Atomically cache one complete episode; resume verifies cached checksums."""
    cache = Path(cache)
    source_path = Path(job["source_file"])
    _check_source(source_path, job["source_identity"])
    if cache.exists():
        record = json.loads((cache / "record.json").read_text())
        if record["job"] != job:
            raise ValueError(f"Cached episode differs from conversion plan: {cache}")
        for name, checksum in record["artifacts_sha256"].items():
            path = cache / name
            if not path.is_file() or file_sha256(path) != checksum:
                raise ValueError(f"Incomplete or corrupt cached artifact: {path}")
        return record
    length, demo_id = job["demo"]["frames"], job["demo"]["demo_id"]
    with h5py.File(source_path, "r") as handle:
        group = handle[f"data/{demo_id}"]
        pose = _numeric(group, "obs/ee_states", (length, 6))
        gripper = _numeric(group, "obs/gripper_states", (length, 2))
        state = np.concatenate([pose, gripper], axis=1)
        original = _numeric(group, "actions", (length, 7))
        if vector_fingerprint(state) != job["demo"]["state_sha256"]:
            raise ValueError(f"State differs from audit: {job['source_episode_key']}")
        if vector_fingerprint(original) != job["demo"]["actions_sha256"]:
            raise ValueError(f"Actions differ from audit: {job['source_episode_key']}")
        if not np.isin(original[:, -1], [-1, 1]).all():
            raise ValueError(f"Expected -1/+1 simulator gripper commands: {job['source_episode_key']}")
        action = original.astype(np.float32)
        action[:, -1] = (1 - action[:, -1]) / 2
        dones = _numeric(group, "dones", (length,))
        if not np.isin(dones, [0, 1]).all() or not dones[-1] or dones[:-1].any():
            raise ValueError(f"Expected only final done flag: {job['source_episode_key']}")
        for key in CAMERAS.values():
            if key not in group or group[key].shape != (length, 256, 256, 3) or group[key].dtype != np.uint8:
                raise ValueError(f"Invalid RGB dataset: {group.name}/{key}")
        state = state.astype(np.float32)
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("State/action cannot be represented as finite float32")
        data = pd.DataFrame(
            {
                "observation.state": list(state),
                "action": list(action),
                "timestamp": np.arange(length, dtype=np.float64) / FPS,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, job["episode_index"], dtype=np.int64),
                "index": np.arange(job["frame_offset"], job["frame_offset"] + length, dtype=np.int64),
                "task_index": np.full(length, job["task_index"], dtype=np.int64),
                "next.done": dones.astype(bool),
                "source_task_id": np.full(length, job["task_number"], dtype=np.int64),
                "source_demo_id": demo_id,
                "source_episode_key": job["source_episode_key"],
                "source_frame_index": np.arange(length, dtype=np.int64),
                "source_action_gripper": original[:, -1].astype(np.float32),
            }
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{cache.name}-", dir=cache.parent) as temporary:
            stage = Path(temporary) / "episode"
            stage.mkdir()
            data.to_parquet(stage / "data.parquet", index=False)
            for i, key in enumerate(CAMERAS.values()):
                _encode(group[key], stage / f"camera_{i}.mp4")
            _check_source(source_path, job["source_identity"])
            record = {
                "job": job,
                "length": length,
                "artifacts_sha256": {
                    name: file_sha256(stage / name) for name in ("data.parquet", "camera_0.mp4", "camera_1.mp4")
                },
            }
            write_json(stage / "record.json", record)
            stage.rename(cache)
    return record


def finalize(stage, work, plan):
    stage.mkdir()
    (stage / "meta").mkdir()
    episodes, mappings, episode_stats, states, actions = [], [], [], [], []
    jobs = plan["jobs"]
    for job in jobs:
        index = job["episode_index"]
        cache = work / "episodes" / f"{index:06d}"
        data = pd.read_parquet(cache / "data.parquet", use_threads=False)
        values = {"episode_chunk": index // plan["chunks_size"], "episode_index": index}
        target = stage / DATA_PATH.format(**values)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(cache / "data.parquet", target)
        for camera, key in enumerate(CAMERAS):
            video = stage / VIDEO_PATH.format(**values, video_key=key)
            video.parent.mkdir(parents=True, exist_ok=True)
            os.link(cache / f"camera_{camera}.mp4", video)
        state, action = np.stack(data["observation.state"]), np.stack(data["action"])
        states.append(state)
        actions.append(action)
        episode_stats.append(
            {
                "episode_index": index,
                "stats": {
                    "observation.state": statistics(state),
                    "action": statistics(action),
                },
            }
        )
        episodes.append({"episode_index": index, "length": len(data), "tasks": [job["instruction"]]})
        mappings.append(
            {
                "episode_index": index,
                "task": job["task_name"],
                "source_task_id": job["task_number"],
                "source_demo_id": job["demo"]["demo_id"],
                "raw_demo_id": job["demo"]["raw_demo_id"],
                "source_episode_key": job["source_episode_key"],
                "source_file": job["source_file"],
                "source_frames": len(data),
                "initial_state_sha256": job["demo"]["initial_state_sha256"],
            }
        )
    task_jobs = {job["task_index"]: job for job in jobs}
    features = {
        key: {"dtype": "int64", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index", "source_task_id", "source_frame_index")
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "ax", "ay", "az", "finger_0", "finger_1"],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "source_action_gripper": {"dtype": "float32", "shape": [1], "names": None},
            **{
                key: {"dtype": "string", "shape": [1], "names": None}
                for key in ("source_demo_id", "source_episode_key")
            },
            **{
                key: {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": FPS,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
                for key in CAMERAS
            },
        }
    )
    total_frames = sum(episode["length"] for episode in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(jobs),
        "total_frames": total_frames,
        "total_tasks": len(task_jobs),
        "total_videos": 2 * len(jobs),
        "total_chunks": (len(jobs) + plan["chunks_size"] - 1) // plan["chunks_size"],
        "chunks_size": plan["chunks_size"],
        "fps": FPS,
        "splits": {"train": f"0:{len(jobs)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    write_json(stage / "meta/info.json", info)
    (stage / "meta/modality.json").write_bytes(MODALITY_PATH.read_bytes())
    write_json(
        stage / "meta/stats.json",
        {"observation.state": statistics(np.concatenate(states)), "action": statistics(np.concatenate(actions))},
    )
    write_json(stage / "meta/episode_mapping.json", mappings)
    write_jsonl(stage / "meta/episodes.jsonl", episodes)
    write_jsonl(stage / "meta/episodes_stats.jsonl", episode_stats)
    write_jsonl(
        stage / "meta/tasks.jsonl",
        [{"task_index": i, "task": task_jobs[i]["instruction"]} for i in sorted(task_jobs)],
    )
    manifest = {
        "converter": "examples.Libero_Mem.conversion",
        "converter_version": CONVERTER_VERSION,
        "status": "complete",
        "format": "gr00t_lerobot_v2.1",
        "source": plan["source"],
        "audit_sha256": plan["audit_sha256"],
        "metadata_sha256": plan["metadata"]["sha256"],
        "episodes": len(jobs),
        "frames": total_frames,
        "tasks": len(task_jobs),
        "fps": FPS,
        "alignment": "same_source_row",
        "resampled": False,
        "filtered_frames": 0,
        "action_representation": "OSC_POSE_delta_with_dataset_gripper_0_1",
        "gripper_conversion": plan["gripper_conversion"],
        "rgb_conversion": plan["rgb_conversion"],
        "normalization_scope": "all_exported_rows; recompute separately for each training fold",
        "data_config": "examples.Libero.custom_data_config:LiberoDataConfig",
        "video_encoding": plan["video_encoding"],
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


def convert_dataset(
    source,
    output,
    *,
    audit,
    workers=4,
    task_ids=None,
    demo_ids=None,
    chunks_size=1000,
    resume=False,
    dry_run=False,
):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output).expanduser().absolute()
    source = Path(source).expanduser().resolve(strict=True)
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan = conversion_plan(source, audit, task_ids=task_ids, demo_ids=demo_ids, chunks_size=chunks_size)
    if dry_run:
        return {
            "status": "dry_run",
            "source": str(source),
            "output": str(output),
            "episodes": len(plan["jobs"]),
            "frames": sum(job["demo"]["frames"] for job in plan["jobs"]),
            "task_ids": sorted({job["task_number"] for job in plan["jobs"]}),
            "fps": FPS,
        }
    work = output.with_name(f".{output.name}.work")
    if work.exists() and not resume:
        raise FileExistsError(f"Work directory exists; use --resume: {work}")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings_path = work / "configuration.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != plan:
                raise ValueError("Resume source, selection or conversion settings differ")
        else:
            write_json(settings_path, plan)
        jobs = plan["jobs"]
        print(f"Converting {len(jobs)} episodes with {workers} workers; cache: {work}", flush=True)
        if workers == 1:
            for done, job in enumerate(jobs, 1):
                convert_one(job, work / "episodes" / f"{job['episode_index']:06d}")
                print(f"[{done}/{len(jobs)}] {job['source_episode_key']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = {
                    pool.submit(convert_one, job, work / "episodes" / f"{job['episode_index']:06d}"): job
                    for job in jobs
                }
                for done, future in enumerate(as_completed(futures), 1):
                    future.result()
                    print(f"[{done}/{len(jobs)}] {futures[future]['source_episode_key']}", flush=True)
        # Ensure no source input changed after its worker completed.
        _check_source(source / "metainfo.json", plan["metadata"])
        for job in jobs:
            _check_source(Path(job["source_file"]), job["source_identity"])
        if file_sha256(MODALITY_PATH) != plan["modality_sha256"]:
            raise ValueError("LIBERO modality metadata changed during conversion")
        with tempfile.TemporaryDirectory(prefix="publish-", dir=work) as temporary:
            stage = Path(temporary) / "dataset"
            manifest = finalize(stage, work, plan)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {output}")
            stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="LIBERO-Mem Full HDF5 directory")
    parser.add_argument("--audit", type=Path, required=True, help="Completed unit-1 source inventory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(1, 11))
    parser.add_argument(
        "--demo-ids", type=int, nargs="+", help="Original demo numbers, required in every selected task"
    )
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate the plan without creating output or decoding RGB"
    )
    args = parser.parse_args(argv)
    result = convert_dataset(
        args.source,
        args.output,
        audit=args.audit,
        workers=args.workers,
        task_ids=args.task_ids,
        demo_ids=args.demo_ids,
        chunks_size=args.chunks_size,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
