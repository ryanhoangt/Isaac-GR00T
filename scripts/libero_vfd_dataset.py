"""Score existing LIBERO LeRobot demonstrations without importing a simulator.

Raw images/state are passed through the monitor's checkpoint-owned eval transforms.
Demonstration actions are never used to construct the actor's flow trajectory.
"""

import ast
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from scripts.libero_vfd import SnapshotVFDPolicy, calibrate, flow_grid


SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "long": "libero_10"}
STATE_SLICES = {
    "x": (0, 1),
    "y": (1, 2),
    "z": (2, 3),
    "roll": (3, 4),
    "pitch": (4, 5),
    "yaw": (5, 6),
    "gripper": (6, 8),
}
VIDEO_KEYS = {"video.image": "observation.images.image", "video.wrist_image": "observation.images.wrist_image"}
DATA_KIND = "expert_demonstration"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def task_languages(task_map_path):
    """Read LIBERO's literal task table, without importing LIBERO/MuJoCo/OpenGL."""
    tree = ast.parse(Path(task_map_path).read_text())
    tables = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets)
    ]
    if len(tables) != 1:
        raise ValueError("Expected one literal libero_task_map")
    result = {}
    for suite in SUITES.values():
        names = tables[0][suite]
        languages = [re.sub(r"^.*?SCENE\d+_", "", name).replace("_", " ") for name in names]
        if len(languages) != 10 or len(set(languages)) != 10:
            raise ValueError(f"Invalid task table for {suite}")
        result[suite] = {language: task for task, language in enumerate(languages)}
    return result


def episode_path(root, info, episode, video_key=None):
    values = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
    if video_key is not None:
        values["video_key"] = video_key
    return root / info["video_path" if video_key else "data_path"].format(**values)


def catalog(root, suite, languages):
    root = Path(root).resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    for key, (start, end) in STATE_SLICES.items():
        if modality["state"][key]["start"] != start or modality["state"][key]["end"] != end:
            raise ValueError(f"Unexpected LIBERO state layout: {key}")
    for key, original in VIDEO_KEYS.items():
        if modality["video"][key.removeprefix("video.")]["original_key"] != original:
            raise ValueError(f"Unexpected LIBERO video layout: {key}")
    tasks = {int(item["task_index"]): item["task"] for item in read_jsonl(root / "meta/tasks.jsonl")}
    episodes, seen = [], set()
    for item in read_jsonl(root / "meta/episodes.jsonl"):
        index, length = int(item["episode_index"]), int(item["length"])
        if index in seen or index < 0 or length < 1 or len(item["tasks"]) != 1:
            raise ValueError(f"Invalid/duplicate dataset episode: {item}")
        seen.add(index)
        language = item["tasks"][0]
        if language not in languages[suite] or language not in tasks.values():
            raise ValueError(f"Unknown task language in {suite}: {language}")
        episodes.append(
            {
                "suite": suite,
                "task_id": languages[suite][language],
                "language": language,
                "dataset_episode_index": index,
                "dataset_path": str(root),
                "length": length,
            }
        )
    return info, tasks, episodes


def select_episodes(episodes, task_ids, per_task, start, seed):
    """Select evenly per task; selection is independent of scores and outcomes."""
    groups = defaultdict(list)
    for item in episodes:
        groups[item["task_id"]].append(item)
    selected = []
    for task in task_ids:
        items = sorted(groups[task], key=lambda item: item["dataset_episode_index"])
        order = np.random.default_rng(np.random.SeedSequence([seed, task])).permutation(len(items))
        stop = len(items) if per_task == 0 else start + per_task
        if start >= len(items) or stop > len(items):
            raise ValueError(f"task {task}: requested episodes {start}:{stop}, only {len(items)} available")
        selected.extend(items[index] for index in order[start:stop])
    return selected


def decode_video(path, timestamps, fps):
    """Decode once, selecting the closest preceding PTS like torchvision_av.

    Dataset videos already have the training orientation. Do not rotate them again.
    """
    import av

    images, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError(f"Missing video timestamp: {path}")
            images.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.time))
    times = np.asarray(times)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError(f"Empty or nonmonotonic video: {path}")
    if timestamps[0] < times[0] - 1 / fps or timestamps[-1] > times[-1] + 1 / fps:
        raise ValueError(f"Video does not cover episode timestamps: {path}")
    indices = np.maximum(np.searchsorted(times, timestamps, side="right") - 1, 0)
    return np.stack(images)[indices]


def load_episode(item, info, tasks):
    import pandas as pd

    root, index = Path(item["dataset_path"]), item["dataset_episode_index"]
    path = episode_path(root, info, index)
    table = pd.read_parquet(
        path, columns=["observation.state", "timestamp", "task_index", "episode_index", "frame_index"]
    )
    if len(table) != item["length"] or set(table["episode_index"]) != {index}:
        raise ValueError(f"Episode identity/length mismatch: {path}")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"Non-contiguous frame indices: {path}")
    if {tasks[int(task)] for task in table["task_index"]} != {item["language"]}:
        raise ValueError(f"Episode language mismatch: {path}")
    state = np.stack(table["observation.state"]).astype(np.float32)
    timestamps = table["timestamp"].to_numpy(dtype=float)
    if state.shape != (len(table), 8) or not np.isfinite(state).all():
        raise ValueError(f"Invalid LIBERO state: {path}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Invalid episode timestamps: {path}")
    videos, sources = {}, []
    for key, original in VIDEO_KEYS.items():
        video = episode_path(root, info, index, original).resolve(strict=True)
        videos[key] = decode_video(video, timestamps, info["fps"])
        sources.append(str(video))
    # Resolved video identities survive dataset aliases and cleaned episode reindexing.
    identity = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    return state, videos, timestamps, identity


def observation_at(state, videos, index, language):
    observation = {key: values[index : index + 1] for key, values in videos.items()}
    observation.update(
        {f"state.{key}": state[index : index + 1, start:end] for key, (start, end) in STATE_SLICES.items()}
    )
    observation["annotation.human.action.task_description"] = [language]
    return observation


def score_dataset(args):
    actor, scorer = args.actor.resolve(strict=True), args.scorer.resolve(strict=True)
    if actor == scorer:
        raise ValueError("actor and scorer must differ")
    if (
        args.episodes_per_task < 0
        or args.episode_start < 0
        or (args.max_frames is not None and args.max_frames < 1)
    ):
        raise ValueError("invalid episode selection/frame limit")
    for seed in (args.seed, args.selection_seed, args.uncertainty_seed):
        if not 0 <= seed < 2**32:
            raise ValueError("seeds must be in [0, 2**32)")
    if not 0 < args.quantile < 1 or args.min_successes < 2:
        raise ValueError("invalid calibration quantile/count")
    if not args.assume_successful_demos:
        raise ValueError("Pass --assume-successful-demos only for a dataset of successful demonstrations")
    times = [float(value) for value in args.flow_times.split(",")] if args.flow_times else None
    flow_grid(args.denoising_steps, times)
    if args.num_action_samples < 1:
        raise ValueError("num_action_samples must be positive")
    task_ids = list(range(10)) if args.task_ids == "all" else [int(value) for value in args.task_ids.split(",")]
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(task not in range(10) for task in task_ids):
        raise ValueError("task IDs must be distinct values from 0 through 9")
    suites = [SUITES.get(suite, suite) for suite in args.suites]
    if len(set(suites)) != len(suites) or any(suite not in SUITES.values() for suite in suites):
        raise ValueError("select distinct LIBERO suites")
    repo = Path(__file__).resolve().parents[1]
    task_map_path = (
        args.task_map
        or Path(os.environ.get("LIBERO_REPO", repo.parent / "LIBERO"))
        / "libero/libero/benchmark/libero_suite_task_map.py"
    )
    languages = task_languages(task_map_path)
    catalogs, selected = {}, []
    aliases = {suite: alias for alias, suite in SUITES.items()}
    for suite in suites:
        catalogs[suite] = catalog(args.dataset_root / aliases[suite], suite, languages)
        selected.extend(
            select_episodes(
                catalogs[suite][2], task_ids, args.episodes_per_task, args.episode_start, args.selection_seed
            )
        )
    if not args.score_only and args.max_frames is None:
        count = len(selected) if args.scope == "global" else args.episodes_per_task
        if args.scope == "task" and args.episodes_per_task == 0:
            count = min(
                sum(item["suite"] == suite and item["task_id"] == task for item in selected)
                for suite in suites
                for task in task_ids
            )
        if count < args.min_successes:
            raise ValueError(f"Only {count} demos per threshold; need {args.min_successes}, or use --score-only")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    settings = {**vars(args), "actor": actor, "scorer": scorer, "task_map": task_map_path}
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    plan = {
        "settings": settings,
        "selected_episodes": selected,
        "planned_frames": sum(min(item["length"], args.max_frames or item["length"]) for item in selected),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    for key, value in {
        "USE_TF": "0",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_MODULES_CACHE": str(repo / ".cache/huggingface/modules"),
    }.items():
        os.environ.setdefault(key, value)
    import torch
    from transformers import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; offline VFD scoring still needs a GPU")
    if args.gpu_memory_limit_gb:
        fraction = args.gpu_memory_limit_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory
        if not 0 < fraction <= 1:
            raise ValueError("invalid GPU allocator limit")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    output.mkdir(parents=True)
    manifest = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **plan,
        "data_kind": DATA_KIND,
        "smoke_only": args.max_frames is not None,
        "success_basis": "User asserts successful demonstrations; no policy outcomes measured",
        "official_initial_state_overlap": "unknown; dataset metadata has no official initial-state mapping",
        "preprocessing": "checkpoint eval transforms; dataset images already oriented; no demo actions used",
    }
    write_json(output / "manifest.json", manifest)
    completed, scored, started = [], 0, time.monotonic()
    try:
        monitor = SnapshotVFDPolicy(
            actor,
            scorer,
            args.data_config,
            args.denoising_steps,
            args.num_action_samples,
            times,
            args.uncertainty_seed,
        )
        set_seed(args.seed)
        manifest.update(score_signature=monitor.signature, status="running")
        write_json(output / "manifest.json", manifest)
        for item in selected:
            info, tasks, _ = catalogs[item["suite"]]
            state, videos, timestamps, identity = load_episode(item, info, tasks)
            prefix = (
                output
                / f"task_{item['suite']}_{item['task_id']:02d}"
                / f"episode{item['dataset_episode_index']:06d}"
            )
            prefix.parent.mkdir(exist_ok=True)
            metadata = {
                **item,
                "episode_kind": DATA_KIND,
                "dataset_episode_id": identity,
                "success": True,
                "complete": False,
                "initial_state_index": None,
                "success_basis": manifest["success_basis"],
            }
            scores, samples = [], []
            limit = min(len(state), args.max_frames or len(state))
            try:
                with prefix.with_suffix(".jsonl").open("w") as stream:
                    for index in range(limit):
                        observation = observation_at(state, videos, index, item["language"])
                        reference = None
                        if args.verify_actor and scored == 0:
                            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                                reference = monitor.actor.get_action(copy.deepcopy(observation))
                        chunk, score, sample_scores = monitor.get_action_with_vfd(observation)
                        if reference is not None:
                            error = max(float(np.max(np.abs(chunk[key] - reference[key]))) for key in reference)
                            for key in reference:
                                np.testing.assert_allclose(chunk[key], reference[key], rtol=1e-3, atol=1e-4)
                            manifest["actor_parity_max_abs"] = error
                            print(f"Actor parity PASS: max absolute action difference={error:.6g}", flush=True)
                        if not np.isfinite(score) or not np.isfinite(sample_scores).all():
                            raise FloatingPointError("non-finite demo VFD score")
                        scores.append(score)
                        samples.append(sample_scores)
                        stream.write(json.dumps({"frame_index": index, "uncertainty": score}) + "\n")
                        stream.flush()
                        scored += 1
                        if scored % 100 == 0:
                            elapsed = time.monotonic() - started
                            eta = elapsed / scored * (plan["planned_frames"] - scored)
                            print(
                                f"frames={scored}/{plan['planned_frames']} elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f}",
                                flush=True,
                            )
                metadata["complete"] = True
            finally:
                metadata["action_steps"] = len(scores)
                np.savez_compressed(
                    prefix.with_suffix(".npz"),
                    uncertainty=np.asarray(scores),
                    sample_scores=np.asarray(samples),
                    timestamps=timestamps[: len(scores)],
                )
                write_json(prefix.with_suffix(".json"), metadata)
            completed.append({**metadata, "peak_vfd": float(max(scores))})
            write_json(output / "summary.json", {"data_kind": DATA_KIND, "episodes": completed})
            print(
                f"demo={len(completed)}/{len(selected)} suite={item['suite']} task={item['task_id']} episode={item['dataset_episode_index']} frames={len(scores)} peak_vfd={max(scores):.6g}",
                flush=True,
            )
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest.update(
            completed_episodes=len(completed),
            elapsed_seconds=time.monotonic() - started,
            gpu_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
            gpu_peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
        )
        write_json(output / "manifest.json", manifest)
    if not args.score_only and not manifest["smoke_only"]:
        result = calibrate([output], output / "threshold.json", args.quantile, args.min_successes, args.scope)
        print("Thresholds:", json.dumps(result["thresholds"], indent=2), flush=True)
    else:
        print("Scores saved; no threshold fitted (score-only or shortened smoke run).", flush=True)
    return manifest
