"""Evaluate remote reactive/memory GR00T with held-out, all16 or primary protocols.

Run with python -m examples.RoboMME.eval. Planning and fake-environment tests
do not import RoboMME, Torch or a model. Simulator imports happen only at launch.
"""

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
from numbers import Integral
import sys
import tempfile
import time

import numpy as np

from examples.RoboMME.adapter import OBSERVATION_BUFFERS, RoboMMEGr00tClient
from examples.RoboMME.eval_protocol import PRIMARY_HELD_OUT, training_protocol, verify_memory_server
from examples.RoboMME.memory_client import RoboMMEMemoryClient


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _task_list(value, field):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{field} must be a nonempty list of task names")
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate task names in {field}")
    return sorted(value)


def build_plan(args):
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Expected a completed extraction manifest")
    extra = {}
    if args.protocol == "held-out":
        train_tasks = _task_list(manifest.get("selected_tasks"), "selected_tasks")
        held_out = _task_list(manifest.get("excluded_tasks"), "excluded_tasks")
    else:
        train_tasks, held_out, extra = training_protocol(args, manifest_path, manifest)
    if set(train_tasks) & set(held_out):
        raise ValueError("Training and held-out tasks overlap")
    group = args.task_group or ("held-out" if args.protocol == "held-out" else "all")
    if args.protocol == "held-out" and group != "held-out":
        raise ValueError("Legacy held-out protocol only permits tasks excluded from training")
    allowed = {"all": train_tasks + held_out, "seen": train_tasks, "held-out": held_out}[group]
    tasks = _task_list(args.tasks, "--tasks") if args.tasks is not None else sorted(allowed)
    if not tasks:
        raise ValueError(f"No tasks available in task group {group}")
    if set(tasks) - set(allowed):
        raise ValueError(
            "--tasks may only select tasks excluded from training"
            if group == "held-out"
            else f"--tasks must belong to the selected {group} task group"
        )
    episode_ids = args.episode_indices if args.episode_indices is not None else list(range(args.num_episodes))
    if (
        not episode_ids
        or any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError("Select unique nonnegative episode indices or a positive --num-episodes")
    if not 1 <= args.exec_horizon <= 16 or args.max_steps < 1 or args.video_fps < 1:
        raise ValueError("exec-horizon must be 1..16; max-steps and video-fps must be positive")
    if not 0 <= args.seed < 2**32 or not np.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("seed must be in [0, 2**32); timeout must be positive and finite")
    if not args.checkpoint.strip():
        raise ValueError("--checkpoint must identify the checkpoint served remotely")
    if args.policy_type == "memory":
        if not 0 <= args.server_seed < 2**32 or args.denoising_steps < 1:
            raise ValueError("server-seed must be in [0, 2**32); denoising-steps must be positive")
    if args.protocol != "held-out" or args.policy_type == "memory":
        extra.update(
            protocol=args.protocol,
            policy_type=args.policy_type,
            task_group=group,
            server_seed=args.server_seed,
            denoising_steps=args.denoising_steps,
            task_groups={
                "seen": sorted(set(tasks) & set(train_tasks)),
                "held_out": sorted(set(tasks) & set(held_out)),
                "primary_comparison": sorted(set(tasks) & PRIMARY_HELD_OUT),
            },
        )
    if args.config is not None:
        extra.update(
            config_file=str(args.config.resolve()),
            config_file_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        )
    root = args.benchmark_root.expanduser().resolve(strict=True)
    metadata_hashes = {}
    for task in tasks:
        path = root / "src/robomme/env_metadata/test" / f"record_dataset_{task}_metadata.json"
        metadata = read_json(path)
        available = {
            row["episode"] for row in metadata["records"] if row.get("task", metadata.get("env_id")) == task
        }
        if set(episode_ids) - available:
            raise ValueError(f"{task}: missing test episode IDs {sorted(set(episode_ids) - available)}")
        metadata_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        **extra,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": manifest.get("split"),
        "training_tasks": train_tasks,
        "held_out_tasks": held_out,
        "tasks": tasks,
        "episode_indices": sorted(episode_ids),
        "checkpoint": args.checkpoint,
        "benchmark_root": str(root),
        "test_metadata_sha256": metadata_hashes,
        "dataset": "test",
        "action_space": "joint_angle",
        "exec_horizon": args.exec_horizon,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "server_url": args.server_url,
        "timeout": args.timeout,
        "save_video": not args.no_video,
        "video_fps": args.video_fps,
    }


def open_video(path, fps):
    import imageio.v2 as imageio

    return imageio.get_writer(str(path), fps=fps)


def record_observations(writer, obs, *, initial=False):
    if writer is None:
        return
    front, wrist = obs["front_rgb_list"], obs["wrist_rgb_list"]
    if not len(front) or len(front) != len(wrist):
        raise ValueError("Video camera buffers must have equal nonzero lengths")
    for index, (image, wrist_image) in enumerate(zip(front, wrist)):
        frame = np.concatenate([image, wrist_image], axis=1)
        if initial and index < len(front) - 1:
            for edge in (frame[:4], frame[-4:], frame[:, :4], frame[:, -4:]):
                edge[:] = [255, 0, 0]
        writer.append_data(frame)


def run_episode(
    client,
    builder,
    episode_id,
    *,
    exec_horizon,
    max_steps,
    video_path=None,
    video_fps=10,
    video_writer_factory=open_video,
):
    """Run one episode with a fresh queue, a strict execution budget and cleanup."""
    result = {
        "episode_index": episode_id,
        "outcome": "error",
        "steps": 0,
        "task_goal": None,
        "video_path": None,
        "error": None,
        "input_frame_counts": [],
    }
    env = writer = None
    start = time.monotonic()
    try:
        if not client.reset().get("reset_finished", False):
            raise RuntimeError("Policy reset did not finish")
        env = builder.make_env_for_episode(episode_id)
        obs, info = env.reset()
        goal = info["task_goal"]
        result["task_goal"] = goal[0] if isinstance(goal, (list, tuple)) else goal
        inputs = {"task_goal": goal, "is_first_step": True, **{key: list(obs[key]) for key in OBSERVATION_BUFFERS}}
        if video_path is not None:
            writer = video_writer_factory(video_path, video_fps)
            result["video_path"] = str(video_path)
        record_observations(writer, obs, initial=True)
        actions = deque()
        while result["steps"] < max_steps:
            if not actions:
                result["input_frame_counts"].append(len(inputs["front_rgb_list"]))
                chunk = np.asarray(client.infer(inputs)["actions"], dtype=np.float32)
                if (
                    chunk.ndim != 2
                    or chunk.shape[1] != 8
                    or len(chunk) < exec_horizon
                    or not np.isfinite(chunk).all()
                ):
                    raise ValueError("Expected finite (H, 8) actions with H >= exec_horizon")
                actions.extend(chunk[:exec_horizon])
                inputs.update({key: [] for key in OBSERVATION_BUFFERS})
                inputs["is_first_step"] = False
            obs, _, terminated, truncated, info = env.step(actions.popleft())
            result["steps"] += 1
            if info.get("status") == "error":
                raise RuntimeError(info.get("error_message", "Benchmark step reported an error"))
            record_observations(writer, obs)
            for key in OBSERVATION_BUFFERS:
                inputs[key].extend(obs[key])
            if bool(terminated) or bool(truncated):
                outcome = info.get("status")
                if outcome not in ("success", "fail", "timeout"):
                    if bool(truncated):
                        outcome = "timeout"
                    else:
                        raise ValueError(f"Unknown terminal status: {outcome}")
                result["outcome"] = outcome
                break
        else:
            result["outcome"] = "timeout"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for name, resource in (("video", writer), ("environment", env)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    result.setdefault("cleanup_errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
        result["elapsed_seconds"] = time.monotonic() - start
    return result


def summarize(config, episodes):
    def counts(rows, expected):
        outcomes = Counter(row["outcome"] for row in rows)
        return {
            "expected_episodes": expected,
            "completed_episodes": len(rows),
            **{key: outcomes[key] for key in ("success", "fail", "timeout", "error")},
            "cleanup_errors": sum(bool(row.get("cleanup_errors")) for row in rows),
            "success_rate": outcomes["success"] / len(rows) if rows else None,
        }

    total = len(config["tasks"]) * len(config["episode_indices"])
    result = {
        "complete": len(episodes) == total,
        "overall": counts(episodes, total),
        "per_task": {
            task: counts([row for row in episodes if row["task"] == task], len(config["episode_indices"]))
            for task in config["tasks"]
        },
    }
    if "task_groups" in config:
        result["by_task_group"] = {
            name: {
                "tasks": tasks,
                **counts(
                    [row for row in episodes if row["task"] in tasks], len(tasks) * len(config["episode_indices"])
                ),
            }
            for name, tasks in config["task_groups"].items()
        }
    return result


def evaluate(config, output, client, builder_factory, *, resume=False, video_writer_factory=open_video):
    """Persist each result; resume only runs with identical declared configuration."""
    output = Path(output)
    config_hash = fingerprint(config)
    if resume:
        if read_json(output / "config.json") != config:
            raise ValueError("Resume configuration differs from the saved run")
        progress = read_json(output / "progress.json")
        if progress["config_sha256"] != config_hash:
            raise ValueError("Progress configuration fingerprint does not match")
        episodes = progress["episodes"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        write_json(output / "config.json", config)
    expected = {(task, index) for task in config["tasks"] for index in config["episode_indices"]}
    done = {(row["task"], row["episode_index"]) for row in episodes}
    if (
        len(done) != len(episodes)
        or not done <= expected
        or any(row["outcome"] not in ("success", "fail", "timeout", "error") for row in episodes)
    ):
        raise ValueError("Invalid or duplicate episode results in progress file")
    if config["save_video"]:
        (output / "videos").mkdir(exist_ok=True)

    def save():
        write_json(output / "progress.json", {"config_sha256": config_hash, "episodes": episodes})
        metrics = summarize(config, episodes)
        write_json(output / "metrics.json", metrics)
        return metrics

    save()
    for task in config["tasks"]:
        remaining = [index for index in config["episode_indices"] if (task, index) not in done]
        if not remaining:
            continue
        builder = builder_factory(
            env_id=task,
            dataset="test",
            action_space="joint_angle",
            max_steps=config["max_steps"],
            gui_render=False,
        )
        for index in remaining:
            video_path = output / "videos" / f"{task}_ep{index:04d}.mp4" if config["save_video"] else None
            result = run_episode(
                client,
                builder,
                index,
                exec_horizon=config["exec_horizon"],
                max_steps=config["max_steps"],
                video_path=video_path,
                video_fps=config["video_fps"],
                video_writer_factory=video_writer_factory,
            )
            result["task"] = task
            if "task_groups" in config:
                result["task_group"] = "seen" if task in config["training_tasks"] else "held_out"
            episodes.append(result)
            save()
            print(f"{task} episode {index}: {result['outcome']} ({result['steps']} steps)", flush=True)
    return summarize(config, episodes)


def load_benchmark(root, seed):
    # Select this clone explicitly instead of a different editable installation.
    sys.path.insert(0, str(Path(root) / "src"))
    import robomme

    if not Path(robomme.__file__).resolve().is_relative_to(Path(root)):
        raise RuntimeError(f"Imported RoboMME from a different checkout: {robomme.__file__}")
    from robomme.software_render import enable_if_needed

    enable_if_needed()
    import robomme.robomme_env  # noqa: F401 - register environments
    import torch

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return BenchmarkEnvBuilder


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    config_path = preliminary.parse_known_args(argv)[0].config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Evaluation YAML; CLI flags override its values")
    parser.add_argument("--manifest", type=Path, help="Training dataset's conversion/extraction manifest")
    parser.add_argument("--checkpoint", help="Identity/path of the checkpoint served remotely")
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", choices=("held-out", "all16", "primary"), default="held-out")
    parser.add_argument("--policy-type", choices=("reactive", "memory"), default="reactive")
    parser.add_argument("--task-group", choices=("all", "seen", "held-out"))
    parser.add_argument("--task-index", type=Path, help="Training episode-to-task index for all16/primary")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tasks", nargs="+", help="Optional subset of the selected task group")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--num-episodes", type=int, default=50)
    selection.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--exec-horizon", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-seed", type=int, help="Expected memory server startup seed (default: --seed)")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--video-fps", type=int, default=10)
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--no-video", action="store_true")
    video.add_argument("--video", dest="no_video", action="store_false")
    parser.set_defaults(no_video=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    defaults = {}
    if config_path is not None:
        import yaml

        defaults = yaml.safe_load(config_path.read_text())
        fields = {
            action.dest: action
            for action in parser._actions
            if action.dest not in ("help", "config", "resume", "dry_run")
        }
        if not isinstance(defaults, dict) or set(defaults) - set(fields):
            parser.error("Evaluation YAML contains unknown fields or is not a mapping")
        if defaults.get("episode_indices") is not None and defaults.get("num_episodes") is not None:
            parser.error("YAML must choose episode_indices or num_episodes")
        for key, value in defaults.items():
            action = fields[key]
            if value is None:
                if key not in {
                    "task_index",
                    "task_group",
                    "tasks",
                    "episode_indices",
                    "server_seed",
                    "checkpoint",
                    "manifest",
                    "benchmark_root",
                    "output_dir",
                }:
                    parser.error(f"{key} must not be null")
                continue
            if action.type is Path:
                if not isinstance(value, str):
                    parser.error(f"{key} must be a path string")
                defaults[key] = Path(value).expanduser()
            elif action.nargs == "+":
                if not isinstance(value, list) or any(
                    isinstance(item, bool) or not isinstance(item, int if action.type is int else str)
                    for item in value
                ):
                    parser.error(f"{key} must be a list of {action.type or str}")
            elif action.type in (int, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (action.type is int and not isinstance(value, int))
                ):
                    parser.error(f"{key} has the wrong numeric type")
            elif key == "no_video":
                if not isinstance(value, bool):
                    parser.error("no_video must be boolean")
            elif not isinstance(value, str):
                parser.error(f"{key} must be a string")
            if action.choices is not None and value not in action.choices:
                parser.error(f"Invalid {key}: {value}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    flags = {item.split("=", 1)[0] for item in argv}
    if "--num-episodes" in flags:
        args.episode_indices = None
    if args.server_seed is None:
        args.server_seed = args.seed
    for key in ("manifest", "checkpoint", "benchmark_root", "output_dir"):
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required via CLI or YAML")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = build_plan(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    client_type = RoboMMEGr00tClient
    if args.policy_type == "memory":
        config["server_provenance"] = verify_memory_server(config)
        client_type = RoboMMEMemoryClient
    builder = load_benchmark(config["benchmark_root"], config["seed"])
    with client_type(config["server_url"], timeout=config["timeout"]) as client:
        metrics = evaluate(config, args.output_dir, client, builder, resume=args.resume)
    print(json.dumps(metrics, indent=2))
    return int(metrics["overall"]["error"] > 0 or metrics["overall"]["cleanup_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
