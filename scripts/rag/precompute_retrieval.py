"""Recompute RA-VLA retrieval from saved two-camera LIBERO MP4s.

These are diagnostics from decoded video, not the retrieval trace of the original
evaluation. MP4 compression and batched inference can change near-tied rankings.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
CACHE_VERSION = 1
DEFAULT_RUNS = [f"{subset}_20trials" for subset in ("spatial", "object", "goal", "long")]


@dataclass
class RunPlan:
    source: Path
    output: Path
    checkpoint: Path
    retriever: Path
    memory: Path
    suite: str
    horizon: int
    stride: int
    warmup: int
    policy_k: int
    available_episodes: int
    episodes: list
    signature: dict


def option(command, name, default=None):
    """Read the last occurrence of a flag from a saved argv list."""
    value = default
    for i, arg in enumerate(command):
        if arg == name:
            value = command[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_stamp(directory):
    paths = sorted(
        {*directory.glob("*.json"), *directory.glob("*.safetensors"), *directory.glob("pytorch_model*.bin")}
    )
    metadata = directory / "metadata.pt"
    if metadata.exists():
        paths.append(metadata)
    return [stamp(path) for path in paths]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending_", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2)
            file.write("\n")
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".precompute.lock").open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another precompute process is using {directory}") from None
        yield


def plan_run(name, args):
    source = Path(name)
    if not source.is_absolute():
        source = REPO / source if (REPO / source).is_dir() else REPO / args.input_root / source
    source = source.resolve()
    summary = json.loads((source / "rollouts/summary.json").read_text())
    commands = json.loads((source / "commands.json").read_text())
    cwd = Path(commands.get("cwd", REPO))

    def saved_path(value):
        if value is None:
            raise ValueError(f"Missing model path in {source / 'commands.json'}")
        return (cwd / value).resolve()

    checkpoint = saved_path(option(commands["server"], "--checkpoint-path"))
    retriever = saved_path(option(commands["server"], "--retriever-path"))
    memory_value = option(commands["server"], "--memory-path")
    memory = (
        saved_path(memory_value)
        if memory_value
        else retriever / (option(commands["server"], "--memory-name", "memory") + ".pt")
    )
    for path in (checkpoint / "config.json", checkpoint / "metadata.pt", retriever / "config.json", memory):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads((checkpoint / "config.json").read_text())
    horizon = int(config["action_head_cfg"]["action_horizon"])
    stride = args.frame_stride or horizon
    episodes = []
    identities = set()
    all_episodes = summary["episodes"]
    if len(all_episodes) != summary["num_episodes"]:
        raise ValueError(f"Episode count disagrees with summary: {source}")
    for ordinal, episode in enumerate(all_episodes, start=1):
        identity = (episode["task_id"], episode["episode_index"])
        if identity in identities:
            raise ValueError(f"Duplicate episode {identity} in {source}")
        identities.add(identity)
        if args.task_ids is not None and episode["task_id"] not in args.task_ids:
            continue
        if args.episode_indices is not None and episode["episode_index"] not in args.episode_indices:
            continue
        if episode["control_steps"] <= 0:
            raise ValueError(f"Episode {identity} has no control steps")
        videos = list((source / "rollouts").glob(f"episode={ordinal}--success={bool(episode['success'])}--*.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected one video for episode {ordinal} in {source}; found {len(videos)}")
        episodes.append({**episode, "ordinal": ordinal, "video": stamp(videos[0])})
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise ValueError(f"No episodes selected in {source}")

    preprocessing_files = [
        "examples/Libero/custom_data_config.py",
        "gr00t/model/transforms.py",
        "gr00t/data/transform/video.py",
        "gr00t/data/transform/base.py",
        "gr00t/data/transform/concat.py",
        "gr00t/rag/retriever.py",
    ]
    preprocessing = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in preprocessing_files}
    for path in sorted((REPO / "gr00t/model/backbone/eagle2_hg_model").glob("*.json")):
        preprocessing[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = {
        "cache_version": CACHE_VERSION,
        "checkpoint": model_stamp(checkpoint),
        "retriever": model_stamp(retriever),
        "memory": stamp(memory),
        "preprocessing": preprocessing,
        "stride": stride,
        "top_k": args.top_k,
        "policy_horizon": horizon,
        "query_source": "decoded_mp4_recomputed",
    }
    return RunPlan(
        source,
        (REPO / args.output_dir / source.name).resolve(),
        checkpoint,
        retriever,
        memory,
        summary["task_suite_name"],
        horizon,
        stride,
        int(option(commands["eval"], "--num-steps-wait", "10")),
        int(option(commands["server"], "--k", "1")),
        len(all_episodes),
        episodes,
        signature,
    )


def cache_path(plan, episode):
    return plan.output / f"episode_{episode['ordinal']:06d}.npz"


def episode_signature(plan, episode):
    return digest({"run": plan.signature, "episode": episode})


def validate_cache(path, fingerprint, episode, stride):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("source files or computation settings changed")
            frames = np.arange(0, episode["control_steps"], stride)
            if not np.array_equal(data["frame_indices"], frames):
                raise ValueError("query frame indices disagree with the episode")
            scores, indices = data["topk_scores"], data["topk_indices"]
            if scores.shape != indices.shape or scores.shape != (len(frames), metadata["effective_top_k"]):
                raise ValueError("invalid top-K array shapes")
            if data["queries"].shape != (len(frames), metadata["embedding_dim"]):
                raise ValueError("invalid query embedding shape")
            if not np.isfinite(scores).all() or not np.isfinite(data["queries"]).all():
                raise ValueError("cache contains non-finite values")
            if indices.min() < 0 or indices.max() >= metadata["memory_candidates"]:
                raise ValueError("retrieval indices out of range")
        return metadata
    except Exception as error:
        raise ValueError(f"Cannot reuse {path}: {error}. Use --overwrite to recompute.") from error


class Encoder:
    def __init__(self, plan, device):
        import torch
        from examples.Libero.custom_data_config import LiberoDataConfig
        from gr00t.rag.ravla import RAVLA, RAVLADataCollator
        from gr00t.rag.retriever import RetrieverWrapperWithHead

        self.torch = torch
        self.device = device
        self.transforms = LiberoDataConfig().transform()
        metadata = torch.load(plan.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
        self.transforms.set_metadata(metadata["new_embodiment"])
        self.transforms.eval()
        self.collator = RAVLADataCollator()
        self.vla = RAVLA.from_pretrained(
            str(plan.checkpoint), torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self.retriever = RetrieverWrapperWithHead.from_pretrained(
            str(plan.retriever), torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        # Only retrieval keys are moved to GPU, not images/actions/backbone features.
        self.memory = torch.load(plan.memory, map_location="cpu", mmap=True, weights_only=True)

    def encode(self, frames, task, keys, top_k):
        import numpy as np

        features = []
        for frame in frames:
            middle = frame.shape[1] // 2
            # The MP4 already stores get_libero_image()'s rotated RGB views.
            observation = {
                "video.image": np.ascontiguousarray(frame[:, :middle])[None],
                "video.wrist_image": np.ascontiguousarray(frame[:, middle:])[None],
                "annotation.human.action.task_description": [task],
            }
            # State is intentionally absent: retrieval uses only image/text tokens.
            features.append(self.transforms(observation))
        inputs = self.collator(features)
        if any(value != task for value in inputs.pop("tasks")):
            raise ValueError("Collator task text disagrees with the memory task")
        with self.torch.inference_mode():
            queries = self.retriever(self.vla, inputs)
            # Preserve the original retriever's matmul dtype and ranking rule.
            scores, indices = (queries @ keys.T).topk(top_k)
        return queries.float().cpu().numpy(), scores.float().cpu().numpy(), indices.cpu().numpy()


def compute_episode(plan, episode, encoder, args):
    import av
    import numpy as np

    task = episode["task_description"]
    if task not in encoder.memory:
        raise KeyError(f"Task not found in memory: {task!r}")
    keys = encoder.memory[task]["keys"].to(encoder.device)
    top_k = min(args.top_k, len(keys))
    expected_frames = episode["control_steps"]
    frame_indices, pending, queries, scores, indices = [], [], [], [], []
    inference_seconds = 0.0

    def flush():
        nonlocal inference_seconds
        if pending:
            started = time.perf_counter()
            q, s, i = encoder.encode(pending, task, keys, top_k)
            inference_seconds += time.perf_counter() - started
            queries.append(q)
            scores.append(s)
            indices.append(i)
            pending.clear()

    started = time.perf_counter()
    decoded = 0
    with av.open(episode["video"]["path"]) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = args.decode_threads
        fps = float(stream.average_rate) if stream.average_rate else None
        if stream.width % 2 != 0:
            raise ValueError("Expected an even-width, side-by-side two-camera MP4")
        for index, frame in enumerate(video.decode(stream)):
            decoded += 1
            if index % plan.stride == 0:
                pending.append(frame.to_ndarray(format="rgb24"))
                frame_indices.append(index)
                if len(pending) == args.batch_size:
                    flush()
        flush()
    if decoded != expected_frames:
        raise ValueError(
            f"Video has {decoded} frames but summary says {expected_frames}: {episode['video']['path']}"
        )
    elapsed = time.perf_counter() - started
    query_array = np.concatenate(queries)
    metadata = {
        "cache_version": CACHE_VERSION,
        "query_source": "decoded_mp4_recomputed",
        "is_original_retrieval_trace": False,
        "fingerprint": episode_signature(plan, episode),
        "run": plan.source.name,
        "suite": plan.suite,
        "episode": episode,
        "checkpoint": str(plan.checkpoint),
        "retriever": str(plan.retriever),
        "memory": str(plan.memory),
        "policy_k": plan.policy_k,
        "policy_action_horizon": plan.horizon,
        "frame_stride": plan.stride,
        "simulator_warmup_steps": plan.warmup,
        "requested_top_k": args.top_k,
        "effective_top_k": top_k,
        "memory_candidates": len(keys),
        "embedding_dim": query_array.shape[1],
        "score_torch_dtype": str(keys.dtype),
        "batch_size": args.batch_size,
        "decoded_frames": decoded,
        "video_fps": fps,
        "num_queries": len(frame_indices),
        "processing_seconds": elapsed,
        "preprocess_and_inference_seconds": inference_seconds,
    }
    destination = cache_path(plan, episode)
    with tempfile.NamedTemporaryFile(dir=plan.output, prefix=".pending_", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez_compressed(
                file,
                metadata_json=np.asarray(json.dumps(metadata)),
                frame_indices=np.asarray(frame_indices, dtype=np.int64),
                simulator_steps=np.asarray(frame_indices, dtype=np.int64) + plan.warmup,
                queries=query_array,
                topk_scores=np.concatenate(scores),
                topk_indices=np.concatenate(indices),
            )
            file.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return metadata


def process_run(plan, args):
    import torch

    started = time.perf_counter()
    manifest = {
        "cache_version": CACHE_VERSION,
        "query_source": "decoded_mp4_recomputed",
        "run": str(plan.source),
        "suite": plan.suite,
        "signature": plan.signature,
        "available_episodes": plan.available_episodes,
        "selected_episodes": len(plan.episodes),
        "status": "running",
        "computed": 0,
        "skipped": 0,
        "episodes": [],
        "model_load_seconds": 0.0,
        "processing_seconds": 0.0,
        "new_queries": 0,
    }
    encoder = None
    try:
        # Validate every existing result before allocating GPU memory.
        existing = {}
        if not args.overwrite:
            for episode in plan.episodes:
                path = cache_path(plan, episode)
                if path.exists():
                    existing[episode["ordinal"]] = validate_cache(
                        path, episode_signature(plan, episode), episode, plan.stride
                    )
        for number, episode in enumerate(plan.episodes, start=1):
            metadata = existing.get(episode["ordinal"])
            if metadata is not None:
                manifest["skipped"] += 1
                action = "skip"
            else:
                if encoder is None:
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "CUDA unavailable. Run this command on your GPU allocation, not the login node."
                        )
                    loading = time.perf_counter()
                    torch.manual_seed(0)
                    encoder = Encoder(plan, f"cuda:{args.cuda}")
                    manifest["model_load_seconds"] = time.perf_counter() - loading
                metadata = compute_episode(plan, episode, encoder, args)
                manifest["computed"] += 1
                manifest["new_queries"] += metadata["num_queries"]
                manifest["processing_seconds"] += metadata["processing_seconds"]
                action = "done"
            manifest["episodes"].append(
                {
                    "ordinal": episode["ordinal"],
                    "task_id": episode["task_id"],
                    "cache": cache_path(plan, episode).name,
                    "queries": metadata["num_queries"],
                }
            )
            atomic_json(plan.output / "manifest.json", manifest)
            print(
                f"[{plan.source.name} {number}/{len(plan.episodes)}] {action} "
                f"episode {episode['ordinal']}: {metadata['num_queries']} queries",
                flush=True,
            )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        atomic_json(plan.output / "manifest.json", manifest)
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run names or directories")
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ravla_eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_cache"))
    parser.add_argument("--top-k", type=int, default=20, help="Diagnostic candidates; does not change policy K")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, help="Default: original policy's action horizon")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--episode-indices", type=int, nargs="+", help="Zero-based trial indices within each task")
    parser.add_argument("--max-episodes", type=int, help="Limit episodes PER RUN for a small benchmark")
    parser.add_argument("--overwrite", action="store_true", help="Recompute instead of resuming valid caches")
    parser.add_argument(
        "--dry-run", action="store_true", help="Check files/counts without loading models or writing output"
    )
    args = parser.parse_args()
    for name in ("top_k", "batch_size", "decode_threads", "frame_stride", "max_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.cuda < 0 or any(index < 0 for index in (args.task_ids or []) + (args.episode_indices or [])):
        parser.error("GPU, task, and episode indices must be nonnegative")
    try:
        plans = [plan_run(name, args) for name in args.runs]
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if len({plan.output for plan in plans}) != len(plans):
        parser.error("Run names must be unique to avoid output collisions")
    total = 0
    for plan in plans:
        count = sum(math.ceil(episode["control_steps"] / plan.stride) for episode in plan.episodes)
        total += count
        print(
            f"{plan.source.name}: {len(plan.episodes)} episodes, {count} queries, stride={plan.stride}\n"
            f"  memory={plan.memory}\n  output={plan.output}",
            flush=True,
        )
    print(f"TOTAL: {sum(len(p.episodes) for p in plans)} episodes, {total} queries", flush=True)
    if args.dry_run:
        return
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("USE_TF", "0")
    results = []
    for plan in plans:
        with run_lock(plan.output):
            results.append(process_run(plan, args))
    seconds = sum(result["processing_seconds"] for result in results)
    queries = sum(result["new_queries"] for result in results)
    print(
        json.dumps(
            {
                "computed_episodes": sum(r["computed"] for r in results),
                "skipped_episodes": sum(r["skipped"] for r in results),
                "new_queries": queries,
                "processing_seconds": seconds,
                "queries_per_second": queries / seconds if seconds else None,
                "model_load_seconds": sum(r["model_load_seconds"] for r in results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
