"""Export offline LIBERO positive candidates using the original RA-VLA sampler."""

import argparse
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import time

import numpy as np

from scripts.rag.precompute_retrieval import REPO, atomic_json, digest, run_lock, stamp


ID_SUITES = ("libero_goal", "libero_object", "libero_spatial")
DEFAULT_DATA = REPO / "ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets"


class ParquetDataset:
    """Small adapter for get_aligned_frame_ids, avoiding model/video transforms."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.tasks = {row["task_index"]: row["task"] for row in read_jsonl(self.root / "meta/tasks.jsonl")}
        self.episodes = {row["episode_index"]: row for row in read_jsonl(self.root / "meta/episodes.jsonl")}
        self.trajectory_ids = {episode: episode for episode in self.episodes}

    def path(self, episode, kind="data_path", **kwargs):
        return self.root / self.info[kind].format(
            episode_chunk=episode // self.info.get("chunks_size", 1000), episode_index=episode, **kwargs
        )

    @lru_cache(maxsize=8)
    def get_trajectory_data(self, episode):
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.path(episode), columns=["action", "observation.state", "task_index", "frame_index"]
        )
        if table["frame_index"].to_pylist() != list(range(len(table))):
            raise ValueError(f"Non-contiguous frame IDs: {self.path(episode)}")
        if len(table) != self.episodes[episode]["length"]:
            raise ValueError(f"Episode length disagrees with metadata: {self.path(episode)}")
        tasks = {self.tasks[index] for index in table["task_index"].to_pylist()}
        if tasks != set(self.episodes[episode]["tasks"]) or len(tasks) != 1:
            raise ValueError(f"Episode task disagrees with metadata: {self.path(episode)}")
        values = table.to_pandas()
        actions = np.stack(values["action"])
        if actions.shape != (len(table), 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite 7D LIBERO actions: {self.path(episode)}")
        return values

    def source(self, episode):
        videos = {}
        for camera, key in (("External", "observation.images.image"), ("Wrist", "observation.images.wrist_image")):
            path = self.path(episode, "video_path", video_key=key)
            if path.is_file():
                videos[camera] = stamp(path)
        return {
            "episode": episode,
            "length": self.episodes[episode]["length"],
            "fps": self.info["fps"],
            "parquet": stamp(self.path(episode)),
            "videos": videos,
        }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def record_seed(seed, identity):
    return int(digest([seed, identity])[:16], 16)


def build_plan(args):
    datasets, plan, skipped = {}, [], []
    for suite in args.suites:
        dataset = datasets[suite] = ParquetDataset(args.dataset_root / suite)
        task_ids = sorted(dataset.tasks) if args.task_ids is None else sorted(set(args.task_ids))
        if any(task not in dataset.tasks for task in task_ids):
            raise ValueError(f"Unknown task IDs for {suite}: {task_ids}")
        if args.tasks_per_suite:
            rng = random.Random(record_seed(args.seed, suite))
            task_ids = sorted(rng.sample(task_ids, min(args.tasks_per_suite, len(task_ids))))
        for task in task_ids:
            ids = sorted(
                episode
                for episode, meta in dataset.episodes.items()
                if meta["tasks"] == [dataset.tasks[task]] and meta["length"] >= args.num_frames
            )
            pairs = list(combinations(ids, 2))
            rng = random.Random(record_seed(args.seed, f"{suite}:{task}"))
            selected = rng.sample(pairs, min(args.pairs_per_task, len(pairs)))
            if not selected:
                skipped.append(
                    {
                        "suite": suite,
                        "task_id": task,
                        "reason": "Fewer than two demonstrations long enough for num_frames",
                    }
                )
            for a, b in sorted(selected):
                identity = f"{suite}:task{task}:ep{a}:ep{b}"
                plan.append(
                    {
                        "id": identity,
                        "suite": suite,
                        "task_id": task,
                        "task": dataset.tasks[task],
                        "episode_a": a,
                        "episode_b": b,
                        "seed": record_seed(args.seed, identity),
                    }
                )
    if not plan:
        raise ValueError("No demonstration pairs selected")
    return datasets, plan, skipped


def mine_pair(dataset, item, count):
    # This is the only mining call. Keep the authors' DTW/filter/sampling intact.
    from gr00t.rag.utils import get_aligned_frame_ids

    rng_state = random.getstate()
    try:
        random.seed(item["seed"])
        frames_a, frames_b = get_aligned_frame_ids(dataset, item["episode_a"], item["episode_b"], count)
    finally:
        random.setstate(rng_state)
    source_a, source_b = dataset.source(item["episode_a"]), dataset.source(item["episode_b"])
    progress = np.rint(np.asarray(frames_a) / max(source_a["length"] - 1, 1) * (source_b["length"] - 1)).astype(
        int
    )
    return {
        **item,
        "dataset": str(dataset.root),
        "source_a": source_a,
        "source_b": source_b,
        "frame_pairs": [[int(a), int(b)] for a, b in zip(frames_a, frames_b)],
        "progress_frame_pairs": [[int(a), int(b)] for a, b in zip(frames_a, progress)],
        "label_status": "action-DTW positive candidates; not human-verified",
    }


def validate_record(record, item, count):
    if any(record[key] != value for key, value in item.items()):
        raise ValueError("Saved record disagrees with the selected plan")
    pairs = np.asarray(record["frame_pairs"])
    if pairs.shape != (count, 2) or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Invalid saved frame pairs")
    if np.any(pairs < 0) or np.any(np.diff(pairs[:, 0]) <= 0) or np.any(np.diff(pairs[:, 1]) < 0):
        raise ValueError("Invalid pair ordering")
    for column, side in enumerate(("source_a", "source_b")):
        if np.any(pairs[:, column] >= record[side]["length"]):
            raise ValueError("Saved frame ID outside demonstration")


def read_records(path, repair=False):
    """Ignore an unfinished final line; optionally truncate it before resuming."""
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    complete = raw[: raw.rfind(b"\n") + 1] if raw and not raw.endswith(b"\n") else raw
    if repair and complete != raw:
        with Path(path).open("r+b") as file:
            file.truncate(len(complete))
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def export(args):
    datasets, plan, skipped = build_plan(args)
    for suite in args.suites:
        selected = [item for item in plan if item["suite"] == suite]
        print(
            f"{suite}: {len(set(item['task_id'] for item in selected))} tasks, {len(selected)} demo pairs, {len(selected) * args.num_frames} frame pairs",
            flush=True,
        )
    if args.dry_run:
        return
    sources = sorted(
        {str(datasets[item["suite"]].path(item[side])) for item in plan for side in ("episode_a", "episode_b")}
    )
    signature = {
        "schema_version": 1,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "plan": plan,
        "source_parquets": [stamp(path) for path in sources],
        "dataset_metadata": [
            stamp(dataset.root / f"meta/{name}")
            for dataset in datasets.values()
            for name in ("info.json", "tasks.jsonl", "episodes.jsonl")
        ],
        "sampler_sha256": hashlib.sha256((REPO / "gr00t/rag/utils.py").read_bytes()).hexdigest(),
    }
    fingerprint = digest(signature)
    with run_lock(args.output_dir):
        meta_path, pairs_path = args.output_dir / "metadata.json", args.output_dir / "pairs.jsonl"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Inputs/settings changed; use a different --output-dir")
        elif pairs_path.exists() and pairs_path.stat().st_size:
            raise ValueError("pairs.jsonl exists without metadata; use a different --output-dir")
        existing = read_records(pairs_path, repair=True)
        if len(existing) > len(plan):
            raise ValueError("More saved records than planned")
        for record, item in zip(existing, plan):
            validate_record(record, item, args.num_frames)
        metadata = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "status": "running",
            "held_out_suite": "libero_10",
            "algorithm": "gr00t.rag.utils.get_aligned_frame_ids (unmodified)",
            "signature": signature,
            "planned_demo_pairs": len(plan),
            "completed_demo_pairs": len(existing),
            "num_frames_per_pair": args.num_frames,
            "skipped_tasks": skipped,
            "note": "G/O/S demonstrations only. No retriever training or Long evaluation. Samples use raw action DTW, not visual similarity.",
        }
        atomic_json(meta_path, metadata)
        started = time.perf_counter()
        try:
            with pairs_path.open("a") as file:
                for item in plan[len(existing) :]:
                    record = mine_pair(datasets[item["suite"]], item, args.num_frames)
                    validate_record(record, item, args.num_frames)
                    file.write(json.dumps(record) + "\n")
                    file.flush()
                    metadata["completed_demo_pairs"] += 1
                    atomic_json(meta_path, metadata)
                    print(f"[{metadata['completed_demo_pairs']}/{len(plan)}] {item['id']}", flush=True)
            metadata["status"] = "complete"
        except BaseException as error:
            metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            metadata["error"] = str(error)
            raise
        finally:
            metadata["last_invocation_seconds"] = time.perf_counter() - started
            metadata["frame_pair_count"] = metadata["completed_demo_pairs"] * args.num_frames
            atomic_json(meta_path, metadata)
        print(
            f"Saved {metadata['completed_demo_pairs']} demo pairs / {metadata['frame_pair_count']} frame pairs to {args.output_dir}; skipped {len(existing)} cached demo pairs",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--suites", choices=ID_SUITES, nargs="+", default=list(ID_SUITES))
    parser.add_argument("--task-ids", type=int, nargs="+", help="Zero-based IDs, applied to each selected suite")
    parser.add_argument(
        "--tasks-per-suite", type=int, default=0, help="0=all tasks; positive value for a small test"
    )
    parser.add_argument(
        "--pairs-per-task", type=int, default=10, help="Unique demonstration pairs, capped by availability"
    )
    parser.add_argument("--num-frames", type=int, default=20, help="Positive frame pairs per demonstration pair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/dtw_pairs/gos_offline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(args.pairs_per_task, args.num_frames) < 1
        or args.tasks_per_suite < 0
        or any(task < 0 for task in args.task_ids or [])
    ):
        parser.error("Counts must be positive; task IDs/task limit must be nonnegative")
    if len(set(args.suites)) != len(args.suites):
        parser.error("Duplicate suites")
    try:
        export(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
