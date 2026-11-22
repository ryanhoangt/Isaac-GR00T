"""Export offline LIBERO/RoboMME candidates using the online RA-VLA DTW sampler."""

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

    def __init__(self, root, dataset_kind="libero"):
        self.root = Path(root).resolve()
        self.dataset_path = self.root
        self.dataset_kind = dataset_kind
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.tasks = {row["task_index"]: row["task"] for row in read_jsonl(self.root / "meta/tasks.jsonl")}
        self.episodes = {row["episode_index"]: row for row in read_jsonl(self.root / "meta/episodes.jsonl")}
        self.trajectory_ids = {episode: episode for episode in self.episodes}
        self.trajectory_lengths = [row["length"] for row in self.episodes.values()]
        self.task_groups = {}
        for episode, row in self.episodes.items():
            if len(row["tasks"]) != 1 or row["tasks"][0] not in self.tasks.values():
                raise ValueError(f"Expected one known instruction per episode: {episode}")
            self.task_groups.setdefault(row["tasks"][0], []).append(episode)
        self.action_column, self.state_column, self.action_dimension = "action", "observation.state", 7
        self.video_keys = {"External": "observation.images.image", "Wrist": "observation.images.wrist_image"}
        self.sampling_report = None
        if dataset_kind == "robomme":
            modality = json.loads((self.root / "meta/modality.json").read_text())
            for kind, expected in (("action", "actions"), ("state", "state")):
                columns = {field["original_key"] for field in modality[kind].values()}
                if columns != {expected} or self.info["features"][expected]["shape"] != [8]:
                    raise ValueError(f"Expected RoboMME {expected} as a single 8D column")
            self.action_column, self.state_column, self.action_dimension = "actions", "state", 8
            self.video_keys = {
                name: modality["video"][key]["original_key"]
                for name, key in (("External", "image"), ("Wrist", "wrist_image"))
            }

    def path(self, episode, kind="data_path", **kwargs):
        return self.root / self.info[kind].format(
            episode_chunk=episode // self.info.get("chunks_size", 1000), episode_index=episode, **kwargs
        )

    @lru_cache(maxsize=8)
    def get_trajectory_data(self, episode):
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.path(episode), columns=[self.action_column, self.state_column, "task_index", "frame_index"]
        )
        if table["frame_index"].to_pylist() != list(range(len(table))):
            raise ValueError(f"Non-contiguous frame IDs: {self.path(episode)}")
        if len(table) != self.episodes[episode]["length"]:
            raise ValueError(f"Episode length disagrees with metadata: {self.path(episode)}")
        tasks = {self.tasks[index] for index in table["task_index"].to_pylist()}
        if tasks != set(self.episodes[episode]["tasks"]) or len(tasks) != 1:
            raise ValueError(f"Episode task disagrees with metadata: {self.path(episode)}")
        values = table.to_pandas()
        actions = np.stack(values[self.action_column])
        if actions.shape != (len(table), self.action_dimension) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite {self.action_dimension}D actions: {self.path(episode)}")
        state = np.stack(values[self.state_column])
        if state.ndim != 2 or state.shape[1] < 3 or not np.isfinite(state).all():
            raise ValueError(f"Expected finite state vectors: {self.path(episode)}")
        if self.dataset_kind == "robomme" and state.shape != (len(table), 8):
            raise ValueError(f"Expected 8D RoboMME state: {self.path(episode)}")
        return values

    def source(self, episode):
        videos = {}
        for camera, key in self.video_keys.items():
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
    if getattr(args, "dataset_kind", "libero") == "robomme":
        return build_robomme_plan(args)
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


def build_robomme_plan(args):
    from gr00t.rag.data import get_alignment_groups

    dataset = ParquetDataset(args.dataset_path, "robomme")
    groups, report = get_alignment_groups(dataset, args.num_frames, args.task_map)
    if report["grouping"] != "task_and_instruction":
        raise ValueError("RoboMME requires an episode-task index; supply --task-map")
    dataset.sampling_report = report
    instruction_ids = {instruction: index for index, instruction in dataset.tasks.items()}
    if len(instruction_ids) != len(dataset.tasks):
        raise ValueError("Instruction strings must have unique task_index IDs")
    families = {key[0] for key in groups} | {row["task"] for row in report["skipped_groups"]}
    if args.tasks and set(args.tasks) - families:
        raise ValueError(f"Unknown canonical tasks: {sorted(set(args.tasks) - families)}")
    if args.task_ids and set(args.task_ids) - set(dataset.tasks):
        raise ValueError("Unknown instruction task IDs")

    def selected(task, instruction):
        return (not args.tasks or task in args.tasks) and (
            args.task_ids is None or instruction_ids[instruction] in args.task_ids
        )

    keys = sorted(key for key in groups if selected(*key))
    if args.tasks_per_suite:
        rng = random.Random(record_seed(args.seed, "robomme"))
        keys = sorted(rng.sample(keys, min(args.tasks_per_suite, len(keys))))
    skipped = [
        {"suite": "robomme", **row}
        for row in report["skipped_groups"]
        if selected(row["task"], row["instruction"])
    ]
    plan = []
    for task, instruction in keys:
        task_id = instruction_ids[instruction]
        identity_prefix = f"robomme:{task}:task{task_id}"
        pairs = list(combinations(sorted(groups[task, instruction]), 2))
        rng = random.Random(record_seed(args.seed, identity_prefix))
        for a, b in sorted(rng.sample(pairs, min(args.pairs_per_task, len(pairs)))):
            identity = f"{identity_prefix}:ep{a}:ep{b}"
            plan.append(
                {
                    "id": identity,
                    "suite": "robomme",
                    "task_id": task_id,
                    "task": task,
                    "instruction": instruction,
                    "episode_a": a,
                    "episode_b": b,
                    "seed": record_seed(args.seed, identity),
                }
            )
    if not plan:
        raise ValueError("No eligible RoboMME demonstration pairs selected")
    return {"robomme": dataset}, plan, skipped


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
    record = {
        **item,
        "dataset": str(dataset.root),
        "source_a": source_a,
        "source_b": source_b,
        "frame_pairs": [[int(a), int(b)] for a, b in zip(frames_a, frames_b)],
        "progress_frame_pairs": [[int(a), int(b)] for a, b in zip(frames_a, progress)],
        "label_status": "action-DTW positive candidates; not human-verified",
    }
    if dataset.dataset_kind == "robomme":
        record["observation_schema"] = {
            "dataset_kind": "robomme",
            "action_column": dataset.action_column,
            "state_column": dataset.state_column,
            "action_names": [f"Joint {index + 1} target (rad)" for index in range(7)] + ["Gripper target"],
            "state_metric": "joint_rmse_rad",
        }
    return record


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
    for suite in datasets:
        selected = [item for item in plan if item["suite"] == suite]
        groups = {(item["task"], item.get("instruction")) for item in selected}
        print(
            f"{suite}: {len(groups)} alignment groups, {len(selected)} demo pairs, {len(selected) * args.num_frames} frame pairs",
            flush=True,
        )
        if datasets[suite].sampling_report is not None:
            report = datasets[suite].sampling_report
            print(
                f"Dataset: {report['eligible_groups']}/{report['total_groups']} eligible groups; "
                f"{report['skipped_group_count']} groups skipped (dataset unchanged)",
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
    reports = [dataset.sampling_report for dataset in datasets.values() if dataset.sampling_report is not None]
    if reports:
        signature["dataset_kind"] = "robomme"
        signature["sampling_reports"] = reports
        signature["grouping_sha256"] = hashlib.sha256((REPO / "gr00t/rag/data.py").read_bytes()).hexdigest()
        signature["dataset_metadata"].extend(
            stamp(path)
            for dataset in datasets.values()
            for path in (dataset.root / "meta/modality.json", Path(dataset.sampling_report["task_map"]))
        )
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
            "held_out_suite": None if reports else "libero_10",
            "algorithm": "gr00t.rag.utils.get_aligned_frame_ids (online raw-action DTW)",
            "signature": signature,
            "planned_demo_pairs": len(plan),
            "completed_demo_pairs": len(existing),
            "num_frames_per_pair": args.num_frames,
            "skipped_tasks": skipped,
            "note": (
                "Selected RoboMME dataset only; (canonical task, exact instruction) groups. "
                "No held-out split inferred. No model training/evaluation."
                if reports
                else "G/O/S demonstrations only. No retriever training or Long evaluation. Samples use raw action DTW, not visual similarity."
            ),
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
    parser.add_argument("--dataset-kind", choices=("libero", "robomme"), default="libero")
    parser.add_argument("--dataset-path", type=Path, help="One RoboMME v2.1 dataset directory")
    parser.add_argument(
        "--task-map", type=Path, help="RoboMME episode-task index; defaults to meta/episode_tasks.json"
    )
    parser.add_argument("--tasks", nargs="+", help="RoboMME canonical task names (not instruction strings)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--suites", choices=ID_SUITES, nargs="+")
    parser.add_argument(
        "--task-ids", type=int, nargs="+", help="LeRobot instruction IDs, applied to each selected dataset"
    )
    parser.add_argument(
        "--tasks-per-suite",
        type=int,
        default=0,
        help="0=all eligible groups; positive value limits groups per dataset",
    )
    parser.add_argument(
        "--pairs-per-task",
        type=int,
        default=10,
        help="Unique demo pairs per alignment group, capped by availability",
    )
    parser.add_argument("--num-frames", type=int, default=20, help="Positive frame pairs per demonstration pair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(args.pairs_per_task, args.num_frames) < 1
        or args.tasks_per_suite < 0
        or any(task < 0 for task in args.task_ids or [])
    ):
        parser.error("Counts must be positive; task IDs/task limit must be nonnegative")
    if args.dataset_kind == "robomme":
        if args.dataset_path is None or args.suites is not None:
            parser.error("RoboMME requires --dataset-path and does not use --suites")
    elif args.dataset_path is not None or args.task_map is not None or args.tasks is not None:
        parser.error("--dataset-path, --task-map and --tasks require --dataset-kind robomme")
    args.suites = args.suites or list(ID_SUITES)
    args.output_dir = args.output_dir or REPO / "outputs/dtw_pairs" / (
        "robomme_offline" if args.dataset_kind == "robomme" else "gos_offline"
    )
    if len(set(args.suites)) != len(args.suites):
        parser.error("Duplicate suites")
    try:
        export(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
