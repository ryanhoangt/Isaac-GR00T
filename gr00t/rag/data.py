from typing import List
import hashlib
import json
from pathlib import Path
import random
from torch.utils.data import Sampler
from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.rag.utils import random_distribute_n_to_k, get_aligned_frame_ids


def _alignment_task_names(dataset, task_map):
    """Validate an optional portable episode-to-task index without reading HDF5."""
    root = Path(dataset.dataset_path)
    path = Path(task_map).expanduser() if task_map is not None else root / "meta/episode_tasks.json"
    if not path.is_file():
        if task_map is not None:
            raise FileNotFoundError(path)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if (
            manifest.get("converter") == "scripts/convert_robomme_lerobot.py"
            or manifest.get("extractor") == "scripts/extract_robomme_tasks.py"
        ):
            raise ValueError(f"RoboMME requires an episode-task index; supply --task-map for {root}")
        return None, None
    index = json.loads(path.read_text())
    names = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
    actual = {name: hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest() for name in names}
    if index.get("version") != 1 or index.get("source_metadata_sha256") != actual:
        raise ValueError(f"Task index does not match dataset metadata: {path}")
    entries = {row["episode_index"]: row for row in index["episodes"]}
    if len(entries) != len(index["episodes"]) or set(entries) != set(map(int, dataset.trajectory_ids)):
        raise ValueError("Task index must cover every dataset episode exactly once")
    for episode, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
        row = entries[int(episode)]
        if row["length"] != int(length) or not isinstance(row["task"], str) or not row["task"].strip():
            raise ValueError(f"Invalid task index entry for episode {episode}")
    return {episode: row["task"] for episode, row in entries.items()}, path.resolve()


def get_alignment_groups(dataset, num_frames, task_map=None):
    """Build a retriever-only view; leave all policy data and task_groups intact.

    With an episode-task index, keys are (task family, exact instruction).
    Otherwise retain LIBERO's instruction grouping. A positive pair needs two
    different demonstrations, each long enough to sample num_frames matches.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    owners, map_path = _alignment_task_names(dataset, task_map)
    lengths = {
        int(episode): int(length) for episode, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths)
    }
    groups = {}
    for instruction, episodes in dataset.task_groups.items():
        for episode in episodes:
            episode = int(episode)
            key = (owners[episode], instruction) if owners is not None else instruction
            groups.setdefault(key, []).append(episode)
    eligible, skipped = {}, []
    for key, episodes in groups.items():
        candidates = [episode for episode in episodes if lengths[episode] >= num_frames]
        if len(candidates) >= 2:
            eligible[key] = candidates
        else:
            skipped.append(
                {
                    "task": key[0] if owners is not None else None,
                    "instruction": key[1] if owners is not None else key,
                    "episode_indices": episodes,
                    "long_enough_episode_indices": candidates,
                    "reason": "fewer_than_two_demonstrations_long_enough",
                }
            )
    report = {
        "dataset": str(Path(dataset.dataset_path).resolve()),
        "grouping": "task_and_instruction" if owners is not None else "instruction",
        "task_map": str(map_path) if map_path is not None else None,
        "task_map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest() if map_path is not None else None,
        "num_frames": num_frames,
        "total_groups": len(groups),
        "eligible_groups": len(eligible),
        "skipped_group_count": len(skipped),
        "total_demonstrations": len(lengths),
        "eligible_demonstrations": sum(len(episodes) for episodes in eligible.values()),
        "short_demonstrations": [episode for episode, length in lengths.items() if length < num_frames],
        "skipped_groups": skipped,
    }
    return eligible, report


class FramePathsMixtureDataset(LeRobotMixtureDataset):
    def __init__(self, *args, memory_task_keys=False, task_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_frame_paths = self._get_all_frame_paths()
        self._all_frame_paths_inverted = {
            frame_path: index for index, frame_path in enumerate(self.all_frame_paths)
        }
        if task_map is not None and (not memory_task_keys or len(self.datasets) != 1):
            raise ValueError("An explicit task map requires one dataset with memory_task_keys enabled")
        mappings = (
            [_alignment_task_names(dataset, task_map) for dataset in self.datasets]
            if memory_task_keys
            else [(None, None) for _ in self.datasets]
        )
        self.memory_task_owners = [owners for owners, _ in mappings]
        self.memory_task_map_paths = [path for _, path in mappings]
        modes = {"canonical_task" if owners is not None else "instruction" for owners, _ in mappings}
        if len(modes) != 1:
            raise ValueError("Cannot mix canonical-task and instruction-keyed datasets")
        self.memory_key_type = modes.pop()
        self.required_memory_keys = set()
        for dataset, owners in zip(self.datasets, self.memory_task_owners):
            self.required_memory_keys.update(owners.values() if owners is not None else dataset.task_groups)

    @property
    def all_frame_paths(self):
        return self._all_frame_paths

    @property
    def all_frame_paths_inverted(self):
        return self._all_frame_paths_inverted

    def _get_all_frame_paths(self):
        all_frame_paths = []
        for dataset_id, dataset in enumerate(self.datasets):
            for traj_id, frame_id in dataset.all_steps:
                all_frame_paths.append((dataset_id, traj_id, frame_id))
        return all_frame_paths

    def __len__(self):
        return len(self.all_frame_paths)

    def __getitem__(self, index: int):
        dataset_id, traj_id, frame_id = self.all_frame_paths[index]
        dataset = self.datasets[dataset_id]
        feature = dataset.transforms(dataset.get_step_data(traj_id, frame_id))
        owners = self.memory_task_owners[dataset_id]
        if owners is not None:
            feature["memory_key"] = owners[int(traj_id)]
        return feature


class AlignedFramesBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        mixture_dataset: FramePathsMixtureDataset,
        num_trajs: int,
        num_frames: int,
        num_iterations: int,
        *,
        task_map=None,
    ):
        if min(num_trajs, num_frames, num_iterations) < 1:
            raise ValueError("Sampling counts and iterations must be positive")
        if task_map is not None and len(mixture_dataset.datasets) != 1:
            raise ValueError("An explicit --task-map requires a single --dataset-paths directory")
        self.mixture_dataset = mixture_dataset
        self.num_trajs = num_trajs
        self.num_frames = num_frames
        self.num_iterations = num_iterations
        self.alignment_groups = []
        self.sampling_report = []
        for dataset in mixture_dataset.datasets:
            groups, report = get_alignment_groups(dataset, num_frames, task_map)
            if num_trajs > len(groups):
                raise ValueError(
                    f"num_trajs={num_trajs} exceeds {len(groups)} eligible alignment groups in "
                    f"{dataset.dataset_path}; each group needs two demos with at least {num_frames} frames"
                )
            self.alignment_groups.append(groups)
            self.sampling_report.append(report)

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        for _ in range(self.num_iterations):
            frame_paths1, frame_paths2 = [], []
            num_trajs_list = random_distribute_n_to_k(self.num_trajs, len(self.mixture_dataset.datasets))

            for dataset_id, (dataset, num_trajs) in enumerate(zip(self.mixture_dataset.datasets, num_trajs_list)):
                groups = self.alignment_groups[dataset_id]
                tasks = list(groups)
                tasks = random.sample(tasks, num_trajs)

                for task in tasks:
                    traj_ids = groups[task]
                    traj_id1, traj_id2 = random.sample(traj_ids, 2)
                    frame_ids1, frame_ids2 = get_aligned_frame_ids(dataset, traj_id1, traj_id2, self.num_frames)
                    frame_paths1.extend([(dataset_id, traj_id1, frame_id1) for frame_id1 in frame_ids1])
                    frame_paths2.extend([(dataset_id, traj_id2, frame_id2) for frame_id2 in frame_ids2])

            frame_paths = frame_paths1 + frame_paths2
            data_indices = [
                self.mixture_dataset.all_frame_paths_inverted[frame_path] for frame_path in frame_paths
            ]

            yield data_indices
