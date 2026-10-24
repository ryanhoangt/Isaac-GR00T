from typing import List
import random
from torch.utils.data import Sampler
from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.rag.utils import random_distribute_n_to_k, get_aligned_frame_ids


class FramePathsMixtureDataset(LeRobotMixtureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_frame_paths = self._get_all_frame_paths()
        self._all_frame_paths_inverted = {
            frame_path: index for index, frame_path in enumerate(self.all_frame_paths)}

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
        return dataset.transforms(dataset.get_step_data(traj_id, frame_id))


class AlignedFramesBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        mixture_dataset: FramePathsMixtureDataset,
        num_trajs: int,
        num_frames: int,
        num_iterations: int,
    ):
        self.mixture_dataset = mixture_dataset
        self.num_trajs = num_trajs
        self.num_frames = num_frames
        self.num_iterations = num_iterations

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        for _ in range(self.num_iterations):
            frame_paths1, frame_paths2 = [], []
            num_trajs_list = random_distribute_n_to_k(
                self.num_trajs, len(self.mixture_dataset.datasets))

            for dataset_id, (dataset, num_trajs) in enumerate(zip(self.mixture_dataset.datasets, num_trajs_list)):
                tasks = list(dataset.task_groups.keys())
                tasks = random.sample(tasks, num_trajs)

                for task in tasks:
                    traj_ids = dataset.task_groups[task]
                    traj_id1, traj_id2 = random.sample(traj_ids, 2)
                    frame_ids1, frame_ids2 = get_aligned_frame_ids(dataset, traj_id1, traj_id2, self.num_frames)
                    frame_paths1.extend([(dataset_id, traj_id1, frame_id1) for frame_id1 in frame_ids1])
                    frame_paths2.extend([(dataset_id, traj_id2, frame_id2) for frame_id2 in frame_ids2])

            frame_paths = frame_paths1 + frame_paths2
            data_indices = [
                self.mixture_dataset.all_frame_paths_inverted[frame_path] for frame_path in frame_paths]

            yield data_indices
