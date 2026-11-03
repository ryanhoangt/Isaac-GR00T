"""Same-scene hindsight data and episode-level held-out goal-control validation."""

import numpy as np
from torch.utils.data import Dataset

from gr00t.rag.ravla import RAVLADataCollator


def split_tasks(mixture, tasks_per_suite=1, heldout_demos=3):
    training, validation, inventory = [], [], []
    for dataset_id, dataset in enumerate(mixture.datasets):
        chosen = sorted(dataset.task_groups)[:tasks_per_suite]
        for task in chosen:
            episodes = sorted(int(e) for e in dataset.task_groups[task])
            if len(episodes) <= heldout_demos:
                raise ValueError("Not enough demonstrations for the goal-control split")
            validation_ids = set(episodes[:heldout_demos])
            training_ids = set(episodes[heldout_demos:])
            training.extend(
                (dataset_id, ep, frame)
                for d, ep, frame in mixture.all_frame_paths
                if d == dataset_id and int(ep) in training_ids
            )
            validation.extend((dataset_id, ep) for ep in sorted(validation_ids))
            inventory.append(
                dict(
                    dataset=dataset_id,
                    task=task,
                    train_episode_ids=sorted(training_ids),
                    validation_episode_ids=sorted(validation_ids),
                )
            )
    return training, validation, inventory


class SameDemoGoals(Dataset):
    def __init__(self, mixture, paths, lookahead=16):
        self.mixture, self.paths, self.lookahead = mixture, list(paths), lookahead
        if lookahead < 1:
            raise ValueError("lookahead must be positive")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        dataset_id, episode, frame = self.paths[index]
        dataset = self.mixture.datasets[dataset_id]
        length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(episode)])
        goal = min(frame + self.lookahead, length - 1)
        return dict(
            current=dataset.transforms(dataset.get_step_data(episode, frame)),
            goal=dataset.transforms(dataset.get_step_data(episode, goal)),
            provenance=dict(
                dataset=dataset_id,
                episode=int(episode),
                frame=int(frame),
                goal_episode=int(episode),
                goal_frame=int(goal),
                length=length,
            ),
        )


class GoalCollator:
    def __init__(self):
        self.base = RAVLADataCollator()

    def __call__(self, samples):
        return dict(
            current=self.base([s["current"] for s in samples]),
            goal=self.base([s["goal"] for s in samples]),
            provenance=[s["provenance"] for s in samples],
        )


def validation_paths(mixture, episodes, states_per_demo=20):
    paths = []
    for dataset_id, episode in episodes:
        dataset = mixture.datasets[dataset_id]
        length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(episode)])
        # Exclude repeated terminal-action labels from the diagnostic MSE.
        last = max(0, length - 16)
        frames = np.unique(np.linspace(0, last, min(states_per_demo, last + 1)).astype(int))
        paths.extend((dataset_id, episode, int(frame)) for frame in frames)
    return paths
