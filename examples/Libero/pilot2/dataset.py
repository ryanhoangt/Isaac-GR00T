"""Train-only paired goals on top of the repository LeRobot loader."""

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from .data import LANGUAGE_KEY, pair_goal, perturb_state, read_jsonl, sample_mode


class GoalDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mapping = {item["episode_index"]: item["task_id"] for item in read_mapping(self.dataset_path)}
        self.episodes = {
            item["episode_index"]: {**item, "task_id": mapping[item["episode_index"]]}
            for item in read_jsonl(self.dataset_path / "meta/episodes.jsonl")
        }

    def __getitem__(self, index):
        episode, step = self.all_steps[index]
        # NumPy's worker-global RNG is seeded by PyTorch's DataLoader. Avoid a
        # copied per-dataset Generator that would duplicate samples across workers.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))
        target, target_step, cross = pair_goal(self.episodes, episode, step, rng)
        raw = self.get_step_data(episode, step)
        goal = self.get_step_data(target, target_step)["video.image"].copy()
        if cross:
            raw = perturb_state(raw, rng)
        mode = sample_mode(rng)
        raw["subgoal.image"] = goal if mode != "language" else np.zeros_like(goal)
        raw["goal_mode_id"] = np.int64(mode != "language")
        if mode == "goal":
            raw[LANGUAGE_KEY] = np.array([""])
        return self.transforms(raw)


def read_mapping(root):
    import json

    mapping = root / "meta/episode_mapping.json"
    if not mapping.is_file():
        raise ValueError("Pilot 2 requires a prepared split, including episode_mapping.json")
    return json.loads(mapping.read_text())
