"""Use a separate image-only Eagle conversation for the target milestone."""

import numpy as np
from PIL import Image

from gr00t.model.transforms import GR00TTransform
from examples.Libero.custom_data_config import LiberoDataConfig


class PilotLanguageTransform(GR00TTransform):
    def _prepare_language(self, data):
        language = super()._prepare_language(data)
        if isinstance(language, np.ndarray):
            language = str(language.reshape(-1)[0])
        return language


class GoalTransform(PilotLanguageTransform):
    def apply_single(self, data):
        mode = int(np.asarray(data.get("goal_mode_id", 0)).item())
        if mode not in (0, 1):
            raise ValueError("goal_mode_id must be 0 or 1")
        result = super().apply_single(data)
        goal = data.get("subgoal.image")
        if mode and goal is None:
            raise ValueError("Active goal mode requires subgoal.image")
        if mode:
            goal = np.asarray(goal)
            if goal.ndim != 4 or goal.shape[0] != 1 or goal.shape[-1] != 3:
                raise ValueError("subgoal.image must have shape (1, H, W, 3)")
            image = np.asarray(Image.fromarray(goal[0].astype(np.uint8)).resize((224, 224)))
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        content = self._apply_vlm_processing(
            {"images": image.transpose(2, 0, 1)[None, None], "language": "Target subgoal view."}
        )["eagle_content"]
        result["goal_eagle_content"] = content
        result["goal_mode_id"] = np.int64(mode)
        return result


class BaselineDataConfig(LiberoDataConfig):
    """Vanilla architecture with the same string prompt handling as the dual model."""

    model_transform = PilotLanguageTransform

    def transform(self):
        transforms = super().transform()
        original = transforms.transforms[-1]
        transforms.transforms[-1] = self.model_transform(
            state_horizon=original.state_horizon,
            action_horizon=original.action_horizon,
            max_state_dim=original.max_state_dim,
            max_action_dim=original.max_action_dim,
            eagle_processor=original.eagle_processor,
        )
        return transforms


class GoalDataConfig(BaselineDataConfig):
    model_transform = GoalTransform
