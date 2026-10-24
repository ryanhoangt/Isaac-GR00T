import numpy as np
import torch
from gr00t.model.policy import BasePolicy


class VLAPolicy(BasePolicy):
    def __init__(self, modality_configs, transforms, embodiment_tag, collator, vla):
        self.modality_configs = modality_configs
        self.transforms = transforms
        self.embodiment_tag = embodiment_tag
        self.collator = collator
        self.vla = vla

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observations):
        """
        observations = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray,  # (T, D)
            "annotation.<>": list[str],  # (T, )
        }
        unnormalized_actions = {
            "action.<>": np.ndarray,  # (T, D)
        }
        """
        with torch.inference_mode():
            data = observations.copy()
            normalized_inputs = self.collator([self.transforms(data)])
            normalized_actions = self.vla.get_action(normalized_inputs).action_pred
            normalized_actions = {"action": normalized_actions[0].float().cpu()}
            unnormalized_actions = self.transforms.unapply(normalized_actions)
        return unnormalized_actions
