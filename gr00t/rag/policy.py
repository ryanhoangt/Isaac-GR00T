import torch
from gr00t.model.policy import BasePolicy


class VLAPolicy(BasePolicy):
    def __init__(self, modality_configs, transforms, embodiment_tag, collator, vla, memory_key_type="instruction"):
        self.modality_configs = modality_configs
        self.transforms = transforms
        self.embodiment_tag = embodiment_tag
        self.collator = collator
        self.vla = vla
        self.memory_key_type = memory_key_type

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
            key = data.pop("memory_key", None)
            if self.memory_key_type == "canonical_task":
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Canonical RA-VLA inference requires memory_key")
                if key not in self.vla.memory:
                    raise ValueError(f"No memory bank for canonical task: {key}")
            elif key is not None:
                raise ValueError("This checkpoint uses instruction memory keys, not canonical task IDs")
            feature = self.transforms(data)
            if self.memory_key_type == "canonical_task":
                feature["memory_key"] = key
            normalized_inputs = self.collator([feature])
            normalized_actions = self.vla.get_action(normalized_inputs).action_pred
            normalized_actions = {"action": normalized_actions[0].float().cpu()}
            unnormalized_actions = self.transforms.unapply(normalized_actions)
        return unnormalized_actions
