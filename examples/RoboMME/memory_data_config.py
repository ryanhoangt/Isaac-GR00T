"""The existing RoboMME transforms, with explicit memory-field passthrough."""

import numpy as np

from examples.RoboMME.custom_data_config import RoboMMEDataConfig
from gr00t.model.transforms import GR00TTransform


MEMORY_KEYS = ("static_image_emb", "static_pos_emb", "static_mask")


class RoboMMEMemoryTransform(GR00TTransform):
    """Keep cached memory separate from VLM image augmentation/processing.

    GR00TTransform.apply_single builds a new dictionary and drops unknown keys.
    Its batch path calls this method too, so the stock collator can stack the
    three fixed-size memory arrays in both single- and batched-input paths.
    """

    def apply_single(self, data):
        missing = set(MEMORY_KEYS) - data.keys()
        if missing:
            raise ValueError(f"Missing RoboMME memory fields: {sorted(missing)}")
        images, positions, mask = (data[key] for key in MEMORY_KEYS)
        if not all(isinstance(value, np.ndarray) for value in (images, positions, mask)):
            raise ValueError("Memory fields must be NumPy arrays")
        if images.ndim != 2 or positions.ndim != 2 or mask.ndim != 1:
            raise ValueError("Expected memory arrays [tokens,image_dim], [tokens,pos_dim], [tokens]")
        if images.shape[0] != positions.shape[0] or images.shape[0] != mask.shape[0] or not mask.any():
            raise ValueError("Memory lengths must agree and contain at least one valid token")
        if images.dtype != np.float32 or positions.dtype != np.float32 or mask.dtype != np.bool_:
            raise ValueError("Memory features/positions must be float32 and the mask must be bool")
        result = super().apply_single(data)
        result.update({key: data[key] for key in MEMORY_KEYS})
        return result


class RoboMMEMemoryDataConfig(RoboMMEDataConfig):
    """Same current views, instruction, state/actions and normalization as vanilla."""

    def transform(self):
        transforms = super().transform()
        original = transforms.transforms[-1]
        transforms.transforms[-1] = RoboMMEMemoryTransform(
            **{name: getattr(original, name) for name in type(original).model_fields}
        )
        return transforms
