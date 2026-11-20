"""Check normalized targets, padding, inverse actions and the real Eagle collator."""

import copy

import numpy as np
import pytest
import torch

from gr00t.data.schema import DatasetMetadata
from gr00t.experiment.data_config import load_data_config
from gr00t.model.transforms import DefaultDataCollator


CONFIG = "examples.RoboMME.custom_data_config:RoboMMEDataConfig"


def stats(low, high):
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    return dict(min=low, max=high, mean=(low + high) / 2, std=(high - low) / 4, q01=low, q99=high)


@pytest.fixture
def example():
    config = load_data_config(CONFIG)
    action_low = np.array([-2, -1, -3, -4, -1, -2, -3], np.float32)
    action_high = np.array([2, 3, 1, -1, 2, 5, 3], np.float32)
    metadata = DatasetMetadata.model_validate(
        {
            "embodiment_tag": "new_embodiment",
            "statistics": {
                "state": {"joint_position": stats([-1] * 7, [1] * 7), "gripper": stats([0], [0.04])},
                "action": {"joint_position": stats(action_low, action_high), "gripper": stats([-1], [1])},
            },
            "modalities": {
                "video": {
                    key: {"resolution": [256, 256], "channels": 3, "fps": 10} for key in ("image", "wrist_image")
                },
                **{
                    modality: {
                        key: {"absolute": True, "continuous": True, "shape": [size]}
                        for key, size in (("joint_position", 7), ("gripper", 1))
                    }
                    for modality in ("state", "action")
                },
            },
        }
    )
    front, wrist = np.zeros((1, 256, 256, 3), np.uint8), np.zeros((1, 256, 256, 3), np.uint8)
    front[..., 0], wrist[..., 2] = 255, 255
    joint_actions = action_low + np.linspace(0, 1, 16, dtype=np.float32)[:, None] * (action_high - action_low)
    raw = {
        "video.image": front,
        "video.wrist_image": wrist,
        "state.joint_position": np.array([[-0.5, 0, 0.5, 1, -1, 0.25, -0.25]], np.float32),
        "state.gripper": np.array([[0.01]], np.float32),
        "action.joint_position": joint_actions,
        "action.gripper": np.array([-1, 1] * 8, np.float32)[:, None],
        "annotation.human.action.task_description": ["Pick the red cube"],
    }
    transforms = config.transform()
    transforms.set_metadata(metadata)
    return config, transforms, raw


def test_absolute_targets_masks_and_inverse(example):
    config, transforms, raw = example
    modalities = config.modality_config()
    assert modalities["video"].delta_indices == modalities["state"].delta_indices == [0]
    assert modalities["action"].delta_indices == list(range(16))
    assert modalities["language"].modality_keys == ["annotation.human.action.task_description"]
    sample = transforms(copy.deepcopy(raw))
    expected_action = np.repeat(np.linspace(-1, 1, 16, dtype=np.float32)[:, None], 7, axis=1)
    np.testing.assert_allclose(sample["action"][:, :7], expected_action, atol=2e-7)
    np.testing.assert_array_equal(sample["action"][:, 7:8], raw["action.gripper"])
    np.testing.assert_allclose(sample["state"][:, :7], raw["state.joint_position"], atol=1e-7)
    np.testing.assert_allclose(sample["state"][:, 7], [-0.5], atol=1e-7)
    assert sample["state"].shape == (1, 64) and sample["action"].shape == (16, 32)
    for key in ("state", "action"):
        assert sample[key].dtype == np.float32 and np.isfinite(sample[key]).all()
        assert sample[f"{key}_mask"][..., :8].all()
        assert not sample[f"{key}_mask"][..., 8:].any()
        assert (sample[key][..., 8:] == 0).all()
    assert sample["embodiment_id"] == 31
    # Model output has a batch dimension and must recover physical absolute targets.
    restored = transforms.unapply({"action": torch.from_numpy(sample["action"][None].copy())})
    for key in config.action_keys:
        assert restored[key].shape == (1, 16, raw[key].shape[-1])
        np.testing.assert_allclose(restored[key][0], raw[key], atol=5e-7)


def test_inference_without_actions_preserves_view_and_language_order(example):
    config, transforms, raw = example
    transforms.eval()
    observation = {key: value for key, value in raw.items() if key not in config.action_keys}
    sample = transforms(copy.deepcopy(observation))
    assert "action" not in sample and "action_mask" not in sample
    assert sample["state"].shape == (1, 64)
    images = sample["eagle_content"]["image_inputs"]
    assert len(images) == 2
    assert [image.size for image in images] == [(224, 224), (224, 224)]
    assert [int(np.asarray(image).mean(axis=(0, 1)).argmax()) for image in images] == [0, 2]
    assert "Pick the red cube" in sample["eagle_content"]["text_list"][0]


def test_training_batch_uses_real_collator(example):
    _, transforms, raw = example
    batch = DefaultDataCollator()([transforms(copy.deepcopy(raw)), transforms(copy.deepcopy(raw))])
    assert batch["state"].shape == (2, 1, 64)
    assert batch["action"].shape == (2, 16, 32)
    assert batch["action_mask"].sum().item() == 2 * 16 * 8
    assert batch["state_mask"].sum().item() == 2 * 8
    assert batch["embodiment_id"].tolist() == [31, 31]
    assert batch["eagle_input_ids"].shape[0] == 2
    assert batch["eagle_attention_mask"].shape == batch["eagle_input_ids"].shape
    assert batch["eagle_pixel_values"].numel() > 0
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
            assert torch.isfinite(value).all()
