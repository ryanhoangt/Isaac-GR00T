import numpy as np
import pytest
import torch

from scripts.rag.rekey_memory import memory_inputs, save_memory, verify_payload


def test_cached_pixels_bypass_image_processing_without_mutation():
    pixels = torch.randn(3, 2, 3, 8, 8)
    copy = pixels.clone()

    def transform(observation):
        assert observation["video"].shape == (1, 2, 8, 8, 3)
        assert observation["video"].dtype == np.uint8
        assert observation["annotation.human.action.task_description"] == ["test task"]
        return {"state": np.zeros((1, 64))}

    def collator(features):
        return {
            "eagle_pixel_values": torch.zeros(len(features) * 2, 3, 8, 8),
            "eagle_input_ids": torch.ones(len(features), 10, dtype=torch.long),
        }

    inputs = memory_inputs("test task", pixels, transform, collator)
    assert torch.equal(inputs["eagle_pixel_values"], pixels.flatten(0, 1))
    assert torch.equal(pixels, copy)
    assert inputs["eagle_input_ids"].shape == (3, 10)


def test_non_key_payload_is_verified_after_save(tmp_path):
    original = {
        "task": {
            "keys": torch.zeros(2, 4),
            "action": torch.randn(2, 16, 7),
            "pixel_values": torch.randn(2, 2, 3, 8, 8),
        }
    }
    rewritten = {"task": {**original["task"], "keys": torch.ones(2, 4)}}
    path = tmp_path / "memory.pt"
    assert save_memory(original, rewritten, path) > 0
    loaded = torch.load(path, mmap=True, weights_only=True)
    assert verify_payload(original, loaded) > 0
    loaded["task"]["action"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="Payload changed"):
        verify_payload(original, loaded)
