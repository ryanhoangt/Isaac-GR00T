"""Check serving config/provenance and canonical policy routing on CPU."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rag.memory import file_fingerprint
from gr00t.rag.policy import VLAPolicy
from gr00t.rag.serving import load_serving_memory, resolve_serving_config


@pytest.fixture
def serving(tmp_path):
    module = importlib.import_module("scripts.rag.4_serve_ravla")
    root, checkpoint = tmp_path / "retriever", tmp_path / "checkpoint_1"
    root.mkdir()
    checkpoint.mkdir()
    data_config = "examples.RoboMME.custom_data_config:RoboMMEDataConfig"
    (checkpoint / "config.json").write_text(
        json.dumps({"ravla_data_config": data_config, "ravla_memory_key_type": "canonical_task"})
    )
    (checkpoint / "model.safetensors").write_bytes(b"policy weights")
    (root / "model.safetensors").write_bytes(b"retriever weights")
    (root / "config.json").write_text("{}")
    metadata = {"new_embodiment": {"stats": "train statistics"}}
    torch.save(metadata, checkpoint / "metadata.pt")
    torch.save(metadata, root / "metadata.pt")
    bank = {
        "keys": torch.zeros(3, 8),
        "action": torch.zeros(3, 16, 32),
        "pixel_values": torch.zeros(3, 2, 3, 2, 2),
        "backbone_features": torch.zeros(3, 4, 8),
        "backbone_attention_mask": torch.ones(3, 4, dtype=torch.bool),
    }
    path = root / "heldout.pt"
    torch.save({"UnseenTask": bank}, path)
    manifest = {
        "status": "complete",
        "config": {"data_config": data_config},
        "plan": {
            "key_type": "canonical_task",
            "is_smoke_sample": False,
            "banks": [{"key": "UnseenTask", "demos": [{"episode_index": i} for i in range(3)]}],
        },
        "retriever": [
            file_fingerprint(root / name) for name in ("config.json", "metadata.pt", "model.safetensors")
        ],
        "output": {"bytes": path.stat().st_size},
        "candidate_counts": {"UnseenTask": 3},
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    config = module.ArgsConfig(checkpoint_path=str(checkpoint), retriever_path=str(root), memory_path=str(path))
    return config, metadata, manifest


def test_infers_checkpoint_config_and_accepts_separate_heldout_memory(serving):
    config, metadata, _ = serving
    saved, mode = resolve_serving_config(config)
    assert mode == "canonical_task" and config.data_config == saved["ravla_data_config"]
    memory, info = load_serving_memory(config, mode, metadata, "cpu")
    assert set(memory) == {"UnseenTask"}
    assert info["memory_demo_counts"] == {"UnseenTask": 3}
    assert info["memory_tasks"] == ["UnseenTask"] and len(info["source_fingerprint"]) == 64
    assert load_serving_memory(config, mode, metadata, "cpu")[1] == info


@pytest.mark.parametrize(
    "problem", ["metadata", "retriever", "capped", "key_type", "missing_manifest", "data_config"]
)
def test_rejects_mismatched_serving_inputs(serving, problem):
    config, metadata, manifest = serving
    resolve_serving_config(config)
    if problem == "metadata":
        metadata["new_embodiment"]["stats"] = "heldout refitted statistics"
    elif problem == "retriever":
        Path(config.retriever_path, "model.safetensors").write_bytes(b"new weights")
    elif problem == "capped":
        manifest["plan"]["is_smoke_sample"] = True
    elif problem == "key_type":
        manifest["plan"]["key_type"] = "instruction"
    elif problem == "data_config":
        manifest["config"]["data_config"] = "wrong"
    path = Path(config.memory_path).with_suffix(".json")
    path.write_text(json.dumps(manifest))
    if problem == "missing_manifest":
        path.unlink()
    with pytest.raises(ValueError):
        load_serving_memory(config, "canonical_task", metadata, "cpu")


def test_preserves_legacy_instruction_serving_defaults(serving):
    config, metadata, _ = serving
    Path(config.checkpoint_path, "config.json").write_text("{}")
    Path(config.memory_path).with_suffix(".json").unlink()
    _, mode = resolve_serving_config(config)
    assert mode == "instruction" and "LiberoDataConfig" in config.data_config
    assert load_serving_memory(config, mode, metadata, "cpu")[1]["memory_key_type"] == "instruction"


def test_rejects_explicit_config_and_resource_mismatch(serving):
    config, _, _ = serving
    config.data_config = "wrong"
    with pytest.raises(ValueError, match="disagrees"):
        resolve_serving_config(config)
    config.data_config = None
    config.gpu_memory_fraction = 1.1
    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        resolve_serving_config(config)


def test_policy_routes_metadata_without_changing_language_or_inverse_actions():
    calls, transformed = [], []

    class Transform:
        def __call__(self, data):
            assert "memory_key" not in data
            transformed.append(data.copy())
            return {"language": data["language"]}

        def unapply(self, actions):
            return {
                "action.joint_position": actions["action"][:, :7].numpy() + 1.0,
                "action.gripper": actions["action"][:, 7:].numpy() - 1.0,
            }

    def predict(inputs):
        calls.append(inputs)
        return SimpleNamespace(action_pred=torch.zeros(1, 16, 8))

    policy = VLAPolicy(
        {},
        Transform(),
        "new_embodiment",
        lambda rows: {"tasks": [row["memory_key"] for row in rows], "language": rows[0]["language"]},
        SimpleNamespace(memory={"A": {}, "B": {}}, get_action=predict),
        memory_key_type="canonical_task",
    )
    for key in ("A", "B"):
        observation = {"memory_key": key, "language": "same instruction"}
        result = policy.get_action(observation)
        assert calls[-1]["tasks"] == [key] and calls[-1]["language"] == "same instruction"
        assert observation["memory_key"] == key
        np.testing.assert_array_equal(result["action.joint_position"], np.ones((16, 7)))
        np.testing.assert_array_equal(result["action.gripper"], -np.ones((16, 1)))
    for value in ({"language": "same instruction"}, {"memory_key": "unknown", "language": "same instruction"}):
        with pytest.raises(ValueError):
            policy.get_action(value)
    assert len(transformed) == len(calls) == 2
