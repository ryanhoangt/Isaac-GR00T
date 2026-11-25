"""Validate memory provenance and training-only normalization before loading models."""

import json
from types import SimpleNamespace

import pytest
import torch

from gr00t.rag.memory import file_fingerprint
from gr00t.rag.training import load_training_memory, resolve_policy_config, set_training_metadata


@pytest.fixture
def inputs(tmp_path):
    root, dataset_root = tmp_path / "retriever", tmp_path / "dataset"
    root.mkdir()
    (dataset_root / "meta").mkdir(parents=True)
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        (dataset_root / "meta" / name).write_text("{}\n")
    trained = {
        "data_config": "examples.RoboMME.custom_data_config:RoboMMEDataConfig",
        "pretrained_model_path": "nvidia/GR00T-N1.5-3B",
    }
    (root / "training_config.json").write_text(json.dumps(trained))
    (root / "config.json").write_text("{}")
    (root / "metadata.pt").write_bytes(b"training normalization")
    (root / "model.safetensors").write_bytes(b"trained retriever")
    bank = {
        "keys": torch.ones(2, 8),
        "action": torch.zeros(2, 16, 32),
        "action_mask": torch.ones(2, 16, 32, dtype=torch.bool),
        "pixel_values": torch.zeros(2, 2, 3, 2, 2),
        "backbone_features": torch.zeros(2, 3, 8),
        "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    path = root / "memory.pt"
    torch.save({"task": bank}, path)
    manifest = {
        "status": "complete",
        "config": trained,
        "output": {"bytes": path.stat().st_size},
        "candidate_counts": {"task": 2},
        "retriever": [
            file_fingerprint(root / name)
            for name in ("metadata.pt", "model.safetensors", "config.json", "training_config.json")
        ],
        "plan": {
            "key_type": "canonical_task",
            "is_smoke_sample": False,
            "datasets": [
                {"metadata": [file_fingerprint(file) for file in sorted((dataset_root / "meta").iterdir())]}
            ],
        },
    }
    path.with_suffix(".json").write_text(json.dumps(manifest))
    config = SimpleNamespace(
        retriever_path=str(root),
        memory_path=str(path),
        memory_name="memory",
        data_config=None,
        pretrained_model_path=None,
        k=1,
    )
    mixture = SimpleNamespace(
        memory_key_type="canonical_task",
        required_memory_keys={"task"},
        datasets=[SimpleNamespace(dataset_path=dataset_root)],
        memory_task_map_paths=[None],
    )
    return config, mixture, manifest


def test_config_inference_memory_coverage_and_portable_fingerprints(inputs):
    config, mixture, manifest = inputs
    resolve_policy_config(config)
    assert config.data_config == manifest["config"]["data_config"]
    # Provenance paths may refer to the old VM; local filenames/content bind the weights.
    for source in manifest["retriever"]:
        source["path"] = "/old/vm/" + source["path"].rsplit("/", 1)[-1]
    from pathlib import Path

    Path(config.memory_path).with_suffix(".json").write_text(json.dumps(manifest))
    memory, report = load_training_memory(config, mixture, "cpu")
    assert set(memory) == {"task"} and report["candidates"] == 2
    config.data_config = "wrong-config"
    with pytest.raises(ValueError, match="disagrees"):
        resolve_policy_config(config)


@pytest.mark.parametrize(
    "problem", ["capped", "unfinished", "retriever", "dataset", "coverage", "size", "missing_manifest"]
)
def test_invalid_training_memory_is_rejected(inputs, problem):
    from pathlib import Path

    config, mixture, manifest = inputs
    root = Path(config.retriever_path)
    if problem == "capped":
        manifest["plan"]["is_smoke_sample"] = True
    elif problem == "unfinished":
        manifest["status"] = "encoding"
    elif problem == "retriever":
        (root / "model.safetensors").write_bytes(b"other retriever")
    elif problem == "dataset":
        (mixture.datasets[0].dataset_path / "meta/info.json").write_text("changed metadata")
    elif problem == "coverage":
        mixture.required_memory_keys.add("unseen task")
    elif problem == "size":
        manifest["output"]["bytes"] += 1
    path = Path(config.memory_path).with_suffix(".json")
    path.write_text(json.dumps(manifest))
    if problem == "missing_manifest":
        path.unlink()
    with pytest.raises(ValueError):
        load_training_memory(config, mixture, "cpu")


def test_legacy_libero_memory_without_manifest_still_loads(inputs):
    from pathlib import Path

    config, mixture, _ = inputs
    Path(config.memory_path).with_suffix(".json").unlink()
    mixture.memory_key_type = "instruction"
    memory, report = load_training_memory(config, mixture, "cpu")
    assert len(memory["task"]["keys"]) == 2 and report["manifest"] is None


def test_training_metadata_is_reused_instead_of_refitting_on_query_data(tmp_path):
    torch.save({"tag": {"normalization": "train statistics"}}, tmp_path / "metadata.pt")
    received = []
    dataset = SimpleNamespace(tag="tag", set_transforms_metadata=received.append)
    mixture = SimpleNamespace(datasets=[dataset], merged_metadata={"tag": "query statistics"})
    set_training_metadata(mixture, tmp_path)
    assert mixture.merged_metadata["tag"] == {"normalization": "train statistics"}
    assert received == [{"normalization": "train statistics"}]
