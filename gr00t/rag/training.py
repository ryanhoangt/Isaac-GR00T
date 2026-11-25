"""Validate policy-training inputs against their frozen retriever and memory."""

import json
from pathlib import Path

import torch

from gr00t.rag.memory import file_fingerprint


def resolve_policy_config(config):
    root = Path(config.retriever_path)
    trained_path = root / "training_config.json"
    trained = json.loads(trained_path.read_text()) if trained_path.is_file() else {}
    config.memory_path = config.memory_path or str(root / f"{config.memory_name}.pt")
    manifest_path = Path(config.memory_path).with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    cached = manifest.get("config", {}) if manifest else {}
    defaults = {
        "data_config": "examples.Libero.custom_data_config:LiberoDataConfig",
        "pretrained_model_path": "nvidia/GR00T-N1.5-3B",
    }
    for name, default in defaults.items():
        value = getattr(config, name) or cached.get(name) or trained.get(name) or default
        if any(source.get(name) is not None and source[name] != value for source in (trained, cached)):
            raise ValueError(f"{name} disagrees with the retriever/memory configuration")
        setattr(config, name, value)
    return manifest


def set_training_metadata(mixture, retriever_path):
    # Pickled DatasetMetadata comes from this trusted local training run.
    metadata = torch.load(Path(retriever_path) / "metadata.pt", map_location="cpu", weights_only=False)
    mixture.merged_metadata = metadata
    for dataset in mixture.datasets:
        dataset.set_transforms_metadata(metadata[dataset.tag])


def load_training_memory(config, mixture, device):
    path = Path(config.memory_path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    canonical = mixture.memory_key_type == "canonical_task"
    if canonical and manifest is None:
        raise ValueError("Canonical-task memory requires its builder manifest")
    if manifest is not None:
        if manifest.get("status") != "complete":
            raise ValueError("Memory build is not complete")
        plan = manifest["plan"]
        if plan["key_type"] != mixture.memory_key_type:
            raise ValueError("Dataset and memory use different key types")
        if plan["is_smoke_sample"]:
            raise ValueError("Use a full memory build for policy training, not a capped smoke sample")
        if path.stat().st_size != manifest["output"]["bytes"]:
            raise ValueError("Memory file size disagrees with manifest")
        for source in manifest["retriever"]:
            # Match by filename/hash so copying the run to another VM remains valid.
            local = Path(config.retriever_path) / Path(source["path"]).name
            if file_fingerprint(local)["sha256"] != source["sha256"]:
                raise ValueError(f"Memory was built with different retriever/normalization: {local}")
        if canonical:
            source_metadata = [
                {Path(item["path"]).name: item["sha256"] for item in report["metadata"]}
                for report in plan["datasets"]
            ]
            for dataset in mixture.datasets:
                root = Path(dataset.dataset_path) / "meta"
                actual = {
                    name: file_fingerprint(root / name)["sha256"]
                    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
                }
                if actual not in source_metadata:
                    raise ValueError("Training dataset metadata differs from the memory source")
    memory = torch.load(path, map_location=device, weights_only=True)
    expected = mixture.required_memory_keys
    missing, extra = expected - memory.keys(), memory.keys() - expected
    if missing or (canonical and extra):
        raise ValueError(f"Memory task coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for key in expected:
        bank = memory[key]
        count = len(bank["keys"])
        if count < config.k:
            raise ValueError(f"{key}: fewer than k={config.k} candidates")
        if any(
            len(bank[name]) != count
            for name in ("action", "pixel_values", "backbone_features", "backbone_attention_mask")
        ):
            raise ValueError(f"{key}: candidate tensor counts disagree")
        if bank["action"].shape[1:] != (16, 32):
            raise ValueError(f"{key}: expected action chunks [N,16,32]")
        if bank["backbone_features"].shape[:2] != bank["backbone_attention_mask"].shape:
            raise ValueError(f"{key}: backbone features/masks disagree")
        if canonical and ("action_mask" not in bank or bank["action_mask"].shape != bank["action"].shape):
            raise ValueError(f"{key}: missing or invalid action masks")
        if manifest and count != manifest["candidate_counts"][key]:
            raise ValueError(f"{key}: candidate count differs from manifest")
    report = {
        "memory_path": str(path.resolve()),
        "key_type": mixture.memory_key_type,
        "banks": len(memory),
        "candidates": sum(len(bank["keys"]) for bank in memory.values()),
        "k": config.k,
        "manifest": file_fingerprint(manifest_path) if manifest else None,
        "task_maps": [file_fingerprint(path) if path else None for path in mixture.memory_task_map_paths],
    }
    return memory, report
