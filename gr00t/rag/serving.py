"""Resolve checkpoint-serving configuration and validate its retrieval memory."""

import json
from pathlib import Path

import torch

from gr00t.rag.memory import file_fingerprint
from scripts.rag.precompute_retrieval import digest, stamp


def resolve_serving_config(config):
    checkpoint = Path(config.checkpoint_path)
    saved = json.loads((checkpoint / "config.json").read_text())
    recorded = saved.get("ravla_data_config")
    if recorded is not None and config.data_config is not None and config.data_config != recorded:
        raise ValueError("data_config disagrees with the policy checkpoint")
    config.data_config = config.data_config or recorded or "examples.Libero.custom_data_config:LiberoDataConfig"
    config.memory_path = config.memory_path or str(Path(config.retriever_path) / f"{config.memory_name}.pt")
    key_type = saved.get("ravla_memory_key_type", "instruction")
    if key_type not in ("instruction", "canonical_task"):
        raise ValueError(f"Unknown checkpoint memory key type: {key_type}")
    if config.denoising_steps < 1 or not 1 <= config.k <= 8:
        raise ValueError("denoising_steps must be positive; k must be 1..8")
    if not 1 <= config.port <= 65535 or config.cuda < 0:
        raise ValueError("Invalid server port or GPU index")
    if config.gpu_memory_fraction is not None and not 0 < config.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0,1]")
    return saved, key_type


def metadata_signature(metadata):
    values = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in metadata.items()
    }
    return digest(values)


def load_serving_memory(config, key_type, checkpoint_metadata, device):
    path = Path(config.memory_path)
    root = Path(config.retriever_path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if key_type == "canonical_task":
        if manifest is None:
            raise ValueError("Canonical-task serving requires the memory builder manifest")
        trained_metadata = torch.load(root / "metadata.pt", map_location="cpu", weights_only=False)
        if metadata_signature(checkpoint_metadata) != metadata_signature(trained_metadata):
            raise ValueError("Checkpoint normalization differs from the retriever/memory training normalization")
    if manifest is not None:
        if manifest.get("status") != "complete" or manifest["plan"]["is_smoke_sample"]:
            raise ValueError("Serving requires a complete, uncapped memory build")
        if manifest["plan"]["key_type"] != key_type:
            raise ValueError("Checkpoint and memory key types disagree")
        if manifest["config"]["data_config"] != config.data_config:
            raise ValueError("Memory and checkpoint data configs disagree")
        if path.stat().st_size != manifest["output"]["bytes"]:
            raise ValueError("Memory file size disagrees with manifest")
        for source in manifest["retriever"]:
            local = root / Path(source["path"]).name
            if file_fingerprint(local)["sha256"] != source["sha256"]:
                raise ValueError(f"Memory belongs to different retriever/normalization: {local}")
    memory = torch.load(path, map_location=device, weights_only=True)
    if not memory:
        raise ValueError("Memory contains no task banks")
    if manifest and set(memory) != {bank["key"] for bank in manifest["plan"]["banks"]}:
        raise ValueError("Memory task keys disagree with manifest")
    for key, bank in memory.items():
        count = len(bank["keys"])
        if count < config.k or any(
            len(bank[name]) != count
            for name in ("action", "pixel_values", "backbone_features", "backbone_attention_mask")
        ):
            raise ValueError(f"Invalid candidate counts for {key}")
        if bank["action"].shape[1:] != (16, 32):
            raise ValueError(f"Invalid padded action chunks for {key}")
        if bank["backbone_features"].shape[:2] != bank["backbone_attention_mask"].shape:
            raise ValueError(f"Invalid context feature masks for {key}")
        if manifest and count != manifest["candidate_counts"][key]:
            raise ValueError(f"Candidate count disagrees with manifest for {key}")
    demos = {bank["key"]: len(bank["demos"]) for bank in manifest["plan"]["banks"]} if manifest else {}
    sources = {
        "checkpoint_config": file_fingerprint(Path(config.checkpoint_path) / "config.json"),
        "checkpoint_metadata": file_fingerprint(Path(config.checkpoint_path) / "metadata.pt"),
        "checkpoint_weights": [stamp(file) for file in sorted(Path(config.checkpoint_path).glob("*.safetensors"))],
        "retriever_weights": [file_fingerprint(file) for file in sorted(root.glob("*.safetensors"))],
        "memory": stamp(path),
        "memory_manifest": file_fingerprint(manifest_path) if manifest else None,
    }
    info = {
        "protocol": "ravla",
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "retriever_path": str(root.resolve()),
        "memory_path": str(path.resolve()),
        "data_config": config.data_config,
        "memory_key_type": key_type,
        "memory_tasks": sorted(memory),
        "memory_demo_counts": demos,
        "memory_candidates": {key: len(bank["keys"]) for key, bank in memory.items()},
        "k": config.k,
        "denoising_steps": config.denoising_steps,
        "source_fingerprint": digest(sources),
        "sources": sources,
    }
    return memory, info
