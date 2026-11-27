"""Memory/cache validation and reproducible training sidecars for GR00T."""

from dataclasses import asdict
import json
import os
from pathlib import Path

import yaml
from transformers.trainer_utils import get_last_checkpoint

from examples.RoboMME.feature_cache import file_sha256, json_fingerprint, write_json_atomic
from examples.RoboMME.memory_dataset import RoboMMEMemoryDataset
from examples.RoboMME.visual_features import memory_vision_signature, validate_memory_vision_signature


MEMORY_DATA_CONFIG = "examples.RoboMME.memory_data_config:RoboMMEMemoryDataConfig"
VANILLA_DATA_CONFIG = "examples.RoboMME.custom_data_config:RoboMMEDataConfig"


def effective_data_config(args):
    """Validate opt-in flags before reading data or loading model weights."""
    if bool(args.memory_config) != bool(args.memory_cache):
        raise ValueError("--memory-config and --memory-cache must be specified together")
    if args.memory_config:
        if args.tune_visual:
            raise ValueError("Cached memory requires frozen vision/connector: use --no-tune-visual")
        if args.lora_rank:
            raise ValueError("FrameSamp+Modul training currently uses full action-head tuning, without LoRA")
        if not args.tune_diffusion_model:
            raise ValueError(
                "Memory training requires --tune-diffusion-model so the new modules receive gradients"
            )
        if args.data_config in ("fourier_gr1_arms_only", VANILLA_DATA_CONFIG):
            return MEMORY_DATA_CONFIG
    elif args.data_config == MEMORY_DATA_CONFIG:
        raise ValueError("RoboMME memory data config requires --memory-config and --memory-cache")
    return args.data_config


def resolve_memory_checkpoint(args):
    """Pin the checkpoint to be validated, loaded and passed to Trainer.train."""
    if not args.resume:
        output = Path(args.output_dir)
        if output.exists() and (
            get_last_checkpoint(str(output))
            or (output / "experiment_cfg/metadata.json").exists()
            or (output / "experiment_cfg/memory_training.json").exists()
            or (output / "config.json").exists()
        ):
            raise FileExistsError("Training output already exists; choose a fresh directory or use --resume")
        return None
    if not Path(args.output_dir).is_dir():
        raise ValueError("--resume requires an existing memory training checkpoint")
    checkpoint = get_last_checkpoint(args.output_dir)
    if checkpoint is None:
        raise ValueError("No checkpoint found for memory training resume")
    if not (Path(checkpoint) / "experiment_cfg/memory_training.json").is_file():
        raise ValueError("Resume checkpoint has no memory training provenance")
    return str(Path(checkpoint).resolve())


def memory_datasets(dataset):
    datasets = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    if not datasets or not all(isinstance(item, RoboMMEMemoryDataset) for item in datasets):
        raise ValueError("Memory training requires RoboMMEMemoryDataset for every mixture component")
    return datasets


def verify_memory_cache(model, dataset):
    """Check the loaded model's actual frozen weights, including on resume."""
    eagle = model.backbone.eagle_model
    if any(
        parameter.requires_grad for module in (eagle.vision_model, eagle.mlp1) for parameter in module.parameters()
    ):
        raise ValueError("Memory cache cannot be used with trainable vision/connector weights")
    for item in memory_datasets(dataset):
        cached = item.memory_encoder_signature
        actual = memory_vision_signature(
            eagle, item.memory_config, compute_dtype=cached["compute_dtype"], device_type=cached["device_type"]
        )
        validate_memory_vision_signature(actual, cached, mismatch_context="Loaded model and memory cache differ")



def memory_run_record(args, dataset, memory_config, data_config, context):
    components = []
    for item in memory_datasets(dataset):
        # Bind normalization, language/task selection and row provenance without
        # hashing all video/feature files again on every resume.
        meta = item.dataset_path / "meta"
        hashes = {
            path.name: file_sha256(path)
            for path in sorted(meta.glob("*"))
            if path.is_file() and path.suffix in (".json", ".jsonl")
        }
        manifest = item.dataset_path / "manifest.json"
        if manifest.is_file():
            hashes["../manifest.json"] = file_sha256(manifest)
        components.append(
            {
                "dataset_path": str(item.dataset_path.resolve()),
                "metadata_sha256": hashes,
                "samples": len(item),
                "normalization": item.metadata.model_dump(mode="json"),
                "cache_path": str(item.memory_cache),
                "cache_encoder_sha256": item.memory_encoder_sha256,
                "cache_manifest_sha256": file_sha256(item.memory_cache / "manifest.json"),
                "source_episode_indices": sorted(item._cache_records),
            }
        )
    launch = asdict(args)
    # Operational controls may change when resuming (e.g. extending max_steps).
    # Optimization, sampling, precision and data identity remain fixed.
    operational = {
        "output_dir",
        "resume",
        "max_steps",
        "num_train_epochs",
        "save_steps",
        "save_strategy",
        "save_total_limit",
        "save_final_model",
        "report_to",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
        "memory_max_open_episodes",
        "memory_config",
        "memory_cache",
        "dataset_path",
        "data_config",
        "dataloader_multiprocessing_context",
    }
    identity = {
        "version": 1,
        "memory_config": memory_config,
        "datasets": components,
        "data_config": data_config,
        "dataloader_multiprocessing_context": context,
        "training": {key: value for key, value in launch.items() if key not in operational},
    }
    # Embed extraction details outside the existing resume identity so old
    # step-5 runs can resume unchanged and new checkpoints serve without cache files.
    cache_specs = []
    for item, component in zip(memory_datasets(dataset), components):
        spec = {
            "encoder": item.memory_encoder_signature,
            "feature_spec": item._cache_manifest["feature_spec"],
            "cache_manifest_sha256": component["cache_manifest_sha256"],
        }
        cache_specs.append({**spec, "spec_sha256": json_fingerprint(spec)})
    return {
        "identity": identity,
        "identity_sha256": json_fingerprint(identity),
        "launch_config": launch,
        "cache_specs": cache_specs,
    }


def prepare_memory_run(args, dataset, memory_config, data_config, context, checkpoint):
    """Reject incompatible resume before rewriting any output sidecars."""
    record = memory_run_record(args, dataset, memory_config, data_config, context)
    output = Path(args.output_dir) / "experiment_cfg"
    if checkpoint is not None:
        saved_path = Path(checkpoint) / "experiment_cfg/memory_training.json"
        saved = json.loads(saved_path.read_text())
        if (
            saved.get("identity_sha256") != json_fingerprint(saved["identity"])
            or saved["identity_sha256"] != record["identity_sha256"]
        ):
            raise ValueError("Memory resume configuration/data/cache differ from the saved checkpoint")
        root_path = output / "memory_training.json"
        if (
            root_path.exists()
            and json.loads(root_path.read_text()).get("identity_sha256") != record["identity_sha256"]
        ):
            raise ValueError("Output memory provenance differs from the resume checkpoint")
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "memory_training.json", record)
        (output / "memory_config.yaml").write_text(yaml.safe_dump(memory_config, sort_keys=False))
    return record
