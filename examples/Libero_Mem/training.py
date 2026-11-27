"""Validated two-fold baseline launch plans using the existing GR00T fine-tuner."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from examples.Libero_Mem.selection import REPO, file_hash, write_summary
from examples.Libero_Mem.tasks import TASK_BY_NAME, get_fold
from scripts.eval_libero_rollout import complete_checkpoint


DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
DEFAULT_BASE = Path(
    "/mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/"
    "869830fc749c35f34771aa5209f923ac57e4564e"
)


@dataclass(frozen=True)
class TrainingConfig:
    output_root: Path
    datasets_root: Path = Path("/mnt/data/vhoangth2/datasets")
    auxiliary_root: Path = REPO / "demo_data/libero_loso_clean"
    base_model: Path = DEFAULT_BASE
    folds: tuple[str, ...] = ("A", "B")
    mode: str = "cotrain"
    seed: int = 42
    max_steps: int = 10000
    save_steps: int = 1000
    save_total_limit: int = 2
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    report_to: str = "tensorboard"
    python: str = sys.executable
    dry_run: bool = False


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dataset(path, *, fold=None):
    path = path.expanduser().resolve(strict=True)
    names = ["meta/info.json", "meta/modality.json", "meta/stats.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
    info = json.loads((path / names[0]).read_text())
    modality = json.loads((path / names[1]).read_text())
    if modality != json.loads((REPO / "examples/Libero/modality.json").read_text()):
        raise ValueError(f"Dataset must use the existing LIBERO modality: {path}")
    if info["codebase_version"] != "v2.1" or info["fps"] != 20:
        raise ValueError(f"Expected LeRobot v2.1 at 20 FPS: {path}")
    for key, size in (("observation.state", 8), ("action", 7)):
        if info["features"][key]["shape"] != [size] or info["features"][key]["dtype"] != "float32":
            raise ValueError(f"Invalid LIBERO state/action feature: {path}/{key}")
    tasks = _lines(path / "meta/tasks.jsonl")
    if fold is None and "source_task_id" in info["features"]:
        raise ValueError(f"Auxiliary input must be standard LIBERO, not a LIBERO-Mem dataset: {path}")
    episodes = _lines(path / "meta/episodes.jsonl")
    if len(tasks) != info["total_tasks"] or len(episodes) != info["total_episodes"]:
        raise ValueError(f"Dataset inventory disagrees with metadata: {path}")
    if sum(episode["length"] for episode in episodes) != info["total_frames"]:
        raise ValueError(f"Dataset frame count disagrees with metadata: {path}")
    for episode in episodes:
        index = episode["episode_index"]
        values = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        if not (path / info["data_path"].format(**values)).is_file():
            raise FileNotFoundError(f"Missing parquet in {path}, episode {index}")
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                if not (path / info["video_path"].format(**values, video_key=key)).is_file():
                    raise FileNotFoundError(f"Missing video in {path}, episode {index}, {key}")
    if fold:
        names += ["manifest.json", "meta/episode_mapping.json", "meta/episodes_stats.jsonl"]
        manifest = json.loads((path / "manifest.json").read_text())
        expected = get_fold(fold)
        if (
            manifest.get("extractor") != "examples.Libero_Mem.splits"
            or manifest.get("status") != "complete"
            or manifest.get("training_fold") != fold
            or manifest.get("normalization_scope") != "selected_training_rows_only"
            or manifest.get("episodes") != len(episodes)
            or manifest.get("frames") != info["total_frames"]
            or any(manifest.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(f"Invalid or mismatched training fold: {path}")
        instructions = {TASK_BY_NAME[name].instruction for name in expected["train_tasks"]}
        mapping = json.loads((path / "meta/episode_mapping.json").read_text())
        if isinstance(mapping, dict):
            mapping = mapping["episodes"]
        if (
            {task["task"] for task in tasks} != instructions
            or len(mapping) != len(episodes)
            or {entry["task"] for entry in mapping} != set(expected["train_tasks"])
        ):
            raise ValueError(f"Training fold contains unexpected tasks/episodes: {path}")
    return {
        "path": str(path),
        "episodes": len(episodes),
        "frames": info["total_frames"],
        "tasks": [task["task"] for task in tasks],
        "role": "libero_mem" if fold else "auxiliary_libero",
        "metadata_sha256": {name: file_hash(path / name) for name in names},
    }


def training_plan(config):
    if (
        not config.folds
        or len(set(config.folds)) != len(config.folds)
        or any(fold not in ("A", "B") for fold in config.folds)
    ):
        raise ValueError("folds must be distinct A/B values")
    if config.mode not in ("cotrain", "mem-only") or config.report_to not in ("tensorboard", "wandb"):
        raise ValueError("Unsupported training mode or reporting backend")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    for name in (
        "max_steps",
        "save_steps",
        "save_total_limit",
        "batch_size",
        "gradient_accumulation_steps",
        "dataloader_prefetch_factor",
    ):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.dataloader_num_workers) is not int or config.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be nonnegative")
    if config.max_steps % config.save_steps:
        raise ValueError("max_steps must be divisible by save_steps to save the final update")
    if config.mode == "cotrain" and config.batch_size % 2:
        raise ValueError("Co-training requires an even batch_size for exact 50/50 batches")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0 or not 0 <= config.warmup_ratio <= 1:
        raise ValueError("Invalid weight_decay or warmup_ratio")
    output = config.output_root.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    base = config.base_model.expanduser().resolve(strict=True)
    if not complete_checkpoint(base):
        raise ValueError(f"Base checkpoint is incomplete: {base}")
    if output.resolve().is_relative_to(base):
        raise ValueError("Training output must be outside the base checkpoint")
    auxiliary = []
    if config.mode == "cotrain":
        auxiliary = [_dataset(config.auxiliary_root / suite) for suite in ("spatial", "object", "goal", "long")]
    plans = []
    for fold in config.folds:
        datasets = [
            _dataset(config.datasets_root / f"libero_mem_fold_{fold.lower()}_train", fold=fold),
            *auxiliary,
        ]
        if any(output.resolve().is_relative_to(Path(dataset["path"])) for dataset in datasets):
            raise ValueError("Training output must be outside input datasets")
        weights = [1.0, *([0.25] * 4 if auxiliary else [])]
        command = [
            config.python,
            str(REPO / "scripts/gr00t_finetune.py"),
            "--base-model-path",
            str(base),
            "--dataset-path",
            *[dataset["path"] for dataset in datasets],
            "--data-config",
            DATA_CONFIG,
            "--embodiment-tag",
            "new_embodiment",
            "--num-gpus",
            "1",
            "--seed",
            str(config.seed),
            "--batch-size",
            str(config.batch_size),
            "--gradient-accumulation-steps",
            str(config.gradient_accumulation_steps),
            "--max-steps",
            str(config.max_steps),
            "--save-strategy",
            "steps",
            "--save-steps",
            str(config.save_steps),
            "--save-total-limit",
            str(config.save_total_limit),
            "--no-save-final-model",
            "--dataloader-num-workers",
            str(config.dataloader_num_workers),
            "--dataloader-prefetch-factor",
            str(config.dataloader_prefetch_factor),
            "--learning-rate",
            str(config.learning_rate),
            "--weight-decay",
            str(config.weight_decay),
            "--warmup-ratio",
            str(config.warmup_ratio),
            "--video-backend",
            "torchvision_av",
            "--no-tune-llm",
            "--no-tune-visual",
            "--tune-projector",
            "--tune-diffusion-model",
            "--report-to",
            config.report_to,
            "--output-dir",
            str(output / f"fold_{fold.lower()}"),
        ]
        if auxiliary:
            command += [
                "--dataset-weights",
                *map(str, weights),
                "--no-balance-dataset-weights",
                "--balanced-primary-batches",
            ]
        plans.append(
            {
                "training_fold": fold,
                **get_fold(fold),
                "datasets": datasets,
                "dataset_weights": weights,
                "dataset_probabilities": [weight / sum(weights) for weight in weights],
                "balanced_primary_batches": bool(auxiliary),
                "command": command,
                "normalization_scope": "training_fold_and_auxiliary_mixture"
                if auxiliary
                else "training_fold_only",
                "percentile_mixing_method": "weighted_average",
                "base_model": str(base),
                "base_config_sha256": file_hash(base / "config.json"),
                "data_config": DATA_CONFIG,
                "embodiment_tag": "new_embodiment",
            }
        )
    return plans


def _checkpoints(path, max_steps):
    metadata = path / "experiment_cfg/metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing normalization metadata: {metadata}")
    found = []
    for checkpoint in sorted(path.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[-1])):
        if not complete_checkpoint(checkpoint):
            raise ValueError(f"Incomplete training checkpoint: {checkpoint}")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if file_hash(checkpoint / "experiment_cfg/metadata.json") != file_hash(metadata):
            raise ValueError(f"Checkpoint normalization differs from the training run: {checkpoint}")
        found.append({"path": str(checkpoint), "global_step": state["global_step"], "epoch": state["epoch"]})
    if not found or found[-1]["global_step"] != max_steps:
        raise ValueError("Final optimizer update was not saved")
    return found


def run_training(config):
    plans = training_plan(config)
    if config.dry_run:
        return {"status": "dry_run", "folds": plans}
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        USE_TF="0",
        NO_ALBUMENTATIONS_UPDATE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_MODULES_CACHE=str(REPO / ".cache/huggingface/modules"),
    )
    env.pop("TRANSFORMERS_CACHE", None)
    subprocess.run(
        [config.python, "-c", "import torch; assert torch.cuda.is_available(), 'Run inside a GPU allocation'"],
        env=env,
        cwd=REPO,
        check=True,
    )
    root = config.output_root.expanduser().absolute()
    root.mkdir(parents=True)
    summary = {"status": "running", "folds": []}
    write_summary(root / "summary.json", summary)
    manifest = None
    try:
        for plan in plans:
            manifest = None
            fold = plan["training_fold"]
            output = root / f"fold_{fold.lower()}"
            record = {"training_fold": fold, "status": "running", "output_dir": str(output)}
            summary["folds"].append(record)
            write_summary(root / "summary.json", summary)
            output.mkdir()
            for dataset in plan["datasets"]:
                for name, digest in dataset["metadata_sha256"].items():
                    if file_hash(Path(dataset["path"]) / name) != digest:
                        raise ValueError("Dataset metadata changed after training planning")
            manifest = {
                "status": "running",
                "method": "vanilla_gr00t",
                **plan,
                "recipe": {
                    key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
                },
                "seed": config.seed,
                "data_seed": config.seed,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
                "compute_dtype": "bfloat16",
                "tf32": True,
            }
            write_summary(output / "run_manifest.json", manifest)
            print(f"Training fold {fold}; log: {output / 'train.log'}", flush=True)
            with (output / "train.log").open("w") as log:
                subprocess.run(
                    plan["command"], env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            checkpoints = _checkpoints(output, config.max_steps)
            write_summary(output / "checkpoints.json", checkpoints)
            manifest["status"] = record["status"] = "completed"
            record["checkpoints"] = checkpoints
            write_summary(output / "run_manifest.json", manifest)
            write_summary(root / "summary.json", summary)
        summary["status"] = "completed"
        write_summary(root / "summary.json", summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if summary["folds"]:
            summary["folds"][-1]["status"] = "failed"
        if manifest is not None:
            manifest["status"] = "failed"
            manifest["error"] = summary["error"]
            write_summary(output / "run_manifest.json", manifest)
        write_summary(root / "summary.json", summary)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=TrainingConfig.datasets_root)
    parser.add_argument("--auxiliary-root", type=Path, default=TrainingConfig.auxiliary_root)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--folds", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--mode", choices=("cotrain", "mem-only"), default="cotrain")
    for name in (
        "seed",
        "max_steps",
        "save_steps",
        "save_total_limit",
        "batch_size",
        "gradient_accumulation_steps",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(TrainingConfig, name))
    for name in ("learning_rate", "weight_decay", "warmup_ratio"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(TrainingConfig, name))
    parser.add_argument("--report-to", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    args["folds"] = tuple(args["folds"])
    print(json.dumps(run_training(TrainingConfig(**args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
