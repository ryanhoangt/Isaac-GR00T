"""Train RA-VLA with validated demo memory and optional bounded step runs."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import List, Literal

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import tyro

from gr00t.data.dataset import CachedLeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.rag.data import FramePathsMixtureDataset
from gr00t.rag.ravla import RAVLADataCollator, RAVLA
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.training import load_training_memory, resolve_policy_config, set_training_metadata
from gr00t.rag.utils import seed_everything, get_elapsed_time, log_message
from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset

EMBODIMENT_TAG = "new_embodiment"
VIDEO_BACKEND = "torchvision_av"


@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0
    dataset_paths: List[str] = field(default_factory=list)
    retriever_path: str = "models/retriever/default"
    memory_name: str = "memory"
    memory_path: str | None = None
    data_config: str | None = None
    pretrained_model_path: str | None = None
    task_map: str | None = None
    vla_path: str = "models/ravla/default"

    k: int = 1
    margin: float = 0.01
    lambda_: float = 0.1
    batch_size: int = 32
    num_epochs: int = 3
    max_steps: int | None = None
    """Positive optimizer-step budget overriding num_epochs; useful for smoke tests."""
    num_workers: int = 2
    cache_video: bool = False
    video_cache_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_norm: float = 1.0
    save_start_epoch: int = 1
    save_interval: int = 1

    wandb_enable: bool = True
    wandb_project: str = "icl-vla-uncertainty"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] | None = None
    log_freq: int = 20


def main(config: ArgsConfig):
    if config.log_freq < 1:
        raise ValueError("log_freq must be positive")
    if (
        not config.dataset_paths
        or min(
            config.batch_size,
            config.num_epochs,
            config.video_cache_size,
            config.save_start_epoch,
            config.save_interval,
        )
        < 1
    ):
        raise ValueError("Supply datasets and positive batch/epoch/cache/save counts")
    if config.num_workers < 0 or config.cuda < 0 or not 1 <= config.k <= 8:
        raise ValueError("Invalid workers/GPU index; k must be between 1 and 8")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.batch_size < 2 and config.lambda_ > 0:
        raise ValueError("Adherence training needs batch_size >= 2 for context replacement")
    output = Path(config.vla_path)
    if (output / "training_config.json").exists() or list(output.glob("checkpoint*")):
        raise FileExistsError("Training output already exists; choose a fresh --vla-path")
    resolve_policy_config(config)
    output.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "base_model": config.pretrained_model_path,
        "embodiment_tag": EMBODIMENT_TAG,
        "video_backend": VIDEO_BACKEND,
    }
    (output / "training_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    run = None
    exit_code = 1
    try:
        if config.wandb_enable:
            import wandb

            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_name or os.environ.get("WANDB_NAME") or output.name,
                group=config.wandb_group,
                mode=config.wandb_mode,
                dir=str(output.resolve()),
                config=run_config,
            )
            run.define_metric("global_step")
            run.define_metric("train/*", step_metric="global_step")
            run.define_metric("epoch")
            run.define_metric("epoch/*", step_metric="epoch")
        result = _train(config, run)
        exit_code = 0
        return result
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def _train(config: ArgsConfig, run=None):
    # Logging initialization must not shift the training RNG stream.
    seed_everything(config.seed)
    started = perf_counter()
    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    data_config_cls = load_data_config(config.data_config)
    dataset_class = CachedLeRobotSingleDataset if config.cache_video else BoundedLeRobotVideoDataset
    dataset_kwargs = {} if config.cache_video else {"cache_size": config.video_cache_size}
    data_mixture = []
    for path in config.dataset_paths:
        dataset = dataset_class(
            dataset_path=path,
            modality_configs=data_config_cls.modality_config(),
            transforms=data_config_cls.transform(),
            embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
            video_backend=VIDEO_BACKEND,
            **dataset_kwargs,
        )
        data_mixture.append((dataset, 1.0))
    mixture = FramePathsMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        metadata_config={"percentile_mixing_method": "weighted_average"},
        memory_task_keys=True,
        task_map=config.task_map,
    )
    set_training_metadata(mixture, config.retriever_path)
    mixture.save_metadata(config.vla_path)
    canonical = mixture.memory_key_type == "canonical_task"
    collator = RAVLADataCollator(require_memory_key=canonical)
    dataloader = DataLoader(
        mixture,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        drop_last=True,
        **({"prefetch_factor": 1} if config.num_workers else {}),
    )
    if len(dataloader) == 0:
        raise ValueError("No training batches; reduce batch_size or supply more data")
    total_steps = config.max_steps or config.num_epochs * len(dataloader)
    epochs = math.ceil(total_steps / len(dataloader))
    memory, memory_report = load_training_memory(config, mixture, device)
    output = Path(config.vla_path)
    (output / "memory_validation.json").write_text(json.dumps(memory_report, indent=2) + "\n")
    retriever = (
        RetrieverWrapperWithHead.from_pretrained(
            config.retriever_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        .eval()
        .requires_grad_(False)
    )
    vla = RAVLA.from_pretrained(
        config.pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        margin=config.margin,
        lambda_=config.lambda_,
        mask_padding=canonical,
    ).train()
    vla.backbone.eval()
    vla.set_retrieval_components(retriever, memory, config.k)
    vla.init_new_weights()
    vla.compute_dtype = vla.config.compute_dtype = "bfloat16"
    vla.config.ravla_memory_key_type = mixture.memory_key_type
    vla.config.ravla_data_config = config.data_config

    parameters = [p for p in vla.parameters() if p.requires_grad]
    trainable = [(name, p) for name, p in vla.named_parameters() if p.requires_grad]
    tracked = next(
        ((name, p) for name, p in trainable if name.endswith("ret_action_encoder.weight")), trainable[0]
    )
    initial_parameter = tracked[1].detach().clone()
    retriever_sample = next(retriever.parameters())
    initial_retriever = retriever_sample.detach().clone()
    backbone_sample = next(vla.backbone.parameters())
    initial_backbone = backbone_sample.detach().clone()
    if any(p.requires_grad for p in vla.backbone.parameters()):
        raise ValueError("Cached memory requires a frozen policy backbone")
    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, total_steps)
    if run is not None:
        run.config.update(
            {
                "steps_per_epoch": len(dataloader),
                "total_steps": total_steps,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "device": str(device),
                "memory_key_type": mixture.memory_key_type,
            }
        )
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")
    global_step, completed_epochs = 0, 0
    checkpoints = []
    last_values = {}
    print(
        f"Training {total_steps} optimizer steps; key_type={mixture.memory_key_type}; "
        f"memory_candidates={memory_report['candidates']}; batch_size={config.batch_size}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        start_time = perf_counter()
        sums = {"loss": 0.0, "mse_loss": 0.0, "margin_loss": 0.0}
        batches = 0
        for batch_index, inputs in enumerate(tqdm(dataloader), start=1):
            if global_step == 0 and isinstance(inputs, dict):
                (output / "first_batch.json").write_text(
                    json.dumps(
                        {
                            "tasks": inputs["tasks"],
                            "action_shape": list(inputs["action"].shape),
                            "state_shape": list(inputs["state"].shape),
                            "action_dimensions": inputs["action_mask"][0].any(0).nonzero().flatten().tolist(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            losses = vla(inputs)
            values = {"loss": losses.loss, "mse_loss": losses.mse_loss, "margin_loss": losses.margin_loss}
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
            learning_rate = optimizer.param_groups[0]["lr"]
            losses.loss.backward()
            grad_norm = clip_grad_norm_(parameters, config.max_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            batches += 1
            last_values = {key: float(value.detach()) for key, value in values.items()}
            for key, value in last_values.items():
                sums[key] += value
            record = {
                "step": global_step,
                "epoch": epoch,
                **last_values,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
            }
            with metrics_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            last_batch = batch_index == len(dataloader) or global_step == total_steps
            if run is not None and (global_step == 1 or global_step % config.log_freq == 0 or last_batch):
                run.log(
                    {
                        "global_step": global_step,
                        "train/epoch": epoch - 1 + batch_index / len(dataloader),
                        **{f"train/{key}": value for key, value in last_values.items()},
                        "train/learning_rate": learning_rate,
                        "train/grad_norm": float(grad_norm),
                    }
                )
            if global_step >= total_steps:
                break
        full_epoch = batches == len(dataloader)
        completed_epochs += int(full_epoch)
        elapsed = perf_counter() - start_time
        means = {key: value / batches for key, value in sums.items()}
        log_message(
            f"[Epoch {epoch}/{epochs}] Loss: {means['loss']:.4f} | "
            f"{get_elapsed_time(start_time, perf_counter())} | steps={global_step}",
            config.vla_path,
        )
        if run is not None:
            run.log(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    **{f"epoch/{key}": value for key, value in means.items()},
                    "epoch/seconds": elapsed,
                }
            )
            run.summary.update({"completed_epochs": completed_epochs, "global_step": global_step})
        scheduled = full_epoch and epoch >= config.save_start_epoch and epoch % config.save_interval == 0
        if scheduled or global_step == total_steps:
            name = f"checkpoint_{epoch}" if full_epoch else f"checkpoint_step_{global_step}"
            checkpoint = output / name
            mixture.save_metadata(str(checkpoint))
            vla.save_pretrained(str(checkpoint))
            checkpoints.append(str(checkpoint.resolve()))
            if run is not None:
                run.summary["last_checkpoint"] = checkpoints[-1]
        if global_step >= total_steps:
            break
    report = {
        "optimizer_steps": global_step,
        "completed_epochs": completed_epochs,
        "partial_epoch": not full_epoch,
        "final_losses": last_values,
        "checkpoints": checkpoints,
        "tracked_parameter": tracked[0],
        "parameter_max_abs_change": float((tracked[1].detach() - initial_parameter).abs().max()),
        "retriever_frozen": not any(p.requires_grad for p in retriever.parameters()),
        "retriever_has_gradients": any(p.grad is not None for p in retriever.parameters()),
        "retriever_sample_unchanged": torch.equal(retriever_sample.detach(), initial_retriever),
        "backbone_frozen": not any(p.requires_grad for p in vla.backbone.parameters()),
        "backbone_has_gradients": any(p.grad is not None for p in vla.backbone.parameters()),
        "backbone_sample_unchanged": torch.equal(backbone_sample.detach(), initial_backbone),
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else None,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda"
        else None,
        "seconds": perf_counter() - started,
    }
    (output / "completed.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
