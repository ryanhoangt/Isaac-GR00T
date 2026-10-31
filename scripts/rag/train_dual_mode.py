"""Train an opt-in dual-mode RA-VLA pilot on a Slurm GPU worker."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import tyro

from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.rag.data import FramePathsMixtureDataset
from gr00t.rag.dual_mode import DualModeRAVLA
from gr00t.rag.dual_mode_data import DualModeCollator, DualModeDataset, prepare_training_batch
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything
from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset


DATA_ROOT = "ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets"


@dataclass
class Args:
    output: Path
    dataset_paths: list[str] = field(
        default_factory=lambda: [f"{DATA_ROOT}/libero_{s}" for s in ("goal", "object", "spatial")]
    )
    retriever: Path = Path("ra-vla-official-assets/libero_gos")
    memory_name: str = "memory_gos"
    base_model: str = "nvidia/GR00T-N1.5-3B"
    variant: Literal["dual", "action"] = "dual"
    seed: int = 42
    batch_size: int = 32
    gradient_accumulation: int = 1
    num_workers: int = 2
    video_cache_size: int = 6
    num_epochs: int = 3
    max_steps: int = 0
    lookahead: int = 16
    negative_gap: int = 32
    margin: float = 0.01
    adherence_weight: float = 0.1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    save_steps: int = 1000
    log_steps: int = 10
    resume: Path | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_entity: str | None = None
    wandb_name: str | None = None


def load_data(args):
    config = LiberoDataConfig()
    datasets = [
        (
            BoundedLeRobotVideoDataset(
                dataset_path=path,
                modality_configs=config.modality_config(),
                transforms=config.transform(),
                embodiment_tag=EmbodimentTag("new_embodiment"),
                video_backend="torchvision_av",
                cache_size=args.video_cache_size,
            ),
            1.0,
        )
        for path in args.dataset_paths
    ]
    mixture = FramePathsMixtureDataset(
        data_mixture=datasets,
        mode="train",
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    # Actions in released memory are normalized with these exact statistics.
    mixture.load_metadata(str(args.retriever))
    return mixture


def check_worker():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run training with srun inside a GPU allocation; do not run on the login node")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA worker is required for pretrained RA-VLA training")


def main(args: Args):
    check_worker()
    for name in (
        "batch_size",
        "gradient_accumulation",
        "num_epochs",
        "save_steps",
        "log_steps",
        "lookahead",
        "negative_gap",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps < 0 or args.num_workers < 0:
        raise ValueError("max_steps and num_workers must be nonnegative")
    if len(args.dataset_paths) != 3 or {Path(p).name for p in args.dataset_paths} != {
        "libero_goal",
        "libero_object",
        "libero_spatial",
    }:
        raise ValueError("LOSO training requires exactly Goal/Object/Spatial; Long is held out")
    recipe = asdict(args)
    recipe.update(
        protocol_version=1,
        goal_probability=0.5 if args.variant == "dual" else 0.0,
        k=1,
        denoising_steps=4,
        effective_batch=args.batch_size * args.gradient_accumulation,
        frozen_vision=True,
        frozen_llm=True,
        retrieved_proprioception=False,
        goal_pairing="cross-demo same-task DTW + lookahead",
        negative="same-demo temporal gap",
        normalization="released GOS retriever metadata",
        wandb_project="icl-vla-uncertainty",
    )
    recipe = json.loads(json.dumps(recipe, default=str))
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "training_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
    else:
        saved = json.loads((args.output / "training_config.json").read_text())
        flexible = {"resume", "wandb_mode", "wandb_entity", "wandb_name", "num_workers"}
        if {k: v for k, v in saved.items() if k not in flexible} != {
            k: v for k, v in recipe.items() if k not in flexible
        }:
            raise ValueError("Resume recipe differs from the original run")
    run = None
    exit_code = 1
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity=args.wandb_entity,
                name=args.wandb_name or args.output.name,
                group="dual_mode_libero_long_loso",
                mode=args.wandb_mode,
                dir=str(args.output.resolve()),
                config=recipe,
            )
            run.define_metric("step")
            run.define_metric("train/*", step_metric="step")
        train(args, run)
        exit_code = 0
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def train(args, run):
    seed_everything(args.seed)
    mixture = load_data(args)
    mixture.save_metadata(str(args.output))
    dataset = DualModeDataset(
        mixture,
        seed=args.seed,
        goal_probability=0.5 if args.variant == "dual" else 0.0,
        lookahead=args.lookahead,
        negative_gap=args.negative_gap,
    )
    steps_per_epoch = math.ceil((len(dataset) // args.batch_size) / args.gradient_accumulation)
    total_steps = args.max_steps or args.num_epochs * steps_per_epoch
    if not steps_per_epoch or total_steps > args.num_epochs * steps_per_epoch:
        raise ValueError("Empty loader or max_steps exceeds the specified epoch budget")
    collator = DualModeCollator()
    device = torch.device("cuda:0")
    retriever = (
        RetrieverWrapperWithHead.from_pretrained(
            args.retriever,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        .eval()
        .requires_grad_(False)
    )
    memory = torch.load(
        args.retriever / f"{args.memory_name}.pt", map_location="cpu", mmap=True, weights_only=True
    )
    model = DualModeRAVLA.from_pretrained(
        str(args.resume) if args.resume is not None else args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        tune_llm=False,
        tune_visual=False,
        margin=args.margin,
        lambda_=args.adherence_weight,
    ).train()
    if args.resume is None:
        model.init_new_weights()
        torch.nn.init.normal_(model.action_head.goal_mode_token, std=0.02)
        torch.nn.init.normal_(model.action_head.goal_view_embedding, std=0.02)
    model.action_head.num_inference_timesteps = 4
    model.action_head.config.num_inference_timesteps = 4
    model.config.action_head_cfg = model.action_head.config.to_dict()
    model.config.backbone_cfg.update(tune_llm=False, tune_visual=False)
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Feature sharing requires a fully frozen backbone")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, total_steps)
    epoch_start, batch_start, step = 0, 0, 0
    if args.resume is not None:
        state = torch.load(args.resume / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch_start, batch_start, step = state["epoch"], state["next_batch"], state["step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    inventory = {
        str(d.dataset_path): {task: len(ids) for task, ids in d.task_groups.items()} for d in mixture.datasets
    }
    info = {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_frames": len(dataset),
        "dataset_counts": inventory,
        "trainable_parameters": sum(p.numel() for p in params),
    }
    (args.output / "data_inventory.json").write_text(json.dumps(info, indent=2) + "\n")
    if run is not None:
        run.config.update(info)
    print(json.dumps({"worker": os.uname().nodename, **info}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    last_saved = step if args.resume is not None else -1

    def save(epoch, next_batch):
        nonlocal last_saved
        directory = args.output / f"checkpoint-{step}"
        temporary = args.output / f".checkpoint-{step}.tmp"
        if directory.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {directory} or its temporary directory")
        mixture.save_metadata(str(temporary))
        model.save_pretrained(temporary)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_batch": next_batch,
                "step": step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            temporary / "trainer_state.pt",
        )
        (temporary / "training_config.json").write_text((args.output / "training_config.json").read_text())
        temporary.rename(directory)
        last_saved = step
        if run is not None:
            run.summary["last_checkpoint"] = str(directory.resolve())

    for epoch in range(epoch_start, args.num_epochs):
        if step >= total_steps:
            break
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=collator,
            drop_last=True,
            **({"prefetch_factor": 1} if args.num_workers else {}),
        )
        window_metrics = []
        for batch_index, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
            if epoch == epoch_start and batch_index < batch_start:
                continue
            # Correct scaling for an incomplete accumulation window at epoch end.
            window_start = batch_index // args.gradient_accumulation * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            inputs = prepare_training_batch(
                model, retriever, memory, batch, negative_seed=args.seed + epoch * len(loader) + batch_index
            )
            losses = model(inputs)
            if not torch.isfinite(losses.loss):
                raise FloatingPointError("Non-finite training loss")
            (losses.loss / window_size).backward()
            window_metrics.append({key: float(value.detach()) for key, value in losses.items()})
            if (batch_index + 1) % args.gradient_accumulation and batch_index + 1 != len(loader):
                continue
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("Non-finite gradient norm")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {
                key: sum(row[key] for row in window_metrics) / len(window_metrics) for key in window_metrics[0]
            }
            for mode in ("action", "goal"):
                count = sum(row[f"{mode}_count"] for row in window_metrics)
                for suffix in ("mse", "negative_mse", "margin"):
                    key = f"{mode}_{suffix}"
                    metrics[key] = sum(row[key] * row[f"{mode}_count"] for row in window_metrics) / max(count, 1)
                metrics[f"{mode}_count"] = count
            metrics.update(learning_rate=lr, grad_norm=float(grad), epoch=epoch + (batch_index + 1) / len(loader))
            window_metrics.clear()
            if step == 1 or step % args.log_steps == 0 or step == total_steps:
                print(json.dumps({"step": step, **metrics}), flush=True)
                if run is not None:
                    run.log({"step": step, **{f"train/{key}": value for key, value in metrics.items()}})
            if step == 1:
                (args.output / "pairing_examples.json").write_text(
                    json.dumps(batch["provenance"], indent=2) + "\n"
                )
            if step % args.save_steps == 0:
                save(epoch, batch_index + 1)
            if step >= total_steps:
                if last_saved != step:
                    save(epoch, batch_index + 1)
                break
        batch_start = 0
    if step and last_saved != step:
        save(args.num_epochs, 0)
    completed_path = args.output / "completed.json"
    previous = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    completed_path.write_text(
        json.dumps(
            {
                "steps": step,
                "gpu_peak_gib": max(previous.get("gpu_peak_gib", 0), torch.cuda.max_memory_allocated() / 1024**3),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
