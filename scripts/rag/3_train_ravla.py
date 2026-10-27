import json
import os
from pathlib import Path
from typing import List, Literal
from dataclasses import asdict, dataclass, field
from time import perf_counter
import tyro
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from gr00t.experiment.data_config import load_data_config
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import CachedLeRobotSingleDataset
from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset

from gr00t.rag.data import FramePathsMixtureDataset
from gr00t.rag.ravla import RAVLADataCollator, RAVLA
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything, get_elapsed_time, log_message

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"
VIDEO_BACKEND = "torchvision_av"
PRETRAINED_VLA_PATH = "nvidia/GR00T-N1.5-3B"

@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0

    dataset_paths: List[str] = field(default_factory=list)
    retriever_path: str = "models/retriever/default"
    memory_name: str = "memory"
    vla_path: str = "models/ravla/default"

    k: int = 1
    margin: float = 0.01
    lambda_: float = 0.1

    batch_size: int = 32
    num_epochs: int = 3
    num_workers: int = 16
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
    if config.video_cache_size < 1:
        raise ValueError("video_cache_size must be positive")
    output = Path(config.vla_path)
    output.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "base_model": PRETRAINED_VLA_PATH,
        "data_config": DATA_CONFIG,
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
        _train(config, run)
        exit_code = 0
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def _train(config: ArgsConfig, run=None):
    # Initialize logging before seeding so SDK initialization cannot shift training RNGs.
    seed_everything(config.seed)
    device = torch.device(f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    data_config_cls = load_data_config(DATA_CONFIG)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(EMBODIMENT_TAG)

    dataset_class = CachedLeRobotSingleDataset if config.cache_video else BoundedLeRobotVideoDataset
    video_mode = "full RAM cache" if config.cache_video else f"bounded cache ({config.video_cache_size} videos/dataset/worker)"
    print(f"Video loading: {video_mode}", flush=True)
    dataset_kwargs = {} if config.cache_video else {"cache_size": config.video_cache_size}
    data_mixture = []
    for dataset_path in config.dataset_paths:
        dataset = dataset_class(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=VIDEO_BACKEND,
            **dataset_kwargs,
        )
        data_mixture.append((dataset, 1.0))

    mixture_dataset = FramePathsMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    mixture_dataset.save_metadata(config.vla_path)

    collator = RAVLADataCollator()

    dataloader = DataLoader(
        mixture_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    retriever = RetrieverWrapperWithHead.from_pretrained(
        config.retriever_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    memory = torch.load(f"{config.retriever_path}/{config.memory_name}.pt", map_location=device)

    vla = RAVLA.from_pretrained(
        PRETRAINED_VLA_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        margin=config.margin,
        lambda_=config.lambda_,
    ).train()
    vla.set_retrieval_components(retriever, memory, config.k)
    vla.init_new_weights()

    vla.compute_dtype = "bfloat16"
    vla.config.compute_dtype = "bfloat16"

    parameters = [p for p in vla.parameters() if p.requires_grad]
    num_iterations = config.num_epochs * len(dataloader)
    if len(dataloader) == 0:
        raise ValueError("No training batches; reduce batch_size or supply more data")
    if run is not None:
        run.config.update({
            "steps_per_epoch": len(dataloader),
            "total_steps": num_iterations,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "device": str(device),
        })

    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, num_iterations)
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        start_time = perf_counter()
        train_mse_loss, train_margin_loss, train_loss = 0, 0, 0

        for batch_index, inputs in enumerate(tqdm(dataloader), start=1):
            losses = vla(inputs)
            mse_loss, margin_loss, loss = losses.mse_loss, losses.margin_loss, losses.loss

            learning_rate = optimizer.param_groups[0]["lr"]
            loss.backward()
            grad_norm = clip_grad_norm_(parameters, config.max_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            mse_value, margin_value, loss_value = mse_loss.item(), margin_loss.item(), loss.item()
            train_mse_loss += mse_value
            train_margin_loss += margin_value
            train_loss += loss_value
            if run is not None and (
                global_step == 1 or global_step % config.log_freq == 0 or batch_index == len(dataloader)
            ):
                run.log({
                    "global_step": global_step,
                    "train/epoch": epoch - 1 + batch_index / len(dataloader),
                    "train/loss": loss_value,
                    "train/mse_loss": mse_value,
                    "train/margin_loss": margin_value,
                    "train/learning_rate": learning_rate,
                    "train/grad_norm": float(grad_norm),
                })

        train_mse_loss /= len(dataloader)
        train_margin_loss /= len(dataloader)
        train_loss /= len(dataloader)

        end_time = perf_counter()
        elapsed_time = get_elapsed_time(start_time, end_time)

        message = (
            f"[Epoch {epoch:2}/{config.num_epochs}] "
            f"Loss: {train_loss:6.4f} | {elapsed_time}"
        )
        log_message(message, config.vla_path)
        if run is not None:
            metrics = {
                "global_step": global_step,
                "epoch": epoch,
                "epoch/loss": train_loss,
                "epoch/mse_loss": train_mse_loss,
                "epoch/margin_loss": train_margin_loss,
                "epoch/seconds": end_time - start_time,
            }
            if device.type == "cuda":
                metrics["epoch/gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            run.log(metrics)
            run.summary.update({"completed_epochs": epoch, "global_step": global_step})

        if epoch >= config.save_start_epoch and epoch % config.save_interval == 0:
            checkpoint_path = f"{config.vla_path}/checkpoint_{epoch}"
            mixture_dataset.save_metadata(checkpoint_path)
            vla.save_pretrained(checkpoint_path)
            if run is not None:
                run.summary["last_checkpoint"] = str(Path(checkpoint_path).resolve())


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
