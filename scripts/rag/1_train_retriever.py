from typing import List
from dataclasses import dataclass, field, asdict
from time import perf_counter
import json
from pathlib import Path
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
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.gr00t_n1 import GR00T_N1_5

from gr00t.rag.data import FramePathsMixtureDataset, AlignedFramesBatchSampler
from gr00t.rag.retriever import get_config, RetrieverWrapperWithHead
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
    data_config: str = DATA_CONFIG
    pretrained_model_path: str = PRETRAINED_VLA_PATH

    num_trajs: int = 10
    num_frames: int = 20
    num_iterations: int = 10 * 75  # 1 epoch ~ (30 * 50 / 2 / N_traj) iterations
    num_workers: int = 2
    cache_video: bool = False
    video_cache_size: int = 2

    hidden_size: int = 512
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    num_hidden_layers: int = 2

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    temperature: float = 1e-2
    max_norm: float = 1.0

    print_interval: int = 25


def main(config: ArgsConfig):
    if not config.dataset_paths:
        raise ValueError("Supply at least one --dataset-paths directory")
    if (
        min(
            config.num_trajs,
            config.num_frames,
            config.num_iterations,
            config.print_interval,
            config.video_cache_size,
        )
        < 1
    ):
        raise ValueError("Sampling, iteration, logging and cache counts must be positive")
    if config.num_workers < 0 or config.temperature <= 0:
        raise ValueError("num_workers must be nonnegative and temperature must be positive")
    if config.num_trajs * config.num_frames < 2:
        raise ValueError("Need at least two positive pairs per batch for contrastive negatives")
    output = Path(config.retriever_path)
    if list(output.glob("*.safetensors")) or list(output.glob("pytorch_model*.bin")):
        raise ValueError(f"Checkpoint already exists in {output}; choose a fresh --retriever-path")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    seed_everything(config.seed)
    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu")

    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    embodiment_tag = EmbodimentTag(EMBODIMENT_TAG)

    data_mixture = []
    dataset_class = CachedLeRobotSingleDataset if config.cache_video else BoundedLeRobotVideoDataset
    dataset_kwargs = {} if config.cache_video else {"cache_size": config.video_cache_size}
    for dataset_path in config.dataset_paths:
        # Transforms keep mutable dataset metadata: give each dataset its own instance.
        dataset = dataset_class(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=data_config_cls.transform(),
            embodiment_tag=embodiment_tag,
            video_backend=VIDEO_BACKEND,
            **dataset_kwargs,
        )
        if config.num_trajs > len(dataset.task_groups):
            raise ValueError(
                f"num_trajs exceeds task count in {dataset_path}; the sampler may assign all groups to one suite"
            )
        if min(map(len, dataset.task_groups.values())) < 2:
            raise ValueError(f"Every task needs at least two demonstrations: {dataset_path}")
        if min(dataset.trajectory_lengths) < config.num_frames:
            raise ValueError(f"num_frames exceeds the shortest demonstration in {dataset_path}")
        data_mixture.append((dataset, 1.0))

    mixture_dataset = FramePathsMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    mixture_dataset.save_metadata(config.retriever_path)

    batch_sampler = AlignedFramesBatchSampler(
        mixture_dataset, config.num_trajs, config.num_frames, config.num_iterations
    )
    collator = DefaultDataCollator()

    dataloader = DataLoader(
        mixture_dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        # Bound queued CPU batches; default prefetch=2 can retain several large batches.
        **({"prefetch_factor": 1} if config.num_workers else {}),
    )

    vla = GR00T_N1_5.from_pretrained(
        config.pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    vla.requires_grad_(False)

    vla.compute_dtype = "bfloat16"
    vla.config.compute_dtype = "bfloat16"
    llm_hidden_size = vla.backbone.eagle_model.config.text_config.hidden_size

    args = asdict(config)
    args["llm_hidden_size"] = llm_hidden_size
    bert_config = get_config(args)

    retriever = RetrieverWrapperWithHead(bert_config)
    retriever = retriever.to(device, torch.bfloat16).train()
    initial_linear = retriever.linear.weight.detach().clone()

    optimizer = AdamW(retriever.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, config.num_iterations)

    start_time = perf_counter()
    train_loss = 0
    interval_steps = 0
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")
    print(
        f"Training {config.num_iterations} optimizer steps; {2 * config.num_trajs * config.num_frames} observations/batch; bounded_video_cache={not config.cache_video}",
        flush=True,
    )
    for iteration, inputs in tqdm(enumerate(dataloader, start=1), total=config.num_iterations):
        vl_embeds = retriever(vla, inputs)
        loss = retriever.compute_simclr_loss(vl_embeds, config.temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")

        loss.backward()
        grad_norm = clip_grad_norm_(retriever.parameters(), config.max_norm, error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        interval_steps += 1
        with metrics_path.open("a") as file:
            file.write(
                json.dumps(
                    {
                        "step": iteration,
                        "loss": loss.item(),
                        "grad_norm": float(grad_norm),
                        "learning_rate": learning_rate,
                        "observations": len(vl_embeds),
                    }
                )
                + "\n"
            )

        if iteration % config.print_interval == 0 or iteration == config.num_iterations:
            train_loss /= interval_steps
            end_time = perf_counter()
            elapsed_time = get_elapsed_time(start_time, end_time)

            message = f"[Iter {iteration:3}/{config.num_iterations}] Loss: {train_loss:6.4f} | {elapsed_time}"
            log_message(message, config.retriever_path)

            start_time = perf_counter()
            train_loss = 0
            interval_steps = 0

    retriever.save_pretrained(config.retriever_path)
    (output / "completed.json").write_text(
        json.dumps(
            {
                "optimizer_steps": iteration,
                "final_loss": loss.item(),
                "linear_weight_max_abs_change": float(
                    (retriever.linear.weight.detach() - initial_linear).abs().max()
                ),
                "base_model_frozen": not any(parameter.requires_grad for parameter in vla.parameters()),
                "base_model_has_gradients": any(parameter.grad is not None for parameter in vla.parameters()),
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else None,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
