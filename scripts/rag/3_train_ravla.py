from typing import List
from dataclasses import dataclass, field
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

    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_norm: float = 1.0

    save_start_epoch: int = 1
    save_interval: int = 1


def main(config: ArgsConfig):
    seed_everything(config.seed)
    device = torch.device(f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    data_config_cls = load_data_config(DATA_CONFIG)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(EMBODIMENT_TAG)

    data_mixture = []
    for dataset_path in config.dataset_paths:
        dataset = CachedLeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=VIDEO_BACKEND,
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

    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, num_iterations)

    for epoch in range(1, config.num_epochs + 1):
        start_time = perf_counter()
        train_mse_loss, train_margin_loss, train_loss = 0, 0, 0

        for inputs in tqdm(dataloader):
            losses = vla(inputs)
            mse_loss, margin_loss, loss = losses.mse_loss, losses.margin_loss, losses.loss

            loss.backward()
            clip_grad_norm_(parameters, config.max_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_mse_loss += mse_loss.item()
            train_margin_loss += margin_loss.item()
            train_loss += loss.item()

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

        if epoch >= config.save_start_epoch and epoch % config.save_interval == 0:
            checkpoint_path = f"{config.vla_path}/checkpoint_{epoch}"
            mixture_dataset.save_metadata(checkpoint_path)
            vla.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
