from typing import List
from dataclasses import dataclass, field, asdict
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
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.gr00t_n1 import GR00T_N1_5

from gr00t.rag.data import FramePathsMixtureDataset, AlignedFramesBatchSampler
from gr00t.rag.retriever import get_config, RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything, get_elapsed_time, log_message

DATA_CONFIG = "examples.UR5e.custom_data_config:Ur5eDataConfig"
EMBODIMENT_TAG = "new_embodiment"
VIDEO_BACKEND = "torchvision_av"
PRETRAINED_VLA_PATH = "nvidia/GR00T-N1.5-3B"

@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0

    dataset_paths: List[str] = field(default_factory=list)
    retriever_path: str = "models/retriever/default"

    num_trajs: int = 10
    num_frames: int = 20
    num_iterations: int = 10 * 75  # 1 epoch ~ (30 * 50 / 2 / N_traj) iterations
    num_workers: int = 16

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
    mixture_dataset.save_metadata(config.retriever_path)

    batch_sampler = AlignedFramesBatchSampler(
        mixture_dataset, config.num_trajs, config.num_frames, config.num_iterations)
    collator = DefaultDataCollator()

    dataloader = DataLoader(
        mixture_dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    vla = GR00T_N1_5.from_pretrained(
        PRETRAINED_VLA_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    vla.compute_dtype = "bfloat16"
    vla.config.compute_dtype = "bfloat16"
    llm_hidden_size = vla.backbone.eagle_model.config.text_config.hidden_size

    args = asdict(config)
    args["llm_hidden_size"] = llm_hidden_size
    bert_config = get_config(args)

    retriever = RetrieverWrapperWithHead(bert_config)
    retriever = retriever.to(device, torch.bfloat16).train()

    optimizer = AdamW(
        retriever.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, config.num_iterations)

    start_time = perf_counter()
    train_loss = 0
    for iteration, inputs in tqdm(enumerate(dataloader, start=1), total=config.num_iterations):
        vl_embeds = retriever(vla, inputs)
        loss = retriever.compute_simclr_loss(vl_embeds, config.temperature)

        loss.backward()
        clip_grad_norm_(retriever.parameters(), config.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if iteration % config.print_interval == 0:
            train_loss /= config.print_interval
            end_time = perf_counter()
            elapsed_time = get_elapsed_time(start_time, end_time)

            message = (
                f"[Iter {iteration:3}/{config.num_iterations}] "
                f"Loss: {train_loss:6.4f} | {elapsed_time}"
            )
            log_message(message, config.retriever_path)

            start_time = perf_counter()
            train_loss = 0

    retriever.save_pretrained(config.retriever_path)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
