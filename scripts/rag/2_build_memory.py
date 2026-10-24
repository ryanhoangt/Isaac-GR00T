from typing import List
from dataclasses import dataclass, field
import tyro
from tqdm import tqdm

import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import CachedLeRobotSingleDataset, LeRobotMixtureDataset
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.gr00t_n1 import GR00T_N1_5

from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything, get_key_frames

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"
VIDEO_BACKEND = "torchvision_av"
PRETRAINED_VLA_PATH = "nvidia/GR00T-N1.5-3B"

@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0

    retriever_path: str = "models/retriever/default"
    dataset_paths: List[str] = field(default_factory=list)
    memory_name: str = "memory"

    num_trajs_per_task: int = 3
    frame_stride: int = 4


def main(config: ArgsConfig):
    seed_everything(config.seed)
    device = torch.device(f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    data_config_cls = load_data_config(DATA_CONFIG)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(EMBODIMENT_TAG)

    transforms.eval()
    transforms.transforms[-1].train()

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

    mixture_dataset = LeRobotMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    mixture_dataset.load_metadata(config.retriever_path)
    collator = DefaultDataCollator()

    vla = GR00T_N1_5.from_pretrained(
        PRETRAINED_VLA_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    vla.compute_dtype = "bfloat16"
    vla.config.compute_dtype = "bfloat16"

    retriever = RetrieverWrapperWithHead.from_pretrained(
        config.retriever_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    memory = {}
    for dataset in mixture_dataset.datasets:
        tasks = list(dataset.task_groups.keys())

        for task in tqdm(tasks):
            data = {
                "pixel_values": [], "action": [], "keys": [],
                "backbone_features": [], "backbone_attention_mask": []}

            key_frames = get_key_frames(
                dataset, task, config.num_trajs_per_task, config.frame_stride, transforms)

            for i in range(0, len(key_frames), 100):
                inputs = collator(key_frames[i:i+100])

                data["pixel_values"].append(inputs["eagle_pixel_values"].unflatten(0, (-1, 2)))
                data["action"].append(inputs["action"])

                with torch.inference_mode():
                    vl_embeds = retriever(vla, inputs)
                data["keys"].append(vl_embeds)

                with torch.inference_mode():
                    backbone_inputs, _ = vla.prepare_input(inputs)
                    backbone_outputs = vla.backbone(backbone_inputs)
                data["backbone_features"].append(backbone_outputs["backbone_features"])
                data["backbone_attention_mask"].append(backbone_outputs["backbone_attention_mask"])

            memory[task] = {key: torch.cat(value) for key, value in data.items()}

    torch.save(memory, f"{config.retriever_path}/{config.memory_name}.pt")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
