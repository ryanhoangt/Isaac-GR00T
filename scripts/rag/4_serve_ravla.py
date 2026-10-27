from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import tyro
import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.data.schema import EmbodimentTag
from gr00t.eval.robot import RobotInferenceServer

from gr00t.rag.utils import seed_everything
from gr00t.rag.ravla import RAVLADataCollator, RAVLA
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.policy import VLAPolicy

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"

@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0

    retriever_path: str = "models/retriever/default"
    memory_name: str = "memory"
    memory_path: Optional[str] = None
    checkpoint_path: str = "models/ravla/default/checkpoint_3"

    denoising_steps: int = 4
    k: int = 1

    host: str = "0.0.0.0"
    port: int = 8071
    server_id: Optional[str] = None


def main(config: ArgsConfig):
    seed_everything(config.seed)
    device = torch.device(f'cuda:{config.cuda}' if torch.cuda.is_available() else 'cpu')

    data_config_cls = load_data_config(DATA_CONFIG)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(EMBODIMENT_TAG)

    metadata = torch.load(f"{config.checkpoint_path}/metadata.pt")
    transforms.set_metadata(metadata[embodiment_tag.value])
    transforms.eval()

    collator = RAVLADataCollator()

    retriever = RetrieverWrapperWithHead.from_pretrained(
        config.retriever_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    memory_path = (
        Path(config.memory_path) if config.memory_path is not None
        else Path(config.retriever_path) / f"{config.memory_name}.pt"
    )
    memory = torch.load(memory_path, map_location=device)

    vla = RAVLA.from_pretrained(
        config.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    vla.set_retrieval_components(retriever, memory, config.k)

    vla.action_head.num_inference_timesteps = config.denoising_steps
    vla.compute_dtype = "bfloat16"
    vla.config.compute_dtype = "bfloat16"

    policy = VLAPolicy(
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        collator=collator,
        vla=vla,
    )

    server = RobotInferenceServer(policy, config.host, config.port)
    server.register_endpoint(
        "server_info",
        lambda: {"server_id": config.server_id, "pid": os.getpid(),
                 "checkpoint_path": str(Path(config.checkpoint_path).resolve())},
        requires_input=False,
    )
    server.run()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
