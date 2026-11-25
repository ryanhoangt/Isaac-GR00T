"""Serve RA-VLA checkpoints over the existing ZeroMQ inference protocol."""

from dataclasses import dataclass
import os
from typing import Optional

import torch
import tyro

from gr00t.data.schema import EmbodimentTag
from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.rag.policy import VLAPolicy
from gr00t.rag.ravla import RAVLADataCollator, RAVLA
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.serving import load_serving_memory, resolve_serving_config
from gr00t.rag.utils import seed_everything


@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0
    retriever_path: str = "models/retriever/default"
    memory_name: str = "memory"
    memory_path: Optional[str] = None
    checkpoint_path: str = "models/ravla/default/checkpoint_3"
    data_config: Optional[str] = None
    """Infer from policy checkpoint, falling back to LIBERO for legacy models."""
    denoising_steps: int = 4
    k: int = 1
    host: str = "0.0.0.0"
    port: int = 8071
    server_id: Optional[str] = None
    gpu_memory_fraction: Optional[float] = None
    """Optional process VRAM cap for a server sharing a GPU with training."""


def build_policy(config):
    _, key_type = resolve_serving_config(config)
    seed_everything(config.seed)
    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(config.cuda)
        if config.gpu_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, config.cuda)
    elif config.gpu_memory_fraction is not None:
        raise ValueError("gpu_memory_fraction requires a CUDA device")
    metadata = torch.load(f"{config.checkpoint_path}/metadata.pt", map_location="cpu", weights_only=False)
    memory, info = load_serving_memory(config, key_type, metadata, device)
    data_config = load_data_config(config.data_config)
    transforms = data_config.transform()
    tag = EmbodimentTag("new_embodiment")
    transforms.set_metadata(metadata[tag.value])
    transforms.eval()
    retriever = (
        RetrieverWrapperWithHead.from_pretrained(
            config.retriever_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        .eval()
        .requires_grad_(False)
    )
    vla = (
        RAVLA.from_pretrained(
            config.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        .eval()
        .requires_grad_(False)
    )
    vla.set_retrieval_components(retriever, memory, config.k)
    vla.action_head.num_inference_timesteps = config.denoising_steps
    vla.compute_dtype = vla.config.compute_dtype = "bfloat16"
    modalities = data_config.modality_config()
    policy = VLAPolicy(
        modality_configs=modalities,
        transforms=transforms,
        embodiment_tag=tag,
        collator=RAVLADataCollator(require_memory_key=key_type == "canonical_task"),
        vla=vla,
        memory_key_type=key_type,
    )
    info.update(
        {
            "server_id": config.server_id,
            "pid": os.getpid(),
            "action_horizon": vla.action_head.config.action_horizon,
            "padded_action_dim": vla.action_head.config.action_dim,
            "action_keys": modalities["action"].modality_keys,
            "padding_mask_enabled": bool(getattr(vla.config, "ravla_mask_padding", False)),
        }
    )
    return policy, info


def main(config):
    policy, info = build_policy(config)
    server = RobotInferenceServer(policy, config.host, config.port)

    def server_info():
        result = dict(info)
        if torch.cuda.is_available():
            result["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(config.cuda) / 1024**3
            result["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(config.cuda) / 1024**3
        return result

    server.register_endpoint("server_info", server_info, requires_input=False)
    try:
        server.run()
    finally:
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
