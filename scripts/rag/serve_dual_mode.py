"""Serve a trained dual-mode checkpoint using a fixed action/goal mode."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import tyro

from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.eval.robot import RobotInferenceServer
from gr00t.rag.dual_mode import DualModeRAVLA
from gr00t.rag.dual_mode_memory import DualModePolicy, MilestoneMemory, module_digest
from gr00t.rag.ravla import RAVLA, RAVLADataCollator
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything
from scripts.rag.train_dual_mode import check_worker


@dataclass
class Args:
    checkpoint: Path
    memory: Path
    retriever: Path = Path("ra-vla-official-assets/libero_gos")
    mode: int = 0
    lookahead: int = 16
    port: int = 8085
    host: str = "127.0.0.1"
    seed: int = 0


def load_policy(args):
    check_worker()
    seed_everything(args.seed)
    metadata = torch.load(args.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    config = LiberoDataConfig()
    transforms = config.transform()
    transforms.set_metadata(metadata["new_embodiment"])
    transforms.eval()
    memory = MilestoneMemory(args.memory, normalization=metadata)
    saved_config = json.loads((args.checkpoint / "config.json").read_text())
    dual = saved_config["action_head_cfg"].get("dual_mode", False)
    if args.mode and not dual:
        raise ValueError("Goal mode requires a checkpoint trained with dual-mode conditioning")
    model_cls = DualModeRAVLA if dual else RAVLA
    model = model_cls.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0", tune_llm=False, tune_visual=False
    ).eval()
    retriever = RetrieverWrapperWithHead.from_pretrained(
        args.retriever, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    if memory.manifest["backbone_sha256"] != module_digest(model.backbone):
        raise ValueError("Memory/checkpoint frozen backbones differ")
    if memory.manifest["retriever_sha256"] != module_digest(retriever):
        raise ValueError("Memory/query retriever weights differ")
    return DualModePolicy(
        model,
        retriever,
        memory,
        transforms,
        RAVLADataCollator(),
        config.modality_config(),
        mode=args.mode,
        lookahead=args.lookahead,
    )


def main(args):
    policy = load_policy(args)
    server = RobotInferenceServer(policy, args.host, args.port)
    server.register_endpoint("get_action_with_context", policy.get_action_with_context)
    server.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
