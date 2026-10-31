"""Worker smoke check: load a saved checkpoint, infer both modes, test action leakage."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import tyro

from gr00t.rag.dual_mode_data import encode_context, pack_context
from scripts.rag.serve_dual_mode import Args as ServeArgs, load_policy


@dataclass
class Args:
    checkpoint: Path
    memory: Path
    output: Path
    retriever: Path = Path("ra-vla-official-assets/libero_gos")


def main(args):
    policy = load_policy(ServeArgs(checkpoint=args.checkpoint, memory=args.memory, retriever=args.retriever))
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset

    config = LiberoDataConfig()
    dataset = BoundedLeRobotVideoDataset(
        dataset_path=policy.memory.manifest["dataset_path"],
        modality_configs=config.modality_config(),
        transforms=policy.transforms,
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="torchvision_av",
        cache_size=6,
    )
    # Construction sets dataset statistics; restore the checkpoint's statistics.
    metadata = torch.load(args.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    policy.transforms.set_metadata(metadata["new_embodiment"])
    policy.transforms.eval()
    task = next(iter(policy.memory.manifest["tasks"]))
    episode = policy.memory.manifest["tasks"][task]["episode_ids"][0]
    observation = dataset.get_step_data(episode, 0)
    entry = policy.memory[task]
    policy.transforms.transforms[-1].train()
    expected = policy.collator([policy.transforms(observation.copy())])["action"][0]
    torch.testing.assert_close(entry["action"][0], expected, rtol=0, atol=0)
    policy.transforms.eval()
    with torch.inference_mode():
        outputs = {}
        for mode in (0, 1):
            policy.mode = mode
            torch.manual_seed(9)
            result = policy.get_action_with_context(observation)
            assert all(np.isfinite(value).all() for value in result["action"].values())
            outputs[str(mode)] = {key: value for key, value in result.items() if key != "action"}
        batch = policy.collator([policy.transforms(observation.copy())])
        batch.pop("tasks")
        visual = encode_context(policy.vla, batch)
        entry = policy.memory[task]
        context, actions = pack_context(
            [(entry["backbone_features"][0], entry["backbone_attention_mask"][0], entry["action"][0])],
            policy.vla.device,
            policy.vla.action_head.dtype,
        )
        batch.update(ret_actions=actions, ret_backbone_output=context, context_mode=torch.ones(1))
        _, action_input = policy.vla.prepare_input(batch)
        action_input["flow_source"] = torch.randn_like(actions)
        first = policy.vla.action_head.get_action(visual, action_input).action_pred
        action_input.ret_actions.mul_(1000).add_(100)
        second = policy.vla.action_head.get_action(visual, action_input).action_pred
        if not torch.equal(first, second):
            raise AssertionError("Expert actions leaked into goal-mode inference")
    report = {
        "worker": __import__("os").uname().nodename,
        "checkpoint": str(args.checkpoint),
        "both_modes_finite": True,
        "goal_action_leakage_max_abs": float((first - second).abs().max()),
        "cached_action_normalization_verified": True,
        "context": outputs,
        "goal_token_norm": float(policy.vla.action_head.goal_mode_token.float().norm()),
        "goal_view_norm": float(policy.vla.action_head.goal_view_embedding.float().norm()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
