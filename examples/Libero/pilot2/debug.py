"""CPU-only small-scale exercise of the real flow head and goal masking."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


def tiny_head():
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHead,
        FlowmatchingActionHeadConfig,
    )

    config = FlowmatchingActionHeadConfig(
        hidden_size=16,
        input_embedding_dim=16,
        backbone_embedding_dim=8,
        action_dim=32,
        action_horizon=16,
        max_state_dim=64,
        max_num_embodiments=2,
        num_inference_timesteps=4,
        num_target_vision_tokens=2,
        max_seq_len=64,
        goal_conditioning=True,
        use_vlln=True,
        diffusion_model_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "output_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "final_dropout": False,
            "cross_attention_dim": 8,
            "interleave_self_attention": True,
            "max_num_positional_embeddings": 64,
        },
        vl_self_attention_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "output_dim": 8,
            "dropout": 0.0,
            "final_dropout": False,
            "max_num_positional_embeddings": 64,
        },
    )
    return FlowmatchingActionHead(config)


def tiny_batch():
    visual = BatchFeature(
        data={
            "backbone_features": torch.randn(3, 4, 8),
            "backbone_attention_mask": torch.tensor([[1, 1, 1, 0]] * 3),
            "goal_features": torch.randn(3, 3, 8),
            "goal_attention_mask": torch.tensor([[1, 1, 0]] * 3),
            "goal_mode_id": torch.tensor([0, 1, 1]),
        }
    )
    mask = torch.zeros(3, 16, 32)
    mask[:, :, :7] = 1
    action = BatchFeature(
        data={
            "state": torch.randn(3, 1, 64),
            "embodiment_id": torch.zeros(3, dtype=torch.long),
            "action": torch.randn(3, 16, 32),
            "action_mask": mask,
        }
    )
    return visual, action


def clone_features(features):
    return BatchFeature(data={key: value.clone() for key, value in features.items()})


def debug(output, steps=3):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    head = tiny_head()
    visual, action = tiny_batch()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    losses = []
    for _ in range(steps):
        head.train()
        optimizer.zero_grad()
        loss = head(clone_features(visual), action)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CPU debug loss")
        loss.backward()
        if head.goal_conditioner.subgoal_view.grad is None:
            raise RuntimeError("Goal conditioning disconnected from the training loss")
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    torch.manual_seed(10)
    prediction = head.get_action(clone_features(visual), action)["action_pred"]
    changed = clone_features(visual)
    changed["goal_features"][0] += 1000
    torch.manual_seed(10)
    counterfactual = head.get_action(changed, action)["action_pred"]
    torch.testing.assert_close(prediction[0], counterfactual[0], rtol=0, atol=0)
    result = {
        "kind": "cpu_debug_only",
        "device": "cpu",
        "real_gr00t_3b_loaded": False,
        "architecture": "repository FlowmatchingActionHead + tiny DiT + synthetic Eagle features",
        "steps": steps,
        "losses": losses,
        "action_shape": list(prediction.shape),
        "inactive_goal_invariance": True,
        "goal_gradient_checked": True,
        "scientific_success_rate": None,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save({"state_dict": head.state_dict(), "config": head.config.to_dict()}, output.with_suffix(".pt"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def debug_data(split, output):
    """Decode real AV1 data, apply all three modes and run the actual collator."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.transforms import collate
    from .dataset import GoalDataset
    from .transforms import GoalDataConfig
    from .data import LANGUAGE_KEY, pair_goal, perturb_state

    config = GoalDataConfig()
    dataset = GoalDataset(
        Path(split).resolve().parent / "train",
        config.modality_config(),
        EmbodimentTag.NEW_EMBODIMENT,
        transforms=config.transform(),
        video_backend="torchvision_av",
    )
    rng = np.random.default_rng(42)
    episode, step = dataset.all_steps[0]
    examples, goals = [], []
    for mode in ("language", "both", "goal"):
        target, target_step, cross = pair_goal(
            dataset.episodes, episode, step, rng, cross_probability=float(mode == "goal")
        )
        raw = dataset.get_step_data(episode, step)
        image = dataset.get_step_data(target, target_step)["video.image"].copy()
        if cross:
            raw = perturb_state(raw, rng)
        raw["subgoal.image"] = image
        raw["goal_mode_id"] = np.int64(mode != "language")
        if mode == "goal":
            raw[LANGUAGE_KEY] = np.array([""])
        examples.append(dataset.transforms(raw))
        goals.append(
            {
                "mode": mode,
                "source_episode": int(episode),
                "goal_episode": int(target),
                "goal_step": int(target_step),
                "cross_demo": cross,
            }
        )
    batch = collate(examples, dataset.transforms.transforms[-1].eagle_processor)
    assert batch["state"].shape == (3, 1, 64)
    assert batch["action"].shape == (3, 16, 32)
    assert batch["action_mask"].sum().item() == 3 * 16 * 7
    assert batch["goal_mode_id"].tolist() == [0, 1, 1]
    assert "Target subgoal view." not in examples[0]["eagle_content"]["text_list"][0]
    result = {
        "kind": "real_data_cpu_pipeline",
        "real_gr00t_3b_loaded": False,
        "train_frames": len(dataset),
        "examples": goals,
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "goal_only_prompt": examples[2]["eagle_content"]["text_list"][0],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
