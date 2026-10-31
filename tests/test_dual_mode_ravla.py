"""Real small RA-VLA heads and DTW; run these tests on a worker."""

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
from gr00t.rag.dual_mode import DualModeActionHead, validate_modes
from gr00t.rag.dual_mode_data import DualModeDataset, dense_alignment, temporal_negative
from gr00t.rag.dual_mode_memory import milestone_index, module_digest
from gr00t.rag.ravla import RAVLAActionHead


def config(dropout=0.0):
    return FlowmatchingActionHeadConfig(
        hidden_size=32,
        input_embedding_dim=32,
        backbone_embedding_dim=32,
        max_num_embodiments=2,
        max_state_dim=8,
        action_dim=8,
        action_horizon=16,
        num_target_vision_tokens=2,
        num_inference_timesteps=4,
        diffusion_model_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "output_dim": 32,
            "num_layers": 16,
            "cross_attention_dim": 32,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "dropout": dropout,
            "positional_embeddings": None,
        },
        vl_self_attention_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "num_layers": 1,
            "dropout": dropout,
            "positional_embeddings": None,
        },
    )


def inputs(batch=2):
    current = BatchFeature(
        data={"backbone_features": torch.randn(batch, 5, 32), "backbone_attention_mask": torch.ones(batch, 5)}
    )
    retrieved = BatchFeature(
        data={"backbone_features": torch.randn(batch, 5, 32), "backbone_attention_mask": torch.ones(batch, 5)}
    )
    actions = BatchFeature(
        data={
            "state": torch.randn(batch, 1, 8),
            "embodiment_id": torch.zeros(batch, dtype=torch.long),
            "ret_actions": torch.randn(batch, 16, 8),
            "ret_backbone_output": retrieved,
            "context_mode": torch.arange(batch) % 2,
            "action": torch.randn(batch, 16, 8),
            "action_mask": torch.ones(batch, 16, 8),
            "negative_valid": torch.ones(batch),
            "neg_ret_actions": torch.randn(batch, 16, 8),
            "neg_backbone_output": BatchFeature(
                data={
                    "backbone_features": torch.randn(batch, 5, 32),
                    "backbone_attention_mask": torch.ones(batch, 5),
                }
            ),
        }
    )
    actions.action_mask[..., 7] = 0
    return current, actions


def test_mode1_matches_original_head_exactly():
    torch.manual_seed(3)
    original = RAVLAActionHead(config()).eval()
    dual = DualModeActionHead(config()).eval()
    missing, unexpected = dual.load_state_dict(original.state_dict(), strict=False)
    assert set(missing) == {"goal_mode_token", "goal_view_embedding"}
    assert not unexpected
    current, data = inputs()
    data.context_mode.zero_()
    expected = original.get_action(deepcopy(current), deepcopy(data)).action_pred
    actual = dual.get_action(deepcopy(current), deepcopy(data)).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_goal_output_has_no_expert_action_leak():
    torch.manual_seed(4)
    head = DualModeActionHead(config()).eval()
    current, data = inputs()
    data.context_mode.fill_(1)
    data["flow_source"] = torch.randn_like(data.ret_actions)
    first = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    data.ret_actions.mul_(1000).add_(500)
    second = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    data.ret_backbone_output.backbone_features.add_(
        torch.randn_like(data.ret_backbone_output.backbone_features) * 2
    )
    third = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    assert (third - second).abs().max() > 1e-6


def test_mixed_source_uses_gaussian_only_in_goal_rows():
    head = DualModeActionHead(config())
    retrieved = torch.randn(4, 16, 8)  # B=2, K=2
    gaussian = torch.randn(2, 16, 8)
    source = head._source(retrieved, torch.tensor([False, True]), gaussian)
    torch.testing.assert_close(source[0], retrieved[:2].mean(0), rtol=0, atol=0)
    torch.testing.assert_close(source[1], gaussian[1], rtol=0, atol=0)


def test_mixed_training_has_finite_losses_and_goal_gradients():
    torch.manual_seed(6)
    head = DualModeActionHead(config(dropout=0.2), margin=0.01, lambda_=0.1).train()
    current, data = inputs()
    losses = head(current, data)
    assert all(torch.isfinite(v).all() for v in losses.values())
    assert losses.goal_count == 1 and losses.action_count == 1
    losses.loss.backward()
    assert head.goal_mode_token.grad.abs().sum() > 0
    assert head.goal_view_embedding.grad.abs().sum() > 0


def test_identical_context_has_identical_loss_with_dropout():
    head = DualModeActionHead(config(dropout=0.3), margin=0.02).train()
    current, data = inputs()
    data["neg_backbone_output"] = deepcopy(data.ret_backbone_output)
    data["neg_ret_actions"] = data.ret_actions.clone()
    losses = head(current, data)
    torch.testing.assert_close(losses.action_mse, losses.action_negative_mse, rtol=0, atol=0)
    torch.testing.assert_close(losses.goal_mse, losses.goal_negative_mse, rtol=0, atol=0)
    assert float(losses.margin_loss) == pytest.approx(0.02, abs=1e-6)


def test_goal_training_does_not_backpropagate_through_expert_actions():
    head = DualModeActionHead(config()).train()
    current, data = inputs()
    data.context_mode.fill_(1)
    data.ret_actions.requires_grad_()
    data.neg_ret_actions.requires_grad_()
    head(current, data).loss.backward()
    assert data.ret_actions.grad is None or not data.ret_actions.grad.any()
    assert data.neg_ret_actions.grad is None or not data.neg_ret_actions.grad.any()
    assert not head.ret_action_encoder.weight.grad.any()


def test_invalid_negatives_do_not_contribute_margin():
    head = DualModeActionHead(config()).train()
    current, data = inputs()
    data.negative_valid.zero_()
    losses = head(current, data)
    assert losses.margin_loss == 0
    torch.testing.assert_close(losses.loss, losses.mse_loss)


@pytest.mark.parametrize("modes", [[2, 0], [-1, 1], [0]])
def test_invalid_modes_fail(modes):
    with pytest.raises(ValueError, match="context_mode"):
        validate_modes(modes, 2, "cpu")


def test_dtw_identity_and_temporal_negatives():
    trajectory = np.arange(50, dtype=float)[:, None]
    np.testing.assert_array_equal(dense_alignment(trajectory, trajectory), np.arange(50))
    for goal in (0, 25, 49):
        negative, valid = temporal_negative(goal, 50, 16, random.Random(2))
        assert valid and abs(goal - negative) >= 16
    assert temporal_negative(2, 5, 16, random.Random(0)) == (2, False)


class TinyDataset:
    task_groups = {"same task": [10, 20]}
    trajectory_tasks = ["same task", "same task"]
    trajectory_lengths = [50, 50]

    def get_trajectory_index(self, episode):
        return [10, 20].index(episode)

    def get_trajectory_data(self, episode):
        return pd.DataFrame({"action": [np.array([float(t)]) for t in range(50)]})

    def get_step_data(self, episode, frame):
        return {"episode": episode, "frame": frame}

    def transforms(self, sample):
        return sample


def test_pairing_is_cross_demo_clamped_and_same_scene_negative():
    class Mixture:
        datasets = [TinyDataset()]
        all_frame_paths = [(0, 10, 0), (0, 20, 49)]

        def __len__(self):
            return len(self.all_frame_paths)

    mixture = Mixture()
    dataset = DualModeDataset(mixture, goal_probability=1, lookahead=16, negative_gap=16)
    first, last = dataset[0], dataset[1]
    assert first["positive"] == {"episode": 20, "frame": 16}
    assert last["positive"] == {"episode": 10, "frame": 49}
    for sample in (first, last):
        assert sample["positive"]["episode"] != sample["current"]["episode"]
        assert sample["positive"]["episode"] == sample["negative"]["episode"]
        assert abs(sample["positive"]["frame"] - sample["negative"]["frame"]) >= 16


def test_milestone_never_crosses_demo_boundary_and_uses_control_steps():
    entry = {
        "episode_id": torch.tensor([10, 10, 10, 20, 20, 20]),
        "frame_index": torch.tensor([0, 4, 9, 0, 4, 9]),
        "trajectory_length": torch.full((6,), 10),
    }
    assert milestone_index(entry, 0, 4) == 1
    assert milestone_index(entry, 0, 8) == 2
    assert milestone_index(entry, 2, 16) == 2
    assert milestone_index(entry, 3, 4) == 4
    entry["frame_index"] = torch.tensor([0, 4, 8, 0, 4, 8])
    with pytest.raises(ValueError, match="terminal"):
        milestone_index(entry, 2, 16)


def test_fingerprint_detects_weight_changes():
    module = torch.nn.Linear(2, 2)
    original = module_digest(module)
    assert module_digest(deepcopy(module)) == original
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert module_digest(module) != original
