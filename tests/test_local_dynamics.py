from copy import deepcopy

import numpy as np
import pytest
import torch

from gr00t.rag.endpoint_data import EndpointCache, collate_cached, windows
from gr00t.rag.goal_control import IsolatedGoalHead, owner_digest
from gr00t.rag.local_dynamics import (
    DeltaMLP,
    DynamicsEnsemble,
    PolicyCodec,
    differentiable_goal_action,
    endpoint_loss,
)
from test_dual_mode_ravla import config, inputs


def codec():
    return PolicyCodec(torch.zeros(8), torch.ones(8), torch.tensor([-1.0] * 6 + [0.0]), torch.ones(7))


def test_codec_accepts_pydantic_dataset_statistics():
    from types import SimpleNamespace
    from pydantic import BaseModel

    class Statistics(BaseModel):
        state: dict
        action: dict

    values = {
        key: {"min": [0.0] * (2 if key == "gripper" else 1), "max": [1.0] * (2 if key == "gripper" else 1)}
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    }
    actions = {key: {"min": [0.0], "max": [1.0]} for key in values}
    metadata = {"new_embodiment": SimpleNamespace(statistics=Statistics(state=values, action=actions))}
    result = PolicyCodec.from_metadata(metadata)
    assert result.state_min.shape == (8,) and result.action_min.shape == (7,)


def test_complete_horizon_requires_observed_endpoint():
    states = np.arange(17 * 8).reshape(17, 8)
    actions = np.zeros((17, 7))
    s, a, delta = windows(states, actions)
    assert s.shape == (1, 8) and a.shape == (1, 16, 7)
    np.testing.assert_equal(delta[0], states[16, :3] - states[0, :3])
    with pytest.raises(ValueError):
        windows(states[:16], actions[:16])


def test_codec_matches_minmax_and_handles_constant_dimensions():
    instance = codec()
    torch.testing.assert_close(instance.state(torch.full((1, 8), 0.5)), torch.zeros(1, 8))
    instance.state_max[0] = 0
    assert instance.state(torch.ones(1, 8))[0, 0] == 0


def test_actions_match_saturation_and_binary_gripper_without_grip_gradient():
    instance = codec()
    action = torch.zeros(1, 16, 7, requires_grad=True)
    result = instance.actions(action, normalized=True)
    assert torch.all(result[..., 6] == 1)
    result.sum().backward()
    assert action.grad[..., :6].abs().sum() > 0
    assert action.grad[..., 6].abs().sum() == 0
    raw = torch.ones(1, 16, 7) * 3
    assert instance.actions(raw)[..., :6].max() == 1


def test_normalization_is_fit_from_supplied_training_rows_only():
    model = DeltaMLP(hidden=16)
    x = torch.randn(20, 120)
    y = torch.randn(20, 3) * 0.01
    model.fit_normalization(x, y)
    torch.testing.assert_close(model.input_mean, x.mean(0))
    before = model.input_mean.clone()
    model(torch.ones(2, 120) * 100)
    assert torch.equal(before, model.input_mean)


def test_frozen_dynamics_propagates_action_gradients_without_parameter_gradients():
    torch.manual_seed(5)
    dynamics = DynamicsEnsemble(count=1, hidden=16).freeze()
    actions = torch.zeros(2, 16, 8, requires_grad=True)
    loss, metrics = endpoint_loss(dynamics, codec(), torch.ones(2, 8) * 0.5, actions, torch.ones(2, 3) * 0.02, 1.0)
    loss.backward()
    assert metrics["endpoint_supported"] == 1
    assert actions.grad.abs().sum() > 0
    assert all(p.grad is None for p in dynamics.parameters())


def test_unsupported_rows_get_no_endpoint_gradient():
    dynamics = DynamicsEnsemble(count=1, hidden=16).freeze()
    actions = torch.zeros(2, 16, 8, requires_grad=True)
    loss, metrics = endpoint_loss(dynamics, codec(), torch.ones(2, 8) * 100, actions, torch.ones(2, 3), 1.0)
    loss.backward()
    assert metrics["endpoint_supported"] == 0 and loss == 0
    assert actions.grad.abs().sum() == 0


def test_differentiable_rollout_matches_actual_inference_and_preserves_mode1():
    torch.manual_seed(9)
    head = IsolatedGoalHead(config()).eval()
    head.initialize_goal_from_owner()
    current, data = inputs()
    noise = torch.randn_like(data.ret_actions)
    data.context_mode.fill_(1)
    data["flow_source"] = noise
    expected = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    before = owner_digest(head)
    head.train()
    prediction = differentiable_goal_action(
        head.goal_branch, current, data.state, data.embodiment_id, data.ret_backbone_output, noise
    )
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    prediction.square().mean().backward()
    assert head.goal_branch.goal_mode_token.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in head.named_parameters() if not name.startswith("goal_branch."))
    assert owner_digest(head) == before and head.goal_branch.training


def test_cache_does_not_clamp_terminal_goals_or_mix_episodes(tmp_path):
    import json

    records = [dict(file="train.pt", split="train", length=18), dict(file="val.pt", split="validation", length=17)]
    (tmp_path / "manifest.json").write_text(json.dumps(dict(status="completed", records=records)))
    train = EndpointCache(tmp_path, "train")
    val = EndpointCache(tmp_path, "validation")
    assert train.paths == [("train.pt", 0), ("train.pt", 1)]
    assert val.paths == [("val.pt", 0)]


def test_cached_collator_preserves_left_padding_masks():
    def sample(length):
        values = {
            key: torch.zeros(1)
            for key in ("state", "embodiment_id", "action", "action_mask", "raw_state", "delta")
        }
        values.update(
            current=dict(backbone_features=torch.ones(length, 3), backbone_attention_mask=torch.ones(length)),
            goal=dict(backbone_features=torch.ones(length + 1, 3), backbone_attention_mask=torch.ones(length + 1)),
        )
        return values

    batch = collate_cached([sample(2), sample(4)])
    assert batch["current"]["backbone_attention_mask"][0].tolist() == [0, 0, 1, 1]
    assert batch["goal"]["backbone_attention_mask"][0].tolist() == [0, 0, 1, 1, 1]
