from copy import deepcopy

import pytest
import torch

from gr00t.rag.goal_control import GoalMSEHead, IsolatedGoalHead, owner_digest
from gr00t.rag.goal_control_data import SameDemoGoals, split_tasks, validation_paths
from gr00t.rag.ravla import RAVLAActionHead
from test_dual_mode_ravla import TinyDataset, config, inputs


def test_warm_owner_exact_parity_and_goal_optimizer_isolation():
    torch.manual_seed(4)
    original = RAVLAActionHead(config()).eval()
    head = IsolatedGoalHead(config()).eval()
    missing, unexpected = head.load_state_dict(original.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("goal_branch.") or k in ("goal_mode_token", "goal_view_embedding") for k in missing)
    head.initialize_goal_from_owner()
    current, data = inputs()
    data.context_mode.zero_()
    expected = original.get_action(deepcopy(current), deepcopy(data)).action_pred
    actual = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before = owner_digest(head)
    head.train()
    assert not head.model.training and head.goal_branch.model.training
    data.context_mode.fill_(1)
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    losses = head(deepcopy(current), deepcopy(data))
    assert losses.margin_loss == 0
    losses.loss.backward()
    assert head.goal_branch.goal_mode_token.grad.abs().sum() > 0
    assert all(p.grad is None for k, p in head.named_parameters() if not k.startswith("goal_branch."))
    optimizer.step()
    assert owner_digest(head) == before
    data.context_mode.zero_()
    actual = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_goal_source_mask_and_state_roundtrip():
    head = IsolatedGoalHead(config()).eval()
    head.initialize_goal_from_owner()
    current, data = inputs()
    data.context_mode.fill_(1)
    data["flow_source"] = torch.randn_like(data.ret_actions)
    expected = head.get_action(deepcopy(current), deepcopy(data)).action_pred
    data.ret_actions.mul_(1000)
    torch.testing.assert_close(
        head.get_action(deepcopy(current), deepcopy(data)).action_pred, expected, rtol=0, atol=0
    )
    reloaded = IsolatedGoalHead(config()).eval()
    reloaded.load_state_dict(head.state_dict())
    torch.testing.assert_close(
        reloaded.get_action(deepcopy(current), deepcopy(data)).action_pred, expected, rtol=0, atol=0
    )
    data.context_mode[0] = 0
    with pytest.raises(ValueError, match="homogeneous"):
        head.get_action(current, data)


def test_same_demo_hindsight_is_exact_and_clamped():
    class Mixture:
        datasets = [TinyDataset()]

    samples = SameDemoGoals(Mixture(), [(0, 10, 0), (0, 20, 49)])
    first, last = samples[0], samples[1]
    assert first["current"] == {"episode": 10, "frame": 0}
    assert first["goal"] == {"episode": 10, "frame": 16}
    assert last["goal"] == {"episode": 20, "frame": 49}
    assert first["provenance"]["goal_episode"] == first["provenance"]["episode"]


def test_beta_sampler_is_fp32_under_bfloat16_checkpoint_initialization():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        head = GoalMSEHead(config())
    finally:
        torch.set_default_dtype(previous)
    assert head.beta_dist.concentration1.dtype == torch.float32
    assert torch.isfinite(head.sample_time(2, dtype=torch.bfloat16, device="cpu")).all()


def test_unused_new_embeddings_are_initialized_for_finite_mode1():
    head = IsolatedGoalHead(config()).eval()
    with torch.no_grad():
        head.goal_view_embedding.fill_(float("nan"))
        head.goal_mode_token.fill_(float("nan"))
    head.initialize_goal_from_owner()
    assert not head.goal_view_embedding.any() and not head.goal_mode_token.any()
    current, data = inputs()
    data.context_mode.zero_()
    assert torch.isfinite(head.get_action(current, data).action_pred).all()


def test_validation_episodes_are_excluded_from_goal_updates():
    class Mixture:
        datasets = [TinyDataset()]
        all_frame_paths = [(0, 10, 0), (0, 10, 20), (0, 20, 0), (0, 20, 49)]

    mixture = Mixture()
    training, validation, inventory = split_tasks(mixture, tasks_per_suite=1, heldout_demos=1)
    assert {episode for _, episode, _ in training} == {20}
    assert validation == [(0, 10)]
    assert inventory[0]["validation_episode_ids"] == [10]
    paths = validation_paths(mixture, validation, states_per_demo=3)
    assert len(paths) == 3 and all(episode == 10 and frame <= 34 for _, episode, frame in paths)
