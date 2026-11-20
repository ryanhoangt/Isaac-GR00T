import numpy as np
import pytest
import torch

from gr00t.rag.dynamics_reliability import (
    ReliabilityFilter,
    acceptance_summary,
    episode_partition,
    nearest_rms,
    select_threshold,
)


def test_episode_partition_is_deterministic_disjoint_and_preserves_audit():
    recipe = dict(split=[dict(train_episode_ids=list(range(40)), validation_episode_ids=[50, 51, 52])])
    result = episode_partition(recipe)
    assert result == episode_partition(recipe)
    groups = [set(result["split"][0][name + "_episode_ids"]) for name in ("train", "calibration", "test", "audit")]
    assert list(map(len, groups)) == [28, 6, 6, 3]
    assert len(set.union(*groups)) == sum(map(len, groups))
    assert result["split"][0]["audit_episode_ids"] == [50, 51, 52]
    assert len(recipe["split"][0]["train_episode_ids"]) == 40


def test_partition_rejects_source_validation_overlap():
    with pytest.raises(ValueError, match="Overlapping"):
        episode_partition(dict(split=[dict(train_episode_ids=list(range(40)), validation_episode_ids=[1])]))


def test_batched_support_distance_matches_brute_force_and_has_no_gradient():
    torch.manual_seed(5)
    query = torch.randn(7, 120, requires_grad=True)
    reference = torch.randn(11, 120)
    expected = (query[:, None] - reference[None]).square().mean(-1).sqrt().min(-1).values
    result = nearest_rms(query, reference, query_batch=3, reference_batch=4)
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad
    torch.testing.assert_close(nearest_rms(reference, reference), torch.zeros(11))


def test_joint_filter_rejects_either_bad_score():
    support = ReliabilityFilter(torch.zeros(1, 120), torch.zeros(120), torch.ones(120), 0.01, 0.5)
    query = torch.stack([torch.zeros(120), torch.ones(120), torch.zeros(120)])
    assert support(query, torch.tensor([0.005, 0.005, 0.02])).tolist() == [True, False, False]


def test_threshold_selected_on_calibration_can_fail_independent_test():
    provenance = [(task, episode, t) for task in range(3) for episode in range(3) for t in range(40)]
    n = len(provenance)
    score = np.tile(np.linspace(0, 1, 120), 3)
    error = np.where(score < 0.4, 0.002, 0.03)
    selected, candidates = select_threshold(error, score, score, provenance)
    assert selected is not None and selected["passed"] and selected["coverage"] == pytest.approx(0.4)
    threshold = selected["std_limit_m"]
    mask = score <= threshold
    final = acceptance_summary(np.full(n, 0.05), mask, provenance)
    assert not final["passed"]
    assert selected["std_limit_m"] == threshold
    assert len(candidates) == 36


def test_filter_cannot_hide_bad_task_or_keep_too_few_episodes():
    provenance = [(task, episode, t) for task in range(3) for episode in range(3) for t in range(40)]
    error = np.asarray([0.03 if task == 2 else 0.002 for task, _, _ in provenance])
    score = np.zeros(len(error))
    selected, _ = select_threshold(error, score, score, provenance)
    assert selected is None
    keep = np.asarray([episode == 0 for _, episode, _ in provenance])
    assert not acceptance_summary(np.full(len(error), 0.001), keep, provenance)["passed"]
