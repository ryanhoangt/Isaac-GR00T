import numpy as np
import pytest

from gr00t.rag.oracle_goal_protocol import (
    array_hash,
    case_seed,
    choose_milestone,
    phase_schema,
    phase_vector,
    summarize_episodes,
)


def test_milestone_follows_physical_state_instead_of_clock():
    reference = np.arange(100, dtype=float)[:, None]
    assert choose_milestone(reference, np.array([20.1]), 10)[:2] == (20, 36)
    assert choose_milestone(reference, np.array([5.0]), 20)[:2] == (20, 36)
    assert choose_milestone(reference, np.array([99.0]), 90)[:2] == (99, 99)


def test_selector_cannot_jump_to_distant_phase():
    reference = np.arange(100, dtype=float)[:, None]
    assert choose_milestone(reference, np.array([99.0]), 0)[:2] == (32, 48)


def test_invalid_state_fails_loudly():
    with pytest.raises(ValueError):
        choose_milestone(np.zeros((10, 2)), np.array([np.nan, 0]), 0)
    with pytest.raises(ValueError):
        choose_milestone(np.zeros((10, 2)), np.zeros(3), 0)


def test_phase_uses_object_state_and_excludes_relative_duplicates():
    obs = dict(
        robot0_eef_pos=np.ones(3) * 0.05,
        robot0_gripper_qpos=np.ones(2) * 0.02,
        mug_pos=np.zeros(3),
        mug_to_robot_pos=np.ones(3),
        robot0_joint_pos=np.zeros(7),
    )
    keys = phase_schema(obs)
    assert keys == ["robot0_eef_pos", "robot0_gripper_qpos", "mug_pos"]
    assert np.array_equal(phase_vector(obs, keys), [1, 1, 1, 1, 1, 0, 0, 0])


def test_no_objects_does_not_silently_degrade_to_ee_only():
    with pytest.raises(ValueError):
        phase_schema(dict(robot0_eef_pos=np.zeros(3)))


def test_hash_checks_content_shape_and_dtype():
    a = np.zeros((2, 2), dtype=np.uint8)
    assert array_hash(a) == array_hash(a.copy())
    assert array_hash(a) != array_hash(a.reshape(4))
    assert array_hash(a) != array_hash(a.astype(float))


def test_rng_deterministic_and_distinct_across_cases():
    assert case_seed(0, "goal", 0, 0) == case_seed(0, "goal", 0, 0)
    assert case_seed(0, "goal", 0, 0) != case_seed(0, "goal", 1, 0)


def test_failed_reference_is_not_counted_as_goal_failure():
    rows = [
        dict(case="action_reference", complete=True, success=False),
        dict(case="action_reference", complete=True, success=True),
        dict(case="goal_oracle", complete=True, eligible=False, skipped=True, success=False),
        dict(case="goal_oracle", complete=True, eligible=True, success=True),
    ]
    summary = summarize_episodes(rows)
    assert summary["baseline_success_pct"] == 50
    assert summary["eligible_goal_episodes"] == 1
    assert summary["conditional_goal_success_pct"] == 100
    assert summary["skipped_goal_episodes"] == 1
