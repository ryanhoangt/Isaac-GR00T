import numpy as np
import pytest

from gr00t.rag.fixed_goal_protocol import (
    REFERENCE_KEYS,
    ROBOT_KEYS,
    action_agreement,
    load_reference,
    observation_errors,
    reference_observation,
    replay_prefix,
    robot_observation,
    target_metrics,
)


def observation(value=0):
    return dict(
        agentview_image=np.full((4, 4, 3), value, dtype=np.uint8),
        robot0_eye_in_hand_image=np.full((4, 4, 3), value, dtype=np.uint8),
        robot0_eef_pos=np.array([value, 0.0, 0.0]),
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        robot0_gripper_qpos=np.zeros(2),
    )


def reference():
    obs = [observation(i) for i in range(4)]
    result = {key: np.asarray([o[k] for o in obs]) for key, k in zip(REFERENCE_KEYS[:-1], ROBOT_KEYS)}
    result["actions"] = np.zeros((3, 7))
    return result


def test_loader_never_reads_privileged_archive_columns(monkeypatch):
    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __getitem__(self, key):
            assert key in REFERENCE_KEYS
            return reference()[key]

        @property
        def files(self):
            raise AssertionError("Must not enumerate privileged fields")

    monkeypatch.setattr(np, "load", lambda *a, **kw: Archive())
    assert set(load_reference("unused", prefix=1, horizon=2)) == set(REFERENCE_KEYS)


def test_hidden_observation_fields_are_discarded():
    obs = observation()
    obs.update(object_position=np.ones(3), simulator_state=np.ones(100))
    assert set(robot_observation(obs)) == set(ROBOT_KEYS)


def test_replay_works_with_step_only_environment():
    class Env:
        index = 0

        def step(self, action):
            assert action == [0] * 7
            self.index += 1
            return observation(self.index), 0, False, {}

        def __getattr__(self, name):
            raise AssertionError("Forbidden simulator access: " + name)

    current, checks = replay_prefix(Env(), observation(), reference(), 2)
    assert len(checks) == 3 and all(c["matched"] for c in checks)
    assert np.array_equal(current["robot0_eef_pos"], [2, 0, 0])


def test_replay_fails_on_wrong_scene():
    with pytest.raises(ValueError, match="frame 0"):
        replay_prefix(None, observation(1), reference(), 0)


def test_replay_checks_wrist_and_proprio_too():
    obs = observation()
    obs["robot0_eye_in_hand_image"][0, 0, 0] = 1
    with pytest.raises(ValueError):
        replay_prefix(None, obs, reference(), 0)
    obs = observation()
    obs["robot0_gripper_qpos"][0] = 0.001
    with pytest.raises(ValueError):
        replay_prefix(None, obs, reference(), 0)


def test_quaternion_sign_does_not_mean_different_pose():
    obs = observation()
    obs["robot0_eef_quat"] *= -1
    errors = observation_errors(obs, observation())
    assert errors["orientation_error_deg"] == 0 and errors["proprio_max_abs"] == 0


def test_relative_metrics_target_one_fixed_frame():
    metrics, errors = target_metrics([observation(0), observation(1)], reference(), 2)
    assert metrics["start_ee_distance_m"] == 2
    assert metrics["end_ee_distance_m"] == 1
    assert metrics["reduction_workspace_mae"] == 1
    assert len(errors) == 2


def test_future_horizon_not_silently_clamped(tmp_path):
    path = tmp_path / "reference.npz"
    np.savez(path, **reference())
    with pytest.raises(ValueError):
        load_reference(path, prefix=2, horizon=2)


def test_reference_actions_are_a_comparator_not_expert_labels():
    errors = action_agreement(np.ones((16, 7)), np.zeros((16, 7)))
    assert errors["reference_action_rmse"] == 1
    assert errors["reference_gripper_rmse"] == 1


def test_reference_order_matches_robot_keys():
    obs = reference_observation(reference(), 1)
    assert set(obs) == set(ROBOT_KEYS)
    assert observation_errors(obs, observation(1))["proprio_max_abs"] == 0
