"""Observation-only fixed-goal diagnosis; no object state or simulator snapshots."""

import numpy as np

from gr00t.rag.oracle_goal_protocol import array_hash


REFERENCE_KEYS = ("agent", "wrist", "ee", "quat", "gripper", "actions")
ROBOT_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def robot_observation(observation):
    return {key: np.asarray(observation[key]).copy() for key in ROBOT_KEYS}


def load_reference(path, prefix=64, horizon=16):
    # Deliberately never enumerate archive.files: older files also contain
    # privileged columns, which this experiment must not read.
    with np.load(path, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in REFERENCE_KEYS}
    if prefix < 0 or horizon < 1 or prefix + horizon >= len(reference["ee"]):
        raise ValueError("Reference does not cover prefix and complete future horizon")
    if len(reference["actions"]) < prefix + horizon:
        raise ValueError("Reference action sequence is too short")
    if any(len(reference[key]) != len(reference["ee"]) for key in REFERENCE_KEYS[:-1]):
        raise ValueError("Reference observation lengths differ")
    if reference["actions"].ndim != 2 or reference["actions"].shape[1] != 7:
        raise ValueError("Expected LIBERO seven-dimensional reference actions")
    if not all(np.isfinite(reference[key]).all() for key in ("ee", "quat", "gripper", "actions")):
        raise ValueError("Nonfinite reference data")
    return reference


def reference_observation(reference, index):
    return dict(zip(ROBOT_KEYS, [reference[key][index] for key in REFERENCE_KEYS[:-1]]))


def observation_errors(current, target):
    result = {}
    for name, key in (("workspace", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
        left, right = np.asarray(current[key]), np.asarray(target[key])
        if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("Expected matching uint8 camera observations")
        error = np.abs(left.astype(np.float64) - right.astype(np.float64))
        result[name + "_mae"] = float(error.mean())
        result[name + "_max_abs"] = float(error.max())
    xyz = np.asarray(current["robot0_eef_pos"]) - np.asarray(target["robot0_eef_pos"])
    grip = np.asarray(current["robot0_gripper_qpos"]) - np.asarray(target["robot0_gripper_qpos"])
    a, b = np.asarray(current["robot0_eef_quat"]), np.asarray(target["robot0_eef_quat"])
    # q and -q encode the same orientation.
    quaternion_error = min(float(np.abs(a - b).max()), float(np.abs(a + b).max()))
    cosine = float(np.clip(abs(np.dot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), 0, 1))
    result.update(
        ee_distance_m=float(np.linalg.norm(xyz)),
        gripper_distance_m=float(np.linalg.norm(grip)),
        orientation_error_deg=float(np.rad2deg(2 * np.arccos(cosine))),
        proprio_max_abs=max(float(np.abs(xyz).max()), float(np.abs(grip).max()), quaternion_error),
    )
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite observation comparison")
    return result


def observation_identity(observation):
    return {key: array_hash(observation[key]) for key in ROBOT_KEYS}


def replay_prefix(env, observation, reference, prefix, rgb_tolerance=0.0, proprio_tolerance=1e-6):
    """env only needs step(); no state/pose inspection or snapshot restoration."""
    checks = []
    for index in range(prefix + 1):
        observation = robot_observation(observation)
        error = observation_errors(observation, reference_observation(reference, index))
        matched = (
            max(error["workspace_max_abs"], error["wrist_max_abs"]) <= rgb_tolerance
            and error["proprio_max_abs"] <= proprio_tolerance
        )
        checks.append(dict(reference_frame=index, matched=matched, **error))
        if not matched:
            raise ValueError(f"Replay observation mismatch at frame {index}: {error}")
        if index < prefix:
            observation, _, _, _ = env.step(reference["actions"][index].tolist())
    return observation, checks


def target_metrics(trajectory, reference, goal_index):
    goal = reference_observation(reference, goal_index)
    errors = [observation_errors(obs, goal) for obs in trajectory]
    before, after = errors[0], errors[-1]
    result = {}
    for name in ("ee_distance_m", "orientation_error_deg", "gripper_distance_m", "workspace_mae", "wrist_mae"):
        result.update(
            {
                "start_" + name: before[name],
                "end_" + name: after[name],
                "reduction_" + name: before[name] - after[name],
            }
        )
    return result, errors


def action_agreement(actions, reference_actions):
    error = np.asarray(actions, dtype=float) - np.asarray(reference_actions, dtype=float)
    if error.ndim != 2 or error.shape[1] != 7 or not np.isfinite(error).all():
        raise ValueError("Expected matching finite action sequences")
    return dict(
        reference_action_rmse=float(np.sqrt(np.square(error).mean())),
        reference_translation_rmse=float(np.sqrt(np.square(error[:, :3]).mean())),
        reference_rotation_rmse=float(np.sqrt(np.square(error[:, 3:6]).mean())),
        reference_gripper_rmse=float(np.sqrt(np.square(error[:, 6:]).mean())),
        reference_action_max_abs=float(np.abs(error).max()),
    )
