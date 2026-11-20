"""Map RoboMME observations and physical GR00T actions for the reactive baseline.

These functions use only NumPy. Image preprocessing and state/action normalization
belong to Gr00tPolicy and RoboMMEDataConfig, using the checkpoint's metadata.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


OBSERVATION_BUFFERS = (
    "front_rgb_list",
    "wrist_rgb_list",
    "joint_state_list",
    "gripper_state_list",
)


def _float32_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return array


def _rgb_frame(value: Any, name: str) -> np.ndarray:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError(f"{name} must be a uint8 RGB frame with shape (height, width, 3)")
    return frame[None].copy()


def to_gr00t_observation(inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert one benchmark request into an unbatched GR00T observation.

    The four observation buffers must be nonempty and have equal lengths. Only
    their final entries are used, including on the first request when a buffer
    can also contain conditioning-video frames. State is seven joint positions
    and the first of two physical finger positions, matching training.

    Select the first task_goal alternative verbatim. Oracle subgoals, history,
    depth and camera parameters are not inputs to this baseline. Returned arrays
    own their data so preprocessing cannot mutate the benchmark buffers.
    """
    buffers = {}
    for key in OBSERVATION_BUFFERS:
        value = inputs.get(key)
        if (
            not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim == 0)
            or len(value) == 0
        ):
            raise ValueError(f"{key} must be a nonempty observation buffer")
        buffers[key] = value
    if len({len(value) for value in buffers.values()}) != 1:
        raise ValueError("Observation buffers must have equal lengths")

    goal = inputs.get("task_goal")
    if isinstance(goal, (list, tuple)) and goal:
        goal = goal[0]
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("task_goal must be a nonempty string or list of task instructions")

    joints = _float32_array(buffers["joint_state_list"][-1], "joint_state_list[-1]")
    gripper = _float32_array(buffers["gripper_state_list"][-1], "gripper_state_list[-1]")
    if joints.shape != (7,):
        raise ValueError("joint_state_list[-1] must have shape (7,)")
    if gripper.shape != (2,):
        raise ValueError("gripper_state_list[-1] must have shape (2,) for the two finger positions")

    return {
        "video.image": _rgb_frame(buffers["front_rgb_list"][-1], "front_rgb_list[-1]"),
        "video.wrist_image": _rgb_frame(buffers["wrist_rgb_list"][-1], "wrist_rgb_list[-1]"),
        "state.joint_position": joints[None].copy(),
        "state.gripper": gripper[None, :1].copy(),
        "annotation.human.action.task_description": np.array([goal]),
    }


def to_robomme_actions(prediction: Mapping[str, Any]) -> np.ndarray:
    """Pack unbatched, unnormalized Gr00tPolicy output into float32 (H, 8).

    Input fields must have shapes (H, 7) and (H, 1), with a common nonzero
    horizon (16 for the current config). Joints are already absolute targets;
    gripper is already in the command scale learned from -1/+1 labels. Values
    are preserved without another inverse transform, delta conversion, clipping
    or gripper threshold. The benchmark wrapper handles stick-task dimensions.
    """
    fields = ("action.joint_position", "action.gripper")
    for key in fields:
        if key not in prediction:
            raise ValueError(f"Missing physical action field: {key}")
    joints = _float32_array(prediction[fields[0]], fields[0])
    gripper = _float32_array(prediction[fields[1]], fields[1])
    if joints.ndim != 2 or joints.shape[1] != 7 or len(joints) == 0:
        raise ValueError("action.joint_position must have shape (H, 7) with H > 0")
    if gripper.shape != (len(joints), 1):
        raise ValueError("action.gripper must have shape (H, 1) with the same horizon as joint positions")
    return np.concatenate([joints, gripper], axis=1)
