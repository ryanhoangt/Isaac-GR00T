"""Observation/action mapping and HTTP inference for the reactive RoboMME baseline.

Mapping functions use only NumPy; the HTTP client lazily imports requests and
json_numpy. Image preprocessing and state/action normalization belong to
Gr00tPolicy and RoboMMEDataConfig, using the checkpoint's metadata.
"""

from collections.abc import Mapping
import math
from typing import Any
from urllib.parse import urlsplit

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


class RoboMMEGr00tClient:
    """Expose the benchmark infer/reset interface over GR00T's existing /act API.

    The model server owns checkpoint loading and normalization. This client
    keeps no episode memory or action queue, and does not retry inference calls.
    Use as a context manager or call close() when evaluation finishes.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:5555", *, timeout: float = 60.0):
        parsed = urlsplit(server_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("server_url must be an HTTP(S) base URL without a query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")

        import json_numpy
        import requests

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._json_numpy = json_numpy
        self._session = requests.Session()

    def infer(self, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Send the current observation and return {"actions": float32 (H, 8)}.

        HTTP and timeout errors propagate as requests exceptions. Invalid JSON
        or physical action fields raise ValueError. A failed call is not retried
        because inference may already have advanced the server's sampling RNG.
        """
        observation = to_gr00t_observation(inputs)
        response = self._session.post(
            f"{self.server_url}/act",
            data=self._json_numpy.dumps({"observation": observation}),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        prediction = self._json_numpy.loads(response.content)
        if not isinstance(prediction, Mapping):
            raise ValueError("GR00T /act response must be a mapping of physical action fields")
        return {"actions": to_robomme_actions(prediction)}

    def reset(self) -> dict[str, bool]:
        """Acknowledge the episode boundary for this stateless baseline.

        GR00T's HTTP API has no reset endpoint. No request is sent and the
        server's sampling RNG is not reseeded. The rollout runner owns its queue.
        """
        return {"reset_finished": True}

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RoboMMEGr00tClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
