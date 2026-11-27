"""Shared LIBERO observation/action mapping, independent of model and simulator imports."""

from collections.abc import Mapping

import numpy as np

from examples.Libero.eval.utils import get_libero_image, normalize_gripper_action, quat2axisangle


ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def to_gr00t_observation(obs, language):
    """Single current observation; normalization and resizing stay on the server."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a nonempty instruction")
    vectors = {}
    for key, size in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2)):
        array = np.asarray(obs[key])
        if array.shape != (size,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{key} must be a finite numeric vector of length {size}")
        vectors[key] = array
    xyz = vectors["robot0_eef_pos"]
    rpy = quat2axisangle(vectors["robot0_eef_quat"].copy())
    image, wrist = get_libero_image(obs)
    for frame in (image, wrist):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("LIBERO camera observations must be uint8 HWC RGB frames")
    result = {
        "video.image": image[None].copy(),
        "video.wrist_image": wrist[None].copy(),
        "state.gripper": vectors["robot0_gripper_qpos"][None].copy(),
        "annotation.human.action.task_description": [language],
    }
    for key, value in zip(ACTION_KEYS[:6], [*xyz, *rpy]):
        result[f"state.{key}"] = np.array([[value]])
    return result


def to_libero_actions(prediction):
    """Physical action fields -> (H, 7), using LIBERO's existing gripper convention."""
    if not isinstance(prediction, Mapping):
        raise ValueError("Prediction must be a mapping of physical action fields")
    components = []
    for key in ACTION_KEYS:
        name = f"action.{key}"
        if name not in prediction:
            raise ValueError(f"Missing physical action field: {name}")
        array = np.asarray(prediction[name])
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != 1 or len(array) == 0 or array.dtype.kind not in "fiu":
            raise ValueError(f"{name} must have shape (H, 1) or (H,) with H > 0")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite float32 values")
        components.append(array)
    if len({len(array) for array in components}) != 1:
        raise ValueError("Action components must have the same horizon")
    return normalize_gripper_action(np.concatenate(components, axis=1), binarize=True)
