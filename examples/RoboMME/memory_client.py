"""NumPy-only benchmark mapping and HTTP client for stateful memory inference."""

from collections.abc import Mapping

import numpy as np

from examples.RoboMME.adapter import RoboMMEGr00tClient, _float32_array, _rgb_frame, to_gr00t_observation


PROTOCOL = "robomme-memory-v1"


def to_memory_observation(inputs):
    observation = to_gr00t_observation(inputs)
    first = inputs.get("is_first_step")
    if not isinstance(first, (bool, np.bool_)):
        raise ValueError("is_first_step must be boolean")
    front = [_rgb_frame(frame, "front_rgb_list frame")[0] for frame in inputs["front_rgb_list"]]
    if any(frame.shape != front[0].shape for frame in front):
        raise ValueError("New front frames must have a consistent shape")
    return {"observation": observation, "front_frames": np.stack(front), "is_first_step": bool(first)}


class RoboMMEMemoryClient(RoboMMEGr00tClient):
    """One client/active episode; explicit reset and ordered delta-frame requests.

    Every input buffer contains only newly observed frames. On the first request
    these include the conditioning video and current frame. No call is retried.
    A failed/ambiguous HTTP request requires reset before another inference.
    """

    def __init__(self, server_url="http://127.0.0.1:8000", *, timeout=60.0):
        super().__init__(server_url, timeout=timeout)
        self._episode_id = None
        self._request_id = 0
        self._frame_count = 0

    def _post(self, endpoint, payload):
        response = self._session.post(
            f"{self.server_url}/{endpoint}",
            data=self._json_numpy.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        value = self._json_numpy.loads(response.content)
        if not isinstance(value, Mapping):
            raise ValueError("Memory server response must be a mapping")
        return value

    def reset(self):
        self._episode_id = None
        value = self._post("reset", {})
        if (
            value.get("reset_finished") is not True
            or value.get("protocol") != PROTOCOL
            or not isinstance(value.get("episode_id"), str)
            or not value["episode_id"]
        ):
            raise ValueError("Invalid memory reset response")
        self._episode_id = value["episode_id"]
        self._request_id = self._frame_count = 0
        return {"reset_finished": True}

    def infer(self, inputs):
        if self._episode_id is None:
            raise RuntimeError("Call reset() before memory inference or after a failed request")
        # Validate before sending: local input errors do not advance server state.
        mapped = to_memory_observation(inputs)
        if mapped["is_first_step"] != (self._request_id == 0):
            raise ValueError("is_first_step disagrees with the current episode/request")
        payload = {
            **mapped,
            "episode_id": self._episode_id,
            "request_id": self._request_id,
            "start_frame_index": self._frame_count,
        }
        expected_count = self._frame_count + len(mapped["front_frames"])
        try:
            value = self._post("act", payload)
            if (
                value.get("episode_id") != self._episode_id
                or value.get("request_id") != self._request_id
                or value.get("frame_count") != expected_count
                or value.get("protocol") != PROTOCOL
            ):
                raise ValueError("Memory server episode/request/frame acknowledgement differs")
            actions = _float32_array(value.get("actions"), "actions")
            if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions):
                raise ValueError("Expected finite physical actions [horizon,8]")
        except BaseException:
            self._episode_id = None
            raise
        self._request_id += 1
        self._frame_count = expected_count
        return {"actions": actions}
