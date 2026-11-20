"""Check mapping and the real GR00T HTTP protocol with a fake CPU policy."""

import copy
import multiprocessing
import socket
import time

import numpy as np
import pytest

from examples.RoboMME.adapter import RoboMMEGr00tClient, to_gr00t_observation, to_robomme_actions


@pytest.fixture
def benchmark_inputs():
    front = np.zeros((12, 16, 3), np.uint8)
    wrist = np.zeros_like(front)
    front[..., 0], wrist[..., 2] = 255, 255
    return {
        "front_rgb_list": [np.full_like(front, 17), front],
        "wrist_rgb_list": [np.full_like(wrist, 29), wrist],
        "joint_state_list": [np.full(7, 99.0), np.arange(7, dtype=np.float64) / 10],
        "gripper_state_list": [np.array([0.04, 0.04]), np.array([0.012, 0.023])],
        "task_goal": ["Pick the RED cube", "An alternative instruction"],
        "is_first_step": True,
        "simple_subgoal": "Oracle annotation must not become the instruction",
    }


@pytest.mark.parametrize("first_step", [True, False])
def test_latest_frame_and_physical_state(benchmark_inputs, first_step):
    benchmark_inputs["is_first_step"] = first_step
    observation = to_gr00t_observation(benchmark_inputs)
    assert set(observation) == {
        "video.image",
        "video.wrist_image",
        "state.joint_position",
        "state.gripper",
        "annotation.human.action.task_description",
    }
    for source, destination in (("front_rgb_list", "video.image"), ("wrist_rgb_list", "video.wrist_image")):
        assert observation[destination].shape == (1, 12, 16, 3)
        assert observation[destination].dtype == np.uint8
        np.testing.assert_array_equal(observation[destination][0], benchmark_inputs[source][-1])
    assert observation["state.joint_position"].shape == (1, 7)
    assert observation["state.joint_position"].dtype == np.float32
    np.testing.assert_allclose(observation["state.joint_position"][0], benchmark_inputs["joint_state_list"][-1])
    assert observation["state.gripper"].shape == (1, 1)
    assert observation["state.gripper"].dtype == np.float32
    np.testing.assert_allclose(observation["state.gripper"], [[0.012]])
    assert observation["annotation.human.action.task_description"].tolist() == ["Pick the RED cube"]


def test_single_frame_and_stick_state(benchmark_inputs):
    for key in ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list"):
        benchmark_inputs[key] = [benchmark_inputs[key][-1]]
    benchmark_inputs["gripper_state_list"] = [np.zeros(2)]
    benchmark_inputs["task_goal"] = "Draw a pattern"
    observation = to_gr00t_observation(benchmark_inputs)
    np.testing.assert_array_equal(observation["state.gripper"], [[0.0]])
    assert observation["annotation.human.action.task_description"].tolist() == ["Draw a pattern"]


def test_mapping_does_not_share_or_mutate_buffers(benchmark_inputs):
    before = copy.deepcopy(benchmark_inputs)
    observation = to_gr00t_observation(benchmark_inputs)
    for key in ("video.image", "video.wrist_image", "state.joint_position", "state.gripper"):
        observation[key][...] = 0
    for key in ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list"):
        for old, current in zip(before[key], benchmark_inputs[key]):
            np.testing.assert_array_equal(current, old)
    assert benchmark_inputs["task_goal"] == before["task_goal"]


@pytest.mark.parametrize("invalid", [None, [], "", [""], [42]])
def test_invalid_instruction_is_rejected(benchmark_inputs, invalid):
    benchmark_inputs["task_goal"] = invalid
    with pytest.raises(ValueError, match="task_goal"):
        to_gr00t_observation(benchmark_inputs)


@pytest.mark.parametrize("key", ["front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list"])
def test_missing_empty_and_unaligned_buffers_are_rejected(benchmark_inputs, key):
    missing = dict(benchmark_inputs)
    missing.pop(key)
    with pytest.raises(ValueError, match=key):
        to_gr00t_observation(missing)
    benchmark_inputs[key] = []
    with pytest.raises(ValueError, match=key):
        to_gr00t_observation(benchmark_inputs)
    benchmark_inputs[key] = [None]
    with pytest.raises(ValueError, match="equal lengths"):
        to_gr00t_observation(benchmark_inputs)


@pytest.mark.parametrize(
    "key,invalid,message",
    [
        ("front_rgb_list", np.zeros((12, 16, 3), np.float32), "uint8 RGB"),
        ("wrist_rgb_list", np.zeros((3, 12, 16), np.uint8), "uint8 RGB"),
        ("joint_state_list", np.zeros((1, 7)), "shape"),
        ("gripper_state_list", np.zeros(1), "two finger positions"),
        ("joint_state_list", np.full(7, np.nan), "finite"),
        ("gripper_state_list", np.full(2, np.inf), "finite"),
    ],
)
def test_invalid_current_observation_is_rejected(benchmark_inputs, key, invalid, message):
    benchmark_inputs[key][-1] = invalid
    with pytest.raises(ValueError, match=message):
        to_gr00t_observation(benchmark_inputs)


@pytest.mark.parametrize("horizon", [1, 16])
def test_physical_action_order_and_values(horizon):
    joints = np.arange(horizon * 7, dtype=np.float64).reshape(horizon, 7) / 10 - 3
    gripper = np.linspace(-1, 1, horizon)[:, None]
    prediction = {"action.joint_position": joints, "action.gripper": gripper}
    actions = to_robomme_actions(prediction)
    assert actions.shape == (horizon, 8) and actions.dtype == np.float32
    np.testing.assert_allclose(actions[:, :7], joints)
    np.testing.assert_allclose(actions[:, 7:], gripper)
    actions[...] = 0
    np.testing.assert_array_equal(prediction["action.joint_position"], joints)
    np.testing.assert_array_equal(prediction["action.gripper"], gripper)
    assert not np.shares_memory(actions, joints) and not np.shares_memory(actions, gripper)


@pytest.mark.parametrize(
    "prediction,message",
    [
        ({"action": np.zeros((16, 32))}, "Missing physical action field"),
        ({"action.joint_position": np.zeros((1, 16, 7)), "action.gripper": np.zeros((1, 16, 1))}, "shape"),
        ({"action.joint_position": np.zeros((0, 7)), "action.gripper": np.zeros((0, 1))}, "H > 0"),
        ({"action.joint_position": np.zeros((16, 7)), "action.gripper": np.zeros((15, 1))}, "same horizon"),
        ({"action.joint_position": np.zeros((16, 7)), "action.gripper": np.zeros(16)}, "shape"),
        ({"action.joint_position": np.full((16, 7), np.nan), "action.gripper": np.zeros((16, 1))}, "finite"),
        ({"action.joint_position": np.zeros((16, 7)), "action.gripper": np.full((16, 1), np.inf)}, "finite"),
    ],
)
def test_invalid_physical_action_is_rejected(prediction, message):
    with pytest.raises(ValueError, match=message):
        to_robomme_actions(prediction)


def _serve_gr00t_stub(listener, observations):
    # Keep the server's global json_numpy.patch() and heavy model imports in
    # the child process, exactly as in a separate simulator/model environment.
    import uvicorn

    from gr00t.eval.http_server import HTTPInferenceServer

    class FakePolicy:
        def get_action(self, observation):
            observations.put(observation)
            goal = observation["annotation.human.action.task_description"][0]
            if goal == "raise an error":
                raise RuntimeError("stub inference failed")
            if goal == "invalid output":
                return {"action": np.zeros((16, 32), np.float32)}
            return {
                "action.joint_position": observation["state.joint_position"] + np.arange(16)[:, None] / 100,
                "action.gripper": np.linspace(-1, 1, 16, dtype=np.float32)[:, None],
            }

    app = HTTPInferenceServer(FakePolicy(), port=0).app
    uvicorn.Server(uvicorn.Config(app, log_level="error")).run(sockets=[listener])


@pytest.fixture(scope="module")
def gr00t_http_server():
    import requests

    context = multiprocessing.get_context("spawn")
    observations = context.Queue()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = context.Process(target=_serve_gr00t_stub, args=(listener, observations))
        process.start()
        try:
            deadline = time.monotonic() + 40
            while True:
                if not process.is_alive():
                    pytest.fail(f"GR00T stub server exited with code {process.exitcode}")
                try:
                    response = requests.get(f"{url}/health", timeout=0.2)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail("GR00T stub server did not become healthy")
                time.sleep(0.05)
            yield url, observations
        finally:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            observations.close()
            observations.join_thread()


def test_http_round_trip_and_episode_reset(gr00t_http_server, benchmark_inputs):
    url, observations = gr00t_http_server
    benchmark_inputs["task_goal"] = ["Pick the RED cube — lấy khối đỏ"]
    with RoboMMEGr00tClient(url + "/", timeout=5) as client:
        assert client.reset() == {"reset_finished": True}
        for step in range(2):
            expected = to_gr00t_observation(benchmark_inputs)
            result = client.infer(benchmark_inputs)
            received = observations.get(timeout=5)
            assert set(received) == set(expected)
            for key in expected:
                np.testing.assert_array_equal(received[key], expected[key])
                assert received[key].dtype == expected[key].dtype
            assert set(result) == {"actions"}
            assert result["actions"].shape == (16, 8) and result["actions"].dtype == np.float32
            np.testing.assert_allclose(
                result["actions"][:, :7], expected["state.joint_position"] + np.arange(16)[:, None] / 100
            )
            np.testing.assert_allclose(result["actions"][:, 7], np.linspace(-1, 1, 16), atol=1e-7)
            assert client.reset() == {"reset_finished": True}
            # Next episode must use its own observation, without client-side history.
            if step == 0:
                benchmark_inputs = copy.deepcopy(benchmark_inputs)
                benchmark_inputs["joint_state_list"][-1] = np.full(7, -0.75)
                benchmark_inputs["front_rgb_list"][-1][...] = 83
                benchmark_inputs["task_goal"] = ["A different episode"]


@pytest.mark.parametrize(
    "goal,exception,message",
    [
        ("raise an error", "http", "500"),
        ("invalid output", "mapping", "Missing physical action field"),
    ],
)
def test_http_errors_and_invalid_actions(gr00t_http_server, benchmark_inputs, goal, exception, message):
    import requests

    url, observations = gr00t_http_server
    benchmark_inputs["task_goal"] = [goal]
    error = requests.HTTPError if exception == "http" else ValueError
    with RoboMMEGr00tClient(url, timeout=5) as client:
        with pytest.raises(error, match=message):
            client.infer(benchmark_inputs)
    observations.get(timeout=5)


def test_reset_makes_no_http_request_and_inference_errors_are_not_retried(monkeypatch, benchmark_inputs):
    import requests

    calls = []
    with RoboMMEGr00tClient(timeout=0.25) as client:

        def timeout(url, **kwargs):
            calls.append((url, kwargs))
            raise requests.Timeout("stub timeout")

        monkeypatch.setattr(client._session, "post", timeout)
        assert client.reset() == {"reset_finished": True} and not calls
        with pytest.raises(requests.Timeout, match="stub timeout"):
            client.infer(benchmark_inputs)
        assert len(calls) == 1 and calls[0][1]["timeout"] == 0.25


@pytest.mark.parametrize(
    "body,message",
    [
        (b"not JSON", "Expecting value"),
        (b"[]", "must be a mapping"),
    ],
)
def test_invalid_server_response(monkeypatch, benchmark_inputs, body, message):
    import requests

    response = requests.Response()
    response.status_code = 200
    response._content = body
    with RoboMMEGr00tClient() as client:
        monkeypatch.setattr(client._session, "post", lambda *args, **kwargs: response)
        with pytest.raises(ValueError, match=message):
            client.infer(benchmark_inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"server_url": "127.0.0.1:5555"},
        {"server_url": "http://"},
        {"server_url": "http://localhost:5555?query=value"},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
    ],
)
def test_invalid_client_configuration(kwargs):
    with pytest.raises(ValueError):
        RoboMMEGr00tClient(**kwargs)
