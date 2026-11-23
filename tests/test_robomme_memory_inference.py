"""Offline/online history parity, episode boundaries, HTTP and saved-model inference."""

import copy
import json
from pathlib import Path
import threading

import numpy as np
import pytest

import test_robomme_adapter as adapter_tests
import test_robomme_memory_training as training_tests

from examples.RoboMME.feature_cache import json_fingerprint
from examples.RoboMME.memory_client import PROTOCOL, RoboMMEMemoryClient, to_memory_observation
from examples.RoboMME.memory_config import load_memory_config, prepare_frame_sampling
from examples.RoboMME.memory_policy import (
    EpisodeConflict,
    MemoryActionPolicy,
    MemoryEpisodeController,
    OnlineFrameMemory,
    checkpoint_memory_spec,
)
from examples.RoboMME.memory_server import create_memory_app
from examples.RoboMME.memory_training import MEMORY_DATA_CONFIG, prepare_memory_run
from examples.RoboMME.serve_memory import DEFAULT_CONFIG, load_runtime_config, validate_runtime_config


benchmark_inputs = adapter_tests.benchmark_inputs
prepared = training_tests.prepared
setup_training = training_tests.setup_training


def small_config():
    config = load_memory_config()
    config["budget"] = 32
    config["memory_feature"]["img"]["input_dim"] = 4
    return config


class Encoder:
    def encode(self, frames):
        return np.broadcast_to(frames[:, :1, :1, :1].reshape(-1, 1, 1) / 7.0, (len(frames), 16, 4)).copy()


class Policy:
    def __init__(self):
        self.observations = []

    def get_action(self, obs):
        self.observations.append(copy.deepcopy(obs))
        if obs["annotation.human.action.task_description"][0] == "fail":
            raise RuntimeError("policy failure")
        return {
            "action.joint_position": np.tile(obs["state.joint_position"], (16, 1)),
            "action.gripper": np.zeros((16, 1), np.float32),
        }


def make_controller(**kwargs):
    return MemoryEpisodeController(Policy(), Encoder(), small_config(), {"storage_dtype": "float16"}, **kwargs)


def payload(controller, inputs, *, reset=False):
    if reset:
        controller.reset()
    return {
        **to_memory_observation(inputs),
        "episode_id": controller._episode_id,
        "request_id": controller._request_id,
        "start_frame_index": controller.memory.length,
    }


def test_incremental_history_matches_offline_quantization_and_sampling():
    frames = np.stack([np.full((2, 3, 3), index, np.uint8) for index in range(9)])
    offline = Encoder().encode(frames).astype("float16")
    online = OnlineFrameMemory(Encoder(), small_config(), "float16", batch_size=2, max_frames=10)
    for start, stop in ((0, 3), (3, 4), (4, 9)):
        online.append(frames[start:stop])
        for key, value in prepare_frame_sampling(offline, stop - 1, small_config()).items():
            np.testing.assert_array_equal(online.prepare()[key], value)
    online.reset()
    online.append(frames[-1:])
    assert online.length == 1
    assert online.prepare()["static_mask"].sum() == 16


def test_prefill_delta_frames_reset_and_no_deduplication(benchmark_inputs):
    controller = make_controller()
    first = payload(controller, benchmark_inputs, reset=True)
    controller.infer(first)
    assert controller.memory.length == 2
    np.testing.assert_array_equal(
        controller.action_policy.observations[-1]["static_image_emb"],
        controller.memory.prepare()["static_image_emb"],
    )
    benchmark_inputs["is_first_step"] = False
    # Equal pixels still represent new timesteps, and must not be deduplicated.
    controller.infer(payload(controller, benchmark_inputs))
    assert controller.memory.length == 4
    old_id = controller._episode_id
    controller.reset()
    assert controller.memory.length == 0 and controller._episode_id != old_id
    with pytest.raises(EpisodeConflict):
        controller.infer(first)
    benchmark_inputs["is_first_step"] = True
    benchmark_inputs["task_goal"] = "Another episode"
    controller.infer(payload(controller, benchmark_inputs))
    assert controller.memory.length == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("episode_id", "stale"),
        ("request_id", 1),
        ("request_id", True),
        ("start_frame_index", 1),
        ("is_first_step", False),
    ],
)
def test_sequence_errors_do_not_mutate_memory(benchmark_inputs, key, value):
    controller = make_controller()
    request = payload(controller, benchmark_inputs, reset=True)
    request[key] = value
    with pytest.raises(EpisodeConflict):
        controller.infer(request)
    assert controller.memory.length == 0 and not controller._failed
    controller.infer(payload(controller, benchmark_inputs))


def test_bad_frame_capacity_and_goal_do_not_mutate_memory(benchmark_inputs):
    controller = make_controller(max_frames=3)
    request = payload(controller, benchmark_inputs, reset=True)
    request["front_frames"][-1] = 0
    with pytest.raises(ValueError, match="Final"):
        controller.infer(request)
    controller.infer(payload(controller, benchmark_inputs))
    benchmark_inputs["is_first_step"] = False
    with pytest.raises(ValueError, match="max_episode_frames"):
        controller.infer(payload(controller, benchmark_inputs))
    benchmark_inputs["task_goal"] = "Different goal"
    with pytest.raises(EpisodeConflict, match="instruction"):
        controller.infer(payload(controller, benchmark_inputs))
    assert controller.memory.length == 2 and not controller._failed


@pytest.mark.parametrize("kind", ["encode", "overflow", "policy"])
def test_failure_requires_reset_and_partial_encoding_is_not_published(benchmark_inputs, kind):
    controller = make_controller(batch_size=1)

    class BadEncoder:
        calls = 0

        def encode(self, frames):
            self.calls += 1
            if self.calls == 2:
                if kind == "overflow":
                    return np.full((1, 16, 4), 1e10)
                raise RuntimeError("encoding failure")
            return Encoder().encode(frames)

    if kind != "policy":
        controller.memory.encoder = BadEncoder()
    else:
        benchmark_inputs["task_goal"] = "fail"
    with pytest.raises(RuntimeError, match="reset"):
        controller.infer(payload(controller, benchmark_inputs, reset=True))
    assert controller.memory.length == (2 if kind == "policy" else 0)
    with pytest.raises(EpisodeConflict, match="failed"):
        controller.infer(payload(controller, benchmark_inputs))
    controller.reset()
    controller.memory.encoder = Encoder()
    benchmark_inputs["task_goal"] = "recovered"
    controller.infer(payload(controller, benchmark_inputs))


@pytest.fixture
def http_server():
    # A real HTTP connection checks JSON/NumPy handling without TestClient/httpx.
    import socket
    import time

    import requests
    import uvicorn

    controller = make_controller()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(create_memory_app(controller), log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    if requests.get(url + "/health", timeout=0.2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    pytest.fail("Memory HTTP server failed to start")
                time.sleep(0.02)
            yield url, controller
        finally:
            server.should_exit = True
            thread.join(5)
            assert not thread.is_alive()


def test_http_client_episode_round_trip(http_server, benchmark_inputs):
    url, controller = http_server
    with RoboMMEMemoryClient(url, timeout=3) as client:
        with pytest.raises(RuntimeError, match="reset"):
            client.infer(benchmark_inputs)
        client.reset()
        for first in (True, False):
            benchmark_inputs["is_first_step"] = first
            actions = client.infer(benchmark_inputs)["actions"]
            assert actions.shape == (16, 8) and actions.dtype == np.float32
        assert controller.memory.length == 4
        client.reset()
        benchmark_inputs["is_first_step"] = True
        client.infer(benchmark_inputs)
        assert controller.memory.length == 2
        # Another client resetting the server invalidates the previous token.
        with RoboMMEMemoryClient(url, timeout=3) as other:
            other.reset()
        benchmark_inputs["is_first_step"] = False
        import requests

        with pytest.raises(requests.HTTPError, match="409"):
            client.infer(benchmark_inputs)
        with pytest.raises(RuntimeError, match="reset"):
            client.infer(benchmark_inputs)


def test_http_bad_json_status_and_model_failure(http_server, benchmark_inputs):
    import requests

    url, controller = http_server
    assert requests.post(url + "/act", data="{bad", timeout=3).status_code == 400
    assert requests.post(url + "/reset", json={"unexpected": 1}, timeout=3).status_code == 400
    assert requests.post(url + "/act", json={}, timeout=3).status_code == 400
    assert requests.get(url + "/health", timeout=3).json()["protocol"] == PROTOCOL
    with RoboMMEMemoryClient(url, timeout=3) as client:
        client.reset()
        benchmark_inputs["task_goal"] = "fail"
        with pytest.raises(requests.HTTPError, match="500"):
            client.infer(benchmark_inputs)
        assert controller._failed
        with pytest.raises(RuntimeError, match="reset"):
            client.infer(benchmark_inputs)


def test_ambiguous_network_error_is_not_retried(benchmark_inputs, monkeypatch):
    import requests

    with RoboMMEMemoryClient() as client:
        client._episode_id = "active"
        calls = []

        def timeout(*args, **kwargs):
            calls.append(args)
            raise requests.Timeout("ambiguous")

        monkeypatch.setattr(client._session, "post", timeout)
        with pytest.raises(requests.Timeout):
            client.infer(benchmark_inputs)
        assert len(calls) == 1 and client._episode_id is None


def test_real_checkpoint_online_policy_and_embedded_provenance(setup_training):
    args, model, memory = setup_training
    dataset = training_tests.make_dataset(args, memory)
    model.enable_memory(memory)
    root = Path(args.output_dir)
    record = prepare_memory_run(args, dataset, memory, MEMORY_DATA_CONFIG, "spawn", None)
    model.save_pretrained(root)
    (root / "experiment_cfg/metadata.json").write_text(
        json.dumps({"new_embodiment": dataset.metadata.model_dump(mode="json")})
    )
    # Embedded specs make serving independent of the training cache's existence.
    cache = Path(args.memory_cache)
    manifest = cache / "manifest.json"
    hidden_manifest = cache / "hidden.json"
    manifest.rename(hidden_manifest)
    policy = MemoryActionPolicy(root, device="cpu", denoising_steps=2)
    frames = np.full((3, 32, 32, 3), 73, np.uint8)
    expected = training_tests.FrozenMemoryVisionEncoder(
        model.backbone.eagle_model, memory, device="cpu", compute_dtype="float32"
    ).encode(frames)
    np.testing.assert_allclose(policy.encode(frames), expected, atol=1e-6)
    obs = {
        "video.image": frames[-1:],
        "video.wrist_image": frames[-1:],
        "state.joint_position": np.zeros((1, 7), np.float32),
        "state.gripper": np.zeros((1, 1), np.float32),
        "annotation.human.action.task_description": np.array(["Task 7"]),
    }
    controller = MemoryEpisodeController(policy, policy, memory, policy.cache_spec["feature_spec"], max_frames=10)
    episode = controller.reset()["episode_id"]
    request = {
        "episode_id": episode,
        "request_id": 0,
        "start_frame_index": 0,
        "observation": obs,
        "front_frames": frames,
        "is_first_step": True,
    }
    actions = controller.infer(request)["actions"]
    assert actions.shape == (16, 8) and np.isfinite(actions).all()
    assert not any(parameter.requires_grad for parameter in policy.model.parameters())
    sidecar = root / "experiment_cfg/memory_training.json"
    # Compatibility with pre-step-6 checkpoints still validates the original manifest.
    legacy = copy.deepcopy(record)
    legacy.pop("cache_specs")
    sidecar.write_text(json.dumps(legacy))
    hidden_manifest.rename(manifest)  # Training resume still needs its cache.
    upgraded = prepare_memory_run(args, dataset, memory, MEMORY_DATA_CONFIG, "spawn", str(root))
    assert upgraded["identity_sha256"] == legacy["identity_sha256"]
    assert upgraded["cache_specs"] == record["cache_specs"]
    manifest.rename(hidden_manifest)
    sidecar.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="original cache manifest"):
        checkpoint_memory_spec(root)
    checkpoint_memory_spec(root, hidden_manifest)
    hidden_manifest.write_text(hidden_manifest.read_text() + " ")
    with pytest.raises(ValueError, match="differs"):
        checkpoint_memory_spec(root, hidden_manifest)
    bad = copy.deepcopy(record)
    bad["cache_specs"][0]["encoder"]["compute_dtype"] = "float16"
    sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="checksum"):
        checkpoint_memory_spec(root)
    spec = bad["cache_specs"][0]
    spec["spec_sha256"] = json_fingerprint({key: value for key, value in spec.items() if key != "spec_sha256"})
    sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="provenance"):
        checkpoint_memory_spec(root)
    dataset.close()


def test_runtime_yaml_and_invalid_limits():
    config = load_runtime_config(DEFAULT_CONFIG)
    assert validate_runtime_config(config)["encode_batch_size"] == 16
    for key in ("port", "encode_batch_size", "denoising_steps", "max_episode_frames"):
        with pytest.raises(ValueError, match=key):
            validate_runtime_config({**config, key: 0})
