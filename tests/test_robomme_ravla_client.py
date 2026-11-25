"""Exercise real ZeroMQ transport with a CPU-only task-scoped fake policy."""

import multiprocessing
from pathlib import Path
import socket

import numpy as np
import pytest

from examples.RoboMME.adapter import RoboMMERAVLAClient


def fake_server(port, ready, checkpoint):
    from gr00t.eval.service import BaseInferenceServer

    server = BaseInferenceServer("127.0.0.1", port)
    count = 0
    info = {
        "protocol": "ravla",
        "checkpoint_path": checkpoint,
        "memory_key_type": "canonical_task",
        "memory_tasks": ["A", "B"],
        "memory_demo_counts": {"A": 3, "B": 3},
        "action_horizon": 16,
        "action_keys": ["action.joint_position", "action.gripper"],
    }

    def action(observation):
        nonlocal count
        count += 1
        key = observation["memory_key"]
        if key not in ("A", "B"):
            raise ValueError("Unknown memory bank")
        joints = observation["state.joint_position"] + np.arange(16, dtype=np.float32)[:, None] / 100
        return {
            "action.joint_position": joints + (10 if key == "B" else 0),
            "action.gripper": np.linspace(-1, 1, 16, dtype=np.float32)[:, None],
        }

    def stop():
        server.running = False
        return {"stopped": True}

    server.register_endpoint("get_action", action)
    server.register_endpoint("server_info", lambda: info, requires_input=False)
    server.register_endpoint("count", lambda: {"calls": count}, requires_input=False)
    server.register_endpoint("kill", stop, requires_input=False)
    ready.set()
    try:
        server.run()
    finally:
        server.socket.close(linger=0)
        server.context.term()


@pytest.fixture
def endpoint(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    checkpoint = str((tmp_path / "checkpoint_1").resolve())
    process = context.Process(target=fake_server, args=(port, ready, checkpoint))
    process.start()
    assert ready.wait(15), "Fake server did not start"
    try:
        yield f"tcp://127.0.0.1:{port}", checkpoint
    finally:
        try:
            with RoboMMERAVLAClient(f"tcp://127.0.0.1:{port}", timeout=1) as client:
                client._transport.call_endpoint("kill", requires_input=False)
        finally:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)


def inputs(task="A"):
    return {
        "canonical_task": task,
        "task_goal": ["same instruction"],
        "front_rgb_list": [np.zeros((8, 8, 3), np.uint8)],
        "wrist_rgb_list": [np.zeros((8, 8, 3), np.uint8)],
        "joint_state_list": [np.arange(7, dtype=np.float32) / 10],
        "gripper_state_list": [np.array([0.01, 0.02], np.float32)],
    }


def test_real_rpc_keeps_banks_separate_and_reset_is_stateless(endpoint):
    address, checkpoint = endpoint
    with RoboMMERAVLAClient(address, timeout=5) as client:
        client.validate_server(checkpoint, ["A", "B"])
        first = client.infer(inputs("A"))["actions"]
        assert first.shape == (16, 8) and first.dtype == np.float32
        assert client.reset() == {"reset_finished": True}
        assert client._transport.call_endpoint("count", requires_input=False)["calls"] == 1
        second = client.infer(inputs("B"))["actions"]
        np.testing.assert_allclose(second[:, :7] - first[:, :7], 10, atol=1e-6)
        np.testing.assert_allclose(first[:, 7], np.linspace(-1, 1, 16), atol=1e-7)
        np.testing.assert_array_equal(second[:, 7], first[:, 7])
    client.close()  # Explicit close is idempotent.


def test_preflight_rejects_checkpoint_support_and_budget_mismatch(endpoint):
    address, checkpoint = endpoint
    with RoboMMERAVLAClient(address, timeout=5) as client:
        with pytest.raises(ValueError, match="checkpoint differs"):
            client.validate_server(str(Path(checkpoint).parent / "checkpoint_2"), ["A"])
        with pytest.raises(ValueError, match="missing task banks"):
            client.validate_server(checkpoint, ["heldout"])
        with pytest.raises(ValueError, match="support budget"):
            client.validate_server(checkpoint, ["A"], support_demos_per_task=5)
        assert client._transport.call_endpoint("count", requires_input=False)["calls"] == 0


def test_failed_inference_is_not_retried_and_connection_recovers(endpoint):
    address, _ = endpoint
    with RoboMMERAVLAClient(address, timeout=5) as client:
        with pytest.raises(ValueError, match="canonical_task"):
            request = inputs()
            request.pop("canonical_task")
            client.infer(request)
        with pytest.raises(RuntimeError, match="Unknown memory bank"):
            client.infer(inputs("unknown"))
        assert client._transport.call_endpoint("count", requires_input=False)["calls"] == 1
        assert client.infer(inputs())["actions"].shape == (16, 8)


@pytest.mark.parametrize(
    "address", ["http://localhost:8071", "tcp://localhost", "tcp://localhost:0", "tcp://host:8071/path"]
)
def test_invalid_transport_address_is_rejected(address):
    with pytest.raises(ValueError):
        RoboMMERAVLAClient.parse_address(address)
