"""Recorder integrity and evaluator chunk/RPC routing without a simulator."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.Libero.eval.uncertainty_recording import EpisodeUncertaintyRecorder


def signature():
    return {"score_id": "test", "action_horizon": 16}


def result():
    return {
        "action": {"data": np.zeros((16, 7))},
        "u_ctx": 0.1,
        "score_id": "test",
        "context": {"indices": [2], "similarities": [0.8]},
        "weights": np.asarray([0.25, 0.75]),
    }


def test_incremental_partial_recording_and_exclusive_episode_ownership(tmp_path):
    directory = tmp_path / "episode"
    recorder = EpisodeUncertaintyRecorder(directory, 3, 0, signature(), True)
    with pytest.raises(FileExistsError):
        EpisodeUncertaintyRecorder(directory, 3, 0, signature())
    obs = {
        "video.image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "annotation.human.action.task_description": ["task"],
    }
    recorder.record(0, obs, result(), np.zeros((16, 7)))
    assert json.loads((directory / "episode.json").read_text())["complete"] is False
    line = json.loads((directory / "queries.jsonl").read_text())
    assert "action" not in line and line["control_step"] == 0
    with np.load(directory / line["observation_path"], allow_pickle=False) as data:
        assert data["annotation.human.action.task_description"].tolist() == ["task"]
    with pytest.raises(ValueError):
        recorder.record(0, obs, result(), np.zeros((16, 7)))
    with pytest.raises(ValueError):
        recorder.finish(False, 32)
    recorder.record(16, obs, result(), np.zeros((16, 7)))
    recorder.finish(False, 21)
    metadata = json.loads((directory / "episode.json").read_text())
    assert (
        metadata["complete"]
        and not metadata["success"]
        and metadata["num_queries"] == 2
    )
    assert not list(directory.glob("*.pending"))


@pytest.fixture
def evaluator(monkeypatch):
    # Only environment imports are stubbed. Exercise the actual policy wrapper.
    fake = ModuleType("libero.libero")
    fake.benchmark = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "libero", ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", fake)
    utils = ModuleType("examples.Libero.eval.utils")
    for name in [
        "get_libero_dummy_action",
        "get_libero_env",
        "get_libero_image",
        "normalize_gripper_action",
        "quat2axisangle",
        "save_rollout_video",
    ]:
        setattr(utils, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = (
        Path(__file__).resolve().parents[1] / "examples/Libero/eval/run_libero_eval.py"
    )
    spec = importlib.util.spec_from_file_location("context_recording_eval", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["baseline", "score", "diagnostics"])
def test_scores_only_at_chunk_boundaries_and_executes_returned_actor_actions(
    evaluator, monkeypatch, tmp_path, mode
):
    import gr00t.eval.service as service

    endpoints = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def call_endpoint(self, name, *args, **kwargs):
            endpoints.append(name)
            return signature() if name == "get_score_signature" else result()

        def get_action(self, obs):
            endpoints.append("get_action")
            return result()["action"]

    monkeypatch.setattr(service, "ExternalRobotInferenceClient", Client)
    wrapper = evaluator.GR00TPolicy(
        recording_root=tmp_path if mode != "baseline" else None,
        context_diagnostics=mode == "diagnostics",
    )
    wrapper._process_observation = lambda obs, lang: obs
    wrapper._convert_to_libero_action = lambda action: action["data"]
    wrapper.reset(task_id=3, episode_index=0)
    for _ in range(21):
        np.testing.assert_array_equal(wrapper.get_action({}, "task"), np.zeros(7))
    wrapper.finish_episode(False)
    if mode == "baseline":
        assert endpoints == ["get_action", "get_action"]
        assert not list(tmp_path.iterdir())
    else:
        endpoint = (
            "get_action_with_context_diagnostics"
            if mode == "diagnostics"
            else "get_action_with_uncertainty"
        )
        assert endpoints == ["get_score_signature", endpoint, endpoint]
        rows = [
            json.loads(line)
            for line in (tmp_path / "task03_init000/queries.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["control_step"] for r in rows] == [0, 16]
        metadata = json.loads((tmp_path / "task03_init000/episode.json").read_text())
        assert metadata["control_steps"] == 21 and metadata["complete"]
