"""Exercise offline video sampling and resumable cache writes without a GPU."""

import json
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from scripts.rag import precompute_retrieval as precompute


def make_video(path, count=35):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for index in range(count):
            image = np.zeros((32, 64, 3), dtype=np.uint8)
            image[:, :32, 0] = 50 + index * 4
            image[:, 32:, 2] = 200 - index * 3
            image[:8, :8, 1] = 180
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FakeEncoder:
    device = "cpu"

    def __init__(self):
        self.memory = {"test task": {"keys": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])}}
        self.frames = []

    def encode(self, frames, task, keys, top_k):
        self.frames.extend(frame.copy() for frame in frames)
        queries = np.array(
            [[frame[:, :32, 0].mean(), frame[:, 32:, 2].mean()] for frame in frames], dtype=np.float32
        )
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        scores = queries @ keys.numpy().T
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        return queries, np.take_along_axis(scores, indices, axis=1), indices


@pytest.fixture
def sample(tmp_path):
    video = tmp_path / "rollout.mp4"
    make_video(video)
    output = tmp_path / "cache"
    output.mkdir()
    episode = {
        "ordinal": 1,
        "task_id": 0,
        "episode_index": 0,
        "task_description": "test task",
        "success": False,
        "control_steps": 35,
        "video": precompute.stamp(video),
    }
    plan = precompute.RunPlan(
        tmp_path,
        output,
        tmp_path,
        tmp_path,
        tmp_path / "memory.pt",
        "libero_goal",
        16,
        16,
        10,
        1,
        1,
        [episode],
        {"test": 1},
    )
    args = SimpleNamespace(top_k=20, batch_size=2, decode_threads=1, overwrite=False, cuda=0)
    return plan, episode, args


def test_decode_alignment_cache_and_cpu_resume(sample, monkeypatch):
    plan, episode, args = sample
    encoder = FakeEncoder()
    metadata = precompute.compute_episode(plan, episode, encoder, args)
    with av.open(episode["video"]["path"]) as video:
        reference = [frame.to_ndarray(format="rgb24") for frame in video.decode(video.streams.video[0])]
    assert len(encoder.frames) == 3
    for actual, expected in zip(encoder.frames, [reference[0], reference[16], reference[32]]):
        np.testing.assert_array_equal(actual, expected)
    assert metadata["effective_top_k"] == 3  # Clip diagnostic K to available candidates.
    assert metadata["is_original_retrieval_trace"] is False
    destination = precompute.cache_path(plan, episode)
    with np.load(destination, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["frame_indices"], [0, 16, 32])
        np.testing.assert_array_equal(cache["simulator_steps"], [10, 26, 42])
        assert cache["topk_indices"].shape == (3, 3)
        assert np.all(np.diff(cache["topk_scores"], axis=1) <= 0)
        assert json.loads(cache["metadata_json"].item())["num_queries"] == 3
    precompute.validate_cache(destination, precompute.episode_signature(plan, episode), episode, 16)

    def forbidden(*_args):
        raise AssertionError("A fully cached run must not load a model")

    monkeypatch.setattr(precompute, "Encoder", forbidden)
    result = precompute.process_run(plan, args)
    assert result["status"] == "complete"
    assert result["skipped"] == 1
    assert result["computed"] == 0
    assert result["model_load_seconds"] == 0


def test_video_length_mismatch_does_not_publish_cache(sample):
    plan, episode, args = sample
    episode["control_steps"] = 36
    with pytest.raises(ValueError, match="Video has 35 frames but summary says 36"):
        precompute.compute_episode(plan, episode, FakeEncoder(), args)
    assert not precompute.cache_path(plan, episode).exists()


def test_changed_provenance_refuses_resume(sample):
    plan, episode, args = sample
    precompute.compute_episode(plan, episode, FakeEncoder(), args)
    with pytest.raises(ValueError, match="source files or computation settings changed"):
        precompute.validate_cache(precompute.cache_path(plan, episode), "changed", episode, 16)
