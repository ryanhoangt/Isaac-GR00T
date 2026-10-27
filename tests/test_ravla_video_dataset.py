"""Check bounded memory behavior and frame selection against full video decoding."""

from pathlib import Path

import av
import numpy as np
import pytest

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.rag import video_dataset


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    def initialize(self, **kwargs):
        self._trajectory_lengths = np.array([96, 96, 96])
        self._delta_indices = {"video.image": np.array([-1, 0, 1])}
        self.video_backend = "pyav"
        self.video_backend_kwargs = {}

    monkeypatch.setattr(LeRobotSingleDataset, "__init__", initialize)
    monkeypatch.setattr(LeRobotSingleDataset, "get_trajectory_index", lambda self, trajectory: int(trajectory))
    monkeypatch.setattr(
        LeRobotSingleDataset, "get_video_path",
        lambda self, trajectory, key: tmp_path / f"{trajectory}_{key}.mp4",
    )
    return video_dataset.BoundedLeRobotVideoDataset(cache_size=2)


def test_frame_indices_clamp_and_repeated_reads_reuse_decoding(dataset, monkeypatch):
    calls = []
    frames = np.arange(96 * 2 * 2 * 3, dtype=np.uint8).reshape(96, 2, 2, 3)

    def decode(path, **kwargs):
        calls.append(path)
        return frames

    monkeypatch.setattr(video_dataset, "get_all_frames", decode)
    np.testing.assert_array_equal(dataset.get_video(0, "video.image", 0), frames[[0, 0, 1]])
    np.testing.assert_array_equal(dataset.get_video(0, "video.image", 95), frames[[94, 95, 95]])
    assert len(calls) == 1


def test_lru_evicts_oldest_video_and_stays_bounded(dataset, monkeypatch):
    calls = []

    def decode(path, **kwargs):
        calls.append(Path(path).name)
        return np.zeros((96, 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(video_dataset, "get_all_frames", decode)
    for trajectory in [0, 1, 0, 2, 1]:
        dataset.get_video(trajectory, "video.image", 0)
        assert len(dataset._video_cache) <= 2
    assert calls == ["0_image.mp4", "1_image.mp4", "2_image.mp4", "1_image.mp4"]


def test_bad_frame_count_is_rejected(dataset, monkeypatch):
    monkeypatch.setattr(video_dataset, "get_all_frames", lambda *a, **k: np.zeros((95, 2, 2, 3)))
    with pytest.raises(ValueError, match="decoded 95, metadata 96"):
        dataset.get_video(0, "video.image", 0)


def test_zero_cache_size_is_rejected():
    with pytest.raises(ValueError, match="cache_size must be positive"):
        video_dataset.BoundedLeRobotVideoDataset(cache_size=0)


def test_encoded_video_matches_integer_frame_selection(dataset):
    path = dataset.get_video_path(0, "image")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(96):
            pixels = np.full((16, 16, 3), index * 2, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    expected = video_dataset.get_all_frames(str(path), video_backend="pyav")
    # These include times vulnerable to float timestamp rounding in the old on-demand reader.
    for frame in [0, 1, 71, 77, 95]:
        indices = np.clip(np.array([-1, 0, 1]) + frame, 0, 95)
        np.testing.assert_array_equal(dataset.get_video(0, "video.image", frame), expected[indices])
