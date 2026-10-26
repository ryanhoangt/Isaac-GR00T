"""Exercise viewer decoding, memory rendering and interactive navigation on CPU."""

import json
from pathlib import Path
import sys

import av
import numpy as np
import pytest
import torch

from scripts.rag import retrieval_viewer_data as data


@pytest.fixture
def viewer_cache(tmp_path):
    source = tmp_path / "eval"
    rollouts = source / "rollouts"
    rollouts.mkdir(parents=True)
    video_path = rollouts / "episode.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for frame_index in range(35):
            image = np.zeros((32, 64, 3), dtype=np.uint8)
            image[:, :32, 0] = 70 + frame_index * 4
            image[:, 32:, 2] = 210 - frame_index * 3
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    memory_path = tmp_path / "memory.pt"
    torch.save(
        {
            task: {"pixel_values": torch.zeros(3, 2, 3, 8, 8), "action": torch.zeros(3, 16, 32)}
            for task in ("task zero", "task one")
        },
        memory_path,
    )
    directory = tmp_path / "cache" / "sample"
    directory.mkdir(parents=True)
    summary = []
    manifest = []
    for ordinal, (task_id, success) in enumerate([(0, True), (0, False), (1, True)], 1):
        episode = {
            "task_id": task_id,
            "task_description": ["task zero", "task one"][task_id],
            "episode_index": ordinal - 1,
            "success": success,
            "control_steps": 35,
        }
        summary.append(episode)
        filename = f"episode_{ordinal:06d}.npz"
        manifest.append({"ordinal": ordinal, "task_id": task_id, "cache": filename, "queries": 3})
        metadata = {
            "episode": {**episode, "video": {"path": str(video_path)}},
            "memory": str(memory_path),
            "suite": "libero_goal",
            "policy_k": 1,
            "memory_candidates": 3,
            "decoded_frames": 35,
            "video_fps": 30,
            "is_original_retrieval_trace": False,
        }
        np.savez_compressed(
            directory / filename,
            metadata_json=json.dumps(metadata),
            frame_indices=[0, 16, 32],
            simulator_steps=[10, 26, 42],
            topk_scores=np.array([[0.9, 0.8], [0.8, 0.79], [0.9, 0.5]], dtype=np.float32),
            topk_indices=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        )
    (rollouts / "summary.json").write_text(json.dumps({"episodes": summary}))
    stat = memory_path.stat()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run": str(source),
                "status": "complete",
                "signature": {"memory": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}},
                "episodes": manifest,
            }
        )
    )
    return directory


def test_exact_video_frame_alignment_and_camera_split(viewer_cache):
    episode = data.read_episode(viewer_cache / "episode_000001.npz")
    video = episode["metadata"]["episode"]["video"]["path"]
    frames = data.decode_frames(video, episode["frame_indices"], 35)
    with av.open(video) as container:
        reference = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    np.testing.assert_array_equal(frames, np.stack([reference[0], reference[16], reference[32]]))
    np.testing.assert_array_equal(data.rollout_view(frames[1], "External"), frames[1, :, :32])
    np.testing.assert_array_equal(data.rollout_view(frames[1], "Wrist"), frames[1, :, 32:])
    with pytest.raises(ValueError, match="alignment mismatch"):
        data.decode_frames(video, [0, 16], 36)


def test_memory_normalization_and_rank_validation(viewer_cache):
    rgb = np.full((2, 8, 8, 3), [0, 128, 255], dtype=np.uint8)
    pixels = (rgb.astype(np.float32) / 255 - 0.5) / 0.5
    np.testing.assert_array_equal(data.memory_images(pixels.transpose(0, 3, 1, 2), [0.5] * 3, [0.5] * 3), rgb)
    episode = data.read_episode(viewer_cache / "episode_000001.npz")
    trace = data.score_trace(episode)
    assert trace["margin"].idxmin() == 1
    assert trace["top1_memory"].tolist() == [0, 1, 2]
    episode["topk_indices"][0, 0] = 3
    metadata = episode.pop("metadata")
    broken = viewer_cache / "broken.npz"
    np.savez(broken, metadata_json=json.dumps(metadata), **episode)
    with pytest.raises(ValueError, match="outside task memory"):
        data.read_episode(broken)


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def test_streamlit_navigation_and_empty_filter(viewer_cache, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    monkeypatch.setattr(sys, "argv", ["retrieval_viewer.py", "--cache-root", str(viewer_cache.parent)])
    app_path = Path(__file__).resolve().parents[1] / "scripts/rag/retrieval_viewer.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)
    assert not app.exception and not app.error
    assert app.session_state["query"] == 0
    widget(app, "button", "Next →").click().run()
    assert not app.exception and not app.error
    assert app.session_state["query"] == 1
    widget(app, "radio", "Camera").set_value("Wrist").run()
    assert not app.exception and not app.error
    widget(app, "selectbox", "Episode").select(2).run()
    assert not app.exception and not app.error
    assert app.session_state["query"] == 0
    widget(app, "selectbox", "Task").select(1).run()
    assert not app.exception and not app.error
    widget(app, "radio", "Outcome").set_value("Failure").run()
    assert not app.exception and not app.error
    assert any("No cached episodes match" in item.value for item in app.info)
