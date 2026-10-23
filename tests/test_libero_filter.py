"""Check that removing a demo preserves frame/action alignment and normalization."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_libero_clean import filter_suite


def test_filter_suite_preserves_alignment(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "clean"
    (source / "meta").mkdir(parents=True)
    info = {
        "total_episodes": 3,
        "total_frames": 6,
        "total_videos": 3,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"image": {"dtype": "video"}},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/modality.json").write_text("{}")
    (source / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n')
    episodes = []
    for episode, action in enumerate((0.0, 100.0, 2.0)):
        values = {"episode_chunk": 0, "episode_index": episode}
        parquet = source / info["data_path"].format(**values)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "episode_index": [episode] * 2,
            "index": [2 * episode, 2 * episode + 1],
            "frame_index": [0, 1],
            "timestamp": [0.0, 0.05],
            "action": [np.array([action])] * 2,
            "observation.state": [np.array([episode, 0.0]), np.array([episode, 1.0])],
        }).to_parquet(parquet, index=False)
        video = source / info["video_path"].format(**values, video_key="image")
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"original video {episode}".encode())
        episodes.append({"episode_index": episode, "length": 2, "tasks": ["test"]})
    (source / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))

    result = filter_suite(source, destination, {1})
    assert result == {"excluded_original_episodes": [1], "episodes": 2, "frames": 4}
    clean_info = json.loads((destination / "meta/info.json").read_text())
    assert clean_info["splits"] == {"train": "0:2"}
    assert clean_info["total_videos"] == 2
    second = pd.read_parquet(destination / info["data_path"].format(episode_chunk=0, episode_index=1))
    assert second["episode_index"].tolist() == [1, 1]
    assert second["index"].tolist() == [2, 3]
    assert second["timestamp"].tolist() == [0.0, 0.05]
    np.testing.assert_array_equal(np.stack(second["action"]), [[2.0], [2.0]])
    np.testing.assert_array_equal(np.stack(second["observation.state"]), [[2.0, 0.0], [2.0, 1.0]])
    video = destination / info["video_path"].format(episode_chunk=0, episode_index=1, video_key="image")
    assert video.read_bytes() == b"original video 2"
    stats = json.loads((destination / "meta/stats.json").read_text())
    assert stats["action"]["mean"] == [1.0]
    assert stats["action"]["max"] == [2.0]
    assert (source / "meta/info.json").read_text() == json.dumps(info)

    with pytest.raises(ValueError, match="Unknown excluded"):
        filter_suite(source, tmp_path / "invalid", {99})
