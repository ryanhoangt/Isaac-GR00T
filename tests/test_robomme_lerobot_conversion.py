"""Exercise local image-parquet conversion, multi-episode loading and resume."""

import io
import json

import av
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import pytest

from scripts import convert_robomme_lerobot as conversion


def image_value(level):
    buffer = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), level, np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def make_source(root, flags=None):
    flags = [True, True, False, False, False] if flags is None else flags
    (root / "meta").mkdir(parents=True)
    features = {
        key: {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]}
        for key in ("image", "wrist_image")
    }
    for key in ("state", "actions"):
        features[key] = {"dtype": "float32", "shape": [8], "names": [key]}
    for key in (
        "timestamp",
        "index",
        "episode_index",
        "frame_index",
        "task_index",
        "exec_start_idx",
        "step_idx",
        "epis_idx",
    ):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    features["is_demo"] = {"dtype": "bool", "shape": [1], "names": None}
    annotations = ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online", "grounded_subgoal_online")
    for key in annotations:
        features[key] = {"dtype": "string", "shape": [1], "names": None}
    episodes = []
    # Sparse IDs and a one-episode chunk size exercise both input and output chunk lookup.
    for episode_id, demo, task_id, offset in ((3, flags, 7, 0), (1001, [False, False], 42, 10)):
        length = len(demo)
        data = pd.DataFrame(
            {
                "image": [image_value(i * 30) for i in range(length)],
                "wrist_image": [image_value(i * 30 + 20) for i in range(length)],
                "state": [np.array([100 + offset + i] * 7 + [0.01], np.float32) for i in range(length)],
                "actions": [np.array([10 + offset + i] * 7 + [(-1) ** i], np.float32) for i in range(length)],
                "timestamp": np.arange(length, dtype=np.float32) / 10,
                "frame_index": np.arange(length, dtype=np.int64),
                "index": np.arange(length, dtype=np.int64) + episode_id * 10,
                "episode_index": np.full(length, episode_id),
                "epis_idx": np.full(length, episode_id),
                "task_index": np.full(length, task_id),
                "is_demo": demo,
                "exec_start_idx": np.full(length, sum(demo)),
                "step_idx": np.arange(length),
                **{key: [f"{key} {i}" for i in range(length)] for key in annotations},
            }
        )
        path = root / conversion.DATA_PATH.format(episode_chunk=episode_id, episode_index=episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        episodes.append({"episode_index": episode_id, "length": length, "tasks": [f"task {task_id}"]})
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "fps": 10,
        "chunks_size": 1,
        "total_episodes": 2,
        "total_frames": sum(ep["length"] for ep in episodes),
        "total_tasks": 2,
        "data_path": conversion.DATA_PATH,
        "video_path": conversion.VIDEO_PATH,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    (root / "meta/tasks.jsonl").write_text(
        '{"task_index": 7, "task": "Task Seven"}\n{"task_index": 42, "task": "Task Forty Two"}\n'
    )


def modality_configs():
    from gr00t.data.dataset import ModalityConfig

    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["video.image", "video.wrist_image"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["state.joint_position", "state.gripper"]),
        "action": ModalityConfig(
            delta_indices=list(range(4)), modality_keys=["action.joint_position", "action.gripper"]
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    }


def test_parallel_conversion_preserves_pairing_and_loads_gr00t(tmp_path):
    from gr00t.data.dataset import LeRobotSingleDataset

    source, output = tmp_path / "source", tmp_path / "converted"
    make_source(source)
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = conversion.convert_dataset(source, output, workers=2)
    assert result["episodes"] == 2 and result["frames"] == 5 and result["excluded_demo_frames"] == 2
    frames = [
        pd.read_parquet(output / conversion.DATA_PATH.format(episode_chunk=i, episode_index=i)) for i in (0, 1)
    ]
    assert frames[0]["source_frame_index"].tolist() == [2, 3, 4]
    assert frames[0]["step_idx"].tolist() == [2, 3, 4]
    assert frames[0]["simple_subgoal"].tolist() == ["simple_subgoal 2", "simple_subgoal 3", "simple_subgoal 4"]
    assert frames[0]["source_exec_start_idx"].tolist() == [2, 2, 2]
    assert frames[0]["exec_start_idx"].tolist() == [0, 0, 0]
    np.testing.assert_allclose(frames[0]["timestamp"], [0, 0.1, 0.2])
    np.testing.assert_array_equal(np.stack(frames[0]["actions"])[:, 0], [12, 13, 14])
    np.testing.assert_array_equal(np.stack(frames[0]["state"])[:, 0], [102, 103, 104])
    assert frames[1]["index"].tolist() == [3, 4]
    assert frames[1]["source_episode_index"].tolist() == [1001, 1001]
    assert frames[1]["task_index"].tolist() == [1, 1]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_chunks"] == 2 and info["total_videos"] == 4
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["actions"]["mean"][0] == np.mean([12, 13, 14, 20, 21])
    np.testing.assert_allclose(stats["actions"]["q99"][0], np.quantile([12, 13, 14, 20, 21], 0.99))
    dataset = LeRobotSingleDataset(
        output,
        modality_configs(),
        "new_embodiment",
        video_backend="decord",
        video_backend_kwargs={"num_threads": 1},
    )
    assert len(dataset) == 5
    for index in range(5):
        sample = dataset[index]
        if index < 3:
            expected = np.minimum(np.arange(index, index + 4), 2) + 12
        else:
            expected = np.minimum(np.arange(index - 3, index + 1), 1) + 20
        np.testing.assert_array_equal(sample["action.joint_position"][:, 0], expected)
        assert sample["annotation.human.action.task_description"] == [
            "Task Seven" if index < 3 else "Task Forty Two"
        ]
    for episode, length, first in ((0, 3, 60), (1, 2, 0)):
        for key, offset in (("image", 0), ("wrist_image", 20)):
            video = output / conversion.VIDEO_PATH.format(
                episode_chunk=episode, episode_index=episode, video_key=key
            )
            with av.open(str(video)) as container:
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            assert len(decoded) == length
            np.testing.assert_allclose(
                [image.mean() for image in decoded], np.arange(length) * 30 + first + offset, atol=3
            )
    assert all(path.read_bytes() == content for path, content in original.items())
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        conversion.convert_dataset(source, output, workers=1, resume=True)


@pytest.mark.parametrize("change_source", [False, True])
def test_resume_after_interrupted_publication(tmp_path, monkeypatch, change_source):
    source, output = tmp_path / "source", tmp_path / "converted"
    make_source(source)

    def interrupted(*args):
        raise RuntimeError("Simulated interruption before publication")

    with monkeypatch.context() as patch:
        patch.setattr(conversion, "finalize", interrupted)
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            conversion.convert_dataset(source, output, workers=1)
    assert not output.exists()
    work = output.with_name(f".{output.name}.work")
    cached = work / "episodes/000000/image.mp4"
    timestamp = cached.stat().st_mtime_ns
    if change_source:
        path = source / "meta/tasks.jsonl"
        path.write_text(path.read_text().replace("Task Seven", "Different instruction"))
        with pytest.raises(ValueError, match="settings differ"):
            conversion.convert_dataset(source, output, workers=1, resume=True)
    else:
        result = conversion.convert_dataset(source, output, workers=2, resume=True)
        assert result["frames"] == 5
    assert cached.stat().st_mtime_ns == timestamp


@pytest.mark.parametrize(
    "flags, message", [([True, True], "no execution frames"), ([False, True], "demo frame after execution")]
)
def test_invalid_demo_sequence(tmp_path, flags, message):
    source, output = tmp_path / "source", tmp_path / "invalid"
    make_source(source, flags)
    with pytest.raises(ValueError, match=message):
        conversion.convert_dataset(source, output, workers=1, episode_indices=[3])
    assert not output.exists()


def test_subset_and_v3_rejection(tmp_path):
    source, output = tmp_path / "source", tmp_path / "subset"
    make_source(source)
    result = conversion.convert_dataset(source, output, workers=1, episode_indices=[1001])
    assert result["frames"] == 2 and result["episodes"] == 1
    tasks = [json.loads(line) for line in (output / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "Task Forty Two"}]
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v2.x"):
        conversion.convert_dataset(source, tmp_path / "v3", workers=1)


def test_corrupt_image_does_not_publish_dataset(tmp_path):
    source, output = tmp_path / "source", tmp_path / "broken"
    make_source(source)
    path = source / conversion.DATA_PATH.format(episode_chunk=3, episode_index=3)
    data = pd.read_parquet(path)
    data.at[2, "image"] = {"bytes": b"broken image", "path": None}
    data.to_parquet(path, index=False)
    with pytest.raises(UnidentifiedImageError):
        conversion.convert_dataset(source, output, workers=1, episode_indices=[3])
    assert not output.exists()
    assert not list(output.with_name(f".{output.name}.work").glob("episodes/.000000-*"))
