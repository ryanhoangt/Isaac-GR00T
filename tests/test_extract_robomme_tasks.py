"""Exercise all five task splits, shared-instruction indexing and independent copies."""

import json
import shutil

import av
import h5py
import numpy as np
import pandas as pd
import pytest

from scripts import extract_robomme_tasks as extraction
from scripts.convert_robomme_lerobot import DATA_PATH, VIDEO_PATH, modality_metadata


def write_video(path, length, level):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for _ in range(length):
            image = np.full((256, 256, 3), level, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("robomme_task_splits")
    root, raw = temporary / "source", temporary / "hdf5"
    (root / "meta").mkdir(parents=True)
    raw.mkdir()
    episodes, mappings, vocabulary = [], [], {}
    features = {
        key: {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": 10},
        }
        for key in ("image", "wrist_image")
    }
    for key in ("state", "actions"):
        features[key] = {"dtype": "float32", "shape": [8], "names": [key]}
    for key in (
        "index",
        "episode_index",
        "frame_index",
        "task_index",
        "epis_idx",
        "step_idx",
        "exec_start_idx",
        "source_episode_index",
        "source_frame_index",
        "source_task_index",
    ):
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    features["timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
    features["is_demo"] = {"dtype": "bool", "shape": [1], "names": None}
    features["simple_subgoal"] = {"dtype": "string", "shape": [1], "names": None}
    offset = 0
    for task_number, task in enumerate(extraction.TASK_SUITE):
        with h5py.File(raw / f"record_dataset_{task}.h5", "w") as handle:
            for local in range(2):
                episode_id, length = 10 + task_number * 7 + local, 2 + local
                text = (
                    f"shared unmask instruction {local}"
                    if task in ("VideoUnmask", "VideoUnmaskSwap")
                    else f"{task.lower()} instruction {local}"
                )
                task_id = vocabulary.setdefault(text, 80 + len(vocabulary) * 2)
                state = np.stack(
                    [
                        np.array([task_number + local / 10 + t / 100] * 7 + [0.01], np.float32)
                        for t in range(length)
                    ]
                )
                action = state.copy()
                action[:, :7] += 0.5
                action[:, -1] = np.array([-1, 1, -1][:length])
                data = pd.DataFrame(
                    {
                        "state": list(state),
                        "actions": list(action),
                        "timestamp": np.arange(length) / 10,
                        "index": np.arange(offset, offset + length),
                        "episode_index": episode_id,
                        "frame_index": np.arange(length),
                        "task_index": task_id,
                        "epis_idx": episode_id,
                        "step_idx": np.arange(length) + 1,
                        "exec_start_idx": 0,
                        "is_demo": False,
                        "source_episode_index": episode_id + 5000,
                        "source_frame_index": np.arange(length) + 1,
                        "source_task_index": task_id,
                        "simple_subgoal": [f"annotation {t}" for t in range(length)],
                    }
                )
                values = dict(episode_chunk=episode_id // 3, episode_index=episode_id)
                path = root / DATA_PATH.format(**values)
                path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(path, index=False)
                for key in ("image", "wrist_image"):
                    write_video(
                        root / VIDEO_PATH.format(**values, video_key=key), length, task_number * 13 + local
                    )
                episodes.append({"episode_index": episode_id, "length": length, "tasks": [text]})
                mappings.append(
                    {
                        "episode_index": episode_id,
                        "source_episode_index": episode_id + 5000,
                        "source_parquet": f"original/episode_{episode_id + 5000}.parquet",
                        "source_frames": length + 1,
                        "excluded_demo_frames": 1,
                        "exported_frames": length,
                    }
                )
                group = handle.create_group(f"episode_{local}")
                group.create_dataset("setup/task_goal", data=[text.encode()])
                for t in range(length + 1):
                    step = group.create_group(f"timestep_{t}")
                    step.create_dataset("info/is_video_demo", data=t == 0)
                    row_state = state[max(t - 1, 0)] if t else np.full(8, 999, np.float32)
                    row_action = action[max(t - 1, 0)] if t else np.full(8, -999, np.float32)
                    step.create_dataset("obs/joint_state", data=row_state[:7])
                    step.create_dataset("obs/gripper_state", data=np.repeat(row_state[7:], 2))
                    step.create_dataset("action/joint_action", data=row_action)
                offset += length
    extraction.write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "panda",
            "fps": 10,
            "chunks_size": 3,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": offset,
            "total_tasks": len(vocabulary),
            "total_videos": 2 * len(episodes),
            "splits": {"train": f"0:{len(episodes)}"},
        },
    )
    extraction.write_jsonl(root / "meta/episodes.jsonl", episodes[::-1])
    extraction.write_jsonl(
        root / "meta/tasks.jsonl", [{"task_index": index, "task": text} for text, index in vocabulary.items()]
    )
    extraction.write_json(root / "meta/modality.json", modality_metadata())
    extraction.write_json(root / "meta/episode_mapping.json", mappings)
    extraction.write_json(root / "meta/stats.json", {})
    index_path = temporary / "task_index.json"
    index = extraction.build_task_index(root, raw, index_path)
    return root, raw, index_path, index


def test_shared_instructions_resolved_by_hdf5_controls(source_dataset):
    _, _, _, index = source_dataset
    assert set(index["episodes_per_task"]) == set(extraction.TASK_SUITE)
    assert set(index["episodes_per_task"].values()) == {2}
    resolved = [
        row for row in index["episodes"] if row["identification"] == "hdf5_execution_state_action_signature"
    ]
    assert len(resolved) == 4
    assert {row["task"] for row in resolved} == {"VideoUnmask", "VideoUnmaskSwap"}


@pytest.mark.parametrize("split", tuple(extraction.SPLITS))
def test_all_five_splits_copy_independently_and_recompute_stats(source_dataset, tmp_path, split):
    from gr00t.data.dataset import LeRobotSingleDataset
    from examples.RoboMME.custom_data_config import RoboMMEDataConfig

    root, _, index_path, index = source_dataset
    before = extraction.identity(root)
    output = tmp_path / split
    result = extraction.extract_tasks(root, index_path, output, split=split, workers=2)
    expected_tasks = set(extraction.TASK_SUITE) - set(extraction.SPLITS[split])
    assert set(result["selected_tasks"]) == expected_tasks and len(expected_tasks) == 12
    assert result["episodes"] == 24 and result["frames"] == 60
    assert set(result["episodes_per_task"].values()) == {2}
    info = extraction.read_json(output / "meta/info.json")
    episodes = extraction.read_jsonl(output / "meta/episodes.jsonl")
    mappings = extraction.read_json(output / "meta/episode_mapping.json")
    old_entries = {row["episode_index"]: row for row in index["episodes"]}
    assert [ep["episode_index"] for ep in episodes] == list(range(24))
    assert {row["task"] for row in mappings} == expected_tasks
    frames, offset = [], 0
    for episode, mapping in zip(episodes, mappings):
        new_id, old_id = episode["episode_index"], mapping["parent_episode_index"]
        destination, videos = extraction.episode_paths(output, info, new_id, ("image", "wrist_image"))
        source, source_videos = extraction.episode_paths(root, info, old_id, ("image", "wrist_image"))
        original, data = pd.read_parquet(source), pd.read_parquet(destination)
        assert old_entries[old_id]["task"] in expected_tasks
        for key in ("state", "actions"):
            np.testing.assert_array_equal(np.stack(data[key]), np.stack(original[key]))
        assert data["simple_subgoal"].tolist() == original["simple_subgoal"].tolist()
        assert data["source_episode_index"].tolist() == original["source_episode_index"].tolist()
        assert data["index"].tolist() == list(range(offset, offset + len(data)))
        assert not destination.is_symlink() and not destination.samefile(source)
        for key, path in videos.items():
            assert path.read_bytes() == source_videos[key].read_bytes()
            assert not path.is_symlink() and not path.samefile(source_videos[key])
            assert path.stat().st_nlink == 1
        frames.append(data)
        offset += len(data)
    values = np.concatenate([np.stack(frame["actions"]) for frame in frames])
    stats = extraction.read_json(output / "meta/stats.json")
    np.testing.assert_allclose(stats["actions"]["mean"], values.astype(np.float64).mean(0))
    np.testing.assert_allclose(stats["actions"]["q99"], np.quantile(values.astype(np.float64), 0.99, axis=0))
    assert extraction.identity(root) == before
    config = RoboMMEDataConfig()
    dataset = LeRobotSingleDataset(
        output,
        config.modality_config(),
        "new_embodiment",
        video_backend="decord",
        video_backend_kwargs={"num_threads": 1},
    )
    assert len(dataset) == 60
    for position in (0, 59):
        sample = dataset[position]
        assert sample["video.image"].shape == (1, 256, 256, 3)
        assert sample["action.joint_position"].shape == (16, 7)
        assert sample["action.gripper"].shape == (16, 1)


def test_explicit_selection_and_nested_extraction(source_dataset, tmp_path):
    root, _, index_path, _ = source_dataset
    first = tmp_path / "first"
    extraction.extract_tasks(root, index_path, first, tasks=["VideoUnmask", "PatternLock"])
    second = tmp_path / "second"
    result = extraction.extract_tasks(
        first, first / "meta/episode_tasks.json", second, exclude_tasks=["PatternLock"]
    )
    assert result["selected_tasks"] == ["VideoUnmask"] and result["episodes"] == 2
    with pytest.raises(FileExistsError, match="Refusing"):
        extraction.extract_tasks(root, index_path, first, tasks=["VideoUnmask"])


def test_dry_run_and_invalid_mapping(source_dataset, tmp_path):
    root, _, index_path, index = source_dataset
    output = tmp_path / "dry"
    plan = extraction.extract_tasks(root, index_path, output, split="primary", dry_run=True)
    assert plan["episodes"] == 24 and not output.exists()
    bad = json.loads(json.dumps(index))
    bad["source_metadata_sha256"]["info.json"] = "wrong"
    path = tmp_path / "wrong.json"
    extraction.write_json(path, bad)
    with pytest.raises(ValueError, match="does not match"):
        extraction.extract_tasks(root, path, tmp_path / "invalid", tasks=["BinFill"])
    with pytest.raises(ValueError, match="Unknown"):
        extraction.extract_tasks(root, index_path, tmp_path / "unknown", tasks=["NoSuchTask"])


def test_changed_controls_fail_without_publishing(source_dataset, tmp_path):
    root, _, _, index = source_dataset
    bad = json.loads(json.dumps(index))
    bad["episodes"][0]["execution_sha256"] = "wrong"
    path = tmp_path / "wrong_controls.json"
    extraction.write_json(path, bad)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="changed since task indexing"):
        extraction.extract_tasks(root, path, output, tasks=[bad["episodes"][0]["task"]])
    assert not output.exists() and not list(tmp_path.glob(".invalid-*"))


def test_ambiguous_hdf5_identity_is_rejected(source_dataset, tmp_path):
    root, raw, _, _ = source_dataset
    duplicate = tmp_path / "raw"
    shutil.copytree(raw, duplicate)
    with h5py.File(duplicate / "record_dataset_VideoUnmask.h5", "r") as a:
        with h5py.File(duplicate / "record_dataset_VideoUnmaskSwap.h5", "r+") as b:
            for number in range(1, 3):
                for name in ("obs/joint_state", "obs/gripper_state", "action/joint_action"):
                    b[f"episode_0/timestep_{number}/{name}"][...] = a[f"episode_0/timestep_{number}/{name}"][()]
    with pytest.raises(ValueError, match="identity unresolved"):
        extraction.build_task_index(root, duplicate, tmp_path / "ambiguous.json")
    assert not (tmp_path / "ambiguous.json").exists()
