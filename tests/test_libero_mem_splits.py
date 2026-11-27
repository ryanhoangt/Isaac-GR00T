"""Fold isolation, actual file copies, independent statistics and GR00T loading."""

import hashlib
import json
import shutil

import av
import numpy as np
import pandas as pd
import pytest

from examples.Libero_Mem import splits
from examples.Libero_Mem.conversion import CAMERAS, MODALITY_PATH, file_sha256
from examples.Libero_Mem.tasks import TASKS, get_fold
from gr00t.utils.lerobot_conversion import DATA_PATH, VIDEO_PATH, statistics, write_json, write_jsonl


def write_video(path, length, level):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for i in range(length):
            image = np.full((256, 256, 3), level + i, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def canonical_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("libero_mem_split_source")
    (root / "meta").mkdir()
    episodes, mappings, states, actions = [], [], [], []
    offset = 0
    for task in TASKS:
        # Reverse instruction IDs to ensure folds use task identity, not IDs.
        task_index = 10 - task.number
        for demo_number, length in ((1, 3), (3, 2)):
            index = len(episodes)
            source_key = f"{task.name}/demo_{demo_number}"
            state = np.zeros((length, 8), np.float32)
            state[:, :6] = task.number / 10 + np.arange(length, dtype=np.float32)[:, None] / 100
            state[:, 6:] = np.arange(length, dtype=np.float32)[:, None] / 1000
            action = np.zeros((length, 7), np.float32)
            action[:, :6] = task.number / 100 + np.arange(length, dtype=np.float32)[:, None] / 1000
            action[:, -1] = np.resize([0, 1], length)
            data = pd.DataFrame(
                {
                    "observation.state": list(state),
                    "action": list(action),
                    "episode_index": np.full(length, index, np.int64),
                    "frame_index": np.arange(length, dtype=np.int64),
                    "index": np.arange(offset, offset + length, dtype=np.int64),
                    "task_index": np.full(length, task_index, np.int64),
                    "timestamp": np.arange(length, dtype=np.float64) / 20,
                    "next.done": [False] * (length - 1) + [True],
                    "source_task_id": np.full(length, task.number, np.int64),
                    "source_frame_index": np.arange(length, dtype=np.int64),
                    "source_demo_id": f"demo_{demo_number}",
                    "source_episode_key": source_key,
                    "source_action_gripper": 1 - 2 * action[:, -1],
                }
            )
            values = {"episode_index": index, "episode_chunk": index // 3}
            path = root / DATA_PATH.format(**values)
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, index=False)
            for camera, key in enumerate(CAMERAS):
                write_video(
                    root / VIDEO_PATH.format(**values, video_key=key), length, task.number * 10 + camera * 100
                )
            episodes.append({"episode_index": index, "length": length, "tasks": [task.instruction]})
            mappings.append(
                {
                    "episode_index": index,
                    "task": task.name,
                    "source_task_id": task.number,
                    "source_demo_id": f"demo_{demo_number}",
                    "source_episode_key": source_key,
                    "raw_demo_id": f"demo_{demo_number}",
                    "source_frames": length,
                    "initial_state_sha256": hashlib.sha256(source_key.encode()).hexdigest(),
                    "source_file": f"/unused-original-hdf5/{task.filename}",
                }
            )
            states.append(state)
            actions.append(action)
            offset += length
    features = {
        key: {"dtype": "int64", "shape": [1], "names": None}
        for key in ("episode_index", "frame_index", "index", "task_index", "source_task_id", "source_frame_index")
    }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [8], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "source_action_gripper": {"dtype": "float32", "shape": [1], "names": None},
            **{
                key: {"dtype": "string", "shape": [1], "names": None}
                for key in ("source_demo_id", "source_episode_key")
            },
            **{
                key: {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {"video.fps": 20},
                }
                for key in CAMERAS
            },
        }
    )
    info = {
        "codebase_version": "v2.1",
        "fps": 20,
        "robot_type": "franka",
        "chunks_size": 3,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "total_episodes": 20,
        "total_frames": offset,
        "total_tasks": 10,
        "total_videos": 40,
        "total_chunks": 7,
        "splits": {"train": "0:20"},
    }
    write_json(root / "meta/info.json", info)
    (root / "meta/modality.json").write_bytes(MODALITY_PATH.read_bytes())
    # Poison global normalization: extraction must recompute from selected rows.
    global_stats = {
        "observation.state": statistics(np.concatenate(states)),
        "action": statistics(np.concatenate(actions)),
    }
    for stats in global_stats.values():
        stats["min"] = [-999.0] * len(stats["min"])
        stats["max"] = [999.0] * len(stats["max"])
    write_json(root / "meta/stats.json", global_stats)
    write_jsonl(root / "meta/episodes_stats.jsonl", [{"episode_index": i, "stats": {}} for i in range(20)])
    write_jsonl(root / "meta/episodes.jsonl", episodes[::-1])
    write_json(root / "meta/episode_mapping.json", mappings[::-1])
    write_jsonl(
        root / "meta/tasks.jsonl", [{"task_index": 10 - task.number, "task": task.instruction} for task in TASKS]
    )
    write_json(
        root / "manifest.json",
        {
            "converter": "examples.Libero_Mem.conversion",
            "converter_version": 1,
            "status": "complete",
            "format": "gr00t_lerobot_v2.1",
            "episodes": 20,
            "frames": offset,
            "tasks": 10,
        },
    )
    return root


@pytest.fixture
def source(canonical_template, tmp_path):
    root = tmp_path / "canonical"
    shutil.copytree(canonical_template, root)
    return root


def episode_paths(root, info, episode_id):
    values = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"]}
    return (
        root / info["data_path"].format(**values),
        {key: root / info["video_path"].format(**values, video_key=key) for key in CAMERAS},
    )


def test_complementary_folds_are_independent_copies_with_local_statistics(source, tmp_path):
    before = {path: file_sha256(path) for path in source.rglob("*") if path.is_file()}
    canonical_info = json.loads((source / "meta/info.json").read_text())
    all_keys = {
        row["source_episode_key"] for row in json.loads((source / "meta/episode_mapping.json").read_text())
    }
    fold_keys, fold_means = [], []
    for fold in ("A", "B"):
        output = tmp_path / f"fold_{fold}"
        manifest = splits.extract_fold(source, output, training_fold=fold, chunks_size=2)
        assert manifest["episodes"] == 10 and manifest["frames"] == 25
        assert manifest["storage"] == "independent_copies"
        assert manifest["normalization_scope"] == "selected_training_rows_only"
        assert manifest["train_tasks"] == get_fold(fold)["train_tasks"]
        info = json.loads((output / "meta/info.json").read_text())
        assert info["total_tasks"] == 5 and info["total_chunks"] == 5
        mappings = json.loads((output / "meta/episode_mapping.json").read_text())
        assert {row["task"] for row in mappings} == set(get_fold(fold)["train_tasks"])
        fold_keys.append({row["source_episode_key"] for row in mappings})
        states, actions, offset = [], [], 0
        for i, mapping in enumerate(mappings):
            path, videos = episode_paths(output, info, i)
            original, source_videos = episode_paths(source, canonical_info, mapping["canonical_episode_index"])
            data, parent = pd.read_parquet(path), pd.read_parquet(original)
            for key in (
                "observation.state",
                "action",
                "source_frame_index",
                "source_demo_id",
                "source_episode_key",
            ):
                if key in ("observation.state", "action"):
                    np.testing.assert_array_equal(np.stack(data[key]), np.stack(parent[key]))
                else:
                    assert data[key].tolist() == parent[key].tolist()
            assert data["index"].tolist() == list(range(offset, offset + len(data)))
            assert data["episode_index"].unique().tolist() == [i]
            assert data["canonical_index"].tolist() == parent["index"].tolist()
            assert data["canonical_task_index"].tolist() == parent["task_index"].tolist()
            assert not path.samefile(original)
            for key, video in videos.items():
                assert not video.is_symlink() and not video.samefile(source_videos[key])
                assert video.read_bytes() == source_videos[key].read_bytes()
            states.append(np.stack(data["observation.state"]))
            actions.append(np.stack(data["action"]))
            offset += len(data)
        assert not any(path.is_symlink() for path in output.rglob("*"))
        stats = json.loads((output / "meta/stats.json").read_text())
        for key, values in (("observation.state", states), ("action", actions)):
            values = np.concatenate(values).astype(np.float64)
            for stat, expected in (
                ("min", values.min(0)),
                ("max", values.max(0)),
                ("mean", values.mean(0)),
                ("std", values.std(0)),
                ("q01", np.quantile(values, 0.01, axis=0)),
                ("q99", np.quantile(values, 0.99, axis=0)),
            ):
                np.testing.assert_allclose(stats[key][stat], expected, atol=1e-12)
        fold_means.append(stats["observation.state"]["mean"][0])
        assert (output / "meta/modality.json").read_bytes() == MODALITY_PATH.read_bytes()
    assert fold_keys[0].isdisjoint(fold_keys[1]) and fold_keys[0] | fold_keys[1] == all_keys
    assert fold_means[0] != fold_means[1]
    assert all(file_sha256(path) == digest for path, digest in before.items())


def test_fold_survives_source_changes_and_removal(source, tmp_path):
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.data.dataset import LeRobotSingleDataset

    output = tmp_path / "fold_A"
    splits.extract_fold(source, output, training_fold="A")
    info = json.loads((output / "meta/info.json").read_text())
    source_info = json.loads((source / "meta/info.json").read_text())
    _, copied = episode_paths(output, info, 0)
    _, originals = episode_paths(source, source_info, 0)
    before = next(iter(copied.values())).read_bytes()
    next(iter(originals.values())).write_bytes(b"changed canonical video")
    assert next(iter(copied.values())).read_bytes() == before
    source.rename(tmp_path / "moved_source")
    config = LiberoDataConfig()
    dataset = LeRobotSingleDataset(
        output,
        config.modality_config(),
        "new_embodiment",
        video_backend="decord",
        video_backend_kwargs={"num_threads": 1},
    )
    assert len(dataset) == 25
    assert dataset[0]["video.image"].shape == (1, 256, 256, 3)
    assert dataset[len(dataset) - 1]["action.x"].shape == (16, 1)


@pytest.mark.parametrize(
    "defect",
    [
        "source_task_id",
        "source_demo_id",
        "task_index",
        "frame_index",
        "index",
        "timestamp",
        "next.done",
        "nonfinite",
        "gripper",
    ],
)
def test_bad_row_identity_or_vectors_never_publish_output(source, tmp_path, defect):
    info = json.loads((source / "meta/info.json").read_text())
    path, _ = episode_paths(source, info, 0)
    data = pd.read_parquet(path)
    if defect in ("source_task_id", "task_index", "frame_index", "index"):
        data.loc[0, defect] += 1
    elif defect == "source_demo_id":
        data.loc[0, defect] = "demo_99"
    elif defect == "timestamp":
        data.loc[0, defect] = 100.0
    elif defect == "next.done":
        data.loc[0, defect] = True
    elif defect == "nonfinite":
        vectors = np.stack(data["observation.state"])
        vectors[0, 0] = np.nan
        data["observation.state"] = list(vectors)
    elif defect == "gripper":
        data.loc[0, "source_action_gripper"] *= -1
    data.to_parquet(path, index=False)
    output = tmp_path / "bad_fold"
    with pytest.raises(ValueError):
        splits.extract_fold(source, output, training_fold="A")
    assert not output.exists()


def test_wrong_video_length_never_publishes(source, tmp_path):
    info = json.loads((source / "meta/info.json").read_text())
    _, videos = episode_paths(source, info, 0)
    write_video(next(iter(videos.values())), 2, 50)
    output = tmp_path / "bad_video"
    with pytest.raises(ValueError, match="Video header"):
        splits.extract_fold(source, output, training_fold="A")
    assert not output.exists()


def test_missing_held_out_asset_is_detected_in_canonical_inventory(source, tmp_path):
    info = json.loads((source / "meta/info.json").read_text())
    _, videos = episode_paths(source, info, 2)  # T2 is held out for training fold A.
    next(iter(videos.values())).unlink()
    with pytest.raises(ValueError, match="Missing or empty"):
        splits.extract_fold(source, tmp_path / "fold_A", training_fold="A")


def test_dry_run_and_overwrite_protection(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "not_created" / "fold_A"

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not copy videos")

    monkeypatch.setattr(splits, "_copy_video", forbidden)
    splits.main(["--source", str(source), "--output", str(output), "--fold", "A", "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["episodes"] == 10 and result["frames"] == 25 and result["video_bytes_to_copy"] > 0
    assert not output.parent.exists()
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        splits.extract_fold(source, output, training_fold="A")
    with pytest.raises(ValueError, match="outside the canonical"):
        splits.extract_fold(source, source / "inside", training_fold="A")


def test_changed_metadata_during_copy_never_publishes(source, tmp_path, monkeypatch):
    original_copy = splits._copy_video

    def changed(*args, **kwargs):
        result = original_copy(*args, **kwargs)
        with (source / "meta/tasks.jsonl").open("a") as handle:
            handle.write("\n")
        return result

    monkeypatch.setattr(splits, "_copy_video", changed)
    output = tmp_path / "changed_source"
    with pytest.raises(ValueError, match="Source changed"):
        splits.extract_fold(source, output, training_fold="A")
    assert not output.exists()
