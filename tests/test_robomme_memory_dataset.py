"""Real converted data/collator checks for causal RoboMME cached-history samples."""

import copy
import io
import json
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from examples.RoboMME.custom_data_config import RoboMMEDataConfig
from examples.RoboMME.feature_cache import build_feature_cache, cache_plan
from examples.RoboMME.memory_config import load_memory_config, position_embedding_3d, prepare_frame_sampling
from examples.RoboMME.memory_data_config import MEMORY_KEYS, RoboMMEMemoryDataConfig
from examples.RoboMME.memory_dataset import RoboMMEMemoryDataset
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.transforms import DefaultDataCollator
from scripts.convert_robomme_lerobot import convert_dataset


def image_value(level):
    buffer = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), level, np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


class ColorEncoder:
    signature = {"pool_type": "mean", "token_per_image": 16, "image_dim": 4, "weights": "fixture"}

    def encode(self, images):
        return np.broadcast_to(images[:, 0, 0, 0].astype(np.float32)[:, None, None], (len(images), 16, 4)).copy()


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    features = {
        key: {"dtype": "image", "shape": [32, 32, 3], "names": ["height", "width", "channel"]}
        for key in ("image", "wrist_image")
    }
    for name in ("state", "actions"):
        features[name] = {"dtype": "float32", "shape": [8], "names": [name]}
    for name in ("timestamp", "index", "episode_index", "frame_index", "task_index", "exec_start_idx"):
        features[name] = {"dtype": "float32" if name == "timestamp" else "int64", "shape": [1], "names": None}
    features["is_demo"] = {"dtype": "bool", "shape": [1], "names": None}
    episodes = []
    for episode_id, colors, start, task_id in ((3, [10, 20, 30, 40, 50], 2, 7), (1001, [200, 210], 0, 42)):
        length = len(colors)
        data = pd.DataFrame(
            {
                "image": [image_value(color) for color in colors],
                "wrist_image": [image_value(color + 1) for color in colors],
                "state": [
                    np.array([color / 100] * 7 + [0.01 + index * 0.001], np.float32)
                    for index, color in enumerate(colors)
                ],
                "actions": [
                    np.array([color / 10] * 7 + [(-1) ** index], np.float32) for index, color in enumerate(colors)
                ],
                "timestamp": np.arange(length) / 10,
                "index": np.arange(length) + episode_id * 10,
                "episode_index": [episode_id] * length,
                "frame_index": np.arange(length),
                "task_index": [task_id] * length,
                "exec_start_idx": [start] * length,
                "is_demo": np.arange(length) < start,
            }
        )
        path = source / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        episodes.append({"episode_index": episode_id, "length": length, "tasks": [f"Task {task_id}"]})
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "fps": 10,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 7,
        "total_tasks": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
    (source / "meta/tasks.jsonl").write_text(
        '{"task_index":7,"task":"Task 7"}\n{"task_index":42,"task":"Task 42"}\n'
    )
    config = load_memory_config()
    config["memory_feature"]["img"]["input_dim"] = 4
    cache, converted = tmp_path / "cache", tmp_path / "converted"
    build_feature_cache(cache_plan(source, config), cache, ColorEncoder(), batch_size=2)
    convert_dataset(source, converted, workers=1)
    return source, converted, cache, config


def memory_dataset(prepared, **kwargs):
    _, converted, cache, config = prepared
    return RoboMMEMemoryDataset(
        converted, memory_cache=cache, memory_config=config, video_backend="torchvision_av", **kwargs
    )


def parquet_path(prepared, episode=0):
    return prepared[1] / f"data/chunk-000/episode_{episode:06d}.parquet"


def test_original_frame_indices_prefix_and_causal_memory(prepared):
    dataset = memory_dataset(prepared)
    assert isinstance(dataset, LeRobotSingleDataset) and len(dataset) == 5
    first = dataset.get_step_data(0, 0)
    # Converted episode 0 is original episode 3; execution step 0 is source frame 2.
    np.testing.assert_array_equal(first["static_image_emb"][::16, 0][:3], [10, 20, 30])
    assert first["static_mask"].sum() == 3 * 16
    assert not first["static_image_emb"][48:].any()
    np.testing.assert_array_equal(first["static_pos_emb"][:48], position_embedding_3d([0, 1, 2]).reshape(48, 768))
    last = dataset.get_step_data(0, 2)
    np.testing.assert_array_equal(last["static_image_emb"][::16, 0][:5], [10, 20, 30, 40, 50])
    second_episode = dataset.get_step_data(1, 0)
    assert second_episode["static_mask"].sum() == 16
    assert second_episode["static_image_emb"][0, 0] == 200
    assert first["static_image_emb"][0, 0] == 10  # earlier returned arrays retain their data
    with pytest.raises(IndexError):
        dataset.get_step_data(0, -1)
    dataset.close()


def test_vanilla_labels_normalization_metadata_and_instruction_unchanged(prepared):
    dataset = memory_dataset(prepared)
    config = RoboMMEDataConfig()
    vanilla = LeRobotSingleDataset(
        prepared[1],
        config.modality_config(),
        "new_embodiment",
        transforms=config.transform(),
        video_backend="torchvision_av",
    )
    assert dataset.metadata.model_dump(mode="json") == vanilla.metadata.model_dump(mode="json")
    for index in (0, len(dataset) - 1):
        actual, expected = dataset[index], vanilla[index]
        assert set(actual) == set(expected) | set(MEMORY_KEYS)
        for key in ("state", "state_mask", "action", "action_mask", "embodiment_id"):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual["eagle_content"]["text_list"] == expected["eagle_content"]["text_list"]
        np.testing.assert_array_equal(
            dataset.transforms.unapply({"action": torch.from_numpy(actual["action"][None])})[
                "action.joint_position"
            ],
            vanilla.transforms.unapply({"action": torch.from_numpy(expected["action"][None])})[
                "action.joint_position"
            ],
        )
    dataset.close()


def test_real_collator_and_two_worker_dataloader(prepared):
    dataset = memory_dataset(prepared)
    collator = DefaultDataCollator()
    # Process a sample in the parent first; spawned workers must reopen memmaps.
    # The environment's native video/Torch state is unsafe to fork after this.
    _ = dataset[0]
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collator, multiprocessing_context="spawn")
    batches = list(loader)
    assert sum(batch["state"].shape[0] for batch in batches) == 5
    batch = batches[0]
    assert batch["static_image_emb"].shape == (2, 512, 4)
    assert batch["static_pos_emb"].shape == (2, 512, 768)
    assert batch["static_mask"].dtype == torch.bool
    assert batch["static_mask"].sum(1).tolist() == [48, 64]
    assert batch["action"].shape == (2, 16, 32) and batch["eagle_input_ids"].shape[0] == 2
    assert torch.isfinite(batch["static_image_emb"]).all()
    dataset.close()


def _read_memory_in_forked_process(dataset, connection):
    try:
        memory = prepare_frame_sampling(dataset._get_memory_features(3), 2, dataset.memory_config)
        connection.send((os.getpid(), dataset._owner_pid, memory["static_image_emb"][::16, 0][:3]))
    finally:
        dataset.close()
        connection.close()


def test_forked_process_reopens_inherited_memmap(prepared):
    dataset = memory_dataset(prepared)
    inherited = dataset._get_memory_features(3)
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_read_memory_in_forked_process, args=(dataset, child))
    try:
        process.start()
        child.close()
        assert parent.poll(20), "Child did not return memory features"
        pid, owner_pid, values = parent.recv()
        process.join(10)
        assert process.exitcode == 0 and pid == owner_pid and pid != os.getpid()
        np.testing.assert_array_equal(values, [10, 20, 30])
        assert not inherited._mmap.closed  # child cleanup must not close the parent's map
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        child.close()
        dataset.close()


def test_mixture_loader_does_not_drop_memory(prepared):
    dataset = memory_dataset(prepared)
    mixture = LeRobotMixtureDataset(data_mixture=[(dataset, 1.0)], mode="val", seed=42)
    for index in range(len(mixture)):
        _, episode, step = mixture.sample_step(index)
        expected = dataset.get_step_data(episode, step)
        sample = mixture[index]
        for key in MEMORY_KEYS:
            np.testing.assert_array_equal(sample[key], expected[key])
    dataset.close()


def test_external_config_and_batched_transform_keep_memory_without_labels(prepared):
    config = load_data_config("examples.RoboMME.memory_data_config:RoboMMEMemoryDataConfig")
    assert isinstance(config, RoboMMEMemoryDataConfig)
    dataset = memory_dataset(prepared)
    raw = dataset.get_step_data(0, 0)
    dataset.transforms.eval()
    observation = {key: value for key, value in raw.items() if not key.startswith("action.")}
    single = dataset.transforms(copy.deepcopy(observation))
    assert "action" not in single
    batched = {key: np.stack([value, value]) for key, value in observation.items()}
    batch = dataset.transforms(batched)
    for key in MEMORY_KEYS:
        np.testing.assert_array_equal(batch[key][0].numpy(), single[key])
    dataset.close()


def test_subset_uses_only_its_execution_rows_and_stats(prepared, tmp_path):
    source, _, cache, config = prepared
    subset = tmp_path / "subset"
    convert_dataset(source, subset, workers=1, episode_indices=[1001])
    # The excluded episode's features are deliberately unreadable; they must not be opened.
    (cache / "episode_000003/features.npy").write_bytes(b"excluded episode")
    dataset = RoboMMEMemoryDataset(
        subset, memory_cache=cache, memory_config=config, video_backend="torchvision_av"
    )
    assert len(dataset) == 2 and set(dataset._cache_records) == {1001}
    assert set(dataset._episode_bindings) == {0}
    assert dataset[0]["static_image_emb"][0, 0] == 200
    assert dataset.metadata.statistics.action["joint_position"].min[0] == 20
    dataset.close()


@pytest.mark.parametrize("column", ["state", "actions"])
def test_execution_vectors_must_match_cache(prepared, column):
    path = parquet_path(prepared)
    data = pd.read_parquet(path)
    vectors = np.stack(data[column])
    vectors[0, 0] += 0.125
    data[column] = list(vectors)
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=f"Execution {column} differ"):
        memory_dataset(prepared)


@pytest.mark.parametrize(
    "column,values",
    [
        ("source_frame_index", [0, 1, 2]),
        ("source_episode_index", [999, 999, 999]),
        ("source_episode_index", [3, 1001, 3]),
        ("source_exec_start_idx", [0, 0, 0]),
        ("is_demo", [True, False, False]),
    ],
)
def test_wrong_provenance_or_demo_rows_rejected(prepared, column, values):
    path = parquet_path(prepared)
    data = pd.read_parquet(path)
    data[column] = values
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        memory_dataset(prepared)


def test_incomplete_cache_and_incompatible_pooling_rejected(prepared):
    _, _, cache, config = prepared
    changed = copy.deepcopy(config)
    changed["pool_type"] = "max"
    with pytest.raises(ValueError, match="settings differ"):
        memory_dataset(prepared[:3] + (changed,))
    path = cache / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "building"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        memory_dataset(prepared)


def test_corrupt_feature_file_is_rejected_on_first_read(prepared):
    dataset = memory_dataset(prepared)
    path = prepared[2] / "episode_000003/features.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        dataset[0]
    dataset.close()


def test_bounded_maps_pickle_and_checksum_only_once_per_unchanged_file(prepared, monkeypatch):
    import examples.RoboMME.memory_dataset as module

    calls = []
    original = module.load_cached_episode

    def tracked(root, source_id, *, verify_checksum=True):
        calls.append((source_id, verify_checksum))
        return original(root, source_id, verify_checksum=verify_checksum)

    monkeypatch.setattr(module, "load_cached_episode", tracked)
    dataset = memory_dataset(prepared, max_open_episodes=1)
    _ = dataset[0]
    _ = dataset[3]
    _ = dataset[0]
    assert calls == [(3, True), (1001, True), (3, False)]
    assert len(dataset._feature_maps) == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._feature_maps and not restored._verified_files
    np.testing.assert_array_equal(restored[0]["static_image_emb"], dataset[0]["static_image_emb"])
    restored.close()
    dataset.close()


def test_vanilla_transform_cannot_silently_drop_memory(prepared):
    with pytest.raises(ValueError, match="preserve memory"):
        memory_dataset(prepared, transforms=RoboMMEDataConfig().transform())
    dataset = memory_dataset(prepared)
    raw = dataset.get_step_data(0, 0)
    raw.pop("static_mask")
    with pytest.raises(ValueError, match="Missing RoboMME memory"):
        dataset.transforms(raw)
    dataset.close()
