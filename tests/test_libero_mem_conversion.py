"""Small HDF5 conversions, real video decoding, GR00T loading and resume."""

import json
import os

import av
import h5py
import numpy as np
import pandas as pd
import pytest

from examples.Libero_Mem import conversion
from examples.Libero_Mem.source import audit_sources
from examples.Libero_Mem.tasks import TASK_BY_NUMBER


def simulator_rgb(task, frame, camera):
    image = np.zeros((256, 256, 3), np.uint8)
    for y, x, level in ((0, 0, 25), (0, 128, 75), (128, 0, 135), (128, 128, 195)):
        image[y : y + 128, x : x + 128] = [level + frame * 3, task + camera * 80, 40 + frame * 5]
    return image


@pytest.fixture
def sources(tmp_path):
    full, raw = tmp_path / "full", tmp_path / "raw"
    full.mkdir()
    raw.mkdir()
    metadata = {}
    for number in (1, 7):
        task = TASK_BY_NUMBER[number]
        metadata[task.name] = {}
        with h5py.File(raw / task.filename, "w") as handle:
            for demo_number in (1, 3, 101):
                demo = handle.create_group(f"data/demo_{demo_number}")
                initial = np.array([number, demo_number, 0.25, 1.0], np.float64)
                demo.create_dataset("init_state", data=initial)
                demo.create_dataset("states", data=np.repeat(initial[None], 4, axis=0))
                demo.create_dataset("actions", data=np.zeros((4, 7), np.float64))
                demo.attrs["model_file"] = "<mujoco/>"
        with h5py.File(full / task.filename, "w") as handle:
            for demo_number, length in ((1, 4), (3, 3)):
                demo_id = f"demo_{demo_number}"
                demo = handle.create_group(f"data/{demo_id}")
                pose = np.arange(length * 6, dtype=np.float64).reshape(length, 6) / 100 + number / 10
                gripper = np.stack([np.arange(length) / 1000, np.arange(length) / -1000], axis=1)
                actions = np.arange(length * 7, dtype=np.float64).reshape(length, 7) / 100
                actions[-1, :6] = 0  # Preserve terminal stationary rows too.
                actions[:, -1] = np.resize([-1, 1], length)
                demo.create_dataset("actions", data=actions)
                demo.create_dataset("dones", data=np.array([0] * (length - 1) + [1], np.uint8))
                demo.create_dataset("obs/ee_states", data=pose)
                demo.create_dataset("obs/gripper_states", data=gripper)
                demo.create_dataset("obs/joint_states", data=np.zeros((length, 7), np.float64))
                for camera, key in enumerate(conversion.CAMERAS.values()):
                    # Exactly reproduce upstream refinement's image transform.
                    images = [np.rot90(simulator_rgb(number, i, camera)[:, ::-1], 2) for i in range(length)]
                    demo.create_dataset(key, data=np.stack(images))
                metadata[task.name][demo_id] = {
                    "success": True,
                    "task_description": task.instruction,
                    "initial_state": [number, demo_number, 0.25, 1.0],
                    "exo_boxes": [{} for _ in range(length)],
                    "ego_boxes": [{} for _ in range(length)],
                }
    (full / "metainfo.json").write_text(json.dumps(metadata))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(audit_sources(full, raw, task_ids=[1, 7])))
    return full, raw, audit


def convert(sources, output, **kwargs):
    return conversion.convert_dataset(sources[0], output, audit=sources[2], task_ids=[1, 7], **kwargs)


def read_frames(output, episode, chunks_size=2):
    return pd.read_parquet(
        output / conversion.DATA_PATH.format(episode_chunk=episode // chunks_size, episode_index=episode)
    )


def test_parallel_conversion_and_existing_libero_loader(sources, tmp_path):
    from examples.Libero.custom_data_config import LiberoDataConfig
    from examples.Libero.eval.utils import get_libero_image, normalize_gripper_action
    from gr00t.data.dataset import LeRobotSingleDataset

    before = {path: conversion.file_sha256(path) for path in sources[0].iterdir()}
    output = tmp_path / "converted"
    result = convert(sources, output, workers=2, chunks_size=2)
    assert result["episodes"] == 4 and result["frames"] == 14
    assert result["filtered_frames"] == 0 and result["resampled"] is False
    assert (output / "meta/modality.json").read_bytes() == conversion.MODALITY_PATH.read_bytes()
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_chunks"] == 2 and info["total_videos"] == 8 and info["fps"] == 20
    mappings = json.loads((output / "meta/episode_mapping.json").read_text())
    assert [entry["source_demo_id"] for entry in mappings] == ["demo_1", "demo_3", "demo_1", "demo_3"]
    assert [entry["source_task_id"] for entry in mappings] == [1, 1, 7, 7]
    all_state, all_action = [], []
    offset = 0
    for episode, mapping in enumerate(mappings):
        data = read_frames(output, episode)
        with h5py.File(mapping["source_file"], "r") as handle:
            original = handle[f"data/{mapping['source_demo_id']}"]
            state = np.concatenate([original["obs/ee_states"][:], original["obs/gripper_states"][:]], axis=1)
            source_action = original["actions"][:]
        physical_action = np.stack(data["action"])
        # The actual existing LIBERO adapter must recover simulator commands.
        normalize_gripper_action(physical_action, binarize=True)
        np.testing.assert_allclose(physical_action, source_action, atol=1e-7)
        np.testing.assert_array_equal(np.stack(data["observation.state"]), state.astype(np.float32))
        assert data["source_frame_index"].tolist() == list(range(len(data)))
        assert data["index"].tolist() == list(range(offset, offset + len(data)))
        assert data["next.done"].tolist() == [False] * (len(data) - 1) + [True]
        np.testing.assert_allclose(data["timestamp"], np.arange(len(data)) / 20)
        assert data["source_episode_key"].unique().tolist() == [mapping["source_episode_key"]]
        all_state.append(np.stack(data["observation.state"]))
        all_action.append(np.stack(data["action"]))
        for camera, key in enumerate(conversion.CAMERAS):
            path = output / conversion.VIDEO_PATH.format(
                episode_chunk=episode // 2, episode_index=episode, video_key=key
            )
            with av.open(str(path)) as container:
                assert container.streams.video[0].average_rate == 20
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            assert len(decoded) == len(data)
            for i, image in enumerate(decoded):
                raw = simulator_rgb(mapping["source_task_id"], i, camera)
                expected = get_libero_image({"agentview_image": raw, "robot0_eye_in_hand_image": raw})[0]
                assert np.abs(image.astype(float) - expected).mean() < 4
        offset += len(data)
    stats = json.loads((output / "meta/stats.json").read_text())
    for key, values in (("observation.state", all_state), ("action", all_action)):
        values = np.concatenate(values)
        np.testing.assert_allclose(stats[key]["mean"], values.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(stats[key]["q99"], np.quantile(values, 0.99, axis=0), atol=1e-7)
    config = LiberoDataConfig()
    dataset = LeRobotSingleDataset(
        output,
        config.modality_config(),
        "new_embodiment",
        video_backend="decord",
        video_backend_kwargs={"num_threads": 1},
    )
    assert len(dataset) == 14
    for episode in range(4):
        data = read_frames(output, episode)
        sample = dataset.get_step_data(episode, len(data) - 1)
        # Preserve existing LIBERO last-action padding behavior exactly.
        assert sample["action.x"].shape == (16, 1)
        np.testing.assert_allclose(
            np.concatenate([sample[key] for key in config.action_keys], axis=-1),
            np.repeat(np.stack(data["action"])[-1:], 16, axis=0),
        )
        first = dataset.get_step_data(episode, 0)
        np.testing.assert_allclose(
            np.concatenate([first[key] for key in config.action_keys], axis=-1),
            np.stack(data["action"])[np.minimum(np.arange(16), len(data) - 1)],
        )
        assert sample["state.gripper"].shape == (1, 2)
        assert sample["annotation.human.action.task_description"] == [
            TASK_BY_NUMBER[1 if episode < 2 else 7].instruction
        ]
    assert all(conversion.file_sha256(path) == checksum for path, checksum in before.items())


@pytest.mark.parametrize("corrupt_cache", [False, True])
def test_resume_preserves_encoded_episodes_and_detects_corruption(sources, tmp_path, monkeypatch, corrupt_cache):
    output = tmp_path / "converted"
    original_finalize = conversion.finalize

    def interrupted(*args):
        raise RuntimeError("interrupted before publication")

    monkeypatch.setattr(conversion, "finalize", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        convert(sources, output, workers=1, demo_ids=[1])
    assert not output.exists()
    cache = output.with_name(f".{output.name}.work") / "episodes/000000/camera_0.mp4"
    encoded_mtime = cache.stat().st_mtime_ns
    monkeypatch.setattr(conversion, "finalize", original_finalize)
    if corrupt_cache:
        cache.write_bytes(b"corrupt video")
        with pytest.raises(ValueError, match="corrupt cached"):
            convert(sources, output, workers=1, demo_ids=[1], resume=True)
        assert not output.exists()
    else:
        result = convert(sources, output, workers=2, demo_ids=[1], resume=True)
        assert result["episodes"] == 2 and cache.stat().st_mtime_ns == encoded_mtime
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            convert(sources, output, workers=1, demo_ids=[1], resume=True)


def test_resume_rejects_changed_selection(sources, tmp_path, monkeypatch):
    output = tmp_path / "converted"

    def interrupted(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(conversion, "finalize", interrupted)
    with pytest.raises(RuntimeError):
        convert(sources, output, workers=1, demo_ids=[1])
    with pytest.raises(ValueError, match="settings differ"):
        convert(sources, output, workers=1, demo_ids=[3], resume=True)
    with pytest.raises(FileExistsError, match="use --resume"):
        convert(sources, output, workers=1, demo_ids=[1])


def test_dry_run_is_read_only_and_missing_demos_are_explicit(sources, tmp_path, monkeypatch):
    output = tmp_path / "not-created" / "converted"

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not encode episodes")

    monkeypatch.setattr(conversion, "convert_one", forbidden)
    result = convert(sources, output, workers=1, dry_run=True)
    assert result["status"] == "dry_run" and result["episodes"] == 4 and result["frames"] == 14
    assert not output.parent.exists()
    with pytest.raises(ValueError, match="missing from"):
        convert(sources, output, demo_ids=[2], dry_run=True)
    with pytest.raises(ValueError, match="outside the source"):
        convert(sources, sources[0] / "inside-source", dry_run=True)


def test_audit_identity_is_verified(sources, tmp_path):
    stat = (sources[0] / "metainfo.json").stat()
    with (sources[0] / "metainfo.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="differs from audit"):
        convert(sources, tmp_path / "converted", dry_run=True)
    # Same size/mtime with different contents must still fail the metadata hash.
    path = sources[0] / "metainfo.json"
    value = path.read_text()[:-1].replace("True", "true")
    value = value.replace("0.25", "0.26", 1)
    path.write_text(value)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="metadata differs from audit"):
        convert(sources, tmp_path / "converted", dry_run=True)


def test_modified_numeric_payload_fails_even_with_restored_file_identity(sources, tmp_path):
    path = sources[0] / TASK_BY_NUMBER[1].filename
    stat = path.stat()
    with h5py.File(path, "r+") as handle:
        handle["data/demo_1/actions"][0, 0] += 0.1
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Actions differ from audit"):
        convert(sources, tmp_path / "converted", workers=1, demo_ids=[1])


def test_stable_source_keys_across_selections(sources, tmp_path):
    all_jobs = conversion.conversion_plan(sources[0], sources[2], task_ids=[1, 7])["jobs"]
    subset = conversion.conversion_plan(sources[0], sources[2], task_ids=[7], demo_ids=[3])["jobs"]
    assert subset[0]["episode_index"] == 0 and subset[0]["task_index"] == 0
    assert subset[0]["source_episode_key"] == all_jobs[-1]["source_episode_key"]


def test_training_transform_masks_inverse_and_inference(sources, tmp_path):
    import copy

    import torch

    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.data.dataset import LeRobotSingleDataset

    output = tmp_path / "converted"
    convert(sources, output, workers=1, demo_ids=[1])
    config = LiberoDataConfig()
    transforms = config.transform()
    dataset = LeRobotSingleDataset(
        output,
        config.modality_config(),
        "new_embodiment",
        video_backend="decord",
        video_backend_kwargs={"num_threads": 1},
    )
    transforms.set_metadata(dataset.metadata)
    raw = dataset.get_step_data(0, 0)
    transformed = transforms(copy.deepcopy(raw))
    assert transformed["state"].shape == (1, 64)
    assert transformed["action"].shape == (16, 32)
    assert transformed["state_mask"].sum() == 8
    assert transformed["action_mask"].sum() == 16 * 7
    assert np.isfinite(transformed["state"]).all() and np.isfinite(transformed["action"]).all()
    restored = transforms.unapply({"action": torch.from_numpy(transformed["action"][None].copy())})
    for key in config.action_keys:
        np.testing.assert_allclose(restored[key][0], raw[key], atol=1e-6)
    transforms.eval()
    observation = {key: value for key, value in raw.items() if key not in config.action_keys}
    inference = transforms(copy.deepcopy(observation))
    assert "action" not in inference
    assert inference["state"].shape == (1, 64)
