"""Offline calibration invariants: conditioning, task IDs, provenance, and splits."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import libero_vfd_dataset as offline
from scripts.libero_vfd import calibrate, get_threshold, load_run, rollout_id


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets" / "spatial"
    (root / "meta").mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "fps": 20,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
    }
    modality = {
        "state": {key: {"start": start, "end": end} for key, (start, end) in offline.STATE_SLICES.items()},
        "video": {
            key.removeprefix("video."): {"original_key": value} for key, value in offline.VIDEO_KEYS.items()
        },
    }
    mapping = {suite: [f"task_{i}" for i in range(10)] for suite in offline.SUITES.values()}
    mapping["libero_10"][0] = "KITCHEN_SCENE10_task_0"
    task_map = tmp_path / "task_map.py"
    # If this module is imported instead of parsed, the test must fail.
    task_map.write_text(f"raise RuntimeError('must not import LIBERO')\nlibero_task_map = {mapping!r}\n")
    for name, value in [("info.json", info), ("modality.json", modality)]:
        (root / "meta" / name).write_text(json.dumps(value))
    # Dataset-local task 9 is official task 1, deliberately not the same ID.
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 9, "task": "task 1"}) + "\n")
    items = [{"episode_index": i, "tasks": ["task 1"], "length": 3} for i in range(4)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    for item in items:
        index = item["episode_index"]
        path = offline.episode_path(root, info, index)
        path.parent.mkdir(exist_ok=True)
        # No demonstration actions are present: scoring must only need observations.
        pd.DataFrame(
            {
                "observation.state": [np.arange(8, dtype=np.float32) + step for step in range(3)],
                "timestamp": np.arange(3) / 20,
                "task_index": [9] * 3,
                "episode_index": [index] * 3,
                "frame_index": np.arange(3),
            }
        ).to_parquet(path)
        for key in offline.VIDEO_KEYS.values():
            video = offline.episode_path(root, info, index, key)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
    return root, task_map


def arguments(dataset, tmp_path, **overrides):
    root, task_map = dataset
    actor, scorer = tmp_path / "actor", tmp_path / "scorer"
    actor.mkdir(exist_ok=True)
    scorer.mkdir(exist_ok=True)
    values = dict(
        actor=actor,
        scorer=scorer,
        dataset_root=root.parent,
        suites=["spatial"],
        task_map=task_map,
        task_ids="1",
        episodes_per_task=2,
        episode_start=0,
        selection_seed=0,
        seed=0,
        uncertainty_seed=1234,
        denoising_steps=8,
        num_action_samples=1,
        flow_times=None,
        data_config="examples.Libero.custom_data_config:LiberoDataConfig",
        gpu_memory_limit_gb=0,
        max_frames=None,
        verify_actor=False,
        assume_successful_demos=True,
        score_only=False,
        scope="global",
        quantile=0.9,
        min_successes=2,
        output_dir=tmp_path / "scores",
        dry_run=False,
    )
    return SimpleNamespace(**(values | overrides))


def fake_decode(path, timestamps, fps):
    # Distinct camera pixels verify no rotation, channel switch, or view swap.
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    if "wrist" in str(path):
        image = image + 50
    return np.repeat(image[None], len(timestamps), axis=0)


def test_catalog_maps_language_to_official_ids_without_importing_libero(dataset):
    root, task_map = dataset
    languages = offline.task_languages(task_map)
    assert languages["libero_10"]["task 0"] == 0
    _, tasks, episodes = offline.catalog(root, "libero_spatial", languages)
    assert tasks == {9: "task 1"}
    assert {item["task_id"] for item in episodes} == {1}


def test_selection_is_reproducible_and_offsets_do_not_overlap(dataset):
    root, task_map = dataset
    _, _, episodes = offline.catalog(root, "libero_spatial", offline.task_languages(task_map))
    a = offline.select_episodes(episodes, [1], 2, 0, 43)
    b = offline.select_episodes(episodes, [1], 2, 2, 43)
    assert a == offline.select_episodes(episodes, [1], 2, 0, 43)
    assert {e["dataset_episode_index"] for e in a}.isdisjoint(e["dataset_episode_index"] for e in b)
    with pytest.raises(ValueError, match="available"):
        offline.select_episodes(episodes, [1], 5, 0, 43)


def test_raw_conditioning_keeps_images_state_and_language(dataset, monkeypatch):
    root, task_map = dataset
    info, tasks, episodes = offline.catalog(root, "libero_spatial", offline.task_languages(task_map))
    monkeypatch.setattr(offline, "decode_video", fake_decode)
    state, videos, timestamps, identity = offline.load_episode(episodes[0], info, tasks)
    observation = offline.observation_at(state, videos, 1, episodes[0]["language"])
    assert observation["state.gripper"].tolist() == [[7, 8]]
    assert observation["state.roll"].tolist() == [[4]]
    np.testing.assert_array_equal(observation["video.image"][0], np.arange(18).reshape(2, 3, 3))
    np.testing.assert_array_equal(observation["video.wrist_image"][0], observation["video.image"][0] + 50)
    assert observation["annotation.human.action.task_description"] == ["task 1"]
    assert len(identity) == 64 and len(timestamps) == 3
    assert not any(key.startswith("action.") for key in observation)


def test_corrupt_parquet_is_rejected_before_inference(dataset, monkeypatch):
    root, task_map = dataset
    info, tasks, episodes = offline.catalog(root, "libero_spatial", offline.task_languages(task_map))
    path = offline.episode_path(root, info, 0)
    table = pd.read_parquet(path)
    table["task_index"] = [9, 0, 9]
    table.to_parquet(path)
    with pytest.raises((ValueError, KeyError)):
        offline.load_episode(episodes[0], info, tasks)


def test_real_video_timestamp_alignment_and_resource_closure(tmp_path):
    import av

    path = tmp_path / "camera.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for level in [0, 80, 160]:
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), level, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames = offline.decode_video(path, np.array([0, 0.051, 0.1]), 20)
    assert frames.shape == (3, 16, 16, 3)
    np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), [0, 80, 160], atol=5)
    with pytest.raises(ValueError, match="cover"):
        offline.decode_video(path, np.array([0, 2.0]), 20)


@pytest.fixture
def fake_inference(monkeypatch):
    import torch
    import transformers

    observations = []

    class Monitor:
        signature = {"test": "same-pair"}

        def __init__(self, *args):
            pass

        def get_action_with_vfd(self, observation):
            observations.append(observation)
            value = float(observation["state.x"][0, 0])
            return {}, value, np.array([value])

    monkeypatch.setattr(offline, "SnapshotVFDPolicy", Monitor)
    monkeypatch.setattr(offline, "decode_video", fake_decode)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    return observations


def test_end_to_end_scores_then_calibrates_without_simulator(dataset, tmp_path, fake_inference):
    args = arguments(dataset, tmp_path)
    manifest = offline.score_dataset(args)
    assert manifest["status"] == "completed" and manifest["completed_episodes"] == 2
    assert len(fake_inference) == 6  # All three frames of each selected episode.
    calibration = json.loads((args.output_dir / "threshold.json").read_text())
    assert calibration["thresholds"]["global"]["threshold"] == 2
    assert calibration["calibration_data_kind"] == "expert_demonstration"
    assert "unknown" in calibration["official_initial_state_overlap"]
    _, episodes = load_run(args.output_dir)
    assert len(episodes) == 2
    assert all(rollout_id(metadata).startswith("dataset/") for _, metadata, _ in episodes)
    for _, metadata, _ in episodes:
        with pytest.raises(ValueError, match="overlap"):
            get_threshold(calibration, manifest["score_signature"], metadata)
    # Demo index 0 is not official initial state 0; do not fabricate a split claim.
    assert (
        get_threshold(
            calibration,
            manifest["score_signature"],
            {"suite": "libero_spatial", "task_id": 1, "initial_state_index": 0},
        )
        == 2
    )
    with pytest.raises(FileExistsError):
        offline.score_dataset(args)


def test_partial_episode_is_saved_but_cannot_calibrate(dataset, tmp_path, fake_inference, monkeypatch):
    calls = 0

    def interrupted(self, observation):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return {}, 1.0, np.ones(1)

    monkeypatch.setattr(offline.SnapshotVFDPolicy, "get_action_with_vfd", interrupted)
    args = arguments(dataset, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        offline.score_dataset(args)
    manifest, episodes = load_run(args.output_dir)
    assert manifest["status"] == "interrupted" and not episodes
    assert not (args.output_dir / "threshold.json").exists()
    partial = json.loads(next(args.output_dir.glob("task*/episode*.json")).read_text())
    assert partial["action_steps"] == 1 and not partial["complete"]


def test_smoke_run_cannot_fit_threshold(dataset, tmp_path, fake_inference):
    args = arguments(dataset, tmp_path, max_frames=1)
    result = offline.score_dataset(args)
    assert result["smoke_only"] and len(fake_inference) == 2
    assert not (args.output_dir / "threshold.json").exists()
    with pytest.raises(ValueError, match="smoke"):
        calibrate([args.output_dir], tmp_path / "invalid.json", min_successes=2)


def test_dry_run_never_loads_model_or_videos(dataset, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run attempted inference or video decoding")

    monkeypatch.setattr(offline, "SnapshotVFDPolicy", forbidden)
    monkeypatch.setattr(offline, "decode_video", forbidden)
    args = arguments(dataset, tmp_path, dry_run=True)
    plan = offline.score_dataset(args)
    assert plan["planned_frames"] == 6 and not args.output_dir.exists()


def test_success_assumption_and_count_are_checked_before_gpu(dataset, tmp_path):
    with pytest.raises(ValueError, match="assume-successful-demos"):
        offline.score_dataset(arguments(dataset, tmp_path, assume_successful_demos=False))
    with pytest.raises(ValueError, match="need 20"):
        offline.score_dataset(arguments(dataset, tmp_path, min_successes=20))


def test_duplicate_dataset_aliases_and_mixed_calibration_are_rejected(dataset, tmp_path, fake_inference):
    args = arguments(dataset, tmp_path)
    offline.score_dataset(args)
    alias = tmp_path / "alias"
    alias.symlink_to(args.output_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate"):
        calibrate([args.output_dir, alias], tmp_path / "duplicate.json", min_successes=2)
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "manifest.json").write_text(
        json.dumps({"status": "completed", "score_signature": {"test": "same-pair"}})
    )
    task = policy / "task01"
    task.mkdir()
    metadata = {
        "suite": "libero_spatial",
        "task_id": 1,
        "initial_state_index": 0,
        "success": True,
        "complete": True,
    }
    (task / "episode000.json").write_text(json.dumps(metadata))
    np.savez(task / "episode000.npz", uncertainty=np.ones(3))
    with pytest.raises(ValueError, match="do not mix"):
        calibrate([args.output_dir, policy], tmp_path / "mixed.json", min_successes=2)
