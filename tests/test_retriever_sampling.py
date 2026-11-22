"""Check raw action schemas and retriever-only pairing groups on CPU."""

import copy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dtaidistance import dtw_ndim

from gr00t.rag.data import AlignedFramesBatchSampler, get_alignment_groups
from gr00t.rag.utils import get_aligned_frame_ids, get_trajectory_actions


def make_dataset(root, *, column="actions", dimension=8, indexed=True):
    root.mkdir()
    (root / "meta").mkdir()
    lengths = np.array([6, 7, 6, 8, 7, 2])
    owners = ["VideoUnmask", "VideoUnmask", "VideoUnmaskSwap", "VideoUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"]
    instructions = ["shared", "shared", "shared", "shared", "singleton", "shared"]
    episodes = [
        {"episode_index": index, "length": int(length), "tasks": [instructions[index]]}
        for index, length in enumerate(lengths)
    ]
    (root / "meta/info.json").write_text(json.dumps({"features": {column: {"shape": [dimension]}}}))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "shared"}) + "\n")
    (root / "meta/modality.json").write_text(json.dumps({"action": {"joint": {"original_key": column}}}))
    trajectories = {}
    for index, length in enumerate(lengths):
        time = np.linspace(0, 1, length)[:, None]
        actions = np.sin(time * 3 + np.arange(dimension)[None] / 4 + index / 20)
        trajectories[index] = pd.DataFrame({column: list(actions)})
    dataset = SimpleNamespace(
        dataset_path=root,
        trajectory_ids=np.arange(6),
        trajectory_lengths=lengths,
        task_groups={"shared": [0, 1, 2, 3, 5], "singleton": [4]},
        all_steps=[(index, frame) for index, length in enumerate(lengths) for frame in range(length)],
        get_trajectory_data=trajectories.__getitem__,
    )
    if indexed:
        value = {
            "version": 1,
            "source_metadata_sha256": {
                name: hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest()
                for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
            },
            "episodes": [
                {"episode_index": index, "task": owners[index], "length": int(length)}
                for index, length in enumerate(lengths)
            ],
        }
        (root / "meta/episode_tasks.json").write_text(json.dumps(value))
    return dataset, owners, trajectories


def mixture(dataset):
    paths = [(0, episode, frame) for episode, frame in dataset.all_steps]
    return SimpleNamespace(
        datasets=[dataset],
        all_frame_paths=paths,
        all_frame_paths_inverted={path: index for index, path in enumerate(paths)},
    )


def test_compound_groups_separate_shared_instructions_without_dropping_policy_data(tmp_path):
    dataset, _, _ = make_dataset(tmp_path / "robomme")
    original = copy.deepcopy(dataset.task_groups)
    all_steps = dataset.all_steps.copy()
    groups, report = get_alignment_groups(dataset, num_frames=3)
    assert groups == {("VideoUnmask", "shared"): [0, 1], ("VideoUnmaskSwap", "shared"): [2, 3]}
    assert report["total_groups"] == 3 and report["eligible_groups"] == 2
    assert report["total_demonstrations"] == 6 and report["eligible_demonstrations"] == 4
    assert report["short_demonstrations"] == [5]
    assert report["skipped_groups"][0]["episode_indices"] == [4]
    assert report["grouping"] == "task_and_instruction" and len(report["task_map_sha256"]) == 64
    assert dataset.task_groups == original and dataset.all_steps == all_steps
    assert {episode for episode, _ in dataset.all_steps} == set(range(6))
    json.dumps(report)  # Sampling reports are persisted without tuple-key encoding.


def test_sampler_uses_distinct_compatible_demos_and_keeps_positive_halves(tmp_path):
    dataset, owners, _ = make_dataset(tmp_path / "robomme")
    data = mixture(dataset)
    before = random.getstate()
    sampler = AlignedFramesBatchSampler(data, 2, 3, 4)
    assert random.getstate() == before
    try:
        random.seed(42)
        batches = list(sampler)
        random.seed(42)
        assert batches == list(sampler)
    finally:
        random.setstate(before)
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 12
        for left, right in zip(batch[:6], batch[6:]):
            _, episode_a, frame_a = data.all_frame_paths[left]
            _, episode_b, frame_b = data.all_frame_paths[right]
            assert episode_a != episode_b and owners[episode_a] == owners[episode_b]
            assert episode_a in (0, 1, 2, 3) and episode_b in (0, 1, 2, 3)
            assert frame_a < dataset.trajectory_lengths[episode_a]
            assert frame_b < dataset.trajectory_lengths[episode_b]
    assert len(data.all_frame_paths) == sum(dataset.trajectory_lengths)


def test_libero_keeps_instruction_keys_and_group_order(tmp_path):
    dataset, _, _ = make_dataset(tmp_path / "libero", column="action", dimension=7, indexed=False)
    # Two valid groups, as in the original LIBERO sampler.
    dataset.task_groups = {"first": [0, 1], "second": [2, 3]}
    groups, report = get_alignment_groups(dataset, 3)
    assert list(groups) == list(dataset.task_groups)
    assert groups == dataset.task_groups and report["grouping"] == "instruction"
    data = mixture(dataset)
    sampler = AlignedFramesBatchSampler(data, 2, 3, 1)
    previous = random.getstate()
    try:
        random.seed(91)
        actual = next(iter(sampler))
        random.seed(91)
        # The original single-suite distribution consumes no cuts/RNG values.
        random.choices(range(3), k=0)
        left, right = [], []
        for key in random.sample(list(dataset.task_groups), 2):
            first, second = random.sample(dataset.task_groups[key], 2)
            a, b = get_aligned_frame_ids(dataset, first, second, 3)
            left.extend((0, first, frame) for frame in a)
            right.extend((0, second, frame) for frame in b)
        expected = [data.all_frame_paths_inverted[path] for path in left + right]
        assert actual == expected
    finally:
        random.setstate(previous)


@pytest.mark.parametrize("column,dimension", [("action", 7), ("actions", 8)])
def test_dtw_matches_original_raw_action_algorithm(tmp_path, column, dimension):
    dataset, _, trajectories = make_dataset(tmp_path / "data", column=column, dimension=dimension, indexed=False)
    raw = get_trajectory_actions(dataset, 0)
    np.testing.assert_array_equal(raw, np.stack(trajectories[0][column]))
    assert raw.dtype == np.double and raw.shape == (6, dimension)
    path = dtw_ndim.warping_path(np.stack(trajectories[0][column]), np.stack(trajectories[1][column]))
    unique = list(dict((a, b) for a, b in reversed(path)).items())[::-1]
    previous = random.getstate()
    try:
        random.seed(72)
        expected = [unique[index] for index in sorted(random.sample(range(len(unique)), 3))]
        random.seed(72)
        first, second = get_aligned_frame_ids(dataset, 0, 1, 3)
        assert list(zip(first, second)) == expected
    finally:
        random.setstate(previous)


def test_modality_mapping_selects_action_column_when_both_names_exist(tmp_path):
    dataset, _, trajectories = make_dataset(tmp_path / "data", indexed=False)
    trajectories[0]["action"] = [np.zeros(7)] * 6
    with pytest.raises(ValueError, match="unambiguous"):
        get_trajectory_actions(dataset, 0)
    dataset.lerobot_modality_meta = SimpleNamespace(action={"joint": SimpleNamespace(original_key="actions")})
    assert get_trajectory_actions(dataset, 0).shape == (6, 8)


@pytest.mark.parametrize("bad", [np.full((6, 8), np.nan), np.full((6, 8), np.inf), np.zeros(6)])
def test_invalid_raw_actions_rejected_before_dtw(tmp_path, bad):
    dataset, _, trajectories = make_dataset(tmp_path / "data", indexed=False)
    trajectories[0] = pd.DataFrame({"actions": list(bad)})
    with pytest.raises(ValueError, match="finite action vectors"):
        get_trajectory_actions(dataset, 0)


def test_mapping_must_match_source_and_cover_all_episodes(tmp_path):
    dataset, _, _ = make_dataset(tmp_path / "data")
    index = dataset.dataset_path / "meta/episode_tasks.json"
    value = json.loads(index.read_text())
    index.write_text(json.dumps({**value, "episodes": value["episodes"][:-1]}))
    with pytest.raises(ValueError, match="cover every"):
        get_alignment_groups(dataset, 3)
    index.write_text(json.dumps(value))
    (dataset.dataset_path / "meta/tasks.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="does not match"):
        get_alignment_groups(dataset, 3)


def test_all_task_source_requires_explicit_index_and_accepts_portable_map(tmp_path):
    dataset, _, _ = make_dataset(tmp_path / "all16")
    index = dataset.dataset_path / "meta/episode_tasks.json"
    outside = tmp_path / "portable_map.json"
    index.rename(outside)
    (dataset.dataset_path / "manifest.json").write_text(
        json.dumps({"converter": "scripts/convert_robomme_lerobot.py"})
    )
    with pytest.raises(ValueError, match="--task-map"):
        get_alignment_groups(dataset, 3)
    groups, report = get_alignment_groups(dataset, 3, outside)
    assert len(groups) == 2 and report["task_map"] == str(outside)


def test_sampler_validates_eligible_group_capacity(tmp_path):
    dataset, _, _ = make_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="exceeds 2 eligible"):
        AlignedFramesBatchSampler(mixture(dataset), 3, 3, 1)
    with pytest.raises(ValueError, match="exceeds 0 eligible"):
        AlignedFramesBatchSampler(mixture(dataset), 1, 9, 1)
    with pytest.raises(ValueError, match="positive"):
        AlignedFramesBatchSampler(mixture(dataset), 1, 0, 1)
    multiple = SimpleNamespace(datasets=[dataset, dataset])
    with pytest.raises(ValueError, match="single"):
        AlignedFramesBatchSampler(multiple, 1, 3, 1, task_map="unused")
