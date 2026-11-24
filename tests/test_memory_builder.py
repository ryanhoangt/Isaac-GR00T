"""Check task-scoped demo selection, cached sequence masks and memory reload."""

import importlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rag.memory import encode_memory_bank, file_fingerprint, plan_memory, save_memory
from gr00t.rag.retriever import retrieve_from_memory


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "robomme"
    (root / "meta").mkdir(parents=True)
    names = ["VideoUnmask", "VideoUnmask", "VideoUnmaskSwap", "VideoUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"]
    lengths = [8, 9, 10, 8, 9, 2]
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        (root / "meta" / name).write_text("{}\n")
    index = {
        "version": 1,
        "source_metadata_sha256": {
            name: file_fingerprint(root / "meta" / name)["sha256"]
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
        },
        "episodes": [{"episode_index": i, "length": length, "task": names[i]} for i, length in enumerate(lengths)],
    }
    (root / "meta/episode_tasks.json").write_text(json.dumps(index))
    return SimpleNamespace(
        dataset_path=root,
        trajectory_ids=np.arange(6),
        trajectory_lengths=np.array(lengths),
        task_groups={"shared instruction": [0, 1, 2, 3, 5], "singleton instruction": [4]},
    )


def test_canonical_banks_include_singleton_and_short_demos(dataset):
    state = random.getstate()
    plan = plan_memory([dataset], seed=42)
    assert random.getstate() == state
    assert plan == plan_memory([dataset], seed=42)
    assert plan["key_type"] == "canonical_task"
    banks = {bank["key"]: bank for bank in plan["banks"]}
    assert {row["episode_index"] for row in banks["VideoUnmask"]["demos"]} == {0, 1, 4}
    assert {row["episode_index"] for row in banks["VideoUnmaskSwap"]["demos"]} == {2, 3, 5}
    assert next(row for row in banks["VideoUnmaskSwap"]["demos"] if row["episode_index"] == 5)[
        "frame_indices"
    ] == [0]
    assert (
        next(row for row in banks["VideoUnmask"]["demos"] if row["episode_index"] == 4)["instruction"]
        == "singleton instruction"
    )
    filtered = plan_memory([dataset], seed=42, tasks=["VideoUnmask"])
    assert filtered["banks"] == [banks["VideoUnmask"]]
    capped = plan_memory([dataset], seed=42, max_frames_per_demo=1)
    assert capped["num_candidates"] == 6 and capped["is_smoke_sample"]
    json.dumps(plan)


def test_selection_rejects_insufficient_demos_unknown_tasks_and_key_collisions(dataset):
    with pytest.raises(ValueError, match="distinct demos"):
        plan_memory([dataset], num_trajs_per_task=4)
    with pytest.raises(ValueError, match="Unknown"):
        plan_memory([dataset], tasks=["unseen"])
    with pytest.raises(ValueError, match="Duplicate memory key"):
        plan_memory([dataset, dataset])
    with pytest.raises(ValueError, match="positive"):
        plan_memory([dataset], frame_stride=0)
    with pytest.raises(ValueError, match="positive"):
        plan_memory([dataset], max_frames_per_demo=0)


def test_libero_retains_instruction_keys_and_portable_index_is_required_for_robomme(dataset):
    path = dataset.dataset_path / "meta/episode_tasks.json"
    path.unlink()
    dataset.task_groups = {"task A": [0, 1, 4], "task B": [2, 3, 5]}
    plan = plan_memory([dataset])
    assert plan["key_type"] == "instruction"
    assert [bank["key"] for bank in plan["banks"]] == ["task A", "task B"]
    (dataset.dataset_path / "manifest.json").write_text(
        json.dumps({"converter": "scripts/convert_robomme_lerobot.py"})
    )
    with pytest.raises(ValueError, match="--task-map"):
        plan_memory([dataset])


def collate(rows):
    lengths = [2 + row["episode"] % 3 for row in rows]
    mask = torch.zeros(len(rows), max(lengths), dtype=torch.long)
    values = torch.full((len(rows), max(lengths), 2), 9999.0)
    for i, (row, length) in enumerate(zip(rows, lengths)):
        mask[i, -length:] = 1
        values[i, -length:] = row["episode"] + row["frame"] / 100
    action = torch.zeros(len(rows), 16, 32)
    action[:, :, 0] = torch.tensor([row["episode"] for row in rows])[:, None]
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    action_mask[:, :, :8] = True
    return {
        "eagle_pixel_values": torch.zeros(2 * len(rows), 3, 2, 2),
        "action": action,
        "action_mask": action_mask,
        "features": values,
        "mask": mask,
    }


def encode(dataset, bank):
    dataset.get_step_data = lambda episode, frame: {"episode": episode, "frame": frame}
    dataset.transforms = lambda value: value
    vla = SimpleNamespace(
        prepare_input=lambda inputs: (inputs, None),
        backbone=lambda inputs: {
            "backbone_features": inputs["features"],
            "backbone_attention_mask": inputs["mask"],
        },
    )

    def retriever(vla, inputs):
        return torch.tensor([[1.0, 0.0]]).repeat(len(inputs["action"]), 1)

    return encode_memory_bank(dataset, bank, collate, vla, retriever, batch_size=2)


def test_variable_length_cache_and_canonical_lookup_preserve_candidates(dataset, tmp_path):
    plan = plan_memory([dataset], seed=42, max_frames_per_demo=2)
    memory = {bank["key"]: encode(dataset, bank) for bank in plan["banks"]}
    for key, cache in memory.items():
        assert all(value.device.type == "cpu" and not value.requires_grad for value in cache.values())
        for i, (episode, frame) in enumerate(zip(cache["episode_ids"], cache["frame_ids"])):
            valid = cache["backbone_attention_mask"][i].bool()
            assert int(valid.sum()) == 2 + int(episode) % 3
            assert torch.all(cache["backbone_features"][i, valid] == episode + frame / 100)
            assert torch.all(cache["backbone_features"][i, ~valid] == 0)
        assert cache["action"].shape[1:] == (16, 32)
        assert cache["action_mask"][:, :, :8].all() and not cache["action_mask"][:, :, 8:].any()
    # The same instruction cannot cause retrieval from the other task's bank.
    (_, _, actions), (features, masks) = retrieve_from_memory(
        memory,
        1,
        ["VideoUnmask", "VideoUnmaskSwap"],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert int(actions[0, 0, 0]) in (0, 1, 4) and int(actions[1, 0, 0]) in (2, 3, 5)
    assert features.shape[:2] == masks.shape
    path = tmp_path / "memory.pt"
    save_memory(memory, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded.keys() == memory.keys()
    assert all(torch.equal(cache[field], loaded[key][field]) for key, cache in memory.items() for field in cache)
    with pytest.raises(FileExistsError):
        save_memory(memory, path)


def test_failed_validation_does_not_publish_or_leave_temporary_file(tmp_path):
    path = tmp_path / "memory.pt"
    with pytest.raises(ValueError, match="validation failed"):
        save_memory({"task": {"keys": torch.ones(1, 2, requires_grad=True)}}, path)
    assert not path.exists() and not list(tmp_path.glob("pending_memory_*"))


def test_builder_uses_recorded_training_config_and_rejects_mismatch(tmp_path):
    module = importlib.import_module("scripts.rag.2_build_memory")
    trained = {
        "data_config": "examples.RoboMME.custom_data_config:RoboMMEDataConfig",
        "pretrained_model_path": "nvidia/GR00T-N1.5-3B",
    }
    (tmp_path / "training_config.json").write_text(json.dumps(trained))
    config = module.ArgsConfig(retriever_path=str(tmp_path))
    module.resolve_training_config(config)
    assert config.data_config == trained["data_config"]
    assert config.pretrained_model_path == trained["pretrained_model_path"]
    config.data_config = "wrong-config"
    with pytest.raises(ValueError, match="disagrees"):
        module.resolve_training_config(config)


def test_dry_run_avoids_full_video_cache_models_and_writes(dataset, tmp_path, monkeypatch):
    module = importlib.import_module("scripts.rag.2_build_memory")

    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run must not load full videos or GPU models")

    transform = SimpleNamespace(eval=lambda: None, transforms=[SimpleNamespace(train=lambda: None)])
    data_config = SimpleNamespace(transform=lambda: transform, modality_config=lambda: {})
    monkeypatch.setattr(module, "load_data_config", lambda value: data_config)
    monkeypatch.setattr(module, "BoundedLeRobotVideoDataset", lambda **kwargs: dataset)
    monkeypatch.setattr(module, "CachedLeRobotSingleDataset", forbidden)
    monkeypatch.setattr(module.GR00T_N1_5, "from_pretrained", forbidden)
    monkeypatch.setattr(module.torch.cuda, "is_available", forbidden)
    output = tmp_path / "not-created"
    config = module.ArgsConfig(
        dataset_paths=[str(dataset.dataset_path)], retriever_path=str(output), dry_run=True, cache_video=True
    )
    plan = module.main(config)
    assert len(plan["banks"]) == 2 and not output.exists()
