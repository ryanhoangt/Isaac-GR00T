"""Catch the simulator/dataset task permutation that invalidated initial retrieval evals."""

import json
from types import SimpleNamespace

import pytest

from examples.Libero.pilot2.task_mapping import build_task_mapping, validate_retrieval_provenance


def test_reference_tasks_follow_instruction_instead_of_simulator_position(tmp_path):
    (tmp_path / "meta").mkdir()
    tasks = [
        {"task_index": 0, "task": "put the mug on the plate"},
        {"task_index": 7, "task": "put the soup in the basket"},
    ]
    (tmp_path / "meta/tasks.jsonl").write_text("\n".join(json.dumps(row) for row in tasks))
    suite = SimpleNamespace(
        get_task=lambda index: SimpleNamespace(
            name=f"scene{index}", language=["Put the soup in the basket.", "put the mug on the plate"][index]
        )
    )
    mapping = build_task_mapping(suite, tmp_path, [0, 1])
    assert mapping["0"]["reference_task_id"] == 7
    assert mapping["1"]["reference_task_id"] == 0


def test_missing_instruction_fails_instead_of_falling_back_to_task_index(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "mugs"}))
    suite = SimpleNamespace(get_task=lambda index: SimpleNamespace(name="scene", language="soup"))
    with pytest.raises(ValueError, match="No reference dataset task matches"):
        build_task_mapping(suite, tmp_path, [0])


def test_report_rejects_old_goal_runs_and_wrong_task_reference_episodes():
    manifest = {"settings": {"mode": "both", "strategy": "standard"}}
    row = {"task_id": 0, "reference_task_id": 5, "predictions": [{"episode": 15}]}
    with pytest.raises(ValueError, match="Legacy"):
        validate_retrieval_provenance(manifest, row)
    manifest.update(
        task_mapping_version=1, task_mapping={"0": {"reference_task_id": 5}}, reference_episode_tasks={"15": 5}
    )
    validate_retrieval_provenance(manifest, row)
    manifest["reference_episode_tasks"]["15"] = 0
    with pytest.raises(ValueError, match="different task"):
        validate_retrieval_provenance(manifest, row)
