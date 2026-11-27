"""Source identity, sparse episode IDs, schema errors, and split isolation."""

from collections import Counter
import json
import os

import h5py
import numpy as np
import pytest

from examples.Libero_Mem import source
from examples.Libero_Mem.tasks import TASKS, get_fold


def test_task_registry_and_complementary_folds():
    assert len(TASKS) == len({task.name for task in TASKS}) == 10
    for fold, train_numbers in (("A", {1, 3, 5, 7, 9}), ("B", {2, 4, 6, 8, 10})):
        spec = get_fold(fold)
        train = [task for task in TASKS if task.name in spec["train_tasks"]]
        assert {task.number for task in train} == train_numbers
        assert set(spec["train_tasks"]).isdisjoint(spec["held_out_tasks"])
        assert set(spec["train_tasks"]) | set(spec["held_out_tasks"]) == {task.name for task in TASKS}
        assert Counter(task.category for task in train) == {"single": 1, "counting": 2, "sequencing_spatial": 2}
        assert set.union(*(set(task.memory_types) for task in train)) == {"OM", "OS", "OR", "OO"}
    assert get_fold("A")["train_tasks"] == get_fold("B")["held_out_tasks"]
    with pytest.raises(ValueError, match="A or B"):
        get_fold("C")


@pytest.fixture
def sources(tmp_path):
    full, raw = tmp_path / "full", tmp_path / "raw"
    full.mkdir()
    raw.mkdir()
    metadata = {}
    for task in TASKS:
        metadata[task.name] = {}
        with h5py.File(raw / task.filename, "w") as handle:
            data = handle.create_group("data")
            # Numeric order differs from lexical order. IDs need not be dense.
            for number in (1, 2, 3, 10, 11):
                demo = data.create_group(f"demo_{number}")
                initial = np.array([task.number, number, 0.25, 1.0], np.float64)
                demo.create_dataset("init_state", data=initial)
                demo.create_dataset("states", data=np.repeat(initial[None], 3, axis=0))
                actions = np.zeros((3, 7), np.float64)
                actions[:, -1] = [-1, 1, -1]
                demo.create_dataset("actions", data=actions)
                demo.attrs["model_file"] = "<mujoco/>"
        with h5py.File(full / task.filename, "w") as handle:
            data = handle.create_group("data")
            for number in (1, 3):
                demo_id = f"demo_{number}"
                demo = data.create_group(demo_id)
                actions = np.zeros((3, 7), np.float64)
                actions[:, -1] = [-1, 1, -1]
                demo.create_dataset("actions", data=actions)
                demo.create_dataset("dones", data=np.array([0, 0, 1], np.uint8))
                for key, width in source.FULL_VECTORS.items():
                    if key != "actions":
                        demo.create_dataset(key, data=np.zeros((3, width), np.float64))
                for key in source.RGB_KEYS:
                    # Shape-only datasets allocate no RGB payload; the audit
                    # must never read even a single image from these fixtures.
                    demo.create_dataset(key, shape=(3, 256, 256, 3), dtype="uint8")
                demo.create_dataset("obs/agentview_depth", shape=(3, 256, 256), dtype="uint8")
                metadata[task.name][demo_id] = {
                    "success": True,
                    "task_description": task.instruction,
                    "initial_state": [task.number, number, 0.25, 1.0],
                    "exo_boxes": [{}, {}, {}],
                    "ego_boxes": [{}, {}, {}],
                }
    (full / "metainfo.json").write_text(json.dumps(metadata))
    return full, raw


def change_metadata(sources, change):
    path = sources[0] / "metainfo.json"
    metadata = json.loads(path.read_text())
    change(metadata[TASKS[0].name])
    path.write_text(json.dumps(metadata))


def test_all_sources_preserve_identity_and_never_read_images(sources, monkeypatch):
    original = h5py.Dataset.__getitem__

    def guarded(dataset, key):
        if any(token in dataset.name for token in ("rgb", "depth", "seg")):
            raise AssertionError(f"Read an image payload: {dataset.name}")
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    report = source.audit_sources(*sources)
    assert report["all_tasks_audited"] is True
    assert report["totals"] == {"full_demo_count": 20, "raw_demo_count": 50, "full_frames": 60, "raw_frames": 150}
    for task in report["tasks"]:
        assert [demo["demo_id"] for demo in task["full_demos"]] == ["demo_1", "demo_3"]
        assert [demo["demo_id"] for demo in task["raw_demos"]] == [
            "demo_1",
            "demo_2",
            "demo_3",
            "demo_10",
            "demo_11",
        ]
        assert task["raw_only_demo_ids"] == ["demo_2", "demo_10", "demo_11"]
        assert all(demo["demo_id"] == demo["raw_demo_id"] for demo in task["full_demos"])
        assert task["full_demos"][0]["gripper_values"] == [-1.0, 1.0]
        assert task["full_schema_example"]["obs/ee_states"]["shape"] == [3, 6]
        full_states = {demo["initial_state_sha256"] for demo in task["full_demos"]}
        candidate_states = {
            demo["initial_state_sha256"]
            for demo in task["raw_demos"]
            if demo["demo_id"] in task["raw_only_demo_ids"]
        }
        assert full_states.isdisjoint(candidate_states)


def test_same_initial_state_is_excluded_even_with_different_raw_id(sources):
    with h5py.File(sources[1] / TASKS[0].filename, "r+") as handle:
        handle["data/demo_10/init_state"][:] = handle["data/demo_1/init_state"][:]
    report = source.audit_sources(*sources, task_ids=[1])
    assert report["tasks"][0]["raw_only_demo_ids"] == ["demo_2", "demo_11"]
    assert any("duplicate Raw initial states" in warning for warning in report["warnings"])


def test_remapped_id_requires_unique_initial_state(sources):
    change_metadata(sources, lambda demos: demos["demo_3"].update(initial_state=[1, 2, 0.25, 1.0]))
    report = source.audit_sources(*sources, task_ids=[1])
    matched = report["tasks"][0]["full_demos"][1]
    assert matched["raw_demo_id"] == "demo_2"
    assert matched["match_method"] == "unique_initial_state"
    assert any("remapped" in warning for warning in report["warnings"])
    assert report["tasks"][0]["raw_only_demo_ids"] == ["demo_3", "demo_10", "demo_11"]


def test_ambiguous_remapped_id_rejected(sources):
    change_metadata(sources, lambda demos: demos["demo_3"].update(initial_state=[1, 2, 0.25, 1.0]))
    with h5py.File(sources[1] / TASKS[0].filename, "r+") as handle:
        handle["data/demo_10/init_state"][:] = handle["data/demo_2/init_state"][:]
    with pytest.raises(ValueError, match="ambiguous Raw initial-state"):
        source.audit_sources(*sources, task_ids=[1])


@pytest.mark.parametrize(
    "defect",
    [
        "nan_action",
        "nan_state",
        "bad_state_width",
        "bad_rgb_shape",
        "bad_rgb_dtype",
        "bad_done",
        "unaligned_annotation",
        "missing_raw_init",
        "bad_raw_states",
        "missing_xml",
        "bad_demo_id",
    ],
)
def test_invalid_hdf5_fails_explicitly(sources, defect):
    full, raw = sources
    root = raw if defect in ("missing_raw_init", "bad_raw_states", "missing_xml") else full
    with h5py.File(root / TASKS[0].filename, "r+") as handle:
        demo = handle["data/demo_1"]
        if defect == "nan_action":
            demo["actions"][0, 0] = np.nan
        elif defect == "nan_state":
            demo["obs/ee_states"][0, 0] = np.inf
        elif defect == "bad_state_width":
            del demo["obs/ee_states"]
            demo.create_dataset("obs/ee_states", shape=(3, 7), dtype="float64")
        elif defect in ("bad_rgb_shape", "bad_rgb_dtype"):
            del demo["obs/agentview_rgb"]
            demo.create_dataset(
                "obs/agentview_rgb",
                shape=(2 if defect == "bad_rgb_shape" else 3, 256, 256, 3),
                dtype="uint16" if defect == "bad_rgb_dtype" else "uint8",
            )
        elif defect == "bad_done":
            demo["dones"][:] = [1, 0, 1]
        elif defect == "unaligned_annotation":
            demo.create_dataset("annotation", shape=(2,), dtype="float64")
        elif defect == "missing_raw_init":
            del demo["init_state"]
        elif defect == "bad_raw_states":
            del demo["states"]
            demo.create_dataset("states", shape=(2, 4), dtype="float64")
        elif defect == "missing_xml":
            del demo.attrs["model_file"]
        elif defect == "bad_demo_id":
            handle["data"].move("demo_1", "demo_01")
    with pytest.raises(ValueError):
        source.audit_sources(*sources, task_ids=[1])


@pytest.mark.parametrize(
    "defect", ["instruction", "failed", "missing_demo", "boxes", "initial_state", "nonfinite"]
)
def test_invalid_metadata_fails_explicitly(sources, defect):
    def change(demos):
        if defect == "instruction":
            demos["demo_1"]["task_description"] = "wrong task"
        elif defect == "failed":
            demos["demo_1"]["success"] = False
        elif defect == "missing_demo":
            del demos["demo_3"]
        elif defect == "boxes":
            demos["demo_1"]["exo_boxes"] = []
        elif defect == "initial_state":
            demos["demo_1"]["initial_state"] = [99, 99, 99, 99]
        elif defect == "nonfinite":
            demos["demo_1"]["initial_state"][0] = float("nan")

    change_metadata(sources, change)
    with pytest.raises(ValueError):
        source.audit_sources(*sources, task_ids=[1])


def test_duplicate_json_keys_rejected(sources):
    (sources[0] / "metainfo.json").write_text('{"same": {}, "same": {}}')
    with pytest.raises(ValueError, match="Duplicate metadata key"):
        source.audit_sources(*sources, task_ids=[1])


def test_missing_or_unknown_task_files(sources):
    (sources[0] / TASKS[-1].filename).unlink()
    with pytest.raises(FileNotFoundError):
        source.audit_sources(*sources)
    # A deliberately scoped audit can still check the available tasks.
    assert source.audit_sources(*sources, task_ids=[1])["all_tasks_audited"] is False
    (sources[0] / "unknown_demo.hdf5").touch()
    with pytest.raises(ValueError, match="Unknown task files"):
        source.audit_sources(*sources, task_ids=[1])


def test_changing_source_during_audit_is_rejected(sources, monkeypatch):
    original = source._audit_full

    def change_source(*args, **kwargs):
        result = original(*args, **kwargs)
        path = sources[1] / TASKS[0].filename
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(source, "_audit_full", change_source)
    with pytest.raises(ValueError, match="Source file changed"):
        source.audit_sources(*sources, task_ids=[1])


@pytest.mark.parametrize("ids", [[], [1, 1], [0], [11], [True], [1.0]])
def test_bad_selection(sources, ids):
    with pytest.raises(ValueError, match="distinct integers"):
        source.audit_sources(*sources, task_ids=ids)


def test_cli_inventory_and_overwrite_protection(sources, tmp_path, capsys, monkeypatch):
    output = tmp_path / "reports" / "audit.json"
    args = [
        "--full-root",
        str(sources[0]),
        "--raw-root",
        str(sources[1]),
        "--task-ids",
        "7",
        "1",
        "--output",
        str(output),
    ]
    source.main(args)
    report = json.loads(output.read_text())
    summary = json.loads(capsys.readouterr().out)
    assert report["task_ids"] == summary["task_ids"] == [1, 7]
    assert not report["all_tasks_audited"]
    assert report["metadata"]["sha256"]
    assert len(report["tasks"]) == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Should refuse existing output before scanning the sources")

    monkeypatch.setattr(source, "audit_sources", forbidden)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        source.main(args)
