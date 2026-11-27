"""Read-only source audit, without decoding RGB or importing the simulator.

Full demonstrations are refined trajectories, so their action arrays need not
equal Raw actions. Join them using demo IDs AND exact initial-state fingerprints.
Preserve sparse demo IDs and exclude every shared initial state when listing
Raw-only candidates. This module does not choose support or evaluation episodes.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from examples.Libero_Mem.tasks import TASKS, TASK_BY_NUMBER, get_fold


AUDIT_VERSION = 1
FULL_VECTORS = {"actions": 7, "obs/ee_states": 6, "obs/gripper_states": 2, "obs/joint_states": 7}
RGB_KEYS = ("obs/agentview_rgb", "obs/eye_in_hand_rgb")


def vector_fingerprint(value):
    """Exact float64 identity, including shape, independent of JSON formatting."""
    array = np.asarray(value, dtype="<f8")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite values in source vectors")
    digest = hashlib.sha256(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _identity(path):
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _demo_ids(group):
    names = list(group)
    if not names or any(re.fullmatch(r"demo_(0|[1-9][0-9]*)", name) is None for name in names):
        raise ValueError("Expected nonempty, canonical demo_<integer> entries")
    return sorted(names, key=lambda name: int(name[5:]))


def _dataset(group, key, shape, *, dtype_kind="fiu"):
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f"Missing dataset: {group.name}/{key}")
    dataset = group[key]
    if dataset.shape != shape or dataset.dtype.kind not in dtype_kind:
        raise ValueError(f"Invalid dataset {dataset.name}: {dataset.shape}, {dataset.dtype}; expected {shape}")
    return dataset


def _finite(dataset):
    values = dataset[()]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values: {dataset.name}")
    return values


def _schema(group):
    result = {}

    def visit(name, item):
        if isinstance(item, h5py.Dataset):
            result[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

    group.visititems(visit)
    return result


def _actions(group):
    if "actions" not in group or not isinstance(group["actions"], h5py.Dataset):
        raise ValueError(f"Missing actions: {group.name}")
    shape = group["actions"].shape
    if len(shape) != 2 or shape[0] < 1:
        raise ValueError(f"Expected nonempty action trajectory: {group.name}")
    length = shape[0]
    actions = _finite(_dataset(group, "actions", (length, 7)))
    return length, actions


def _audit_raw(data):
    entries = []
    for demo_id in _demo_ids(data):
        group = data[demo_id]
        length, actions = _actions(group)
        if "init_state" not in group or not isinstance(group["init_state"], h5py.Dataset):
            raise ValueError(f"Missing init_state: {group.name}")
        shape = group["init_state"].shape
        if len(shape) != 1 or shape[0] < 1:
            raise ValueError(f"Expected 1D simulator init_state: {group.name}")
        initial = _finite(_dataset(group, "init_state", shape))
        _finite(_dataset(group, "states", (length, len(initial))))
        model_xml = group.attrs.get("model_file")
        if not isinstance(model_xml, (str, bytes)) or not model_xml.strip():
            raise ValueError(f"Missing model_file XML attribute: {group.name}")
        entries.append(
            {
                "demo_id": demo_id,
                "frames": length,
                "initial_state_dim": len(initial),
                "initial_state_sha256": vector_fingerprint(initial),
                "actions_sha256": vector_fingerprint(actions),
                "model_xml_sha256": hashlib.sha256(
                    model_xml.encode() if isinstance(model_xml, str) else model_xml
                ).hexdigest(),
            }
        )
    return entries


def _audit_full(data, metadata, task, raw_entries):
    demo_ids = _demo_ids(data)
    if not isinstance(metadata, dict) or set(demo_ids) != set(metadata):
        raise ValueError(f"Full HDF5 / metadata demo IDs differ: {task.name}")
    by_id = {entry["demo_id"]: entry for entry in raw_entries}
    by_initial = defaultdict(list)
    for entry in raw_entries:
        by_initial[entry["initial_state_sha256"]].append(entry["demo_id"])
    entries = []
    for demo_id in demo_ids:
        group = data[demo_id]
        length, actions = _actions(group)
        vectors = {}
        for key, width in FULL_VECTORS.items():
            vectors[key] = _finite(_dataset(group, key, (length, width))) if key != "actions" else actions
        for key in RGB_KEYS:
            rgb = _dataset(group, key, (length, 256, 256, 3), dtype_kind="u")
            if rgb.dtype != np.dtype("uint8"):
                raise ValueError(f"Expected uint8 RGB: {rgb.name}")
        dones = _finite(_dataset(group, "dones", (length,), dtype_kind="biu"))
        if not np.isin(dones, [0, 1]).all() or not dones[-1] or dones[:-1].any():
            raise ValueError(f"Expected only final done flag: {group.name}")
        # Check temporal alignment of all optional annotations/observations via
        # headers only. In particular, never load RGB/depth/segmentation arrays.
        for key, spec in _schema(group).items():
            if not spec["shape"] or spec["shape"][0] != length:
                raise ValueError(f"Unaligned dataset: {group.name}/{key}")
        meta = metadata[demo_id]
        if not isinstance(meta, dict) or meta.get("success") is not True:
            raise ValueError(f"Expected successful refined demo metadata: {task.name}/{demo_id}")
        if meta.get("task_description") != task.instruction:
            raise ValueError(f"Task instruction mismatch: {task.name}/{demo_id}")
        for key in ("exo_boxes", "ego_boxes"):
            if not isinstance(meta.get(key), list) or len(meta[key]) != length:
                raise ValueError(f"Unaligned {key}: {task.name}/{demo_id}")
        try:
            initial = np.asarray(meta["initial_state"], dtype="<f8")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metadata initial_state: {task.name}/{demo_id}") from exc
        if initial.ndim != 1 or initial.size < 1:
            raise ValueError(f"Invalid metadata initial_state: {task.name}/{demo_id}")
        initial_sha = vector_fingerprint(initial)
        candidates = by_initial[initial_sha]
        if demo_id in by_id and by_id[demo_id]["initial_state_sha256"] == initial_sha:
            raw_id, method = demo_id, "demo_id_and_initial_state"
        elif len(candidates) == 1:
            raw_id, method = candidates[0], "unique_initial_state"
        else:
            raise ValueError(f"Missing or ambiguous Raw initial-state match: {task.name}/{demo_id}")
        entries.append(
            {
                "demo_id": demo_id,
                "frames": length,
                "raw_demo_id": raw_id,
                "match_method": method,
                "initial_state_dim": int(initial.size),
                "initial_state_sha256": initial_sha,
                "state_sha256": vector_fingerprint(
                    np.concatenate([vectors["obs/ee_states"], vectors["obs/gripper_states"]], axis=1)
                ),
                "actions_sha256": vector_fingerprint(actions),
                "gripper_values": np.unique(actions[:, -1]).tolist(),
            }
        )
    return entries


def audit_sources(full_root, raw_root, *, task_ids=None):
    """Audit selected tasks; the default requires all ten canonical task files.

    Numeric trajectories are scanned for finiteness. Images are checked only by
    HDF5 headers. metainfo.json is parsed in RAM; use a CPU compute allocation.
    File size/mtime binds this inventory, not a checksum of the large RGB files.
    """
    full_root = Path(full_root).expanduser().resolve(strict=True)
    raw_root = Path(raw_root).expanduser().resolve(strict=True)
    if full_root == raw_root:
        raise ValueError("Full and Raw directories must differ")
    ids = list(range(1, 11)) if task_ids is None else list(task_ids)
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i not in TASK_BY_NUMBER for i in ids):
        raise ValueError("task_ids must be distinct integers in 1..10")
    tasks = [TASK_BY_NUMBER[i] for i in sorted(ids)]
    filenames = {task.filename for task in TASKS}
    for root in (full_root, raw_root):
        unknown = {path.name for path in root.glob("*.hdf5")} - filenames
        if unknown:
            raise ValueError(f"Unknown task files in {root}: {sorted(unknown)}")
        for task in tasks:
            if not (root / task.filename).is_file():
                raise FileNotFoundError(root / task.filename)
    meta_path = full_root / "metainfo.json"
    metadata_identity = _identity(meta_path)
    with meta_path.open() as handle:
        metadata = json.load(handle, object_pairs_hook=_unique_object)
    metadata_sha256 = _sha256(meta_path)
    if _identity(meta_path) != metadata_identity:
        raise ValueError("Metadata changed during audit")
    if not isinstance(metadata, dict) or set(metadata) - {task.name for task in TASKS}:
        raise ValueError("Unknown or invalid metadata task registry")
    report = {
        "audit_version": AUDIT_VERSION,
        "status": "complete",
        "task_ids": sorted(ids),
        "all_tasks_audited": len(tasks) == len(TASKS),
        "full_root": str(full_root),
        "raw_root": str(raw_root),
        "metadata": {**metadata_identity, "sha256": metadata_sha256},
        "checks": {
            "numeric_trajectories": "all_values",
            "images": "headers_only",
            "raw_full_join": "exact_initial_state",
            "hdf5_content_checksums": False,
        },
        "folds": {fold: get_fold(fold) for fold in ("A", "B")},
        "tasks": [],
        "warnings": [],
    }
    for task in tasks:
        if task.name not in metadata:
            raise ValueError(f"Missing task metadata: {task.name}")
        full_path, raw_path = full_root / task.filename, raw_root / task.filename
        full_identity, raw_identity = _identity(full_path), _identity(raw_path)
        with h5py.File(full_path, "r") as full, h5py.File(raw_path, "r") as raw:
            if "data" not in full or "data" not in raw:
                raise ValueError(f"Missing HDF5 data group: {task.name}")
            raw_entries = _audit_raw(raw["data"])
            full_entries = _audit_full(full["data"], metadata[task.name], task, raw_entries)
            used_states = {entry["initial_state_sha256"] for entry in full_entries}
            raw_only = [
                entry["demo_id"] for entry in raw_entries if entry["initial_state_sha256"] not in used_states
            ]
            record = {
                **asdict(task),
                "name": task.name,
                "full_file": full_identity,
                "raw_file": raw_identity,
                "full_schema_example": _schema(full["data"][full_entries[0]["demo_id"]]),
                "raw_schema_example": _schema(raw["data"][raw_entries[0]["demo_id"]]),
                "full_demo_count": len(full_entries),
                "raw_demo_count": len(raw_entries),
                "full_frames": sum(entry["frames"] for entry in full_entries),
                "raw_frames": sum(entry["frames"] for entry in raw_entries),
                "full_demos": full_entries,
                "raw_demos": raw_entries,
                "raw_only_demo_ids": raw_only,
            }
        if _identity(full_path) != full_identity or _identity(raw_path) != raw_identity:
            raise ValueError(f"Source file changed during audit: {task.name}")
        report["tasks"].append(record)
        if len(full_entries) != 100 or len(raw_entries) != 120:
            report["warnings"].append(
                f"T{task.number}: observed Full={len(full_entries)}, Raw={len(raw_entries)} demos"
            )
        if any(entry["match_method"] != "demo_id_and_initial_state" for entry in full_entries):
            report["warnings"].append(f"T{task.number}: some refined demos have remapped Raw demo IDs")
        for source, entries in (("Full", full_entries), ("Raw", raw_entries)):
            counts = Counter(entry["initial_state_sha256"] for entry in entries)
            if any(count > 1 for count in counts.values()):
                report["warnings"].append(
                    f"T{task.number}: duplicate {source} initial states; split by fingerprint"
                )
    if _identity(meta_path) != metadata_identity:
        raise ValueError("Metadata changed during audit")
    report["totals"] = {
        key: sum(task[key] for task in report["tasks"])
        for key in ("full_demo_count", "raw_demo_count", "full_frames", "raw_frames")
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(1, 11), default=None)
    parser.add_argument("--output", type=Path, help="Optional JSON inventory; existing files are refused")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = audit_sources(args.full_root, args.raw_root, task_ids=args.task_ids)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {key: report[key] for key in ("status", "task_ids", "all_tasks_audited", "totals", "warnings")}
    summary["tasks"] = [
        {key: task[key] for key in ("number", "full_demo_count", "raw_demo_count", "full_frames")}
        for task in report["tasks"]
    ]
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
