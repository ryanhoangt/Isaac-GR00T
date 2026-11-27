"""Shared audited initial-state selection for LIBERO-Mem expert and policy rollouts."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py

from examples.Libero_Mem.source import AUDIT_VERSION, vector_fingerprint
from examples.Libero_Mem.tasks import TASK_BY_NUMBER

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SourceConfig:
    raw_root: Path
    audit: Path
    output_dir: Path
    libero_repo: Path = REPO.parent / "libero-mem"
    task_ids: tuple[int, ...] = tuple(range(1, 11))
    demo_ids: tuple[int, ...] | None = None
    selection: str = "unseen"
    num_episodes: int = 1
    num_steps_wait: int = 20
    seed: int = 0
    renderer: str = "osmesa"
    dry_run: bool = False


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path, expected):
    stat = path.stat()
    if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"Raw source differs from audit: {path}")


def episode_plan(config):
    if config.selection not in ("seen", "unseen") or config.renderer not in ("osmesa", "egl"):
        raise ValueError("Unsupported initial-state selection or renderer")
    for name in ("num_episodes", "num_steps_wait", "seed"):
        value = getattr(config, name)
        if type(value) is not int or value < (1 if name == "num_episodes" else 0):
            raise ValueError(f"Invalid {name}")
    if (
        not config.task_ids
        or len(set(config.task_ids)) != len(config.task_ids)
        or any(type(i) is not int or i not in TASK_BY_NUMBER for i in config.task_ids)
    ):
        raise ValueError("task_ids must be distinct integers in 1..10")
    if config.demo_ids is not None and (
        not config.demo_ids
        or len(set(config.demo_ids)) != len(config.demo_ids)
        or any(type(i) is not int or i < 0 for i in config.demo_ids)
    ):
        raise ValueError("demo_ids must be distinct nonnegative integers")
    report = json.loads(Path(config.audit).read_text())
    if report.get("audit_version") != AUDIT_VERSION or report.get("status") != "complete":
        raise ValueError("Expected a completed unit-1 source audit")
    records = {entry["number"]: entry for entry in report["tasks"]}
    if len(records) != len(report["tasks"]) or set(config.task_ids) - records.keys():
        raise ValueError("Audit does not uniquely cover the selected tasks")
    raw_root = Path(config.raw_root).expanduser().resolve(strict=True)
    libero_repo = Path(config.libero_repo).expanduser().resolve(strict=True)
    jobs = []
    for number in sorted(config.task_ids):
        task, record = TASK_BY_NUMBER[number], records[number]
        if record["name"] != task.name or record["instruction"] != task.instruction:
            raise ValueError("Audited task identity differs from registry")
        raw = sorted(record["raw_demos"], key=lambda item: int(item["demo_id"].removeprefix("demo_")))
        if not raw or len({entry["demo_id"] for entry in raw}) != len(raw):
            raise ValueError(f"Empty or duplicate Raw demo inventory: {task.name}")
        full_states = {entry["initial_state_sha256"] for entry in record["full_demos"]}
        if config.selection == "unseen":
            if len(raw) < 20:
                raise ValueError("Unseen initial-state protocol requires the final twenty Raw demos")
            pool = raw[-20:]
            if any(entry["initial_state_sha256"] in full_states for entry in pool):
                raise ValueError(f"Unseen Raw tail overlaps Full initial states: {task.name}")
        else:
            pool = [entry for entry in raw if entry["initial_state_sha256"] in full_states]
        if config.demo_ids is None:
            if config.num_episodes > len(pool):
                raise ValueError(f"Not enough {config.selection} initial states: {task.name}")
            # Match the author's last-N convention for unseen, first-N for seen.
            selected = (
                pool[-config.num_episodes :] if config.selection == "unseen" else pool[: config.num_episodes]
            )
        else:
            names = {f"demo_{i}" for i in config.demo_ids}
            if names - {entry["demo_id"] for entry in pool}:
                raise ValueError(
                    f"Requested demo is outside the {config.selection} initial-state pool: {task.name}"
                )
            selected = [entry for entry in pool if entry["demo_id"] in names]
        path = raw_root / task.filename
        source_identity(path, record["raw_file"])
        bddl = libero_repo / "libero/libero/bddl_files/libero_mem" / f"{task.name}.bddl"
        bddl_hash = file_hash(bddl)
        for entry in selected:
            jobs.append(
                {
                    "task_number": number,
                    "task": task.name,
                    "instruction": task.instruction,
                    "raw_file": str(path),
                    "raw_identity": record["raw_file"],
                    "demo": entry,
                    "bddl_file": str(bddl),
                    "bddl_sha256": bddl_hash,
                }
            )
    return {
        "protocol": "libero_mem_episode_selection_v1",
        "selection": config.selection,
        "audit_sha256": file_hash(config.audit),
        "jobs": jobs,
    }


def load_initial_state(job):
    """Read only the audited initial state and model identity, never future states/actions."""
    path = Path(job["raw_file"])
    source_identity(path, job["raw_identity"])
    with h5py.File(path, "r") as handle:
        group = handle[f"data/{job['demo']['demo_id']}"]
        initial = group["init_state"][:]
        model_xml = group.attrs["model_file"]
    if initial.ndim != 1 or vector_fingerprint(initial) != job["demo"]["initial_state_sha256"]:
        raise ValueError("Raw initial state differs from audited identity")
    xml_bytes = model_xml.encode() if isinstance(model_xml, str) else model_xml
    if hashlib.sha256(xml_bytes).hexdigest() != job["demo"]["model_xml_sha256"]:
        raise ValueError("Raw model XML differs from audit")
    source_identity(path, job["raw_identity"])
    return initial


def write_summary(path, summary):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
