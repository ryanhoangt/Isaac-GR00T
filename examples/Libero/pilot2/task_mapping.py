"""Match simulator tasks to dataset IDs by instruction, never by list position."""

import json
import re
from pathlib import Path


def normalize_instruction(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_task_mapping(suite, reference_root, task_ids):
    by_instruction = {}
    ids = set()
    for line in (Path(reference_root) / "meta/tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = normalize_instruction(row["task"])
        task_id = int(row["task_index"])
        if key in by_instruction or task_id in ids:
            raise ValueError("Reference task metadata has ambiguous instructions or task IDs")
        by_instruction[key] = task_id
        ids.add(task_id)
    mapping = {}
    for suite_id in task_ids:
        task = suite.get_task(suite_id)
        key = normalize_instruction(task.language)
        if key not in by_instruction:
            raise ValueError(f"No reference dataset task matches simulator instruction: {task.language!r}")
        mapping[str(suite_id)] = {
            "reference_task_id": by_instruction[key],
            "language": task.language,
            "task_name": task.name,
        }
    if len({row["reference_task_id"] for row in mapping.values()}) != len(mapping):
        raise ValueError("Multiple simulator tasks map to the same reference task")
    return mapping


def validate_retrieval_provenance(manifest, row):
    """Reject legacy/wrong-task retrieval before aggregating scientific results."""
    settings = manifest["settings"]
    if settings["mode"] not in ("goal", "both") and settings["strategy"] == "standard":
        return
    if manifest.get("task_mapping_version") != 1:
        raise ValueError("Legacy goal/recovery run lacks instruction-based task mapping; rerun these modes")
    expected = manifest["task_mapping"][str(row["task_id"])]["reference_task_id"]
    if row.get("reference_task_id") != expected:
        raise ValueError("Rollout reference task ID disagrees with the instruction mapping")
    for prediction in row["predictions"]:
        if "episode" in prediction:
            actual = manifest["reference_episode_tasks"][str(prediction["episode"])]
            if actual != expected:
                raise ValueError("Retrieved reference demonstration belongs to a different task")
