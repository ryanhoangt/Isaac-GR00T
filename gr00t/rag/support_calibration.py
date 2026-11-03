"""Expert-only leave-one-demo-out support calibration; no coverage guarantee."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.rag.diagnosis_protocol import digest, instruction_key
from gr00t.rag.dual_mode_diagnosis import support_scores
from gr00t.rag.dual_mode_memory import MilestoneMemory


SCORE_PROTOCOL = "legacy_bf16_argmax_then_fp32_renormalized_cosine_v1"


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def task_fingerprint(entry):
    result = hashlib.sha256()
    for key in ("keys", "episode_id", "frame_index", "trajectory_length"):
        value = entry[key].detach().cpu().contiguous()
        result.update(json.dumps([key, list(value.shape), str(value.dtype)]).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def memory_fingerprint(memory):
    return {
        "manifest_sha256": sha_file(memory.directory / "manifest.json"),
        "tasks": {task: task_fingerprint(memory[task]) for task in memory.manifest["tasks"]},
    }


def weighted_quantile(values, weights, quantile):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not 0 < quantile < 1 or not len(values) or values.shape != weights.shape:
        raise ValueError("Invalid weighted quantile input")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Quantile values/weights must be finite; weights must be positive")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(order) - 1)
    return float(values[order[index]])


@torch.inference_mode()
def calibrate_task(entry, query_stride=16, device="cuda"):
    if query_stride < 1:
        raise ValueError("query_stride must be positive")
    episodes = entry["episode_id"].unique(sorted=True).tolist()
    if len(episodes) != 3 or entry["keys"].dtype != torch.bfloat16:
        raise ValueError("Calibration requires exactly three demos/task and BF16 retrieval keys")
    keys = entry["keys"].to(device)
    if not torch.isfinite(keys).all() or bool((keys.float().norm(dim=-1) == 0).any()):
        raise ValueError("Invalid retrieval keys")
    rows = []
    for episode in episodes:
        indices = torch.where(
            (entry["episode_id"] == episode)
            & (
                (entry["frame_index"] % query_stride == 0)
                | (entry["frame_index"] == entry["trajectory_length"] - 1)
            )
        )[0].tolist()
        if not indices:
            raise ValueError("No query frames for a reference demo")
        eligible = (entry["episode_id"] != episode).to(device)
        for index in indices:
            query = keys[index]
            scores = (query @ keys.T).masked_fill(~eligible, -torch.inf)
            nearest = int(scores.argmax())
            support = support_scores(query, keys[nearest], scores[nearest])
            rows.append(
                dict(
                    query_index=index,
                    query_episode=int(episode),
                    query_frame=int(entry["frame_index"][index]),
                    nearest_index=nearest,
                    retrieved_episode=int(entry["episode_id"][nearest]),
                    retrieved_frame=int(entry["frame_index"][nearest]),
                    weight=1 / (len(episodes) * len(indices)),
                    **support,
                )
            )
    return rows


def build(memory_path, query_stride=16, exit_quantile=0.8, device="cuda"):
    memory = MilestoneMemory(memory_path)
    if query_stride % memory.manifest["frame_stride"] or not 0 < exit_quantile < 0.9:
        raise ValueError("Query stride must be a multiple of cache stride; exit quantile must be below q90")
    inputs = memory_fingerprint(memory)
    settings = dict(
        score_protocol=SCORE_PROTOCOL,
        query_stride=query_stride,
        enter_quantiles=[0.9, 0.95],
        exit_quantile=exit_quantile,
        device_type=torch.device(device).type,
        weighting="equal_demo_mass_inverse_empirical_CDF",
        own_demo_excluded=True,
        query_sampling="stride_plus_terminal",
        calibration_demos=3,
        retrieval_demos=2,
        runtime_demos=3,
    )
    records, all_rows = {}, []
    for task in memory.manifest["tasks"]:
        rows = calibrate_task(memory[task], query_stride, device)
        values, weights = [r["support_fp32"] for r in rows], [r["weight"] for r in rows]
        records[task] = dict(
            episode_ids=memory.manifest["tasks"][task]["episode_ids"],
            samples=len(rows),
            samples_per_demo={
                str(e): sum(r["query_episode"] == e for r in rows)
                for e in memory.manifest["tasks"][task]["episode_ids"]
            },
            q90=weighted_quantile(values, weights, 0.9),
            q95=weighted_quantile(values, weights, 0.95),
            exit=weighted_quantile(values, weights, exit_quantile),
        )
        all_rows.extend(dict(task=task, **row) for row in rows)
    document = dict(
        schema_version=1,
        status="completed",
        settings=settings,
        inputs=inputs,
        tasks=records,
        limitation="Empirical expert-reference threshold, not a conformal coverage or recovery guarantee; LODO uses 2 demos, deployment uses 3.",
    )
    document["signature"] = digest(document)
    return document, all_rows


class Calibration:
    def __init__(self, path, memory):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text())
        doc = self.document
        if doc.get("schema_version") != 1 or doc.get("status") != "completed":
            raise ValueError("Expected completed support calibration")
        if doc.get("signature") != digest({k: v for k, v in doc.items() if k != "signature"}):
            raise ValueError("Calibration content signature differs")
        if doc["settings"]["score_protocol"] != SCORE_PROTOCOL or doc["settings"]["device_type"] != "cuda":
            raise ValueError("Runtime requires the matching CUDA calibration score protocol")
        if doc["inputs"] != memory_fingerprint(memory):
            raise ValueError("Calibration reference memory differs")
        if set(doc["tasks"]) != set(memory.manifest["tasks"]):
            raise ValueError("Calibration task inventory differs")
        self.task_map = {instruction_key(task): task for task in doc["tasks"]}
        if len(self.task_map) != len(doc["tasks"]):
            raise ValueError("Ambiguous calibration task identity")
        for task, thresholds in doc["tasks"].items():
            numbers = [thresholds[k] for k in ("exit", "q90", "q95")]
            if not all(np.isfinite(numbers)) or not 0 <= numbers[0] <= numbers[1] <= numbers[2] <= 2:
                raise ValueError(f"Invalid support thresholds: {task}")

    def thresholds(self, task, variant):
        if variant not in ("q90", "q95"):
            raise ValueError("variant must be q90 or q95")
        value = self.document["tasks"][self.task_map[instruction_key(task)]]
        return value[variant], value["exit"]
