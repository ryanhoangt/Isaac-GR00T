"""Shared deterministic scheduling and command disturbances; no model imports."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def instruction_key(value):
    return " ".join(re.sub(r"[^\w\s]", "", value.lower()).split())


def query_seed(seed, task, initial, step=0):
    value = f"{seed}:{task}:{initial}:{step}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little") % (2**31)


def phase(case, step):
    if case["kind"] == "standard":
        return "goal" if case["strategy"] == "goal" else "action"
    start = case["trigger"] + 4
    if case["trigger"] <= step < start:
        return "perturb"
    duration = 16 if case["strategy"] == "B" else 32
    if start <= step < start + duration:
        return {"A": "action", "B": "copy", "C": "goal"}[case["strategy"]]
    return "action"


def flush_queue(case, step):
    if case["kind"] != "recovery":
        return False
    start = case["trigger"] + 4
    duration = 16 if case["strategy"] == "B" else 32
    return step in (start, start + duration)


def disturbance(seed, scale):
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Disturbance scale must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    while True:
        base = rng.uniform(-0.07, 0.07, 3)
        if 0.05 <= np.linalg.norm(base) <= 0.08:
            return base, base * scale


def perturb_action(action, target, controller):
    if not getattr(controller, "use_delta", True):
        raise ValueError("Cartesian perturbation requires a delta OSC controller")
    low, high = np.asarray(controller.input_min), np.asarray(controller.input_max)
    gain = (np.asarray(controller.output_max)[:3] - np.asarray(controller.output_min)[:3]) / (high[:3] - low[:3])
    if np.any(gain <= 0):
        raise ValueError("Invalid OSC command scaling")
    result = np.asarray(action, dtype=float).copy()
    baseline = np.clip(result[:3], low[:3], high[:3])
    result[:3] = np.clip(result[:3] + target / 4 / gain, low[:3], high[:3])
    delivered = (result[:3] - baseline) * gain
    return result, {
        "delivered_offset_m": delivered.tolist(),
        "clipped": not bool(np.allclose(delivered, target / 4, atol=1e-8)),
    }


def make_cases(stage, triggers=(60,), scales=(0, 1)):
    cases = []
    if stage in ("standard", "all"):
        for name, model, strategy in (
            ("baseline", "baseline", "action"),
            ("dual_action", "dual", "action"),
            ("dual_goal", "dual", "goal"),
        ):
            cases.append(dict(name=name, model=model, kind="standard", strategy=strategy))
    if stage in ("recovery", "all"):
        for trigger in triggers:
            for scale in scales:
                for strategy in "ABC":
                    cases.append(
                        dict(
                            name=f"recovery_{strategy}_t{trigger}_s{scale:g}",
                            model="dual",
                            kind="recovery",
                            strategy=strategy,
                            trigger=trigger,
                            scale=scale,
                        )
                    )
    return cases


def audit_recovery(episodes):
    groups = {}
    for episode in episodes:
        case = episode["case"]
        if case["kind"] == "recovery":
            key = (case["trigger"], case["scale"], episode["task_id"], episode["initial_index"])
            groups.setdefault(key, {})[case["strategy"]] = episode
    complete = 0
    for key, group in groups.items():
        if set(group) != set("ABC"):
            continue
        complete += 1
        for field in (
            "initial_sha256",
            "pre_perturb_sim_sha256",
            "recovery_sim_sha256",
            "prefix_actions_sha256",
            "prefix_ee_sha256",
        ):
            if len({value[field] for value in group.values()}) != 1:
                raise ValueError(f"Recovery pairing mismatch: {key}, {field}")
        if all(value["recovery_eligible"] for value in group.values()):
            b, c = group["B"], group["C"]
            for field in ("anchor_index", "goal_index", "retrieved_episode"):
                if b["recovery_context"][field] != c["recovery_context"][field]:
                    raise ValueError(f"B/C reference mismatch: {key}, {field}")
    return {"complete_ABC_groups": complete, "partial_groups": len(groups) - complete}
