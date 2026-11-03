"""Pure helpers for privileged same-scene goal-controller diagnostics."""

import hashlib
import json

import numpy as np


def array_hash(value):
    value = np.ascontiguousarray(value)
    signature = json.dumps([list(value.shape), str(value.dtype)]).encode()
    return hashlib.sha256(signature + value.tobytes()).hexdigest()


def phase_schema(observation):
    objects = sorted(
        key
        for key in observation
        if key.endswith("_pos")
        and not key.startswith("robot")
        and "_to_" not in key
        and np.asarray(observation[key]).shape == (3,)
    )
    if not objects:
        raise ValueError("Privileged phase alignment requires object positions")
    return ["robot0_eef_pos", "robot0_gripper_qpos", *objects]


def phase_vector(observation, keys):
    # Units: 5 cm for EE/objects, 2 cm for gripper fingers. This is a
    # privileged physical-state proxy, not a semantic phase oracle.
    return np.concatenate(
        [
            np.asarray(observation[key], dtype=np.float64).reshape(-1)
            / (0.02 if key == "robot0_gripper_qpos" else 0.05)
            for key in keys
        ]
    )


def choose_milestone(reference, current, previous, lookahead=16, window=32):
    reference = np.asarray(reference)
    current = np.asarray(current)
    if reference.ndim != 2 or current.shape != reference.shape[1:]:
        raise ValueError("Phase feature shapes differ")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Nonfinite phase features")
    if not 0 <= previous < len(reference) or min(lookahead, window) < 1:
        raise ValueError("Invalid reference pointer/lookahead/window")
    stop = min(previous + window + 1, len(reference))
    errors = np.square(reference[previous:stop] - current).mean(axis=1)
    anchor = previous + int(errors.argmin())
    return anchor, min(anchor + lookahead, len(reference) - 1), float(errors.min())


def case_seed(seed, suite, initial, step):
    value = f"{seed}:{suite}:{initial}:{step}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little") % (2**31)


def summarize_episodes(rows):
    baseline = [r for r in rows if r["case"] == "action_reference" and r.get("complete")]
    goal = [r for r in rows if r["case"] == "goal_oracle" and r.get("complete") and r.get("eligible")]
    return {
        "baseline_episodes": len(baseline),
        "baseline_successes": sum(r["success"] for r in baseline),
        "baseline_success_pct": 100 * sum(r["success"] for r in baseline) / len(baseline) if baseline else None,
        "eligible_goal_episodes": len(goal),
        "goal_successes": sum(r["success"] for r in goal),
        "conditional_goal_success_pct": 100 * sum(r["success"] for r in goal) / len(goal) if goal else None,
        "skipped_goal_episodes": sum(r.get("skipped", False) for r in rows),
        "interpretation": "Goal success is conditional on a successful same-scene Mode-1 reference; not benchmark SR.",
    }
