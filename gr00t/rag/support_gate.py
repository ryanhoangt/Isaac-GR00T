"""Episode-local, frozen-threshold switching. Compatible with simulator Python 3.8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GateConfig:
    max_goal_chunks: int = 2
    cooldown_chunks: int = 1
    enter_consecutive: int = 1

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Gate chunk counts must be positive integers")


def decide(score, high, low, state, config, nearest, milestone):
    """Choose one chunk; cooldown includes the action chunk that ends a burst."""
    if not all(math.isfinite(v) for v in (score, high, low)) or not 0 <= low <= high <= 2:
        raise ValueError("Expected finite support distance and ordered thresholds in [0, 2]")
    if not 0 <= score <= 2:
        raise ValueError("Support distance must be in [0, 2]")
    state = dict(state or {})
    allowed = {"mode", "goal_chunks", "cooldown", "above", "anchor_index", "goal_index"}
    if set(state) - allowed:
        raise ValueError("Unknown gate state fields")
    current = dict(mode="action", goal_chunks=0, cooldown=0, above=0, anchor_index=None, goal_index=None)
    current.update(state)
    if current["mode"] not in ("action", "goal"):
        raise ValueError("Invalid gate mode")
    for field in ("goal_chunks", "cooldown", "above"):
        if type(current[field]) is not int or current[field] < 0:
            raise ValueError("Invalid gate counters")
    if current["goal_chunks"] > config.max_goal_chunks or current["cooldown"] >= config.cooldown_chunks:
        raise ValueError("Gate counters exceed configured bounds")
    if current["mode"] == "goal":
        if current["goal_chunks"] < 1 or any(
            type(current[k]) is not int or current[k] < 0 for k in ("anchor_index", "goal_index")
        ):
            raise ValueError("Goal state requires a fixed anchor and a positive chunk count")
        if score <= low or current["goal_chunks"] == config.max_goal_chunks:
            event = "exit_support" if score <= low else "exit_cap"
            return dict(
                mode="action",
                goal_chunks=0,
                cooldown=config.cooldown_chunks - 1,
                above=0,
                anchor_index=None,
                goal_index=None,
            ), event
        current["goal_chunks"] += 1
        return current, "continue_goal"
    if current["goal_chunks"] or current["anchor_index"] is not None or current["goal_index"] is not None:
        raise ValueError("Action state cannot retain a goal anchor")
    if current["cooldown"]:
        current.update(cooldown=current["cooldown"] - 1, above=0)
        return current, "cooldown"
    current["above"] = current["above"] + 1 if score > high else 0
    if current["above"] >= config.enter_consecutive:
        return dict(
            mode="goal", goal_chunks=1, cooldown=0, above=0, anchor_index=int(nearest), goal_index=int(milestone)
        ), "enter_goal"
    return current, "above_pending" if score > high else "supported_action"
