"""Incremental, per-episode uncertainty recording; usable in the LIBERO env."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n"
    )
    pending.replace(path)


class EpisodeUncertaintyRecorder:
    def __init__(
        self, directory, task_id, episode_index, signature, save_observations=False
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.save_observations = save_observations
        if save_observations:
            (self.directory / "observations").mkdir()
        self.metadata = dict(
            task_id=task_id,
            episode_index=episode_index,
            complete=False,
            score_signature=signature,
            num_queries=0,
            success=None,
        )
        atomic_json(self.directory / "episode.json", self.metadata)
        self.log = (self.directory / "queries.jsonl").open("x")
        self.last_step = -1

    def record(self, step, observation, result, executed_chunk):
        if self.log.closed or step <= self.last_step:
            raise ValueError("query steps must increase in an open episode")
        if (
            not np.isfinite(result["u_ctx"])
            or result["score_id"] != self.metadata["score_signature"]["score_id"]
        ):
            raise ValueError("non-finite score or incompatible score signature")
        index = self.metadata["num_queries"]
        source = None
        if self.save_observations:
            source = f"observations/query{index:05d}.npz"
            with (self.directory / source).open("xb") as file:
                np.savez_compressed(
                    file,
                    **{key: np.asarray(value) for key, value in observation.items()},
                )
        row = {key: value for key, value in result.items() if key != "action"}
        row.update(
            control_step=step,
            query_index=index,
            observation_path=source,
            executed_action_chunk=np.asarray(executed_chunk),
        )
        self.log.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")
        self.log.flush()
        self.last_step = step
        self.metadata["num_queries"] += 1

    def finish(self, success, control_steps):
        if self.log.closed:
            raise ValueError("episode already finalized")
        horizon = self.metadata["score_signature"]["action_horizon"]
        expected = len(range(0, control_steps, horizon))
        if self.metadata["num_queries"] != expected:
            raise ValueError("recorded query count disagrees with executed horizon")
        self.log.close()
        self.metadata.update(
            complete=True, success=bool(success), control_steps=control_steps
        )
        atomic_json(self.directory / "episode.json", self.metadata)
