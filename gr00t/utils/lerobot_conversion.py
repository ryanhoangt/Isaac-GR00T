"""Shared LeRobot v2.1 paths, metadata writers and vector statistics."""

import json

import numpy as np


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        key: value.tolist()
        for key, value in {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "q01": np.quantile(values, 0.01, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
        }.items()
    }
