"""CPU-only readers for the offline retrieval viewer; no model inference."""

import json
from pathlib import Path

import av
import numpy as np


REPO = Path(__file__).resolve().parents[2]
IMAGE_CONFIG = REPO / "gr00t/model/backbone/eagle2_hg_model/preprocessor_config.json"
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def file_token(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns)


def discover_runs(root):
    root = resolve_path(root)
    manifests = (
        [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
    )
    return [str(path.parent) for path in manifests]


def read_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = resolve_path(manifest["run"])
    summary = json.loads((source / "rollouts/summary.json").read_text())
    episodes = []
    for item in manifest["episodes"]:
        path = directory / item["cache"]
        if not path.is_file():
            continue
        ordinal = item["ordinal"]
        episode = summary["episodes"][ordinal - 1]
        if episode["task_id"] != item["task_id"]:
            raise ValueError(f"Task mismatch between manifest and rollout summary: {path}")
        episodes.append({**episode, "ordinal": ordinal, "cache": str(path), "queries": item["queries"]})
    return {"manifest": manifest, "episodes": episodes, "source": str(source)}


def read_episode(path):
    with np.load(path, allow_pickle=False) as cache:
        result = {key: cache[key] for key in ("frame_indices", "simulator_steps", "topk_scores", "topk_indices")}
        result["metadata"] = json.loads(cache["metadata_json"].item())
    scores, indices = result["topk_scores"], result["topk_indices"]
    frames = result["frame_indices"]
    if len(frames) == 0 or scores.ndim != 2 or scores.shape != indices.shape or len(scores) != len(frames):
        raise ValueError(f"Invalid retrieval array shapes in {path}")
    if not np.isfinite(scores).all() or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Invalid retrieval scores/indices in {path}")
    if indices.min() < 0 or indices.max() >= result["metadata"]["memory_candidates"]:
        raise ValueError(f"Retrieval indices outside task memory in {path}")
    return result


def decode_frames(path, frame_indices, expected_frames):
    """Decode sequentially for exact frame-number alignment, including B-frames."""
    wanted = set(int(frame) for frame in frame_indices)
    selected = {}
    decoded = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            decoded += 1
            if index in wanted:
                selected[index] = frame.to_ndarray(format="rgb24")
    if decoded != expected_frames or set(selected) != wanted:
        raise ValueError(f"Video/cache alignment mismatch: decoded {decoded}, expected {expected_frames} frames")
    return np.stack([selected[int(index)] for index in frame_indices])


def memory_images(pixel_values, mean, std):
    """Invert the Eagle image normalization for one segment's two CHW views."""
    if hasattr(pixel_values, "detach"):
        pixel_values = pixel_values.detach().float().cpu().numpy()
    pixels = np.asarray(pixel_values, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[:2] != (2, 3):
        raise ValueError(f"Expected two camera views [2, 3, H, W], got {pixels.shape}")
    rgb = pixels * np.asarray(std, dtype=np.float32)[None, :, None, None]
    rgb += np.asarray(mean, dtype=np.float32)[None, :, None, None]
    return np.rint(np.clip(rgb, 0, 1).transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def paired_views(views, camera):
    if camera == "External":
        return views[0]
    if camera == "Wrist":
        return views[1]
    return np.concatenate(views, axis=1)


def rollout_view(frame, camera):
    if frame.shape[1] % 2:
        raise ValueError("Expected an even-width side-by-side rollout video")
    return paired_views(np.stack(np.split(frame, 2, axis=1)), camera)


def score_trace(episode):
    import pandas as pd

    scores, indices = episode["topk_scores"], episode["topk_indices"]
    trace = pd.DataFrame(
        {"frame": episode["frame_indices"], "top1_score": scores[:, 0], "top1_memory": indices[:, 0]}
    )
    if scores.shape[1] > 1:
        trace["top2_score"] = scores[:, 1]
        trace["margin"] = scores[:, 0] - scores[:, 1]
    return trace
