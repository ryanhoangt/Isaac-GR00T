"""Read-only data and HTML export for the LIBERO VFD viewer; no model imports."""

import base64
import csv
import io
import json
from pathlib import Path

import numpy as np

from scripts.libero_vfd import get_threshold

ASSETS = Path(__file__).with_name("vfd_viewer_assets")


def episode_key(metadata):
    return metadata["suite"], int(metadata["task_id"]), int(metadata["initial_state_index"])


def member_label(path):
    path = Path(path)
    try:
        epoch = json.loads((path / "trainer_state.json").read_text())["epoch"]
        seed = json.loads((path.parent / "run_manifest.json").read_text())["seed"]
        return f"epoch {epoch:g}, seed {seed}"
    except (OSError, KeyError, ValueError):
        return path.name


def discover(root):
    """Index complete policy episodes and matching calibration signatures."""
    runs, calibrations, errors = [], [], []
    for path in sorted(Path(root).glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
            threshold = path.parent / "threshold.json"
            if threshold.is_file():
                calibration = json.loads(threshold.read_text())
                calibrations.append({"path": str(threshold), "data": calibration})
            if manifest.get("data_kind") == "expert_demonstration" or manifest.get("smoke_only"):
                continue
            signature = manifest.get("score_signature")
            if not signature:
                continue
            episodes = {}
            for meta_path in sorted(path.parent.glob("task*/episode*.json")):
                meta = json.loads(meta_path.read_text())
                if not meta.get("complete") or meta.get("episode_kind") == "expert_demonstration":
                    continue
                if not meta_path.with_suffix(".npz").is_file():
                    continue
                key = episode_key(meta)
                if key in episodes:
                    raise ValueError(f"Duplicate episode identity: {key}")
                episodes[key] = {"path": str(meta_path), "metadata": meta}
            if episodes:
                actor, scorer = signature["actor"], signature["scorer"]
                runs.append(
                    {
                        "path": str(path.parent),
                        "manifest": manifest,
                        "episodes": episodes,
                        "label": f"Actor {member_label(actor)} / scorer {member_label(scorer)}",
                    }
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    return runs, calibrations, errors


def matching_calibrations(run, calibrations, kind):
    suite = run["manifest"]["settings"]["suite"]
    return [
        item
        for item in calibrations
        if item["data"].get("score_signature") == run["manifest"]["score_signature"]
        and ((suite in item["data"].get("calibration_suites", [])) == (kind == "target"))
    ]


def read_scores(path):
    with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as arrays:
        scores = arrays["uncertainty"].astype(float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid uncertainty: {path}")
    return scores


def alarm_steps(scores, threshold):
    return np.flatnonzero(np.asarray(scores) > threshold).tolist() if threshold is not None else []


def video_times(path, expected_frames):
    """Map score index to actual MP4 presentation timestamps, not inference time."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        rate = float(stream.average_rate) if stream.average_rate else None
        raw_times = [frame.time for frame in container.decode(video=0)]
    if any(value is None for value in raw_times):
        raise ValueError(f"Video has frames without presentation timestamps: {path}")
    times = [float(value) for value in raw_times]
    if len(times) != expected_frames or not times or not np.isfinite(times).all():
        raise ValueError(f"Video has {len(times)} frames; uncertainty has {expected_frames}: {path}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Video timestamps are not increasing: {path}")
    return times, rate


def encode_image(image):
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def load_panel(run, key, calibrations):
    entry = run["episodes"][key]
    path, metadata = Path(entry["path"]), entry["metadata"]
    scores = read_scores(path)
    thresholds, sources, alarms = {}, {}, {}
    for kind in ("source", "target"):
        item = calibrations.get(kind)
        thresholds[kind] = (
            get_threshold(item["data"], run["manifest"]["score_signature"], metadata) if item else None
        )
        sources[kind] = item["path"] if item else None
        alarms[kind] = alarm_steps(scores, thresholds[kind])
    video, times, rate, warning = path.with_suffix(".mp4"), [], None, None
    video_src = None
    if video.is_file():
        try:
            times, rate = video_times(video, len(scores))
            video_src = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
        except (ValueError, OSError, RuntimeError) as exc:
            warning = f"Exact video alignment unavailable: {exc}"
    snapshots = []
    actions = None
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        if "actions" in arrays:
            actions = arrays["actions"].tolist()
        if video_src is None and "frames" in arrays and "frame_steps" in arrays:
            snapshots = [
                {"step": int(step), "src": encode_image(image)}
                for step, image in zip(arrays["frame_steps"], arrays["frames"])
            ]
    if video_src is None and warning is None:
        warning = "No MP4: showing the nearest saved snapshot, not an exact frame."
    return {
        "label": run["label"],
        "run": run["path"],
        "metadata": metadata,
        "scores": scores.tolist(),
        "thresholds": thresholds,
        "calibration_paths": sources,
        "alarm_steps": alarms,
        "peak_step": int(np.argmax(scores)),
        "actions": actions,
        "video_src": video_src,
        "frame_times": times,
        "video_fps": rate,
        "snapshots": snapshots,
        "warning": warning,
    }


def build_payload(runs, key, selections):
    panels = [load_panel(run, key, chosen) for run, chosen in zip(runs, selections)]
    same_actions = None
    if len(panels) == 2 and all(panel["actions"] is not None for panel in panels):
        a, b = (np.asarray(panel["actions"]) for panel in panels)
        same_actions = a.shape == b.shape and np.array_equal(a, b)
    return {
        "episode": {"suite": key[0], "task_id": key[1], "initial_state_index": key[2]},
        "panels": panels,
        "same_actions": same_actions,
    }


def export_csv(payload):
    stream = io.StringIO()
    fields = [
        "pair",
        "suite",
        "task_id",
        "initial_state_index",
        "step",
        "video_time_seconds",
        "uncertainty",
        "source_threshold",
        "target_threshold",
        "source_alarm",
        "target_alarm",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for panel in payload["panels"]:
        for step, score in enumerate(panel["scores"]):
            thresholds = panel["thresholds"]
            writer.writerow(
                {
                    "pair": panel["label"],
                    **payload["episode"],
                    "step": step,
                    "video_time_seconds": panel["frame_times"][step] if panel["frame_times"] else "",
                    "uncertainty": score,
                    "source_threshold": thresholds["source"],
                    "target_threshold": thresholds["target"],
                    "source_alarm": thresholds["source"] is not None and score > thresholds["source"],
                    "target_alarm": thresholds["target"] is not None and score > thresholds["target"],
                }
            )
    return stream.getvalue()


def standalone_html(payload):
    # Serialize as data, never interpolate dataset instructions as executable HTML.
    data = (
        json.dumps(payload, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    css = (ASSETS / "viewer.css").read_text()
    js = (ASSETS / "viewer.js").read_text().replace("export default function", "function mountViewer", 1)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>LIBERO uncertainty viewer</title><style>{css}</style></head><body><main id='viewer'></main>"
        f"<script>{js}\nmountViewer({{parentElement:document.getElementById('viewer'),data:{data}}});</script>"
        "</body></html>"
    )
