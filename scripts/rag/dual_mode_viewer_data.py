"""CPU-only readers for original dual-mode rollout logs and reference videos."""

from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import av
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def token(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=256)
def _sha(path, signature):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(path):
    return _sha(str(path), token(path))


def instruction(value):
    return " ".join(re.sub(r"[^\w\s]", "", value.lower()).split())


def discover_runs(root):
    root = resolve(root)
    paths = (
        [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
    )
    found = []
    for path in paths:
        try:
            value = json.loads(path.read_text())
            if value.get("cases") and value.get("settings", {}).get("memory"):
                found.append(str(path.parent))
        except (OSError, ValueError):
            continue
    return found


def read_index(run):
    run = resolve(run)
    manifest = json.loads((run / "manifest.json").read_text())
    episodes = []
    for case in manifest["cases"]:
        for path in sorted((run / case["name"]).glob("task*_init*/episode.json")):
            value = json.loads(path.read_text())
            if value.get("complete") and value.get("queries"):
                episodes.append(
                    dict(
                        path=str(path),
                        condition=case["name"],
                        task_id=value["task_id"],
                        task=value["task"],
                        initial_index=value["initial_index"],
                        success=value["success"],
                        control_steps=value["control_steps"],
                    )
                )
    return dict(run=str(run), manifest=manifest, episodes=episodes)


def read_episode(path):
    episode = json.loads(Path(path).read_text())
    steps = [query["control_step"] for query in episode["queries"]]
    if (
        not episode.get("complete")
        or not steps
        or steps[0] != 0
        or steps != sorted(set(steps))
        or steps[-1] >= episode["control_steps"]
    ):
        raise ValueError("Incomplete episode or invalid control-step alignment")
    return episode


def resolve_video(episode_path, episode):
    directory = Path(episode_path).parent
    canonical = directory / "rollout.mp4"
    sidecar = directory / "replay_validation.json"
    if canonical.is_file():
        if sidecar.is_file():
            report = json.loads(sidecar.read_text())
            verify_replay(directory, report)
            expected = report.get("published_video_sha256")
            if expected and sha(canonical) != expected:
                raise ValueError("Published replay video changed since validation")
            return dict(path=str(canonical), origin="Validated action replay", fps=report.get("fps", 20))
        return dict(
            path=str(canonical),
            origin="Original rollout video" if episode.get("video") else "Video origin not recorded",
            fps=20,
        )
    if episode.get("video") and (directory / episode["video"]).is_file():
        return dict(path=str(directory / episode["video"]), origin="Original rollout video", fps=20)
    # Read legacy validated artifacts while older backfills are being migrated.
    run = directory.parent.parent
    for root in ("video_replay_ordered", "video_replay"):
        artifact = run / root / directory.parent.name / directory.name
        if (artifact / "validation.json").is_file():
            report = json.loads((artifact / "validation.json").read_text())
            if report.get("validated") and (artifact / report["video"]).is_file():
                verify_replay(directory, report)
                return dict(
                    path=str(artifact / report["video"]),
                    origin="Validated action replay",
                    fps=report.get("fps", 20),
                )
    return None


def verify_replay(directory, report):
    if not report.get("validated"):
        raise ValueError("Unverified replay is not a valid current-observation source")
    expected = {name: sha(directory / name) for name in ("episode.json", "trace.npz")}
    if report.get("source_sha256") != expected:
        raise ValueError("Replay source differs from the original episode/trace")


def reference_manifest(run):
    manifest = run["manifest"]
    directory = resolve(manifest["settings"]["memory"])
    path = directory / "manifest.json"
    expected = manifest.get("inputs", {}).get("memory")
    if not isinstance(expected, str):
        expected = manifest.get("calibration", {}).get("inputs", {}).get("manifest_sha256")
    if expected and sha(path) != expected:
        raise ValueError("Reference memory differs from the memory bound to this evaluation")
    memory = json.loads(path.read_text())
    if memory.get("status") != "completed":
        raise ValueError("Reference memory is incomplete")
    memory["directory"] = str(directory)
    return memory


def task_record(memory, task):
    matches = [record for name, record in memory["tasks"].items() if instruction(name) == instruction(task)]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely map rollout task language to reference memory")
    return matches[0]


def load_reference(path):
    import torch

    entry = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # Retain only small provenance/action arrays, not VLM features.
    return {key: entry[key] for key in ("episode_id", "frame_index", "trajectory_length", "action")}


def context_rows(episode, query, entry, record):
    def row(index):
        if type(index) is not int or not 0 <= index < len(entry["episode_id"]):
            raise ValueError("Logged memory index is outside the task reference pool")
        result = dict(
            index=index,
            episode=int(entry["episode_id"][index]),
            frame=int(entry["frame_index"][index]),
            length=int(entry["trajectory_length"][index]),
        )
        if result["episode"] not in record["episode_ids"]:
            raise ValueError("Reference demo is outside the recorded task pool")
        return result

    if instruction(query["task"]) != instruction(episode["task"]):
        raise ValueError("Logged query task differs from rollout task")
    nearest, anchor, goal = [row(query[key]) for key in ("nearest_index", "anchor_index", "goal_index")]
    for value, episode_key, frame_key in (
        (nearest, "nearest_episode", "nearest_frame"),
        (anchor, "retrieved_episode", "retrieved_frame"),
        (goal, "retrieved_episode", "goal_frame"),
    ):
        if (value["episode"], value["frame"]) != (query[episode_key], query[frame_key]):
            raise ValueError("Logged demo/frame disagrees with reference memory provenance")
    if goal["episode"] != anchor["episode"] or goal["frame"] < anchor["frame"]:
        raise ValueError("Goal is not a future milestone in the active anchor demo")
    return dict(nearest=nearest, anchor=anchor, goal=goal, mode=query["mode"], goal_active=query["mode"] == "goal")


def reference_video(memory, row, camera):
    dataset = resolve(memory["dataset_path"])
    info = json.loads((dataset / "meta/info.json").read_text())
    episode = row["episode"]
    path = dataset / info["video_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=camera
    )
    if not 0 <= row["frame"] < row["length"]:
        raise ValueError("Reference frame is outside its demonstration")
    return str(path), info["fps"]


def decode_frame(path, index, expected_fps):
    """Seek to a keyframe, then identify the exact CFR frame from integer PTS."""
    if index < 0:
        raise ValueError("Negative video frame")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = stream.average_rate
        if fps is None or abs(float(fps) - expected_fps) > 1e-5:
            raise ValueError("Video frame rate differs from the recorded frame alignment")
        start = stream.start_time or 0
        target = start + int(Fraction(index, 1) / fps / stream.time_base)
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video lacks timestamps required for exact frame alignment")
            position = (frame.pts - start) * frame.time_base * fps
            if position.denominator != 1:
                raise ValueError("Video is not aligned to the expected constant frame grid")
            if position == index:
                return frame.to_ndarray(format="rgb24")
            if position > index:
                break
    raise ValueError(f"Cannot decode exact frame {index}: {path}")


def rollout_views(frame):
    size = frame.shape[1] // 2
    if frame.shape[1] % 2 or frame.shape[0] < size:
        raise ValueError("Expected side-by-side square cameras below an optional banner")
    return np.stack((frame[-size:, :size, :3], frame[-size:, size:, :3]))


def trace_rows(episode):
    return [
        dict(
            step=q["control_step"],
            end=episode["queries"][i + 1]["control_step"]
            if i + 1 < len(episode["queries"])
            else episode["control_steps"],
            support=q["support_fp32"],
            mode=q["mode"],
            event=q.get("gate_event", q.get("phase", q["mode"])),
            enter=q.get("threshold_enter"),
            exit=q.get("threshold_exit"),
            nearest_demo=q["nearest_episode"],
            nearest_frame=q["nearest_frame"],
            anchor_demo=q["retrieved_episode"],
            anchor_frame=q["retrieved_frame"],
            goal_frame=q["goal_frame"],
        )
        for i, q in enumerate(episode["queries"])
    ]


def action_comparison(query, entry):
    predicted = np.asarray(query["normalized_actions"], dtype=float)
    anchor = entry["action"][query["anchor_index"]]
    if hasattr(anchor, "detach"):
        anchor = anchor.detach().float().cpu().numpy()
    anchor = np.asarray(anchor)[:, :7]
    if predicted.shape != anchor.shape or predicted.shape[1] != 7:
        raise ValueError("Predicted/reference chunks have incompatible action shapes")
    return predicted, anchor
