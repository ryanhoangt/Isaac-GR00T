"""Reference retrieval and action scheduling, independent of the simulator."""

import json
from pathlib import Path

import numpy as np

from .data import STATE_KEYS, episode_path


def read_video_frame(path, index):
    import av

    # PyAV's sequential decoding is deliberately conservative for the CPU debug
    # path and AV1 videos. The index stores feature vectors, not entire videos.
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Frame {index} missing in {path}")


class VisionFeatures:
    """Pooled frozen Eagle visual tokens, without language or a goal in the query."""

    def __init__(self, policy):
        self.model = policy.model.backbone.eagle_model
        self.processor = policy.modality_transform.transforms[-1].eagle_processor.image_processor
        self.device = policy.model.device
        self.dtype = next(self.model.vision_model.parameters()).dtype

    def __call__(self, image):
        import torch
        from PIL import Image

        image = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((224, 224))
        inputs = self.processor(images=[image], return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            features = self.model.extract_feature(pixels).float().mean(dim=(0, 1))
        values = features.cpu().numpy()
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm < 1e-10:
            raise ValueError("Invalid retrieval features")
        return values / norm


def build_index(reference, split, feature_encoder, output, stride=1, encoder_id=None):
    import pandas as pd
    import av

    reference, output = Path(reference).resolve(), Path(output)
    if output.suffix != ".npz":
        raise ValueError("Reference index output must end in .npz")
    manifest = json.loads(Path(split).read_text())
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if stride < 1:
        raise ValueError("stride must be positive")
    info = json.loads((reference / "meta/info.json").read_text())
    features, episodes, frames, tasks, positions = [], [], [], [], []
    for item in manifest["reference"]:
        episode = item["episode_index"]
        data = pd.read_parquet(episode_path(reference, info, episode))
        if set(data.task_index.astype(int)) != {item["task_id"]}:
            raise ValueError("Reference task IDs do not match split")
        path = episode_path(reference, info, episode, "observation.images.image")
        decoded = 0
        with av.open(str(path)) as container:
            container.streams.video[0].thread_count = 1
            for step, frame in enumerate(container.decode(video=0)):
                decoded += 1
                if step >= len(data):
                    break
                if step % stride:
                    continue
                features.append(feature_encoder(frame.to_ndarray(format="rgb24")))
                episodes.append(episode)
                frames.append(step)
                tasks.append(item["task_id"])
                positions.append(np.asarray(data["observation.state"].iloc[step])[:3])
        if decoded < len(data):
            raise ValueError(f"Truncated reference video: {path}")
        print(
            f"Indexed reference episode {episode}: {decoded} frames ({len(features)} features total)", flush=True
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            features=np.stack(features),
            episodes=episodes,
            frames=frames,
            tasks=tasks,
            positions=np.stack(positions),
        )
    temporary.replace(output)
    signature = {
        "split_id": manifest["split_id"],
        "reference": str(reference),
        "stride": stride,
        "encoder": encoder_id,
        "feature_type": "pooled_frozen_eagle_visual_tokens",
        "frames": len(frames),
    }
    sidecar = output.with_suffix(".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(signature, indent=2) + "\n")
    temporary.replace(sidecar)
    return signature


class ReferenceBuffer:
    def __init__(self, path, split_id, feature_encoder):
        import pandas as pd

        path = Path(path)
        self.signature = json.loads(path.with_suffix(".json").read_text())
        if self.signature["split_id"] != split_id:
            raise ValueError("Reference index belongs to another train/reference split")
        self.root = Path(self.signature["reference"])
        self.info = json.loads((self.root / "meta/info.json").read_text())
        with np.load(path) as index:
            self.features = index["features"]
            self.episodes, self.frames = index["episodes"], index["frames"]
            self.tasks = index["tasks"]
        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        if np.any(norms < 1e-10) or not np.isfinite(self.features).all():
            raise ValueError("Invalid reference feature index")
        self.features = self.features / norms
        self.encode = feature_encoder
        self.data = {
            int(episode): pd.read_parquet(episode_path(self.root, self.info, episode))
            for episode in np.unique(self.episodes)
        }
        self.positions = {
            int(task): np.concatenate(
                [
                    np.stack(frame["observation.state"])[:, :3]
                    for frame in self.data.values()
                    if int(frame.task_index.iloc[0]) == task
                ]
            )
            for task in np.unique(self.tasks)
        }

    def retrieve(self, image, task, horizon=16):
        query = self.encode(image)
        candidates = np.flatnonzero(self.tasks == task)
        if not len(candidates):
            raise ValueError(f"No reference frames for task {task}")
        idx = candidates[np.argmax(self.features[candidates] @ query)]
        episode, step = int(self.episodes[idx]), int(self.frames[idx])
        data = self.data[episode]
        goal_step = min(step + horizon, len(data) - 1)
        goal = self.image(episode, goal_step)
        action_indices = np.minimum(np.arange(step, step + horizon), len(data) - 1)
        return {
            "episode": episode,
            "frame": step,
            "goal_frame": goal_step,
            "goal_image": goal,
            "goal_position": np.asarray(data["observation.state"].iloc[goal_step])[:3],
            "actions": np.stack(data["action"].iloc[action_indices]),
            "similarity": float(self.features[idx] @ query),
        }

    def image(self, episode, step):
        return read_video_frame(episode_path(self.root, self.info, episode, "observation.images.image"), step)

    def distance(self, position, task):
        return float(np.linalg.norm(self.positions[task] - np.asarray(position), axis=1).min())


def action_matrix(action):
    result = np.concatenate([np.asarray(action[f"action.{key}"]) for key in STATE_KEYS], axis=-1)
    if result.shape != (16, 7) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite (16, 7) action chunk, got {result.shape}")
    return result.astype(np.float32)


def sample_disturbance(rng, scale=1.0):
    # Enforce both constraints in the specification: components <= 7cm and
    # Euclidean target offset between 5 and 8cm.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Perturbation scale must be finite and positive")
    while True:
        value = rng.uniform(-0.07, 0.07, size=3)
        if 0.05 <= np.linalg.norm(value) <= 0.08:
            # Sample the same base vector at every level, then scale it. This
            # preserves both direction and RNG consumption for paired trials.
            return value * scale


def disturbed_command(action, offset_m, controller, steps=4):
    """Convert meters to normalized OSC delta commands; log actual displacement separately."""
    action = np.asarray(action, dtype=np.float64).copy()
    if not getattr(controller, "use_delta", True):
        raise ValueError("Perturbation requires an OSC controller using delta commands")
    low, high = np.asarray(controller.input_min), np.asarray(controller.input_max)
    output_low, output_high = np.asarray(controller.output_min), np.asarray(controller.output_max)
    gain = (output_high[:3] - output_low[:3]) / (high[:3] - low[:3])
    if np.any(gain <= 0):
        raise ValueError("Invalid controller position scaling")
    action[:3] = np.clip(action[:3] + np.asarray(offset_m) / steps / gain, low[:3], high[:3])
    return action


class RecoverySchedule:
    def __init__(self, strategy="standard", trigger=None, perturb_steps=4, recovery_steps=32):
        if strategy not in ("standard", "A", "B", "C"):
            raise ValueError(strategy)
        if strategy != "standard" and trigger is None:
            raise ValueError("Perturbation strategies require a trigger")
        self.strategy, self.trigger = strategy, trigger
        self.perturb_steps, self.recovery_steps = perturb_steps, recovery_steps

    @property
    def recovery_start(self):
        return None if self.trigger is None else self.trigger + self.perturb_steps

    def phase(self, step):
        if self.trigger is None:
            return "standard"
        if self.trigger <= step < self.recovery_start:
            return "perturb"
        duration = 16 if self.strategy == "B" else self.recovery_steps
        if self.recovery_start <= step < self.recovery_start + duration:
            return {"A": "language", "B": "copy", "C": "both"}[self.strategy]
        return "language"

    def invalidate_queue(self, step):
        if self.trigger is None:
            return False
        duration = 16 if self.strategy == "B" else self.recovery_steps
        return step in (self.recovery_start, self.recovery_start + duration)
