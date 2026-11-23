"""Checkpoint-backed, episode-local FrameSamp memory for RoboMME inference."""

from collections.abc import Mapping
from contextlib import nullcontext
import copy
import json
from numbers import Integral
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch

from examples.RoboMME.adapter import _float32_array, _rgb_frame, to_robomme_actions
from examples.RoboMME.feature_cache import file_sha256, json_fingerprint
from examples.RoboMME.memory_client import PROTOCOL
from examples.RoboMME.memory_config import prepare_frame_sampling, validate_memory_config
from examples.RoboMME.memory_data_config import RoboMMEMemoryDataConfig
from examples.RoboMME.visual_features import memory_vision_signature, pool_visual_tokens, preprocess_memory_images
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.policy import Gr00tPolicy


class EpisodeConflict(ValueError):
    """Out-of-order request, stale episode, or inference attempted before reset."""


def checkpoint_memory_spec(model_path, cache_manifest=None):
    """New checkpoints are self-contained; old step-5 sidecars can use their cache."""
    root = Path(model_path).expanduser().resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    memory = config.get("action_head_cfg", {}).get("memory_config")
    if memory is None:
        raise ValueError("Checkpoint does not contain a memory-enabled action head")
    validate_memory_config(memory)
    record = json.loads((root / "experiment_cfg/memory_training.json").read_text())
    identity = record["identity"]
    if record.get("identity_sha256") != json_fingerprint(identity) or identity["memory_config"] != memory:
        raise ValueError("Checkpoint memory architecture/provenance is inconsistent")
    components = identity["datasets"]
    specs = record.get("cache_specs")
    if specs is not None and len(specs) != len(components):
        raise ValueError("Checkpoint cache specification count differs from training datasets")
    resolved = []
    for index, component in enumerate(components):
        if specs is None:
            path = Path(cache_manifest) if cache_manifest else Path(component["cache_path"]) / "manifest.json"
            if not path.is_file():
                raise ValueError("Old checkpoint needs its original cache manifest or --cache-manifest")
            if file_sha256(path) != component["cache_manifest_sha256"]:
                raise ValueError("Original cache manifest differs from checkpoint provenance")
            manifest = json.loads(path.read_text())
            spec = {
                "encoder": manifest["encoder"],
                "feature_spec": manifest["feature_spec"],
                "cache_manifest_sha256": component["cache_manifest_sha256"],
            }
        else:
            spec = copy.deepcopy(specs[index])
            digest = spec.pop("spec_sha256", None)
            if digest != json_fingerprint(spec):
                raise ValueError("Embedded cache specification checksum differs")
        if (
            json_fingerprint(spec["encoder"]) != component["cache_encoder_sha256"]
            or spec["cache_manifest_sha256"] != component["cache_manifest_sha256"]
        ):
            raise ValueError("Cache specification differs from checkpoint provenance")
        expected = {
            "camera": "image",
            "num_views": 1,
            "token_per_image": 16,
            "pool_type": memory["pool_type"],
            "image_dim": memory["memory_feature"]["img"]["input_dim"],
        }
        if any(spec["feature_spec"].get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint visual feature settings differ from its memory architecture")
        if spec["feature_spec"].get("storage_dtype") not in ("float16", "float32"):
            raise ValueError("Unsupported memory feature storage dtype")
        if resolved and (
            spec["encoder"] != resolved[0]["encoder"] or spec["feature_spec"] != resolved[0]["feature_spec"]
        ):
            raise ValueError(
                "A single memory server requires consistent extraction settings across training datasets"
            )
        resolved.append(spec)
    if not resolved:
        raise ValueError("Checkpoint has no training cache specification")
    return root, memory, resolved[0]


class MemoryActionPolicy(Gr00tPolicy):
    """GR00T normalization/inverse transforms plus the shared memory vision path."""

    def __init__(self, model_path, *, device="cuda", denoising_steps=4, cache_manifest=None):
        root, self.memory_config, self.cache_spec = checkpoint_memory_spec(model_path, cache_manifest)
        self._memory_device = torch.device(device)
        if self._memory_device.type != self.cache_spec["encoder"]["device_type"]:
            raise ValueError("Inference device type differs from the training visual cache")
        compute_dtype = self.cache_spec["encoder"]["compute_dtype"]
        if compute_dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Unsupported checkpoint extraction precision")
        if self._memory_device.type == "cpu" and compute_dtype != "float32":
            raise ValueError("This checkpoint's extraction precision requires CUDA")
        self._memory_dtype = getattr(torch, compute_dtype)
        config = RoboMMEMemoryDataConfig()
        super().__init__(
            str(root),
            embodiment_tag="new_embodiment",
            modality_config=config.modality_config(),
            modality_transform=config.transform(),
            denoising_steps=denoising_steps,
            device=device,
        )

    def _load_model(self, model_path):
        # Hash original checkpoint weights before any lower-precision cast, as
        # the offline extractor does. Do not recreate/reinitialize the action head.
        model = GR00T_N1_5.from_pretrained(
            model_path, torch_dtype=torch.float32, tune_visual=False, tune_llm=False
        )
        model.eval().requires_grad_(False)
        if model.action_horizon != len(self._modality_config["action"].delta_indices):
            raise ValueError("Checkpoint action horizon differs from RoboMME data config")
        expected = self.cache_spec["encoder"]
        actual = memory_vision_signature(
            model.backbone.eagle_model,
            self.memory_config,
            compute_dtype=expected["compute_dtype"],
            device_type=expected["device_type"],
        )
        if actual != expected:
            changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
            raise ValueError(f"Checkpoint vision/extraction differs from training: {', '.join(changed)}")
        self.model = model.to(device=self._memory_device, dtype=self._memory_dtype)

    @torch.inference_mode()
    def encode(self, images):
        # json_numpy may return read-only views; own the NumPy storage passed to torch.
        pixels = preprocess_memory_images(np.array(images, copy=True), device=self._memory_device).to(
            self._memory_dtype
        )
        tokens = self.model.backbone.eagle_model.extract_feature(pixels)
        return pool_visual_tokens(tokens.float(), self.memory_config["pool_type"]).cpu().numpy()

    def _get_action_from_normalized_input(self, normalized_input):
        autocast = (
            torch.autocast(self._memory_device.type, dtype=self._memory_dtype)
            if self._memory_dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            return self.model.get_action(normalized_input)["action_pred"].float()


class OnlineFrameMemory:
    """Retain pooled frame features like upstream, with a fixed episode limit.

    FrameSamp's selected set changes as time advances, so previously unselected
    frames remain available. The policy input is still limited to budget tokens.
    """

    def __init__(self, encoder, config, storage_dtype, *, batch_size=16, max_frames=4096):
        validate_memory_config(config)
        for name, value in (("batch_size", batch_size), ("max_frames", max_frames)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if storage_dtype not in ("float16", "float32"):
            raise ValueError("Unsupported feature storage dtype")
        self.encoder, self.config = encoder, copy.deepcopy(config)
        self.batch_size, self.max_frames = batch_size, max_frames
        self.features = np.empty(
            (max_frames, 16, config["memory_feature"]["img"]["input_dim"]), dtype=storage_dtype
        )
        self.length = 0

    def reset(self):
        self.length = 0

    def append(self, frames):
        end = self.length + len(frames)
        if end > self.max_frames:
            raise ValueError(f"Episode exceeds max_episode_frames={self.max_frames}")
        # Publish the new prefix only when every chunk encoded successfully.
        pending = np.empty((len(frames), *self.features.shape[1:]), dtype=self.features.dtype)
        for start in range(0, len(frames), self.batch_size):
            stop = min(start + self.batch_size, len(frames))
            encoded = np.asarray(self.encoder.encode(frames[start:stop]))
            if encoded.shape != pending[start:stop].shape or not np.isfinite(encoded).all():
                raise ValueError("Invalid online visual features")
            with np.errstate(over="ignore", invalid="ignore"):
                pending[start:stop] = encoded
            if not np.isfinite(pending[start:stop]).all():
                raise ValueError("Visual features overflow training storage dtype")
        self.features[self.length : end] = pending
        self.length = end

    def prepare(self):
        if not self.length:
            raise ValueError("No observed frames in memory")
        return prepare_frame_sampling(self.features[: self.length], self.length - 1, self.config)


def validate_observation(observation, frames):
    keys = {
        "video.image",
        "video.wrist_image",
        "state.joint_position",
        "state.gripper",
        "annotation.human.action.task_description",
    }
    if not isinstance(observation, Mapping) or set(observation) != keys:
        raise ValueError("Expected current RoboMME RGB, state and task instruction fields")
    result = {}
    for key in ("video.image", "video.wrist_image"):
        value = np.asarray(observation[key])
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError("Current camera observations require a single frame")
        result[key] = _rgb_frame(value[0], key)
    for key, shape in (("state.joint_position", (1, 7)), ("state.gripper", (1, 1))):
        result[key] = _float32_array(observation[key], key)
        if result[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    goal = np.asarray(observation["annotation.human.action.task_description"])
    if goal.shape != (1,) or not isinstance(goal[0], str) or not goal[0].strip():
        raise ValueError("Expected one nonempty task instruction")
    result["annotation.human.action.task_description"] = goal.copy()
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or not len(frames)
        or min(frames.shape[1:3]) < 1
    ):
        raise ValueError("front_frames must be nonempty uint8 [frames,height,width,3]")
    if not np.array_equal(frames[-1], result["video.image"][0]):
        raise ValueError("Final new front frame must equal the current front observation")
    return result


class MemoryEpisodeController:
    """Serialize reset/infer and guard delta-frame sequencing for one episode."""

    def __init__(
        self, action_policy, encoder, config, feature_spec, *, batch_size=16, max_frames=4096, checkpoint=""
    ):
        self.action_policy = action_policy
        self.memory = OnlineFrameMemory(
            encoder, config, feature_spec["storage_dtype"], batch_size=batch_size, max_frames=max_frames
        )
        self.checkpoint = checkpoint
        self._lock = Lock()
        self._episode_id = None
        self._request_id = 0
        self._failed = False
        self._goal = None

    def reset(self):
        with self._lock:
            self.memory.reset()
            self._episode_id = uuid4().hex
            self._request_id = 0
            self._failed = False
            self._goal = None
            return {"reset_finished": True, "episode_id": self._episode_id, "protocol": PROTOCOL}

    def infer(self, payload):
        with self._lock:
            if not isinstance(payload, Mapping) or set(payload) != {
                "episode_id",
                "request_id",
                "start_frame_index",
                "observation",
                "front_frames",
                "is_first_step",
            }:
                raise ValueError("Malformed memory inference request")
            if self._episode_id is None or payload["episode_id"] != self._episode_id:
                raise EpisodeConflict("Reset is required; request episode does not match the active episode")
            if self._failed:
                raise EpisodeConflict("Previous inference failed; reset is required")
            for key, expected in (("request_id", self._request_id), ("start_frame_index", self.memory.length)):
                value = payload[key]
                if isinstance(value, bool) or not isinstance(value, Integral) or value != expected:
                    raise EpisodeConflict(f"Out-of-order {key}; expected {expected}")
            if not isinstance(payload["is_first_step"], bool) or payload["is_first_step"] != (
                self._request_id == 0
            ):
                raise EpisodeConflict("is_first_step disagrees with the episode/request")
            frames = payload["front_frames"]
            observation = validate_observation(payload["observation"], frames)
            goal = observation["annotation.human.action.task_description"][0]
            if self._goal is not None and goal != self._goal:
                raise EpisodeConflict("Task instruction changed within an episode; reset is required")
            if self.memory.length + len(frames) > self.memory.max_frames:
                raise ValueError("Episode exceeds configured max_episode_frames")
            try:
                self.memory.append(frames)
                observation.update(self.memory.prepare())
                actions = to_robomme_actions(self.action_policy.get_action(observation))
            except Exception as error:
                self._failed = True
                raise RuntimeError("Memory inference failed; reset is required") from error
            result = {
                "actions": actions,
                "episode_id": self._episode_id,
                "request_id": self._request_id,
                "frame_count": self.memory.length,
                "protocol": PROTOCOL,
            }
            self._goal = goal
            self._request_id += 1
            return result
