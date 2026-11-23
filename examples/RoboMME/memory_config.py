"""Configuration and causal sampling shared by offline training and online policy.

The source implementations are mme_vla_suite/shared/{data_utils,posemb_3d}.py.
Unlike a recent-window sampler, FrameSamp evenly samples the entire observed
prefix, including its first and current frames, then right-pads short prefixes.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml


DEFAULT_MEMORY_CONFIG = Path(__file__).parent / "config/perceptual-framesamp-modul.yaml"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_memory_config(config):
    """Validate the supported baseline and report malformed YAML as ValueError."""
    required = {
        "budget",
        "num_views",
        "token_per_image",
        "pool_type",
        "use_pos_emb",
        "use_state_emb",
        "memory_feature",
        "integration_type",
        "memory_token_dim",
        "representation_type",
        "perceptual_memory",
        "attention",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError(f"Memory config must contain exactly these fields: {sorted(required)}")
    nested = (
        ("perceptual_memory", {"type"}),
        ("memory_feature", {"img", "pos"}),
        ("attention", {"num_heads", "head_dim", "rope_base", "output_init_std"}),
    )
    for name, keys in nested:
        if not isinstance(config[name], Mapping) or set(config[name]) != keys:
            raise ValueError(f"{name} must contain exactly these fields: {sorted(keys)}")
    for name, keys in (("img", {"input_dim"}), ("pos", {"input_dim", "hidden_dim"})):
        section = config["memory_feature"][name]
        if not isinstance(section, Mapping) or set(section) != keys:
            raise ValueError(f"memory_feature.{name} must contain {sorted(keys)}")
        for key in keys:
            _positive_integer(section[key], f"memory_feature.{name}.{key}")
    if config["representation_type"] != "perceptual" or config["perceptual_memory"]["type"] != "frame_sampling":
        raise ValueError("Only perceptual frame_sampling is supported")
    if config["integration_type"] != "modulation":
        raise ValueError("Only memory-as-modulator is supported")
    _positive_integer(config["num_views"], "num_views")
    if not isinstance(config["use_pos_emb"], bool) or not isinstance(config["use_state_emb"], bool):
        raise ValueError("use_pos_emb and use_state_emb must be booleans")
    if config["num_views"] != 1 or config["use_state_emb"]:
        raise ValueError("This baseline uses front-view visual memory without historical state")
    for key in ("budget", "token_per_image", "memory_token_dim"):
        _positive_integer(config[key], key)
    if config["token_per_image"] != 16 or config["budget"] % 16:
        raise ValueError("FrameSamp uses 4x4 tokens/frame and a budget divisible by 16")
    if config["pool_type"] not in ("mean", "max"):
        raise ValueError("pool_type must be mean or max")
    if config["memory_feature"]["pos"]["input_dim"] < 12 or config["memory_feature"]["pos"]["input_dim"] % 6:
        raise ValueError("3D position width must be >=12 and divisible by six")
    attention = config["attention"]
    for name in ("num_heads", "head_dim"):
        _positive_integer(attention[name], f"attention.{name}")
    if attention["head_dim"] % 2:
        raise ValueError("Attention needs positive heads and an even head_dim")
    if attention["num_heads"] * attention["head_dim"] != config["memory_token_dim"]:
        raise ValueError("Query heads must span memory_token_dim")
    for name in ("rope_base", "output_init_std"):
        value = attention[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"attention.{name} must be finite and numeric")
    if attention["rope_base"] <= 0 or attention["output_init_std"] < 0:
        raise ValueError("Invalid RoPE base or modulation initialization")
    return config


def load_memory_config(path=DEFAULT_MEMORY_CONFIG):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    return validate_memory_config(config)


def frame_sampling_indices(step_idx, max_frames):
    _positive_integer(max_frames, "max_frames")
    if (
        isinstance(step_idx, bool)
        or not isinstance(step_idx, Integral)
        or not 0 <= step_idx <= np.iinfo(np.int32).max
    ):
        raise ValueError("step_idx must be a nonnegative int32 frame index")
    if step_idx < max_frames:
        return np.arange(step_idx + 1, dtype=np.int32)
    return np.linspace(0, step_idx, max_frames, dtype=np.int32)


def position_embedding_3d(indices, dim=768):
    """Match PosEmb3D: original frame times and 4x4 grid centers (2,6,10,14).

    This is a concatenated sin/cos position feature, distinct from the 1D RoPE
    used inside MemoryAttention.
    """
    _positive_integer(dim, "dim")
    if dim < 12 or dim % 6:
        raise ValueError("3D position width must be >=12 and divisible by six")
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and (indices.dtype.kind not in "iu" or np.any(indices < 0))):
        raise ValueError("Position indices must be a 1D sequence of nonnegative integers")
    width = dim // 6
    omega = np.arange(width, dtype=np.float32) / (width - 1)
    time = np.asarray(indices, np.float32)[:, None] / (10000.0**omega)
    y, x = np.mgrid[:4, :4]
    y = (4 * y.reshape(-1, 1) + 2) / (1000.0**omega)
    x = (4 * x.reshape(-1, 1) + 2) / (1000.0**omega)
    temporal = np.concatenate([np.sin(time), np.cos(time)], axis=-1)
    spatial = np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], axis=-1)
    return np.concatenate(
        [
            np.broadcast_to(temporal[:, None], (len(indices), 16, 2 * width)),
            np.broadcast_to(spatial[None], (len(indices), 16, 4 * width)),
        ],
        axis=-1,
    ).astype(np.float32)


def prepare_frame_sampling(features, step_idx, config):
    """Read only causal entries of an array/memmap [episode_frames,16,img_dim]."""
    validate_memory_config(config)
    capacity = config["budget"] // config["token_per_image"]
    indices = frame_sampling_indices(step_idx, capacity)
    if step_idx >= len(features):
        raise ValueError("Current source frame is absent from memory features")
    image_dim = config["memory_feature"]["img"]["input_dim"]
    pos_dim = config["memory_feature"]["pos"]["input_dim"]
    if features.ndim != 3 or features.shape[1:] != (16, image_dim):
        raise ValueError(f"Expected features of shape [episode_frames,16,{image_dim}]")
    images = np.zeros((capacity, 16, image_dim), np.float32)
    positions = np.zeros((capacity, 16, pos_dim), np.float32)
    images[: len(indices)] = features[indices]
    if not np.isfinite(images[: len(indices)]).all():
        raise ValueError("Observed visual features must be finite")
    positions[: len(indices)] = position_embedding_3d(indices, pos_dim)
    mask = np.arange(capacity) < len(indices)
    return {
        "static_image_emb": images.reshape(config["budget"], image_dim),
        "static_pos_emb": positions.reshape(config["budget"], pos_dim),
        "static_mask": np.repeat(mask, 16),
    }
