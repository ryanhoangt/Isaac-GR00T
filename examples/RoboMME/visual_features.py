"""Shared frozen GR00T vision/connector path for offline and online memory.

Uses Eagle's existing extract_feature implementation, without a language model
or action head. There is no random crop, color jitter or dynamic image tiling
on the memory branch. This follows RoboMME's cached perceptual-memory recipe.
"""

import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PREPROCESSING = {
    "version": "robomme_memory_rgb_v1",
    "size": 224,
    "resize": "bilinear",
    "antialias": True,
    "preserve_aspect_ratio": True,
    "padding": "black_centered",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def preprocess_memory_images(images, device="cpu"):
    """Deterministic resize-with-pad and [-1,1] normalization for RGB uint8."""
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected uint8 images [frames,height,width,3]")
    if images.shape[0] < 1 or min(images.shape[1:3]) < 1:
        raise ValueError("Image batches and dimensions must be nonempty")
    pixels = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(device, torch.float32)
    height, width = images.shape[1:3]
    size = PREPROCESSING["size"]
    ratio = size / max(height, width)
    resized = (max(1, int(height * ratio)), max(1, int(width * ratio)))
    pixels = F.interpolate(pixels, size=resized, mode="bilinear", align_corners=False, antialias=True)
    dh, dw = size - resized[0], size - resized[1]
    pixels = F.pad(pixels, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
    return pixels.div_(127.5).sub_(1.0)


def pool_visual_tokens(tokens, pool_type="mean"):
    """Same non-overlapping 2D pooling as RoboMME pool_tokens_to_size."""
    if tokens.ndim != 3:
        raise ValueError("Expected visual tokens [frames,patches,channels]")
    batch, patches, dim = tokens.shape
    side = math.isqrt(patches)
    if side * side != patches or side < 4 or side % 4:
        raise ValueError("Visual patch grid must be square and divisible into a 4x4 grid")
    grid = tokens.transpose(1, 2).reshape(batch, dim, side, side)
    if pool_type == "mean":
        pooled = F.avg_pool2d(grid, side // 4)
    elif pool_type == "max":
        pooled = F.max_pool2d(grid, side // 4)
    else:
        raise ValueError("pool_type must be mean or max")
    return pooled.flatten(2).transpose(1, 2).contiguous()


def vision_weights_sha256(eagle):
    """Canonical FP32 digest also works on a future live GR00T backbone.

    FP32/BF16 casts of the same original BF16 checkpoint have the same identity.
    Only vision/connector tensors contribute; language/action weights do not.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(eagle.state_dict().items()):
        if not name.startswith(("vision_model.", "mlp1.")):
            continue
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.float()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_eagle_vision(model_path, attention_implementation="flash_attention_2"):
    """Load only vision/connector tensors from a LOCAL GR00T safetensors model.

    Construct Eagle with a supplied SigLIP and a parameter-free language-model
    placeholder. Reuse the repository's connector and extract_feature code.
    Missing/unexpected visual weights fail instead of leaving random tensors.
    """
    from safetensors import safe_open
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
        Eagle2_5_VLForConditionalGeneration,
    )

    model_path = Path(model_path).expanduser().resolve(strict=True)
    model_config = json.loads((model_path / "config.json").read_text())
    if model_config.get("model_type") != "gr00t_n1_5":
        raise ValueError("Expected a local GR00T N1.5 checkpoint")
    # EagleBackbone itself builds from this bundled architecture configuration.
    eagle_path = Path(__file__).resolve().parents[2] / "gr00t/model/backbone/eagle2_hg_model/config.json"
    config = Eagle2_5_VLConfig(**json.loads(eagle_path.read_text()))
    if config.vision_config.model_type != "siglip_vision_model" or config.use_backbone_lora:
        raise ValueError("This cache supports GR00T's unmodified SigLIP vision encoder")
    config._attn_implementation = attention_implementation
    config.vision_config._attn_implementation = attention_implementation
    with torch.device("meta"):
        vision = SiglipVisionModel(config.vision_config)
        eagle = Eagle2_5_VLForConditionalGeneration(config, vision_model=vision, language_model=nn.Identity())

    prefix = "backbone.eagle_model."
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    selected = {
        key: filename
        for key, filename in weight_map.items()
        if key.startswith((prefix + "vision_model.", prefix + "mlp1."))
    }
    if not selected:
        raise ValueError("Checkpoint has no GR00T vision/connector weights")
    state = {}
    for filename in sorted(set(selected.values())):
        with safe_open(model_path / filename, framework="pt", device="cpu") as handle:
            for key in sorted(k for k, shard in selected.items() if shard == filename):
                state[key.removeprefix(prefix)] = handle.get_tensor(key)
    eagle.load_state_dict(state, strict=True, assign=True)
    # SigLIP's non-persistent position_ids are absent from state_dict, so assign
    # cannot materialize them when the module was constructed on meta.
    for module in eagle.modules():
        for name, buffer in list(module.named_buffers(recurse=False)):
            if buffer.is_meta:
                if name != "position_ids" or buffer.ndim != 2 or buffer.shape[0] != 1:
                    raise ValueError(f"Unsupported non-persistent meta buffer: {name}")
                module.register_buffer(name, torch.arange(buffer.shape[-1]).expand(buffer.shape), persistent=False)
    eagle.eval().requires_grad_(False)
    return eagle


def memory_vision_signature(eagle, memory_config, *, compute_dtype, device_type):
    """Inspect live vision weights/implementation without moving or casting them."""
    signature = {
        "vision_weights_sha256": vision_weights_sha256(eagle),
        "vision_config": eagle.config.vision_config.to_dict(),
        "select_layer": eagle.select_layer,
        "use_pixel_shuffle": eagle.use_pixel_shuffle,
        "downsample_ratio": eagle.downsample_ratio,
        "connector_layers": eagle.mlp_connector_layers,
        "extract_feature_sha256": hashlib.sha256(
            inspect.getsource(type(eagle).extract_feature).encode()
        ).hexdigest(),
        "preprocessing": PREPROCESSING,
        "preprocess_sha256": hashlib.sha256(inspect.getsource(preprocess_memory_images).encode()).hexdigest(),
        "pool_sha256": hashlib.sha256(inspect.getsource(pool_visual_tokens).encode()).hexdigest(),
        "pool_compute_dtype": "float32",
        "compute_dtype": compute_dtype,
        "attention_implementation": eagle.config.vision_config._attn_implementation,
        "device_type": device_type,
        "pool_type": memory_config["pool_type"],
        "token_per_image": 16,
        "image_dim": memory_config["memory_feature"]["img"]["input_dim"],
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
    }
    return json.loads(json.dumps(signature, default=str))


class FrozenMemoryVisionEncoder:
    def __init__(self, eagle, memory_config, *, device="cuda", compute_dtype="bfloat16"):
        if compute_dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("compute_dtype must be float32, bfloat16 or float16")
        if any(p.requires_grad for p in eagle.vision_model.parameters()) or any(
            p.requires_grad for p in eagle.mlp1.parameters()
        ):
            raise ValueError("Vision encoder and connector must be frozen before caching")
        self.device = torch.device(device)
        if self.device.type == "cpu" and compute_dtype != "float32":
            raise ValueError("Use compute_dtype=float32 on CPU")
        self.dtype = getattr(torch, compute_dtype)
        self.pool_type = memory_config["pool_type"]
        self.image_dim = memory_config["memory_feature"]["img"]["input_dim"]
        self.signature = memory_vision_signature(
            eagle, memory_config, compute_dtype=compute_dtype, device_type=self.device.type
        )
        self.eagle = eagle.to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode(self, images):
        self.eagle.eval()
        pixels = preprocess_memory_images(images, device=self.device).to(self.dtype)
        tokens = self.eagle.extract_feature(pixels)
        if tokens.shape[-1] != self.image_dim:
            raise ValueError(f"Encoder output width {tokens.shape[-1]} != config width {self.image_dim}")
        # Pool in float32 and store in configured disk dtype separately.
        pooled = pool_visual_tokens(tokens.float(), self.pool_type)
        result = pooled.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Encoder produced non-finite visual features")
        return result


def validate_memory_vision_signature(actual, expected, *, mismatch_context):
    """Require matching extraction signatures, allowing a PyTorch version change."""
    changed = {
        key
        for key in actual.keys() | expected.keys()
        if key not in actual or key not in expected or actual[key] != expected[key]
    }
    if changed == {"torch_version"} and "torch_version" in actual and "torch_version" in expected:
        warnings.warn(
            f"Memory cache was extracted with PyTorch {expected['torch_version']}; "
            f"this process uses PyTorch {actual['torch_version']}. "
            "Frozen visual features may differ numerically across versions.",
            stacklevel=3,
        )
    elif changed:
        raise ValueError(f"{mismatch_context}: {', '.join(sorted(changed))}")
