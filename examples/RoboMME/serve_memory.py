"""Serve a trained memory checkpoint: python -m examples.RoboMME.serve_memory."""

import argparse
from numbers import Integral
from pathlib import Path

from transformers import set_seed
import uvicorn
import yaml

from examples.RoboMME.memory_policy import MemoryActionPolicy, MemoryEpisodeController
from examples.RoboMME.memory_server import create_memory_app


DEFAULT_CONFIG = Path(__file__).parent / "config/memory_inference.yaml"


def load_runtime_config(path):
    config = yaml.safe_load(Path(path).read_text())
    fields = {"host", "port", "device", "seed", "denoising_steps", "encode_batch_size", "max_episode_frames"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError(f"Inference YAML must contain exactly {sorted(fields)}")
    return config


def validate_runtime_config(config):
    for name in ("host", "device"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("port", "seed", "denoising_steps", "encode_batch_size", "max_episode_frames"):
        value = config[name]
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["port"] > 65535 or config["seed"] >= 2**32:
        raise ValueError("port/seed is out of range")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-manifest", help="Original manifest for checkpoints created before step 6")
    for name in ("host", "device"):
        parser.add_argument(f"--{name}")
    for name in ("port", "seed", "denoising-steps", "encode-batch-size", "max-episode-frames"):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    config.update({key: getattr(args, key) for key in config if getattr(args, key) is not None})
    validate_runtime_config(config)
    set_seed(config["seed"])
    policy = MemoryActionPolicy(
        args.model_path,
        device=config["device"],
        denoising_steps=config["denoising_steps"],
        cache_manifest=args.cache_manifest,
    )
    controller = MemoryEpisodeController(
        policy,
        policy,
        policy.memory_config,
        policy.cache_spec["feature_spec"],
        batch_size=config["encode_batch_size"],
        max_frames=config["max_episode_frames"],
        checkpoint=str(policy.model_path),
    )
    # One process and one model: separate servers are required for parallel episodes.
    uvicorn.run(create_memory_app(controller), host=config["host"], port=config["port"], workers=1)


if __name__ == "__main__":
    main()
