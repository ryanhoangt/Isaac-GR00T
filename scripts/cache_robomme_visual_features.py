"""Cache frozen GR00T visual memory for original RoboMME LeRobot v2 episodes.

Run from this repository inside a compute allocation. --dry-run reads metadata
only; the default processes all episodes, keeping both demo and execution rows.
"""

import argparse
import json

from examples.RoboMME.feature_cache import (
    DEFAULT_CACHE_CONFIG,
    build_feature_cache,
    cache_plan,
    load_cache_config,
    validate_cache_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CACHE_CONFIG))
    for name in (
        "source",
        "model-path",
        "output",
        "memory-config",
        "device",
        "compute-dtype",
        "storage-dtype",
        "attention-implementation",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--episode-indices", type=int, nargs="+", help="Original source episode IDs; default: all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_cache_config(args.config)
    for key in config:
        override = getattr(args, key, None)
        if override is not None:
            config[key] = override
    validate_cache_config(config)
    plan = cache_plan(
        config["source"],
        config["memory_config"],
        episode_indices=args.episode_indices,
        storage_dtype=config["storage_dtype"],
    )
    if args.dry_run:
        summary = {key: value for key, value in plan.items() if key != "source_episodes"}
        summary["source_episode_indices"] = [job["source_episode_index"] for job in plan["source_episodes"]]
        summary["run_config"] = config
        print(json.dumps(summary, indent=2))
        return 0
    # Import Torch and model code only when actual feature extraction is requested.
    import torch

    from examples.RoboMME.visual_features import FrozenMemoryVisionEncoder, load_frozen_eagle_vision

    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run extraction inside a GPU allocation")
    eagle = load_frozen_eagle_vision(config["model_path"], config["attention_implementation"])
    encoder = FrozenMemoryVisionEncoder(
        eagle, plan["memory_config"], device=config["device"], compute_dtype=config["compute_dtype"]
    )
    manifest = build_feature_cache(
        plan,
        config["output"],
        encoder,
        batch_size=config["batch_size"],
        resume=args.resume,
        verify=config["verify"],
        run_config=config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "episodes": len(manifest["episodes"]),
                "frames": manifest["completed_frames"],
                "encoder_sha256": manifest["encoder_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
