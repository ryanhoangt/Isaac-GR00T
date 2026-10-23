"""Replay raw video reads around a failed LIBERO training batch on CPU."""

import argparse
import json
from pathlib import Path

import torch
import torchvision

import gr00t.data.dataset as dataset_module
from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", choices=["spatial", "object", "goal", "long"], required=True)
    parser.add_argument("--batch", type=int, required=True, help="Zero-based batch index to replay")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder-threads", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("demo_data/libero_loso"))
    args = parser.parse_args()
    if args.decoder_threads:
        original_reader = torchvision.io.VideoReader

        def bounded_reader(*reader_args, **reader_kwargs):
            reader = original_reader(*reader_args, **reader_kwargs)
            reader.container.streams.video[0].thread_count = args.decoder_threads
            return reader

        torchvision.io.VideoReader = bounded_reader
    datasets = []
    for suite in ("spatial", "object", "goal", "long"):
        if suite == args.held_out:
            continue
        config = LiberoDataConfig()
        datasets.append(
            LeRobotSingleDataset(
                dataset_path=args.data_root / suite,
                modality_configs=config.modality_config(),
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                video_backend="torchvision_av",
            )
        )
    mixture = LeRobotMixtureDataset([(ds, 1.0) for ds in datasets], mode="train", seed=42)
    indices = torch.randperm(len(mixture), generator=torch.Generator().manual_seed(42)).tolist()
    failures = []
    original = dataset_module.get_frames_by_timestamps

    def read_video(path, timestamps, **kwargs):
        try:
            return original(path, timestamps, **kwargs)
        except Exception as exc:
            failure = {"path": path, "timestamps": timestamps.tolist(), "error": repr(exc)}
            failures.append(failure)
            print("DECODE FAILURE", json.dumps(failure), flush=True)
            raise

    dataset_module.get_frames_by_timestamps = read_video
    for batch in range(max(0, args.batch - args.radius), args.batch + args.radius + 1):
        print("REPLAY", args.held_out, "batch", batch, flush=True)
        for index in indices[batch * 32 : (batch + 1) * 32]:
            ds, episode, step = mixture.sample_step(index)
            try:
                ds.get_step_data(episode, step)
            except Exception as exc:
                print("SAMPLE FAILURE", ds.dataset_name, int(episode), int(step), repr(exc), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failures, indent=2) + "\n")
    print("FAILURES", len(failures), flush=True)


if __name__ == "__main__":
    main()
