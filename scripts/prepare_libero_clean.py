"""Create a separate LIBERO dataset view excluding fully validated damaged demos."""

import argparse
import copy
import json
from pathlib import Path
import shutil


def filter_suite(source, destination, excluded):
    import numpy as np
    import pandas as pd

    from gr00t.data.dataset import calculate_dataset_statistics

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    unknown = excluded - {episode["episode_index"] for episode in episodes}
    if unknown:
        raise ValueError(f"Unknown excluded episode IDs: {unknown}")
    kept = [episode for episode in episodes if episode["episode_index"] not in excluded]
    if not kept:
        raise ValueError("Cannot exclude every episode")
    (destination / "meta").mkdir(parents=True)
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copy2(source / "meta" / name, destination / "meta" / name)
    stats_path = source / "meta/episodes_stats.jsonl"
    episode_stats = {
        item["episode_index"]: item
        for item in (json.loads(line) for line in stats_path.read_text().splitlines())
    } if stats_path.exists() else {}
    output_episodes, output_stats, mapping, parquets = [], [], [], []
    frame_offset = 0
    video_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    for new_index, episode in enumerate(kept):
        old_index = episode["episode_index"]
        old_values = {"episode_index": old_index, "episode_chunk": old_index // info["chunks_size"]}
        new_values = {"episode_index": new_index, "episode_chunk": new_index // info["chunks_size"]}
        data = pd.read_parquet(source / info["data_path"].format(**old_values))
        assert len(data) == episode["length"]
        old_offset = int(data["index"].iloc[0])
        data["episode_index"] = np.full(len(data), new_index, dtype=data["episode_index"].dtype)
        data["index"] = np.arange(frame_offset, frame_offset + len(data), dtype=data["index"].dtype)
        output_path = destination / info["data_path"].format(**new_values)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path, index=False)
        parquets.append(output_path)
        for key in video_keys:
            original = (source / info["video_path"].format(**old_values, video_key=key)).resolve(strict=True)
            link = destination / info["video_path"].format(**new_values, video_key=key)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(original)
        output_episodes.append({**episode, "episode_index": new_index})
        if episode_stats:
            item = copy.deepcopy(episode_stats[old_index])
            item["episode_index"] = new_index
            stat = item["stats"]["episode_index"]
            for key in ("min", "max", "mean"):
                stat[key] = [new_index]
            for key in ("min", "max", "mean"):
                item["stats"]["index"][key] = [
                    value + frame_offset - old_offset for value in item["stats"]["index"][key]
                ]
            output_stats.append(item)
        mapping.append({"original_episode_index": old_index, "episode_index": new_index})
        frame_offset += len(data)
    info.update(
        total_episodes=len(kept), total_frames=frame_offset,
        total_videos=len(kept) * len(video_keys),
        total_chunks=(len(kept) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(kept)}"},
    )
    for name, items in (("episodes.jsonl", output_episodes), ("episodes_stats.jsonl", output_stats)):
        if items:
            (destination / "meta" / name).write_text("".join(json.dumps(item) + "\n" for item in items))
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    # Recompute normalization from retained state/action samples only.
    stats = calculate_dataset_statistics(parquets)
    (destination / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (destination / "meta/episode_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return {"excluded_original_episodes": sorted(excluded), "episodes": len(kept), "frames": frame_offset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("demo_data/libero_loso"))
    parser.add_argument("--destination-root", type=Path, default=Path("demo_data/libero_loso_clean"))
    parser.add_argument("--validation-report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.validation_report.read_text())
    if report["checked"] != report["total"]:
        raise ValueError("Video validation must finish before preparing filtered data")
    excluded = {suite: set() for suite in ("spatial", "object", "goal", "long")}
    source_root = args.source_root.absolute()
    for failure in report["failures"]:
        relative = Path(failure["path"]).absolute().relative_to(source_root)
        excluded[relative.parts[0]].add(int(relative.stem.split("_")[-1]))
    args.destination_root.mkdir(parents=True, exist_ok=True)
    manifest = {"validation_report": str(args.validation_report.absolute()), "suites": {}}
    for suite, bad_episodes in excluded.items():
        source = source_root / suite
        destination = args.destination_root / suite
        if bad_episodes:
            summary = filter_suite(source, destination, bad_episodes)
        else:
            destination.symlink_to(source, target_is_directory=True)
            info = json.loads((source / "meta/info.json").read_text())
            summary = {"excluded_original_episodes": [], "episodes": info["total_episodes"],
                       "frames": info["total_frames"]}
        manifest["suites"][suite] = summary
    (args.destination_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
