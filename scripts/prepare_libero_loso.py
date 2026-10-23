"""Stage cached LIBERO datasets without modifying the shared cache."""

import argparse
import json
import shutil
from pathlib import Path


SUITES = {
    "spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "object": "libero_object_no_noops_1.0.0_lerobot",
    "goal": "libero_goal_no_noops_1.0.0_lerobot",
    "long": "libero_10_no_noops_1.0.0_lerobot",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = {}
    for suite, name in SUITES.items():
        source = (args.source_root / name).resolve(strict=True)
        destination = args.destination_root.resolve() / suite
        destination.mkdir(parents=True, exist_ok=True)
        # Keep metadata writable locally; parquet/video assets stay in the cache.
        shutil.copytree(source / "meta", destination / "meta", dirs_exist_ok=True)
        shutil.copy2(repo / "examples/Libero/modality.json", destination / "meta/modality.json")
        for name in ("data", "videos"):
            target = source / name
            if not target.is_dir():
                raise FileNotFoundError(target)
            link = destination / name
            if link.is_symlink() and link.resolve() == target:
                continue
            link.symlink_to(target, target_is_directory=True)
        info = json.loads((destination / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (destination / "meta/episodes.jsonl").read_text().splitlines()]
        if len(episodes) != info["total_episodes"] or info["total_tasks"] != 10:
            raise ValueError(f"Unexpected episode/task counts for {suite}")
        for episode in episodes:
            index = episode["episode_index"]
            values = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            if not (destination / info["data_path"].format(**values)).is_file():
                raise FileNotFoundError(f"Missing parquet for {suite}, episode {index}")
            for key, feature in info["features"].items():
                if feature["dtype"] == "video":
                    video = destination / info["video_path"].format(**values, video_key=key)
                    if not video.is_file():
                        raise FileNotFoundError(video)
        manifest[suite] = {
            "source": str(source),
            "path": str(destination),
            "tasks": info["total_tasks"],
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
        }
    (args.destination_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
