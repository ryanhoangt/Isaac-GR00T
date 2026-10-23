"""Decode every LIBERO video on CPU and report damaged streams without changing data."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time


def check_video(path):
    import av

    count = 0
    last_timestamp = None
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            expected = stream.frames
            for frame in container.decode(video=0):
                count += 1
                last_timestamp = float(frame.time) if frame.time is not None else None
        if count != expected:
            raise ValueError(f"Decoded {count} of {expected} declared frames")
        return None
    except Exception as exc:
        return {"path": path, "decoded_frames": count, "last_timestamp": last_timestamp, "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("demo_data/libero_loso"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_debug/full_validation.json"))
    args = parser.parse_args()
    paths = sorted(str(p) for suite in ("spatial", "object", "goal", "long")
                   for p in (args.root / suite / "videos").rglob("*.mp4"))
    assert paths, "No videos found"
    failures = []
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed, result in enumerate(pool.map(check_video, paths, chunksize=4), start=1):
            if result:
                failures.append(result)
                print("BAD VIDEO", json.dumps(result), flush=True)
            if result or completed % 100 == 0 or completed == len(paths):
                report = {"checked": completed, "total": len(paths), "failures": failures,
                          "elapsed_seconds": round(time.monotonic() - start, 2)}
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("PROGRESS", completed, "/", len(paths), "failures", len(failures), flush=True)


if __name__ == "__main__":
    main()
