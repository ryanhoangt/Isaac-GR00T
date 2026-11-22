"""Verify the offline wrapper uses the original sampler and can resume safely."""

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("dtaidistance")
from gr00t.rag.utils import get_aligned_frame_ids  # noqa: E402
from scripts.rag import mine_dtw_pairs as mining  # noqa: E402


@pytest.fixture
def fixture_args(tmp_path):
    root = tmp_path / "datasets" / "libero_goal"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    tasks = [{"task_index": index, "task": f"task {index}"} for index in range(2)]
    (root / "meta/tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)) + "\n")
    episodes = []
    for episode in range(4):
        task, length = episode // 2, 35 + episode
        episodes.append({"episode_index": episode, "tasks": [f"task {task}"], "length": length})
        actions = np.zeros((length, 7))
        actions[:, 0] = np.sin(np.linspace(0, 3, length))
        actions[:, 6] = np.where(np.arange(length) < length // 2, -1, 1)
        states = np.zeros((length, 8))
        states[:, :3] = np.arange(length)[:, None] * 0.001
        table = pa.table(
            {
                "action": actions.tolist(),
                "observation.state": states.tolist(),
                "frame_index": list(range(length)),
                "task_index": [task] * length,
            }
        )
        path = root / info["data_path"].format(episode_chunk=0, episode_index=episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for camera in ("image", "wrist_image"):
            path = root / info["video_path"].format(
                episode_chunk=0, episode_index=episode, video_key=f"observation.images.{camera}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
                for index in range(length):
                    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
                    pixels[..., 0 if camera == "image" else 2] = 80 + index
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)) + "\n")
    return SimpleNamespace(
        dataset_root=tmp_path / "datasets",
        suites=["libero_goal"],
        task_ids=None,
        tasks_per_suite=0,
        pairs_per_task=1,
        num_frames=5,
        seed=42,
        output_dir=tmp_path / "export",
        dry_run=False,
    )


def test_same_pairs_as_original_and_rng_preserved(fixture_args):
    datasets, plan, _ = mining.build_plan(fixture_args)
    dataset, item = datasets["libero_goal"], plan[0]
    previous = random.getstate()
    actual = mining.mine_pair(dataset, item, fixture_args.num_frames)
    assert random.getstate() == previous
    try:
        random.seed(item["seed"])
        first, second = get_aligned_frame_ids(
            dataset, item["episode_a"], item["episode_b"], fixture_args.num_frames
        )
    finally:
        random.setstate(previous)
    assert actual["frame_pairs"] == [[int(a), int(b)] for a, b in zip(first, second)]
    assert actual["episode_a"] != actual["episode_b"]
    assert {item["suite"] for item in plan} == {"libero_goal"}


def test_resume_repairs_unfinished_line_and_refuses_changed_config(fixture_args):
    mining.export(fixture_args)
    output = fixture_args.output_dir
    before = (output / "pairs.jsonl").read_bytes()
    with (output / "pairs.jsonl").open("ab") as file:
        file.write(b'{"unfinished":')
    mining.export(fixture_args)
    assert (output / "pairs.jsonl").read_bytes() == before
    assert len(mining.read_records(output / "pairs.jsonl")) == 2
    assert json.loads((output / "metadata.json").read_text())["status"] == "complete"
    fixture_args.seed += 1
    with pytest.raises(ValueError, match="Inputs/settings changed"):
        mining.export(fixture_args)


def test_viewer_navigation(fixture_args, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    mining.export(fixture_args)
    monkeypatch.setattr(sys, "argv", ["dtw_pairs_viewer.py", "--pairs-dir", str(fixture_args.output_dir)])
    app_path = Path(__file__).resolve().parents[1] / "scripts/rag/dtw_pairs_viewer.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)

    def widget(kind, label):
        return next(item for item in getattr(app, kind) if item.label == label)

    assert not app.exception and not app.error
    widget("button", "Next →").click().run()
    assert not app.exception and not app.error
    assert app.session_state["pair_index"] == 1
    widget("radio", "Camera").set_value("Wrist").run()
    assert not app.exception and not app.error
    widget("selectbox", "Task").select(1).run()
    assert not app.exception and not app.error
    assert app.session_state["pair_index"] == 0
    widget("number_input", "Action chunk length").set_value(64).run()
    assert not app.exception and not app.error
