"""Verify the offline wrapper uses the original sampler and can resume safely."""

import hashlib
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


@pytest.fixture
def robomme_args(fixture_args):
    """Two families share an instruction, plus one singleton and one short demo."""
    args = fixture_args
    root = args.dataset_root / "libero_goal"
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"] = {key: {"shape": [8]} for key in ("actions", "state")}
    info_path.write_text(json.dumps(info))
    for camera in ("image", "wrist_image"):
        directory = root / "videos/chunk-000"
        (directory / f"observation.images.{camera}").rename(directory / camera)
    (root / "meta/modality.json").write_text(
        json.dumps(
            {
                "action": {
                    "joint_position": {"original_key": "actions", "start": 0, "end": 7},
                    "gripper": {"original_key": "actions", "start": 7, "end": 8},
                },
                "state": {
                    "joint_position": {"original_key": "state", "start": 0, "end": 7},
                    "gripper": {"original_key": "state", "start": 7, "end": 8},
                },
                "video": {key: {"original_key": key} for key in ("image", "wrist_image")},
            }
        )
    )
    tasks = [{"task_index": 0, "task": "shared instruction"}, {"task_index": 1, "task": "singleton"}]
    (root / "meta/tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)) + "\n")
    episodes = mining.read_jsonl(root / "meta/episodes.jsonl")
    for row in episodes:
        row["tasks"] = ["shared instruction"]
        path = root / info["data_path"].format(episode_chunk=0, episode_index=row["episode_index"])
        table = pq.read_table(path)
        actions = np.array(table["action"].to_pylist())
        states = np.zeros((len(table), 8))
        states[:, 6] = row["episode_index"] * 0.7
        states[:, 7] = row["episode_index"] * 0.01
        pq.write_table(
            pa.table(
                {
                    "actions": np.c_[actions[:, :6], actions[:, 0] + 0.5, actions[:, 6]].tolist(),
                    "state": states.tolist(),
                    "frame_index": table["frame_index"],
                    "task_index": [0] * len(table),
                }
            ),
            path,
        )
    for episode, length, instruction, task_id in ((4, 35, "singleton", 1), (5, 2, "shared instruction", 0)):
        episodes.append({"episode_index": episode, "length": length, "tasks": [instruction]})
        path = root / info["data_path"].format(episode_chunk=0, episode_index=episode)
        pq.write_table(
            pa.table(
                {
                    "actions": np.zeros((length, 8)).tolist(),
                    "state": np.zeros((length, 8)).tolist(),
                    "frame_index": list(range(length)),
                    "task_index": [task_id] * length,
                }
            ),
            path,
        )
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)) + "\n")
    owners = ["VideoUnmask", "VideoUnmask", "VideoUnmaskSwap", "VideoUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"]
    index = {
        "version": 1,
        "source_metadata_sha256": {
            name: hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest()
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
        },
        "episodes": [
            {"episode_index": row["episode_index"], "length": row["length"], "task": owners[row["episode_index"]]}
            for row in episodes
        ],
    }
    (root / "meta/episode_tasks.json").write_text(json.dumps(index))
    args.dataset_kind, args.dataset_path, args.task_map, args.tasks = "robomme", root, None, None
    return args


def test_robomme_plan_matches_online_groups_and_preserves_raw_actions(robomme_args):
    from gr00t.rag.data import get_alignment_groups

    datasets, plan, skipped = mining.build_plan(robomme_args)
    dataset = datasets["robomme"]
    groups, report = get_alignment_groups(dataset, robomme_args.num_frames)
    assert report == dataset.sampling_report
    assert report["eligible_groups"] == 2 and report["total_groups"] == 3
    assert report["eligible_demonstrations"] == 4 and report["total_demonstrations"] == 6
    assert report["short_demonstrations"] == [5] and skipped[0]["episode_indices"] == [4]
    assert len(plan) == 2
    for item in plan:
        assert (item["episode_a"], item["episode_b"]) in ((0, 1), (2, 3))
        assert item["episode_a"] in groups[item["task"], item["instruction"]]
        record = mining.mine_pair(dataset, item, robomme_args.num_frames)
        mining.validate_record(record, item, robomme_args.num_frames)
        assert set(record["source_a"]["videos"]) == {"External", "Wrist"}
        assert record["observation_schema"]["state_metric"] == "joint_rmse_rad"
        assert len(record["observation_schema"]["action_names"]) == 8
        previous = random.getstate()
        try:
            random.seed(item["seed"])
            a, b = get_aligned_frame_ids(dataset, item["episode_a"], item["episode_b"], robomme_args.num_frames)
        finally:
            random.setstate(previous)
        assert record["frame_pairs"] == [[int(x), int(y)] for x, y in zip(a, b)]


def test_robomme_filters_and_requires_valid_portable_map(robomme_args, tmp_path):
    robomme_args.tasks = ["VideoUnmaskSwap"]
    _, plan, _ = mining.build_plan(robomme_args)
    assert len(plan) == 1 and plan[0]["task"] == "VideoUnmaskSwap"
    robomme_args.task_ids = [1]
    with pytest.raises(ValueError, match="No eligible"):
        mining.build_plan(robomme_args)
    robomme_args.task_ids = [0]
    index = robomme_args.dataset_path / "meta/episode_tasks.json"
    portable = tmp_path / "task_index.json"
    index.rename(portable)
    with pytest.raises(ValueError, match="--task-map"):
        mining.build_plan(robomme_args)
    robomme_args.task_map = portable
    assert mining.build_plan(robomme_args)[1] == plan
    robomme_args.tasks = ["unknown"]
    with pytest.raises(ValueError, match="Unknown canonical"):
        mining.build_plan(robomme_args)


def test_robomme_resume_binds_task_map(robomme_args):
    mining.export(robomme_args)
    output = robomme_args.output_dir
    before = (output / "pairs.jsonl").read_bytes()
    with (output / "pairs.jsonl").open("ab") as file:
        file.write(b'{"unfinished":')
    mining.export(robomme_args)
    assert (output / "pairs.jsonl").read_bytes() == before
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["held_out_suite"] is None
    assert metadata["signature"]["sampling_reports"][0]["eligible_groups"] == 2
    index = robomme_args.dataset_path / "meta/episode_tasks.json"
    value = json.loads(index.read_text())
    index.write_text(json.dumps({**value, "provenance_note": "changed"}))
    with pytest.raises(ValueError, match="Inputs/settings changed"):
        mining.export(robomme_args)


@pytest.mark.parametrize("column", ["actions", "state"])
def test_robomme_rejects_wrong_dimensions(robomme_args, column):
    dataset = mining.ParquetDataset(robomme_args.dataset_path, "robomme")
    path = dataset.path(0)
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index(column), column, pa.array(np.zeros((len(table), 7)).tolist())
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="8D"):
        dataset.get_trajectory_data(0)


@pytest.mark.parametrize("fixture_name", ["fixture_args", "robomme_args"])
def test_viewer_navigation(fixture_name, request, monkeypatch):
    fixture_args = request.getfixturevalue(fixture_name)
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
    widget("selectbox", "Task").select("VideoUnmaskSwap" if fixture_name == "robomme_args" else 1).run()
    assert not app.exception and not app.error
    assert app.session_state["pair_index"] == 0
    widget("number_input", "Action chunk length").set_value(64).run()
    assert not app.exception and not app.error
    if fixture_name == "robomme_args":
        assert widget("selectbox", "Instruction").value == "shared instruction"
        widget("selectbox", "Action dimension").select(7).run()
        assert not app.exception and not app.error
        assert len(widget("selectbox", "Action dimension").options) == 8
        metrics = {item.label: item.value for item in app.metric}
        # Only joint 7 differs by 0.7 rad; gripper and frame progress do not enter this metric.
        assert metrics["DTW joint-state RMSE"] == f"{0.7 / np.sqrt(7):.4f} rad"
        assert not any("end-effector" in label for label in metrics)
    else:
        assert any(item.label == "DTW end-effector distance" and item.value.endswith(" cm") for item in app.metric)
