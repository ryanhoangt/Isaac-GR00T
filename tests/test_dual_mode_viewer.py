"""Check exact video alignment, pinned context, provenance and UI navigation."""

import copy
import json
from pathlib import Path
import sys

import av
import numpy as np
import pytest
import torch

from scripts.rag import dual_mode_viewer_data as data


def write_video(path, paired=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = (64, 40, "yuv420p") if paired else (32, 32, "yuv420p")
        for index in range(35):
            rgb = np.zeros((stream.height, stream.width, 3), dtype=np.uint8)
            rgb[..., 0] = index * 6
            rgb[..., 1] = 80
            if paired:
                rgb[:8] = 255
                rgb[8:, 32:, 2] = 200
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def rollout(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    info = dict(
        chunks_size=1000,
        fps=20,
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    (dataset / "meta/info.json").write_text(json.dumps(info))
    for episode in (10, 20):
        for camera in data.CAMERAS:
            write_video(
                dataset / info["video_path"].format(episode_chunk=0, episode_index=episode, video_key=camera)
            )
    memory = tmp_path / "memory"
    memory.mkdir()
    entry = dict(
        episode_id=torch.tensor([10, 10, 20, 20]),
        frame_index=torch.tensor([0, 16, 0, 16]),
        trajectory_length=torch.tensor([35] * 4),
        action=torch.zeros(4, 16, 32),
    )
    torch.save(entry, memory / "task-00.pt")
    record = dict(file="task-00.pt", episode_ids=[10, 20], frames=4)
    (memory / "manifest.json").write_text(
        json.dumps(dict(status="completed", dataset_path=str(dataset), tasks={"test manipulation": record}))
    )
    run = tmp_path / "eval" / "example"
    run.mkdir(parents=True)
    manifest = dict(
        settings=dict(memory=str(memory)),
        inputs=dict(memory=data.sha(memory / "manifest.json")),
        cases=[dict(name="adaptive_q90"), dict(name="dual_action")],
    )
    (run / "manifest.json").write_text(json.dumps(manifest))
    query = dict(
        task="test manipulation",
        mode="action",
        control_step=0,
        support_fp32=0.01,
        threshold_enter=0.02,
        threshold_exit=0.015,
        gate_event="stay_action",
        nearest_index=0,
        nearest_episode=10,
        nearest_frame=0,
        anchor_index=0,
        retrieved_episode=10,
        retrieved_frame=0,
        goal_index=1,
        goal_frame=16,
        normalized_actions=np.zeros((16, 7)).tolist(),
    )
    queries = [
        query,
        dict(query, control_step=16, mode="goal", support_fp32=0.04, gate_event="enter_goal"),
        dict(
            query,
            control_step=32,
            mode="goal",
            support_fp32=0.03,
            gate_event="stay_goal",
            nearest_index=2,
            nearest_episode=20,
        ),
    ]
    for case in ("adaptive_q90", "dual_action"):
        for initial in (0, 1):
            directory = run / case / f"task07_init{initial:03d}"
            directory.mkdir(parents=True)
            value = dict(
                complete=True,
                task_id=7,
                task="test manipulation",
                initial_index=initial,
                success=initial == 0,
                control_steps=35,
                queries=copy.deepcopy(queries),
                video="rollout.mp4" if initial == 0 else None,
            )
            if case == "dual_action":
                for q in value["queries"]:
                    q.update(mode="action", gate_event="control_action", nearest_index=0, nearest_episode=10)
            (directory / "episode.json").write_text(json.dumps(value))
            np.savez(directory / "trace.npz", executed_actions=np.zeros((35, 7)), ee=np.zeros((36, 3)))
            if initial == 0:
                write_video(directory / "rollout.mp4", paired=True)
    return run


def test_logged_pinned_goal_and_task_language_mapping(rollout):
    run = data.read_index(rollout)
    assert len(run["episodes"]) == 4
    memory = data.reference_manifest(run)
    record = data.task_record(memory, "Test manipulation!")
    # Simulator task 7 maps to task-00.pt by language, not by numeric task ID.
    entry = data.load_reference(Path(memory["directory"]) / record["file"])
    value = data.read_episode(rollout / "adaptive_q90/task07_init000/episode.json")
    context = data.context_rows(value, value["queries"][2], entry, record)
    assert context["nearest"]["episode"] == 20
    assert context["anchor"]["episode"] == 10
    assert context["goal"]["episode"] == 10 and context["goal"]["frame"] == 16
    assert context["goal_active"]
    assert not data.context_rows(value, value["queries"][0], entry, record)["goal_active"]
    broken = dict(value["queries"][2], goal_frame=32)
    with pytest.raises(ValueError, match="provenance"):
        data.context_rows(value, broken, entry, record)
    p, a = data.action_comparison(value["queries"][0], entry)
    assert p.shape == a.shape == (16, 7)


def test_seek_alignment_and_banner_crop(rollout):
    path = rollout / "adaptive_q90/task07_init000/rollout.mp4"
    with av.open(str(path)) as container:
        reference = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    for index in (0, 16, 32):
        actual = data.decode_frame(path, index, 20)
        np.testing.assert_array_equal(actual, reference[index])
        views = data.rollout_views(actual)
        assert views.shape == (2, 32, 32, 3)
        np.testing.assert_array_equal(views[0], actual[-32:, :32])
    with pytest.raises(ValueError, match="frame rate"):
        data.decode_frame(path, 0, 30)
    with pytest.raises(ValueError, match="exact frame"):
        data.decode_frame(path, 100, 20)


def test_replay_provenance_and_missing_video(rollout):
    path = rollout / "adaptive_q90/task07_init001/episode.json"
    value = data.read_episode(path)
    assert data.resolve_video(path, value) is None
    write_video(path.parent / "rollout.mp4", paired=True)
    report = dict(
        validated=True,
        fps=20,
        source_sha256={name: data.sha(path.parent / name) for name in ("episode.json", "trace.npz")},
        published_video_sha256=data.sha(path.parent / "rollout.mp4"),
    )
    sidecar = path.parent / "replay_validation.json"
    sidecar.write_text(json.dumps(report))
    assert data.resolve_video(path, value)["origin"] == "Validated action replay"
    report["validated"] = False
    sidecar.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Unverified"):
        data.resolve_video(path, value)
    report["validated"] = True
    sidecar.write_text(json.dumps(report))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source differs"):
        data.resolve_video(path, value)


def test_changed_reference_manifest_is_rejected(rollout):
    run = data.read_index(rollout)
    path = Path(run["manifest"]["settings"]["memory"]) / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Reference memory differs"):
        data.reference_manifest(run)


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def test_streamlit_modes_navigation_and_video_pending(rollout, monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr(sys, "argv", ["retrieval_viewer.py", "--view", "dual", "--eval-root", str(rollout.parent)])
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "scripts/rag/retrieval_viewer.py")).run(
        timeout=45
    )
    assert not app.exception and not app.error
    widget(app, "button", "Next mode switch").click().run()
    assert not app.exception and not app.error
    assert app.session_state["dual_query"] == 1
    widget(app, "button", "Next →").click().run()
    assert not app.exception and not app.error
    assert app.session_state["dual_query"] == 2
    widget(app, "radio", "Camera").set_value("Wrist").run()
    assert not app.exception and not app.error
    paths = list(widget(app, "selectbox", "Episode").options)
    assert len(paths) == 2
    widget(app, "selectbox", "Episode").select(str(rollout / "adaptive_q90/task07_init001/episode.json")).run()
    assert app.session_state["dual_query"] == 0
    assert not app.exception and not app.error
    assert any("Video not available" in item.value for item in app.info)
    widget(app, "selectbox", "Condition").select("dual_action").run()
    assert not app.exception and not app.error
    widget(app, "radio", "Outcome").set_value("Success").run()
    assert not app.exception and not app.error
