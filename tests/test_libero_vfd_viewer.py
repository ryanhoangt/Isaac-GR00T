"""Alignment and interactive controls without checkpoints, GPU, or a simulator."""

import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
import pytest

from scripts.libero_vfd_viewer import (
    alarm_steps,
    build_payload,
    discover,
    export_csv,
    matching_calibrations,
    standalone_html,
    video_times,
)

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_video(path, length):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for i in range(length):
            image = np.full((32, 64, 3), i * 30, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def recordings(tmp_path):
    actor, scorer_a, scorer_b = [tmp_path / name / "checkpoint-1" for name in ("actor", "scorer_a", "scorer_b")]
    for path, seed, epoch in [(actor, 42, 2), (scorer_a, 42, 1), (scorer_b, 43, 2)]:
        write_json(path / "trainer_state.json", {"epoch": epoch})
        write_json(path.parent / "run_manifest.json", {"seed": seed})
    base = tmp_path / "recordings"
    for name, scorer, scores, thresholds in [
        ("snapshot", scorer_a, [1, 2, 10, 3, 1, 6], [4, 8]),
        ("seeds", scorer_b, [2, 3, 7, 3, 2, 5], [3, 6]),
    ]:
        signature = {"actor": str(actor), "scorer": str(scorer)}
        root = base / name
        write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "smoke_only": False,
                "created_utc": "1" if name == "snapshot" else "2",
                "score_signature": signature,
                "settings": {"suite": "libero_spatial"},
            },
        )
        for task, success, length in [(0, False, 6), (1, True, 4)]:
            prefix = root / f"task{task:02d}" / "episode000"
            write_json(
                prefix.with_suffix(".json"),
                {
                    "suite": "libero_spatial",
                    "task_id": task,
                    "initial_state_index": 0,
                    "success": success,
                    "complete": True,
                    "language": "pick up the bowl",
                    "action_steps": length,
                },
            )
            indices = np.array([0, length - 1])
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                uncertainty=np.array(scores[:length], float),
                actions=np.zeros((length, 7)),
                frames=np.zeros((2, 32, 64, 3), np.uint8),
                frame_steps=indices,
            )
            write_video(prefix.with_suffix(".mp4"), length)
        # Incomplete episodes never become selectable.
        write_json(root / "task00/episode001.json", {"complete": False})
        for kind, threshold in zip(("source", "target"), thresholds):
            write_json(base / f"{name}_{kind}" / "manifest.json", {"data_kind": "expert_demonstration"})
            write_json(
                base / f"{name}_{kind}" / "threshold.json",
                {
                    "score_signature": signature,
                    "scope": "global",
                    "calibration_ids": [],
                    "calibration_suites": ["libero_spatial"] if kind == "target" else ["libero_goal"],
                    "thresholds": {"global": {"threshold": threshold}},
                },
            )
    return base


def load_pair(recordings):
    runs, calibrations, errors = discover(recordings)
    assert not errors
    runs.sort(key=lambda run: run["manifest"]["created_utc"])
    choices = [
        {kind: matching_calibrations(run, calibrations, kind)[0] for kind in ("source", "target")} for run in runs
    ]
    return runs, choices


def test_discovery_binds_thresholds_to_pair_and_excludes_partial_episodes(recordings):
    runs, choices = load_pair(recordings)
    assert len(runs) == 2 and all(len(run["episodes"]) == 2 for run in runs)
    assert "epoch 1, seed 42" in runs[0]["label"]
    assert "epoch 2, seed 43" in runs[1]["label"]
    payload = build_payload(runs, ("libero_spatial", 0, 0), choices)
    assert [panel["thresholds"] for panel in payload["panels"]] == [
        {"source": 4, "target": 8},
        {"source": 3, "target": 6},
    ]
    assert payload["same_actions"] is True
    assert payload["panels"][1]["alarm_steps"]["source"] == [2, 5]  # Equality at step 1 is not an alarm.
    with pytest.raises(ValueError, match="does not match"):
        build_payload(runs, ("libero_spatial", 0, 0), choices[::-1])


def test_video_frames_align_to_scores_and_mismatch_is_explicit(recordings):
    path = recordings / "snapshot/task00/episode000.mp4"
    times, fps = video_times(path, 6)
    np.testing.assert_allclose(times, np.arange(6) / 10)
    assert fps == 10
    with pytest.raises(ValueError, match="uncertainty has"):
        video_times(path, 7)


def test_missing_video_uses_labelled_snapshots_and_csv_remains_raw(recordings):
    (recordings / "snapshot/task00/episode000.mp4").unlink()
    runs, choices = load_pair(recordings)
    payload = build_payload(runs, ("libero_spatial", 0, 0), choices)
    panel = payload["panels"][0]
    assert panel["video_src"] is None and "not an exact frame" in panel["warning"]
    assert [item["step"] for item in panel["snapshots"]] == [0, 5]
    rows = list(csv.DictReader(io.StringIO(export_csv(payload))))
    assert len(rows) == 12 and float(rows[2]["uncertainty"]) == 10
    assert rows[2]["source_alarm"] == "True" and rows[0]["source_alarm"] == "False"
    assert alarm_steps([3, 4, 5], 4) == [2]


def test_standalone_treats_instructions_as_data_not_script(recordings):
    runs, choices = load_pair(recordings)
    payload = build_payload(runs, ("libero_spatial", 0, 0), choices)
    payload["panels"][0]["metadata"]["language"] = "</script><script>window.injected=true</script>"
    html = standalone_html(payload)
    assert "</script><script>window.injected" not in html
    assert "\\u003c/script" in html and "data:video/mp4;base64," in html
    assert "https://" not in html and "http://" not in html


def test_streamlit_filters_and_default_pair_selection(recordings, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["viewer", "--root", str(recordings)])
    app = AppTest.from_file(str(REPO / "scripts/view_libero_vfd.py")).run(timeout=30)
    assert not app.exception
    assert len([item for item in app.selectbox if item.label == "Left pair"]) == 1
    outcome = next(item for item in app.selectbox if item.label == "Outcome in left recording")
    outcome.select("Success").run()
    assert not app.exception
    selection = next(item for item in app.selectbox if item.label == "Rollout")
    assert selection.value == ("libero_spatial", 1, 0)


@pytest.mark.skipif(os.environ.get("VFD_VIEWER_BROWSER_TESTS") != "1", reason="Opt-in browser test")
def test_browser_seek_playback_notes_and_live_streamlit(recordings, tmp_path):
    from playwright.sync_api import sync_playwright

    runs, choices = load_pair(recordings)
    payload = build_payload(runs, ("libero_spatial", 0, 0), choices)
    html = tmp_path / "viewer.html"
    html.write_text(standalone_html(payload))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        page.wait_for_function(
            "document.querySelectorAll('video').length===2 && [...document.querySelectorAll('video')].every(v=>v.readyState>=2)"
        )
        page.get_by_role("spinbutton", name="Go to timestep").fill("2")
        page.get_by_role("spinbutton", name="Go to timestep").press("Enter")
        page.wait_for_function("[...document.querySelectorAll('video')].every(v=>Math.abs(v.currentTime-.2)<.001)")
        assert page.locator(".reading").all_text_contents() == [
            "U = 10.000 · above source + target threshold",
            "U = 7.000 · above source + target threshold",
        ]
        page.get_by_role("button", name="Frame →", exact=True).click()
        assert page.locator(".step-number").input_value() == "3"
        page.get_by_label("Event note").fill("grasp")
        page.get_by_role("button", name="Mark current step", exact=True).click()
        assert page.get_by_role("button", name="Step 3: grasp").count() == 1
        page.locator(".smooth").check()
        assert (
            "U = 3.000" in page.locator(".reading").first.inner_text()
        )  # Smoothing never replaces the raw readout.
        page.locator(".step-number").fill("0")
        page.locator(".step-number").press("Enter")
        box = page.locator(".vfd canvas").first.bounding_box()
        page.locator(".vfd canvas").first.click(position={"x": 47 + (box["width"] - 59) * 4 / 5, "y": 100})
        assert page.locator(".step-number").input_value() == "4"
        page.locator(".vfd").focus()
        page.keyboard.press("ArrowLeft")
        assert page.locator(".step-number").input_value() == "3"
        page.locator(".step-number").fill("0")
        page.locator(".step-number").press("Enter")
        page.locator(".speed").select_option("0.25")
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_function("Number(document.querySelector('.step-number').value)>=3")
        page.get_by_role("button", name="Pause", exact=True).click()
        assert not errors
        # Exercise the real Streamlit component (including its shadow DOM), not just exported HTML.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (tmp_path / "streamlit.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(REPO / "scripts/view_libero_vfd.py"),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                    "--",
                    "--root",
                    str(recordings),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(100):
                    try:
                        if urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).status == 200:
                            break
                    except OSError:
                        time.sleep(0.1)
                page.goto(f"http://127.0.0.1:{port}")
                page.locator(".step-number").wait_for(timeout=30000)
                assert page.locator(".vfd canvas").count() == 2
                page.locator(".step-number").fill("2")
                page.locator(".step-number").press("Enter")
                assert page.locator(".step-label").inner_text() == "Step 2 / 5"
                assert page.locator(".source").filter(has_text="source: 4.000").count() == 1
                assert page.locator(".target").filter(has_text="target: 6.000").count() == 1
                page.get_by_role("combobox", name="Outcome in left recording", exact=True).click()
                page.get_by_role("option", name="Success", exact=True).click()
                case = page.locator('.vfd[data-episode="libero_spatial/1/0"]')
                case.wait_for()
                assert case.locator(".step-label").inner_text() == "Step 0 / 3"
                assert not errors
            finally:
                process.terminate()
                process.wait(timeout=15)
        browser.close()
