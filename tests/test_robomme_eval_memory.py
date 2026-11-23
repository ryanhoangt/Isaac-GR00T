"""Canonical protocols and real memory HTTP client rollouts without a model/simulator."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from uuid import uuid4

import json_numpy
import numpy as np
import pytest
import yaml

import test_robomme_eval as runner_tests

from examples.RoboMME import eval as evaluation
from examples.RoboMME.eval_protocol import (
    ALL_TASKS,
    INDEX_META,
    PRIMARY_HELD_OUT,
    digest,
    fingerprint,
    verify_memory_server,
)
from examples.RoboMME.memory_client import PROTOCOL


@pytest.fixture(params=["all16", "primary"])
def canonical(request, tmp_path):
    protocol = request.param
    root = tmp_path / "training"
    meta = root / "meta"
    meta.mkdir(parents=True)
    tasks = sorted(ALL_TASKS - PRIMARY_HELD_OUT if protocol == "primary" else ALL_TASKS)
    episodes = [{"episode_index": index, "task": task} for index, task in enumerate(tasks)]
    for name in INDEX_META:
        (meta / name).write_text("{}")
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (meta / "stats.json").write_text('{"source":"selected training rows"}')
    index = {
        "version": 1,
        "source_metadata_sha256": {name: digest(meta / name) for name in INDEX_META},
        "episodes": episodes,
    }
    index_path = meta / "episode_tasks.json"
    index_path.write_text(json.dumps(index))
    manifest = {"status": "complete", "episodes": len(episodes)}
    if protocol == "primary":
        manifest.update(split="primary", selected_tasks=tasks, excluded_tasks=sorted(PRIMARY_HELD_OUT))
    (root / "manifest.json").write_text(json.dumps(manifest))
    benchmark = tmp_path / "benchmark"
    test_meta = benchmark / "src/robomme/env_metadata/test"
    test_meta.mkdir(parents=True)
    for task in ALL_TASKS:
        (test_meta / f"record_dataset_{task}_metadata.json").write_text(
            json.dumps({"env_id": task, "records": [{"episode": index, "task": task} for index in (0, 2)]})
        )
    checkpoint = tmp_path / "checkpoint-1"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    args = [
        "--protocol",
        protocol,
        "--policy-type",
        "memory",
        "--manifest",
        str(root / "manifest.json"),
        "--task-index",
        str(index_path),
        "--benchmark-root",
        str(benchmark),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(tmp_path / "results"),
        "--num-episodes",
        "1",
        "--no-video",
    ]
    plan = evaluation.build_plan(evaluation.parse_args(args))
    memory = {"test_memory": True}
    normalization = {"test_stats": "training only"}
    identity = {
        "memory_config": memory,
        "datasets": [{"metadata_sha256": plan["training_metadata_sha256"], "normalization": normalization}],
    }
    (checkpoint / "experiment_cfg/metadata.json").write_text(json.dumps({"new_embodiment": normalization}))
    (checkpoint / "config.json").write_text(json.dumps({"action_head_cfg": {"memory_config": memory}}))
    (checkpoint / "experiment_cfg/memory_training.json").write_text(
        json.dumps({"identity": identity, "identity_sha256": fingerprint(identity)})
    )
    return args, plan


@pytest.fixture
def memory_http(canonical):
    args, plan = canonical
    root = Path(plan["checkpoint"])
    health = {
        "status": "ok",
        "stateful": True,
        "protocol": PROTOCOL,
        "checkpoint": str(root),
        "checkpoint_config_sha256": digest(root / "config.json"),
        "memory_training_sha256": digest(root / "experiment_cfg/memory_training.json"),
        "normalization_metadata_sha256": digest(root / "experiment_cfg/metadata.json"),
        "runtime": {"seed": 42, "denoising_steps": 4, "max_episode_frames": 4096, "encode_batch_size": 16},
    }
    state = {"requests": [], "resets": 0, "health": health}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value):
            data = json_numpy.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/health"
            self.respond(state["health"])

        def do_POST(self):
            body = json_numpy.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/reset":
                assert body == {}
                state.update(episode_id=uuid4().hex, frame_count=0, request_id=0, resets=state["resets"] + 1)
                self.respond({"reset_finished": True, "episode_id": state["episode_id"], "protocol": PROTOCOL})
            else:
                assert self.path == "/act"
                assert body["episode_id"] == state["episode_id"]
                assert body["start_frame_index"] == state["frame_count"]
                assert body["request_id"] == state["request_id"]
                state["requests"].append(body)
                state["frame_count"] += len(body["front_frames"])
                state["request_id"] += 1
                self.respond(
                    {
                        "protocol": PROTOCOL,
                        "episode_id": state["episode_id"],
                        "request_id": body["request_id"],
                        "frame_count": state["frame_count"],
                        "actions": np.zeros((16, 8), np.float32),
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield args + ["--server-url", url], {**plan, "server_url": url}, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_canonical_task_selection_and_comparison_groups(canonical):
    args, plan = canonical
    assert set(plan["tasks"]) == ALL_TASKS
    assert set(plan["task_groups"]["primary_comparison"]) == PRIMARY_HELD_OUT
    primary = plan["protocol"] == "primary"
    assert len(plan["training_tasks"]) == (12 if primary else 16)
    assert len(plan["held_out_tasks"]) == (4 if primary else 0)
    subset = evaluation.build_plan(evaluation.parse_args(args + ["--tasks", "PickXtimes"]))
    assert subset["task_groups"]["held_out"] == (["PickXtimes"] if primary else [])
    if primary:
        held = evaluation.build_plan(evaluation.parse_args(args + ["--task-group", "held-out"]))
        assert set(held["tasks"]) == PRIMARY_HELD_OUT and held["task_groups"]["seen"] == []
        with pytest.raises(ValueError, match="seen task group"):
            evaluation.build_plan(evaluation.parse_args(args + ["--task-group", "seen", "--tasks", "PickXtimes"]))
    else:
        with pytest.raises(ValueError, match="No tasks"):
            evaluation.build_plan(evaluation.parse_args(args + ["--task-group", "held-out"]))


@pytest.mark.parametrize("change", ["metadata", "missing_episode", "wrong_task"])
def test_training_index_mismatch_rejected(canonical, change):
    args, plan = canonical
    path = Path(plan["task_index"])
    index = json.loads(path.read_text())
    if change == "metadata":
        (Path(plan["training_dataset"]) / "meta/info.json").write_text('{"changed":true}')
    elif change == "missing_episode":
        index["episodes"].pop()
    else:
        index["episodes"][0]["task"] = "Unexpected task"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        evaluation.build_plan(evaluation.parse_args(args))


def test_memory_preflight_binds_training_and_server(memory_http):
    _, plan, state = memory_http
    verified = verify_memory_server(plan)
    assert verified["training_manifest_verified"] and verified["runtime"]["seed"] == 42
    wrong_dataset = {**plan, "training_metadata_sha256": {"stats.json": "different"}}
    with pytest.raises(ValueError, match="different dataset"):
        verify_memory_server(wrong_dataset)
    state["health"]["runtime"]["seed"] = 43
    with pytest.raises(ValueError, match="seed/denoising"):
        verify_memory_server(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("protocol", "reactive"),
        ("checkpoint", "/other"),
        ("memory_training_sha256", "different"),
        ("checkpoint_config_sha256", "different"),
        ("normalization_metadata_sha256", "different"),
    ],
)
def test_wrong_server_is_rejected_before_simulator(memory_http, monkeypatch, key, value):
    args, _, state = memory_http
    state["health"][key] = value
    monkeypatch.setattr(evaluation, "load_benchmark", lambda *args: pytest.fail("Simulator must not load"))
    with pytest.raises(ValueError, match="server"):
        evaluation.main(args)
    assert not evaluation.parse_args(args).output_dir.exists()
    assert state["requests"] == [] and state["resets"] == 0


def test_runner_uses_memory_client_and_saves_split_metrics(memory_http, monkeypatch):
    args, plan, state = memory_http
    environments = []

    def factory(**kwargs):
        assert kwargs["dataset"] == "test" and kwargs["action_space"] == "joint_angle"
        env = runner_tests.FakeEnv(finish_after=3)
        environments.append(env)
        return runner_tests.FakeBuilder(env)

    monkeypatch.setattr(evaluation, "load_benchmark", lambda *args: factory)
    args += ["--tasks", "BinFill", "PickXtimes", "--exec-horizon", "2"]
    assert evaluation.main(args) == 0
    assert len(environments) == 2 and all(env.closed for env in environments)
    assert state["resets"] == 2
    requests = state["requests"]
    assert [row["is_first_step"] for row in requests] == [True, False, True, False]
    assert [row["start_frame_index"] for row in requests] == [0, 2, 0, 2]
    assert [row["front_frames"][:, 0, 0, 0].tolist() for row in requests] == [[17, 80], [81, 82]] * 2
    assert requests[0]["episode_id"] != requests[2]["episode_id"]
    output = evaluation.parse_args(args).output_dir
    saved = evaluation.read_json(output / "config.json")
    assert saved["server_provenance"]["training_manifest_verified"]
    assert saved["exec_horizon"] == 2 and saved["checkpoint"] == plan["checkpoint"]
    metrics = evaluation.read_json(output / "metrics.json")
    assert metrics["complete"] and metrics["overall"]["success"] == 2
    assert metrics["by_task_group"]["primary_comparison"]["tasks"] == ["PickXtimes"]
    assert metrics["by_task_group"]["held_out"]["success"] == (1 if plan["protocol"] == "primary" else 0)
    progress = evaluation.read_json(output / "progress.json")
    assert all(row["input_frame_counts"] == [2, 2] for row in progress["episodes"])
    assert evaluation.main(args + ["--resume"]) == 0
    assert len(state["requests"]) == 4
    state["health"]["runtime"]["max_episode_frames"] = 8192
    with pytest.raises(ValueError, match="Resume configuration differs"):
        evaluation.main(args + ["--resume"])


def test_changed_checkpoint_normalization_is_rejected(canonical):
    _, plan = canonical
    path = Path(plan["checkpoint"]) / "experiment_cfg/metadata.json"
    path.write_text(json.dumps({"new_embodiment": {"wrong_stats": True}}))
    with pytest.raises(ValueError, match="normalization"):
        verify_memory_server(plan)


def test_yaml_presets_overrides_and_no_server_during_dry_run(canonical, monkeypatch, capsys):
    args, plan = canonical
    config = Path("examples/RoboMME/config") / f"eval_memory_{plan['protocol']}.yaml"
    parsed = evaluation.parse_args(["--config", str(config), *args])
    assert parsed.timeout == 120 and parsed.num_episodes == 1 and parsed.no_video
    monkeypatch.setattr(evaluation, "load_benchmark", lambda *args: pytest.fail("No simulator"))
    monkeypatch.setattr(evaluation, "verify_memory_server", lambda *args: pytest.fail("No HTTP/checkpoint load"))
    assert evaluation.main(["--config", str(config), *args, "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["config_file_sha256"] == digest(config)
    assert len(result["tasks"]) == 16


@pytest.mark.parametrize(
    "bad",
    [
        {"typo": 1},
        {"seed": "42"},
        {"max_steps": True},
        {"protocol": None},
        {"protocol": "unknown"},
        {"no_video": "false"},
        {"episode_indices": [True]},
    ],
)
def test_invalid_yaml_is_rejected(tmp_path, bad):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--config", str(path)])


def test_yaml_episode_selection_and_video_overrides(canonical, tmp_path):
    args, _ = canonical
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump({"episode_indices": [2], "no_video": True}))
    parsed = evaluation.parse_args(["--config", str(path), *args[:-1], "--video"])
    assert parsed.episode_indices is None and parsed.num_episodes == 1 and not parsed.no_video
    path.write_text(yaml.safe_dump({"num_episodes": 50}))
    parsed = evaluation.parse_args(["--config", str(path), *args[:-3], "--episode-indices", "2"])
    assert parsed.episode_indices == [2]
