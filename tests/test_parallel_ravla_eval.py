"""Real subprocess/ZeroMQ tests for shared-server routing and process ownership."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import pytest


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/rag/eval_libero_parallel.py"
    spec = importlib.util.spec_from_file_location("parallel_eval_test_module", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan(module, tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""
import os, sys
from gr00t.eval.service import BaseInferenceServer
server = BaseInferenceServer(host="127.0.0.1", port=int(sys.argv[1]))
server.register_endpoint("server_info", lambda: {"server_id": sys.argv[2], "pid": os.getpid()}, requires_input=False)
server.register_endpoint("get_action", lambda data: {"task_id": data["task_id"], "counter": data["counter"]})
server.run()
""")
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json, os, signal, sys, time
from pathlib import Path
from gr00t.eval.service import BaseInferenceClient
port, task, trials, destination = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
destination.mkdir(parents=True)
client = BaseInferenceClient(host="127.0.0.1", port=port, timeout_ms=3000)
start = time.time()
info = client.call_endpoint("server_info", requires_input=False)
if "--crash-server" in sys.argv:
    os.kill(info["pid"], signal.SIGTERM)
    time.sleep(30)
if "--fail" in sys.argv:
    time.sleep(0.2)
    raise RuntimeError("injected worker failure")
for i in range(8):
    response = client.call_endpoint("get_action", {"task_id": task, "counter": i})
    assert response == {"task_id": task, "counter": i}, response
    time.sleep(0.015)
episodes = []
for i in range(trials):
    success = (task + i) % 2 == 0
    episodes.append({"task_id": task, "episode_index": i, "task_description": "task " + str(task),
                     "success": success, "control_steps": 10})
    (destination / f"episode={i+1}--success={success}--task=task_{task}.mp4").write_bytes(b"video")
summary = {"task_suite_name": "libero_10", "task_ids": [task], "num_trials_per_task": trials,
           "num_episodes": trials, "num_successes": sum(e["success"] for e in episodes), "episodes": episodes}
if "--duplicate" in sys.argv:
    summary["episodes"][-1]["episode_index"] = 0
(destination / "summary.json").write_text(json.dumps(summary))
(destination.parent / "worker_info.json").write_text(json.dumps({"start": start, "end": time.time(),
    "cache": os.environ["NUMBA_CACHE_DIR"], "modules": os.environ["HF_MODULES_CACHE"],
    "mpl": os.environ["MPLCONFIGDIR"], "server_pid": info["pid"]}))
client.socket.close(linger=0)
client.context.term()
""")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    output = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    tasks = [0, 1, 2]
    return module.Plan(
        tmp_path,
        output,
        "test",
        "libero_10",
        tasks,
        2,
        2,
        port,
        "test-identity",
        [sys.executable, "-u", str(server), str(port), "test-identity"],
        {
            t: [
                sys.executable,
                str(worker),
                str(port),
                str(t),
                "2",
                str(output / "tasks" / f"task{t:02d}" / "rollouts"),
            ]
            for t in tasks
        },
        env,
        startup_timeout=10,
        poll_interval=0.03,
    )


def assert_children_gone(plan):
    state = json.loads((plan.output / "progress.json").read_text())
    pids = [
        state["server_pid"],
        *[t["pid"] for t in state["tasks"].values() if "pid" in t],
    ]
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_shared_server_routes_requests_and_merges_unique_episodes(module, plan):
    module.run_plan(plan)
    summary = json.loads((plan.output / "rollouts/summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["num_episodes"] == 6
    assert summary["num_successes"] == 3
    assert summary["success_rate"] == 0.5
    assert len({(e["task_id"], e["episode_index"]) for e in summary["episodes"]}) == 6
    videos = list((plan.output / "rollouts").glob("*.mp4"))
    assert len(videos) == 6 and all(p.is_symlink() and p.is_file() for p in videos)
    info = [
        json.loads(
            (plan.output / "tasks" / f"task{t:02d}" / "worker_info.json").read_text()
        )
        for t in plan.task_ids
    ]
    assert len({i["server_pid"] for i in info}) == 1
    for key in ["cache", "modules", "mpl"]:
        assert len({i[key] for i in info}) == 3
    events = sorted([(i["start"], 1) for i in info] + [(i["end"], -1) for i in info])
    active, peak = 0, 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak <= 2
    state = json.loads((plan.output / "progress.json").read_text())
    assert state["episodes_completed"] == 6
    assert_children_gone(plan)


def test_worker_failure_aborts_and_cleans_up(module, plan):
    plan.task_commands[0].append("--fail")
    with pytest.raises(RuntimeError, match="Task 00 exited"):
        module.run_plan(plan)
    state = json.loads((plan.output / "progress.json").read_text())
    assert state["status"] == "failed"
    assert state["tasks"]["0"]["status"] == "failed"
    assert_children_gone(plan)


def test_server_death_aborts_and_cleans_up(module, plan):
    plan.task_commands[0].append("--crash-server")
    with pytest.raises(RuntimeError, match="Policy server died"):
        module.run_plan(plan)
    assert_children_gone(plan)


def test_wrong_server_identity_never_launches_workers(module, plan):
    plan.server_command[-1] = "wrong-identity"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        module.run_plan(plan)
    state = json.loads((plan.output / "progress.json").read_text())
    assert all("pid" not in task for task in state["tasks"].values())
    assert_children_gone(plan)


def test_occupied_port_is_not_reused(module, plan):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", plan.port))
        listener.listen()
        with pytest.raises(OSError):
            module.run_plan(plan)
        assert listener.fileno() >= 0
        assert not plan.output.exists()


def test_duplicate_episodes_refuse_successful_merge(module, plan):
    plan.task_commands[0].append("--duplicate")
    with pytest.raises(ValueError, match="Incomplete or inconsistent"):
        module.run_plan(plan)
    assert json.loads((plan.output / "progress.json").read_text())["status"] == "failed"
    assert_children_gone(plan)


def test_existing_output_is_preserved(module, plan):
    plan.output.mkdir()
    sentinel = plan.output / "sentinel"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        module.run_plan(plan)
    assert sentinel.read_text() == "original"


def test_interrupt_cleans_up_workers_and_server(module, plan, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "running_counts", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run_plan(plan)
    assert_children_gone(plan)
