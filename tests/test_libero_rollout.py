"""Check checkpoint discovery, rollout accounting, and client timeout without a GPU."""

import json
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import zmq

from examples.Libero.eval import run_libero_eval as evaluation
from gr00t.eval.service import BaseInferenceClient
from scripts.eval_libero_rollout import REPO, checkpoint_settings, discover, stop_process, wait_for_server


def make_checkpoint(run, step, complete=True):
    path = run / f"checkpoint-{step}"
    (path / "experiment_cfg").mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "experiment_cfg/metadata.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"test fixture")
    if complete:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return path


def test_future_checkpoint_discovery_and_shell_dry_run(tmp_path):
    run = tmp_path / "path with spaces" / "heldout_goal"
    first = make_checkpoint(run, 2)
    second = make_checkpoint(run, 10)
    make_checkpoint(run, 20, complete=False)
    (run / "run_manifest.json").write_text(json.dumps({"held_out_suite": "goal", "data_config": "test:Config"}))
    assert discover([run]) == [first, second]
    assert discover([run], latest=True) == [second]
    assert discover([run, first]) == [first, second]
    assert checkpoint_settings(first) == ("libero_goal", "test:Config")
    assert checkpoint_settings(first, "long", "other:Config") == ("libero_10", "other:Config")
    result = subprocess.run(
        ["bash", str(REPO / "scripts/eval_libero_rollout.sh"), str(run), "--latest", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "checkpoint-10 -> libero_goal" in result.stdout
    assert "checkpoint-20" not in result.stdout
    assert not list(run.glob("**/summary.json"))


@pytest.mark.parametrize("fail", [False, True])
def test_rollout_summary_and_error_propagation(tmp_path, monkeypatch, fail):
    class Env:
        closed = False

        def seed(self, seed):
            pass

        def reset(self):
            pass

        def set_init_state(self, state):
            return {}

        def step(self, action):
            return {}, 0, True, {}

        def close(self):
            self.closed = True

    class Policy:
        closed = False

        def __init__(self, **kwargs):
            self.policy = SimpleNamespace(close=self.close)

        def get_action(self, obs, language):
            if fail:
                raise RuntimeError("simulated RPC failure")
            return np.zeros(7)

        def close(self):
            Policy.closed = True

    task = SimpleNamespace(language="test task")
    suite = SimpleNamespace(n_tasks=1, get_task=lambda index: task,
                            get_task_init_states=lambda index: [0, 1])
    package = ModuleType("libero")
    module = ModuleType("libero.libero")
    module.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda: suite})
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    env = Env()
    monkeypatch.setattr(evaluation, "get_libero_env", lambda *args, **kwargs: (env, "test task"))
    monkeypatch.setattr(evaluation, "GR00TPolicy", Policy)
    cfg = evaluation.GenerateConfig(task_suite_name="libero_goal", num_trials_per_task=2,
                                    num_steps_wait=1, save_videos=False, output_dir=str(tmp_path))
    if fail:
        with pytest.raises(RuntimeError, match="simulated RPC failure"):
            evaluation.eval_libero(cfg)
    else:
        evaluation.eval_libero(cfg)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == ("failed" if fail else "completed")
    assert summary["total_episodes"] == (0 if fail else 2)
    assert summary["total_successes"] == (0 if fail else 2)
    assert env.closed and Policy.closed


def test_missing_server_times_out_and_closes():
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    client = BaseInferenceClient(host="127.0.0.1", port=port, timeout_ms=20)
    try:
        with pytest.raises(zmq.Again):
            client.call_endpoint("ping", requires_input=False)
        assert not client.ping()
    finally:
        client.close()
    assert client.socket.closed and client.context.closed


def test_managed_server_cleanup_and_startup_failure():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        stop_process(process)
        assert process.poll() is not None
        stop_process(process)  # Cleanup remains safe after a process has exited.
    finally:
        stop_process(process)
    failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
    failed.wait(timeout=10)
    with pytest.raises(RuntimeError, match="Inference server exited"):
        wait_for_server(failed, port=5555, timeout=1)


@pytest.mark.parametrize("wait_steps", [0, 10])
def test_long_rollout_exhausts_action_budget_without_stepping_terminated_env(tmp_path, monkeypatch, wait_steps):
    # Reproduce LIBERO's semantics: internal robosuite termination is separate
    # from the returned `done`, which only signals task success.
    environments = []

    class BudgetEnv:
        def __init__(self, horizon=1000, **kwargs):
            self.horizon = horizon
            self.closed = False
            environments.append(self)

        def seed(self, seed):
            pass

        def reset(self):
            self.timestep = 0
            self.done = False

        def set_init_state(self, state):
            return {}

        def step(self, action):
            if self.done:
                raise ValueError("executing action in terminated episode")
            self.timestep += 1
            self.done = self.timestep >= self.horizon
            return {}, 0, False, {}

        def close(self):
            self.closed = True

    class Policy:
        calls = 0

        def __init__(self, **kwargs):
            self.policy = SimpleNamespace(close=lambda: None)

        def get_action(self, obs, language):
            Policy.calls += 1
            return np.zeros(7)

    task = SimpleNamespace(language="test long task", problem_folder="test", bddl_file="test.bddl")
    suite = SimpleNamespace(n_tasks=1, get_task=lambda index: task,
                            get_task_init_states=lambda index: [0])
    package = ModuleType("libero")
    module = ModuleType("libero.libero")
    module.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": lambda: suite})
    module.get_libero_path = lambda key: str(tmp_path)
    envs = ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = BudgetEnv
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    monkeypatch.setattr(evaluation, "GR00TPolicy", Policy)
    cfg = evaluation.GenerateConfig(task_suite_name="libero_10", num_trials_per_task=1,
                                    num_steps_wait=wait_steps, save_videos=False, output_dir=str(tmp_path))
    evaluation.eval_libero(cfg)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["total_episodes"] == 1
    assert summary["total_successes"] == 0
    assert summary["tasks"][0]["episodes"][0]["action_steps"] == 1000
    assert Policy.calls == 1000
    assert environments[0].horizon == 1000 + wait_steps
    assert environments[0].timestep == 1000 + wait_steps
    assert environments[0].closed
