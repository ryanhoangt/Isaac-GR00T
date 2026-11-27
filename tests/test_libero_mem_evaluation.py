"""Reactive LIBERO-Mem mapping, chunk boundaries, HTTP and rollout accounting."""

from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
import threading

import h5py
import numpy as np
import pytest

from examples.Libero.adapter import ACTION_KEYS, to_gr00t_observation, to_libero_actions
from examples.Libero.eval.run_libero_eval import GR00TPolicy
from examples.Libero_Mem import evaluation
from examples.Libero_Mem.policy import LiberoMemPolicy
from examples.Libero_Mem.selection import SourceConfig
from examples.Libero_Mem.tasks import TASKS
from test_libero_mem_episode import Env, config as config


@pytest.fixture
def observation():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 50,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
        "oracle_subgoal": "Must not become policy input",
    }


def prediction(horizon=4):
    result = {f"action.{key}": np.full((horizon, 1), i / 10) for i, key in enumerate(ACTION_KEYS)}
    result["action.gripper"] = np.linspace(0, 1, horizon)[:, None]
    return result


def test_mapping_matches_existing_libero_and_owns_observation_buffers(observation):
    result = to_gr00t_observation(observation, "Task instruction — tiếng Việt")
    assert len(result) == 10 and "oracle_subgoal" not in result
    np.testing.assert_array_equal(result["video.image"][0], observation["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(
        result["video.wrist_image"][0], observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    np.testing.assert_allclose(result["state.yaw"], [[np.pi / 2]])
    assert result["state.gripper"].shape == (1, 2)
    legacy = object.__new__(GR00TPolicy)
    legacy.headless = True
    for key, value in legacy._process_observation(observation, "Task instruction — tiếng Việt").items():
        np.testing.assert_array_equal(value, result[key])
    result["video.image"][:] = 0
    result["state.gripper"][:] = 0
    assert observation["agentview_image"].max() > 0
    np.testing.assert_array_equal(observation["robot0_gripper_qpos"], [0.02, -0.02])
    observation["robot0_eef_quat"][3] = 1.000001
    to_gr00t_observation(observation, "task")
    assert observation["robot0_eef_quat"][3] == 1.000001


def test_action_mapping_preserves_arm_and_reverses_gripper_once():
    raw = prediction(3)
    actions = to_libero_actions(raw)
    assert actions.shape == (3, 7) and actions.dtype == np.float32
    np.testing.assert_allclose(actions[:, :6], np.tile(np.arange(6) / 10, (3, 1)))
    np.testing.assert_array_equal(actions[:, -1], [1, 0, -1])
    legacy = object.__new__(GR00TPolicy)
    np.testing.assert_array_equal(legacy._convert_to_libero_action(raw), actions[0])
    np.testing.assert_array_equal(raw["action.gripper"].ravel(), [0, 0.5, 1])
    squeezed = {key: value[:, 0] for key, value in raw.items()}
    np.testing.assert_array_equal(to_libero_actions(squeezed), actions)


@pytest.mark.parametrize("mutation", ["missing", "horizon", "batched", "nan", "overflow", "empty"])
def test_bad_action_chunks_are_rejected(mutation):
    raw = prediction()
    if mutation == "missing":
        del raw["action.x"]
    else:
        raw["action.x"] = {
            "horizon": np.zeros((3, 1)),
            "batched": np.zeros((1, 4, 1)),
            "nan": np.full((4, 1), np.nan),
            "overflow": np.full((4, 1), 1e100),
            "empty": np.zeros((0, 1)),
        }[mutation]
    with pytest.raises(ValueError):
        to_libero_actions(raw)


@pytest.fixture
def http_stub():
    import json_numpy

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/act"
            payload = json_numpy.loads(self.rfile.read(int(self.headers["Content-Length"])))
            obs = payload["observation"]
            received.append(obs)
            goal = obs["annotation.human.action.task_description"][0]
            status = 500 if goal == "rpc failure" else 200
            raw = prediction()
            raw["action.x"] = float(obs["state.x"][0, 0]) + np.arange(4)[:, None]
            if goal == "invalid response":
                raw = ["not a mapping"]
            body = json_numpy.dumps(raw).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_chunks_replan_and_reset_without_stale_actions(http_stub, observation):
    url, received = http_stub
    policy = LiberoMemPolicy(url, n_action_steps=2)
    try:
        first = policy.get_action(observation, "task")
        second = policy.get_action(observation, "task")
        assert policy.requests == len(received) == 1 and policy.prediction_horizon == 4
        np.testing.assert_allclose([first[0], second[0]], [0.1, 1.1])
        observation["robot0_eef_pos"][0] = 2
        assert policy.get_action(observation, "task")[0] == 2
        assert policy.requests == len(received) == 2
        policy.reset()  # An old queued x=3 must not reach the new episode.
        assert len(received) == 2 and policy.requests == 0
        observation["robot0_eef_pos"][0] = 5
        assert policy.get_action(observation, "new episode")[0] == 5
        assert policy.requests == 1 and len(received) == 3
        assert "oracle_subgoal" not in received[0]
        assert received[0]["video.image"].dtype == np.uint8
    finally:
        policy.close()


@pytest.mark.parametrize("goal", ["rpc failure", "invalid response"])
def test_http_errors_propagate_without_retry(http_stub, observation, goal):
    import requests

    url, received = http_stub
    policy = LiberoMemPolicy(url)
    try:
        with pytest.raises((requests.HTTPError, ValueError)):
            policy.get_action(observation, goal)
        assert len(received) == 1
    finally:
        policy.close()


def test_short_chunk_and_instruction_change_require_explicit_handling(observation):
    class Client:
        def get_action(self, obs):
            return prediction()

        def close(self):
            pass

    policy = LiberoMemPolicy(n_action_steps=5, client=Client())
    with pytest.raises(ValueError, match="exceeds"):
        policy.get_action(observation, "task")
    policy = LiberoMemPolicy(n_action_steps=2, client=Client())
    policy.get_action(observation, "task")
    with pytest.raises(ValueError, match="Instruction changed"):
        policy.get_action(observation, "different task")


@pytest.fixture
def eval_config(config):  # noqa: F811 -- imported pytest fixture
    # Extend the audited small Raw fixture to all canonical task identities.
    report = json.loads(config.audit.read_text())
    first = report["tasks"][0]
    root = config.libero_repo / "libero/libero/bddl_files/libero_mem"
    for task in TASKS[1:]:
        path = config.raw_root / task.filename
        shutil.copyfile(first["raw_file"]["path"], path)
        stat = path.stat()
        (root / f"{task.name}.bddl").write_text(task.instruction)
        report["tasks"].append(
            {
                **first,
                "number": task.number,
                "name": task.name,
                "instruction": task.instruction,
                "raw_file": {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            }
        )
    config.audit.write_text(json.dumps(report))
    values = asdict(config)
    return evaluation.EvalConfig(
        **{key: values[key] for key in SourceConfig.__dataclass_fields__}, budget_mode="fixed", max_steps=5
    )


@pytest.mark.parametrize("fold,expected", [("A", [2, 4, 6, 8, 10]), ("B", [1, 3, 5, 7, 9])])
def test_fold_task_group_and_initial_state_selection_are_independent(eval_config, fold, expected):
    cfg = replace(eval_config, task_ids=None, training_fold=fold, task_group="held-out", dry_run=True)
    plan = evaluation.run_evaluation(cfg)
    assert [job["task_number"] for job in plan["jobs"]] == expected
    assert {job["demo_id"] for job in plan["jobs"]} == {"demo_24"}
    assert not cfg.output_dir.exists()
    seen = evaluation.run_evaluation(replace(cfg, selection="seen"))
    assert {job["demo_id"] for job in seen["jobs"]} == {"demo_1"}
    with pytest.raises(ValueError, match="outside"):
        evaluation.evaluation_plan(replace(cfg, task_ids=(1 if fold == "A" else 2,)))


class Policy:
    def __init__(self, *args, fail=False, **kwargs):
        self.fail = fail
        self.closed = False
        self.resets = 0

    def reset(self):
        self.requests = 0
        self.prediction_horizon = 16
        self.resets += 1

    def get_action(self, obs, language):
        if self.fail:
            raise RuntimeError("RPC failed")
        self.requests += 1
        return np.zeros(7)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("overshoot", [False, True])
def test_rollout_finishes_budget_and_aggregates_task_metrics(eval_config, overshoot):
    environments = []
    policy = Policy()

    def factory(task, **kwargs):
        assert kwargs["render"] and kwargs["horizon"] == 26
        env = Env(overshoot=overshoot)
        environments.append(env)
        return env

    cfg = replace(eval_config, demo_ids=(23, 24), training_fold="A")
    result = evaluation.run_evaluation(cfg, env_factory=factory, policy_factory=lambda *args, **kwargs: policy)
    assert result["status"] == "completed" and result["active_episode"] is None
    assert result["total_episodes"] == 2 and result["total_successes"] == (0 if overshoot else 2)
    assert policy.resets == 2 and policy.closed
    assert all(env.calls == 25 and env.closed for env in environments)
    assert all(episode["action_steps"] == 5 and episode["truncated"] for episode in result["episodes"])
    assert result["groups"]["task_split"]["train"]["episodes"] == 2
    assert result["groups"]["memory_types"]["OM"]["mean_subgoal_completion_rate"] == 1
    assert result["episodes"][0]["first_success_step"] == 2
    rows = [json.loads(line) for line in next(cfg.output_dir.glob("*.jsonl")).read_text().splitlines()]
    assert rows[0]["environment_done"] and not rows[0]["success"] and len(rows) == 5


def test_rpc_error_marks_failed_run_and_closes_everything(eval_config):
    env, policy = Env(), Policy(fail=True)
    with pytest.raises(RuntimeError, match="RPC failed"):
        evaluation.run_evaluation(
            eval_config, env_factory=lambda *args, **kwargs: env, policy_factory=lambda *args, **kwargs: policy
        )
    result = json.loads((eval_config.output_dir / "summary.json").read_text())
    assert result["status"] == "failed" and result["total_episodes"] == 0
    assert result["active_episode"] == {"task_number": 1, "raw_demo_id": "demo_24"}
    assert env.closed and policy.closed


def test_fixed_budget_never_reads_expert_actions_or_future_states(eval_config, monkeypatch):
    original = h5py.Dataset.__getitem__

    def guarded(dataset, key):
        assert not dataset.name.endswith(("/actions", "/states"))
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    result = evaluation.run_evaluation(
        eval_config, env_factory=lambda *args, **kwargs: Env(), policy_factory=Policy
    )
    assert result["episodes"][0]["full_action_budget"] == 180


def test_benchmark_budget_matches_author_maximum(eval_config, monkeypatch):
    monkeypatch.setitem(evaluation.TASK_ACTION_STEPS, 1, 1)
    result = evaluation.run_evaluation(
        replace(eval_config, budget_mode="benchmark"),
        env_factory=lambda *args, **kwargs: Env(),
        policy_factory=Policy,
    )
    # Three Raw actions, one leading no-op trimmed, twenty tail steps.
    assert result["episodes"][0]["full_action_budget"] == 22


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_group": "held-out"},
        {"n_action_steps": 0},
        {"max_steps": 0},
        {"timeout": float("nan")},
        {"task_ids": (1, 1)},
        {"training_fold": "C"},
        {"budget_mode": "invalid"},
    ],
)
def test_invalid_evaluation_configuration(eval_config, kwargs):
    with pytest.raises(ValueError):
        evaluation.evaluation_plan(replace(eval_config, **kwargs))
