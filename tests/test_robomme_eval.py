"""Check held-out evaluation with fake environments; no simulator or GPU imports."""

import copy
import json

import numpy as np
import pytest

from examples.RoboMME import eval as evaluation


@pytest.fixture
def protocol(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "primary",
                "selected_tasks": ["TrainA"],
                "excluded_tasks": ["HeldA", "HeldB"],
            }
        )
    )
    root = tmp_path / "benchmark"
    metadata = root / "src/robomme/env_metadata/test"
    metadata.mkdir(parents=True)
    for task in ("HeldA", "HeldB"):
        (metadata / f"record_dataset_{task}_metadata.json").write_text(
            json.dumps(
                {
                    "env_id": task,
                    "records": [{"task": task, "episode": index, "seed": index + 510000} for index in (0, 2)],
                }
            )
        )
    return [
        "--manifest",
        str(manifest),
        "--benchmark-root",
        str(root),
        "--checkpoint",
        "declared-checkpoint",
        "--output-dir",
        str(tmp_path / "results"),
        "--num-episodes",
        "1",
        "--no-video",
    ]


def observation(step, *, initial=False):
    values = [17, 80] if initial else [80 + step]
    return {
        "front_rgb_list": [np.full((16, 16, 3), value, np.uint8) for value in values],
        "wrist_rgb_list": [np.full((16, 16, 3), value + 1, np.uint8) for value in values],
        "joint_state_list": [np.full(7, step, np.float32) for _ in values],
        "gripper_state_list": [np.full(2, 0.02, np.float32) for _ in values],
    }


class FakeClient:
    def __init__(self, *, fail=False):
        self.requests = []
        self.resets = 0
        self.fail = fail

    def reset(self):
        self.resets += 1
        return {"reset_finished": True}

    def infer(self, inputs):
        self.requests.append(copy.deepcopy(inputs))
        if self.fail:
            raise TimeoutError("stub timeout")
        return {"actions": np.full((16, 8), len(self.requests), np.float32)}


class FakeEnv:
    def __init__(self, *, finish_after=3, outcome="success", step_error=False):
        self.finish_after, self.outcome, self.step_error = finish_after, outcome, step_error
        self.actions = []
        self.closed = False

    def reset(self):
        return observation(0, initial=True), {"task_goal": ["A task instruction"]}

    def step(self, action):
        self.actions.append(action.copy())
        if self.step_error:
            return None, 0, False, False, {"status": "error", "error_message": "simulator failed"}
        done = len(self.actions) == self.finish_after
        return observation(len(self.actions)), 0, done, False, {"status": self.outcome if done else "ongoing"}

    def close(self):
        self.closed = True


class FakeBuilder:
    def __init__(self, env):
        self.env = env

    def make_env_for_episode(self, episode_id):
        return self.env


def test_plan_selects_only_held_out_and_hashes_metadata(protocol):
    config = evaluation.build_plan(evaluation.parse_args(protocol))
    assert config["tasks"] == ["HeldA", "HeldB"]
    assert config["training_tasks"] == ["TrainA"] and config["episode_indices"] == [0]
    assert "policy_backend" not in config and "support_demos_per_task" not in config
    assert len(config["manifest_sha256"]) == 64
    assert set(config["test_metadata_sha256"]) == set(config["tasks"])
    subset = evaluation.build_plan(evaluation.parse_args(protocol + ["--tasks", "HeldB"]))
    assert subset["tasks"] == ["HeldB"]
    with pytest.raises(ValueError, match="excluded from training"):
        evaluation.build_plan(evaluation.parse_args(protocol + ["--tasks", "TrainA"]))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"status": "partial"}, "completed"),
        ({"selected_tasks": ["HeldA"]}, "overlap"),
        ({"excluded_tasks": []}, "nonempty"),
        ({"excluded_tasks": ["HeldA", "HeldA"]}, "Duplicate"),
    ],
)
def test_invalid_protocol_is_rejected(protocol, change, message):
    path = evaluation.Path(protocol[1])
    value = evaluation.read_json(path)
    path.write_text(json.dumps({**value, **change}))
    with pytest.raises(ValueError, match=message):
        evaluation.build_plan(evaluation.parse_args(protocol))


def test_explicit_episode_ids_and_missing_metadata(protocol):
    args = evaluation.parse_args(protocol)
    args.episode_indices = [2]
    assert evaluation.build_plan(args)["episode_indices"] == [2]
    args.episode_indices = [1]
    with pytest.raises(ValueError, match="missing test episode"):
        evaluation.build_plan(args)
    args.episode_indices = [0, 0]
    with pytest.raises(ValueError, match="unique nonnegative"):
        evaluation.build_plan(args)


@pytest.mark.parametrize(
    "option,value", [("exec_horizon", 17), ("exec_horizon", 0), ("max_steps", 0), ("num_episodes", 0)]
)
def test_invalid_execution_parameters(protocol, option, value):
    args = evaluation.parse_args(protocol)
    setattr(args, option, value)
    with pytest.raises(ValueError):
        evaluation.build_plan(args)


def test_dry_run_does_not_load_simulator_or_write_output(protocol, monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("dry-run must not import the simulator")

    monkeypatch.setattr(evaluation, "load_benchmark", forbidden)
    assert evaluation.main(protocol + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["tasks"] == ["HeldA", "HeldB"]
    assert not evaluation.parse_args(protocol).output_dir.exists()


def test_execution_horizon_replans_and_resets_between_episodes():
    client = FakeClient()
    for index in (0, 1):
        env = FakeEnv()
        result = evaluation.run_episode(client, FakeBuilder(env), index, exec_horizon=2, max_steps=10)
        assert result["outcome"] == "success" and result["steps"] == 3 and env.closed
        assert [float(action[0]) for action in env.actions] == [1 + index * 2, 1 + index * 2, 2 + index * 2]
    assert client.resets == 2 and len(client.requests) == 4
    assert [request["is_first_step"] for request in client.requests] == [True, False, True, False]
    assert [len(request["front_rgb_list"]) for request in client.requests] == [2, 2, 2, 2]
    assert client.requests[1]["joint_state_list"][-1][0] == 2


@pytest.mark.parametrize("outcome", ["success", "fail", "timeout"])
def test_terminal_status_stops_remaining_chunk(outcome):
    env, client = FakeEnv(finish_after=1, outcome=outcome), FakeClient()
    result = evaluation.run_episode(client, FakeBuilder(env), 0, exec_horizon=16, max_steps=10)
    assert result["outcome"] == outcome and len(env.actions) == 1 and env.closed


def test_step_budget_excludes_demo_prefix_and_drops_pending_actions():
    env, client = FakeEnv(finish_after=100), FakeClient()
    result = evaluation.run_episode(client, FakeBuilder(env), 0, exec_horizon=16, max_steps=2)
    assert result["outcome"] == "timeout" and result["steps"] == 2 and env.closed
    assert len(client.requests) == 1 and len(env.actions) == 2


@pytest.mark.parametrize("infer_error", [True, False])
def test_inference_and_structured_environment_errors_close_env(infer_error):
    env, client = FakeEnv(step_error=not infer_error), FakeClient(fail=infer_error)
    result = evaluation.run_episode(client, FakeBuilder(env), 0, exec_horizon=16, max_steps=10)
    assert result["outcome"] == "error" and env.closed
    assert ("timeout" if infer_error else "simulator failed") in result["error"]


def test_video_streams_demo_prefix_and_execution_and_closes(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("imageio_ffmpeg")
    env = FakeEnv(finish_after=1)
    video = tmp_path / "episode.mp4"
    result = evaluation.run_episode(
        FakeClient(), FakeBuilder(env), 0, exec_horizon=16, max_steps=10, video_path=video
    )
    assert result["outcome"] == "success" and env.closed and result["video_path"] == str(video)
    with av.open(str(video)) as handle:
        frames = [frame.to_ndarray(format="rgb24") for frame in handle.decode(video=0)]
    assert len(frames) == 3 and frames[0].shape == (16, 32, 3)
    assert frames[0][0, 0, 0] > 200 and frames[0][0, 0, 1] < 40


def test_results_metrics_and_resume_are_bound_to_configuration(protocol):
    args = evaluation.parse_args(protocol)
    config = evaluation.build_plan(args)
    client, environments = FakeClient(), []

    def factory(**kwargs):
        assert kwargs["dataset"] == "test" and kwargs["action_space"] == "joint_angle"
        env = FakeEnv(finish_after=1, outcome="success" if kwargs["env_id"] == "HeldA" else "fail")
        environments.append(env)
        return FakeBuilder(env)

    metrics = evaluation.evaluate(config, args.output_dir, client, factory)
    assert metrics["complete"] and metrics["overall"]["success_rate"] == 0.5
    assert metrics["overall"]["success"] == metrics["overall"]["fail"] == 1
    assert all(env.closed for env in environments)
    assert len(evaluation.read_json(args.output_dir / "progress.json")["episodes"]) == 2
    assert evaluation.read_json(args.output_dir / "metrics.json") == metrics
    with pytest.raises(FileExistsError):
        evaluation.evaluate(config, args.output_dir, client, factory)
    assert evaluation.evaluate(config, args.output_dir, client, factory, resume=True) == metrics
    assert len(environments) == 2 and client.resets == 2
    with pytest.raises(ValueError, match="differs"):
        evaluation.evaluate(
            {**config, "checkpoint": "another checkpoint"}, args.output_dir, client, factory, resume=True
        )


def test_partial_run_resumes_only_unfinished_episodes(protocol):
    args = evaluation.parse_args(protocol)
    config = evaluation.build_plan(args)
    env = FakeEnv(finish_after=1)

    def interrupt(**kwargs):
        if kwargs["env_id"] == "HeldB":
            raise KeyboardInterrupt
        return FakeBuilder(env)

    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate(config, args.output_dir, FakeClient(), interrupt)
    progress = evaluation.read_json(args.output_dir / "progress.json")
    assert len(progress["episodes"]) == 1 and env.closed
    client = FakeClient()
    metrics = evaluation.evaluate(
        config, args.output_dir, client, lambda **kwargs: FakeBuilder(FakeEnv(finish_after=1)), resume=True
    )
    assert metrics["complete"] and client.resets == 1


def test_errors_remain_in_metric_denominator(protocol):
    args = evaluation.parse_args(protocol)
    config = evaluation.build_plan(args)
    metrics = evaluation.evaluate(
        config, args.output_dir, FakeClient(fail=True), lambda **kwargs: FakeBuilder(FakeEnv())
    )
    assert metrics["complete"] and metrics["overall"]["error"] == 2
    assert metrics["overall"]["success_rate"] == 0.0


def test_ravla_plan_uses_tcp_and_keeps_heldout_restriction(protocol):
    config = evaluation.build_plan(evaluation.parse_args(protocol + ["--policy-backend", "ravla"]))
    assert config["server_url"] == "tcp://127.0.0.1:8071" and config["support_demos_per_task"] == 3
    with pytest.raises(ValueError, match="tcp://"):
        evaluation.build_plan(
            evaluation.parse_args(
                protocol + ["--policy-backend", "ravla", "--server-url", "http://localhost:8000"]
            )
        )
    with pytest.raises(ValueError, match="excluded from training"):
        evaluation.build_plan(evaluation.parse_args(protocol + ["--policy-backend", "ravla", "--tasks", "TrainA"]))


def test_ravla_evaluation_passes_task_identity_on_every_replan(protocol):
    args = evaluation.parse_args(protocol + ["--policy-backend", "ravla", "--exec-horizon", "1"])
    config = evaluation.build_plan(args)
    client = FakeClient()
    metrics = evaluation.evaluate(config, args.output_dir, client, lambda **kwargs: FakeBuilder(FakeEnv()))
    assert metrics["complete"]
    assert [row["canonical_task"] for row in client.requests] == ["HeldA"] * 3 + ["HeldB"] * 3


def test_ravla_preflight_failure_happens_before_simulator_and_output(protocol, monkeypatch):
    class Client(FakeClient):
        parse_address = staticmethod(evaluation.RoboMMERAVLAClient.parse_address)

        def __init__(self, *args, **kwargs):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def validate_server(self, *args, **kwargs):
            raise ValueError("missing support memory")

    def forbidden(*args):
        pytest.fail("Simulator must not load before memory preflight")

    monkeypatch.setattr(evaluation, "RoboMMERAVLAClient", Client)
    monkeypatch.setattr(evaluation, "load_benchmark", forbidden)
    with pytest.raises(ValueError, match="missing support"):
        evaluation.main(protocol + ["--policy-backend", "ravla"])
    assert not evaluation.parse_args(protocol).output_dir.exists()


def test_ravla_main_records_stable_server_identity_for_resume(protocol, monkeypatch):
    class Client(FakeClient):
        parse_address = staticmethod(evaluation.RoboMMERAVLAClient.parse_address)

        def __init__(self, *args, **kwargs):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def validate_server(self, checkpoint, tasks, **kwargs):
            assert kwargs["support_demos_per_task"] == 3
            return {
                "checkpoint_path": checkpoint,
                "retriever_path": "retriever",
                "memory_path": "memory",
                "data_config": "RoboMME",
                "memory_key_type": "canonical_task",
                "memory_tasks": tasks,
                "memory_demo_counts": {task: 3 for task in tasks},
                "k": 1,
                "denoising_steps": 4,
                "source_fingerprint": "stable",
                "pid": 123,
                "server_id": "temporary",
            }

    monkeypatch.setattr(evaluation, "RoboMMERAVLAClient", Client)
    monkeypatch.setattr(evaluation, "load_benchmark", lambda *args: lambda **kwargs: FakeBuilder(FakeEnv()))
    options = protocol + ["--policy-backend", "ravla"]
    assert evaluation.main(options) == 0
    saved = evaluation.read_json(evaluation.parse_args(protocol).output_dir / "config.json")
    assert saved["remote_policy"]["source_fingerprint"] == "stable"
    assert "pid" not in saved["remote_policy"] and "server_id" not in saved["remote_policy"]
    assert evaluation.main(options + ["--resume"]) == 0
