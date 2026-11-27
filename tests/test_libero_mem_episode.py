"""Sequence accounting, audited initial-state pools and expert replay lifecycle."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys

import h5py
import numpy as np
import pytest

from examples.Libero_Mem.episode import LiberoMemEpisodeProtocol
from examples.Libero_Mem.environment import RESET_PROTOCOL, initialize_episode
from examples.Libero_Mem import replay
from examples.Libero_Mem.source import vector_fingerprint
from examples.Libero_Mem.tasks import TASK_BY_NUMBER


class Domain:
    def __init__(self, sequences=None):
        self.sequences = sequences or [[[["on", "bowl", "plate1"]], [["on", "bowl", "plate2"]]]]
        self._overshot = True
        self.tick_calls = 0
        self.pending = False
        self.live = True
        self.reset_subgoal_progress()

    def get_all_goals(self):
        return self.sequences

    def get_goal_sequence_len(self):
        return len(self.sequences[0])

    def reset_subgoal_progress(self):
        self._satisfied_subgoals = []
        self._sub_goal_live_time = self._sub_goal_nonlive_time = 0

    def _check_success(self, inc=False):
        if inc:
            self.tick_calls += 1
            self.pending = True
            self._sub_goal_live_time += 1
        return True  # Deliberately unreliable as a full-task success indicator.

    def _eval_predicate(self, predicate):
        return self.live


class Env:
    def __init__(self, overshoot=False, fail=False, dim=4):
        self.env = Domain()
        self.overshoot = overshoot
        self.fail = fail
        self.closed = False
        self.scored_actions = 0
        self.calls = 0
        self.events = []
        self.state = np.arange(dim, dtype=np.float64)
        self.sim = SimpleNamespace(
            get_state=lambda: SimpleNamespace(flatten=lambda: self.state),
            model=SimpleNamespace(get_xml=lambda: "<compiled_model/>"),
            reset=self.reset_sim,
            set_state_from_flattened=self.restore_state,
            forward=lambda: self.events.append("forward"),
        )

    def reset_sim(self):
        self.events.append("reset_sim")
        self.state[:] = 0

    def restore_state(self, state):
        self.events.append("restore_state")
        self.restored_state = np.array(state, copy=True)
        self.state[:] = state

    def reset_from_xml_string(self, xml):
        self.events.append(("reload_xml", xml))

    def reset(self):
        self.events.append("reset_env")
        self.scored_actions = self.calls = 0
        return {}

    def set_init_state(self, state):
        self.events.append("install_raw_state")
        self.initial = np.array(state, copy=True)
        return {}

    def step(self, action):
        self.calls += 1
        if self.env.pending:
            self.env.pending = False
            if self.fail:
                raise RuntimeError("simulated physics failure")
            self.scored_actions += 1
            progress = min(self.scored_actions, 2)
            self.env._satisfied_subgoals = [["and", *goal] for goal in self.env.sequences[0][:progress]]
            if self.overshoot and self.scored_actions > 2:
                self.env._overshot = True
                self.env.live = False
        return {}, 1.0, True, {}

    def close(self):
        self.closed = True


def test_intermediate_done_is_not_full_success_and_snapshot_is_read_only():
    env = Env()
    protocol = LiberoMemEpisodeProtocol(env)
    assert not env.env._overshot
    _, first = protocol.step([0] * 7)
    assert first["environment_done"] and not first["success"]
    assert first["subgoal_completion_rate"] == 0.5
    ticks = env.env.tick_calls
    for _ in range(5):
        assert protocol.snapshot()["subgoals_completed"] == 1
    assert env.env.tick_calls == ticks == 1
    _, second = protocol.step([0] * 7)
    assert second["success"] and second["first_success_step"] == 2
    assert protocol.snapshot()["first_success_step"] == 2
    assert env.env.tick_calls == protocol.action_steps == 2


def test_overshoot_and_final_predicate_both_matter_and_reset_isolates_episodes():
    env = Env(overshoot=True)
    protocol = LiberoMemEpisodeProtocol(env)
    protocol.step([0] * 7)
    assert protocol.step([0] * 7)[1]["success"]
    last = protocol.step([0] * 7)[1]
    assert last["sequence_complete"] and last["subgoal_completion_rate"] == 1
    assert last["overshot"] and not last["success"]
    env.env._overshot = False
    assert not protocol.snapshot()["success"]  # Final predicate is still false.
    protocol.reset()
    assert not env.env._overshot and env.env._sub_goal_live_time == 0
    assert protocol.snapshot()["subgoals_completed"] == protocol.action_steps == 0
    assert protocol.snapshot()["first_success_step"] is None


def test_alternative_sequences_and_invalid_order():
    a = [[["on", "a", "p1"]], [["on", "b", "p2"]]]
    b = [[["on", "b", "p1"]], [["on", "a", "p2"]]]
    env = Env()
    env.env = Domain([a, b])
    protocol = LiberoMemEpisodeProtocol(env)
    env.env._satisfied_subgoals = [["and", *goal] for goal in b]
    assert protocol.snapshot()["success"]
    env.env._satisfied_subgoals = [["and", *a[1]]]
    with pytest.raises(ValueError, match="ordered prefix"):
        protocol.snapshot()
    with pytest.raises(ValueError, match="sequence-aware domain"):
        LiberoMemEpisodeProtocol(object())
    env.env = Domain([a, b[:1]])
    with pytest.raises(ValueError, match="equal lengths"):
        LiberoMemEpisodeProtocol(env)


def test_episode_initialization_reloads_compiled_xml_and_copies_saved_state():
    env = Env()
    initial = np.array([10, 11, 12, 13], dtype=np.float64)
    _, protocol = initialize_episode(env, initial, num_steps_wait=2)
    assert env.events == [
        "reset_env",
        ("reload_xml", "<compiled_model/>"),
        "reset_sim",
        "restore_state",
        "forward",
        "install_raw_state",
    ]
    # sim.reset() mutated the original array; the saved state must be a copy.
    np.testing.assert_array_equal(env.restored_state, np.arange(4))
    np.testing.assert_array_equal(env.initial, initial)
    assert env.calls == 2 and protocol.action_steps == env.env.tick_calls == 0
    assert not protocol.snapshot()["success"] and not env.env._overshot


@pytest.mark.parametrize("wait", [-1, True, 0.5])
def test_episode_initialization_rejects_invalid_wait(wait):
    env = Env()
    with pytest.raises(ValueError, match="nonnegative integer"):
        initialize_episode(env, np.zeros(4), num_steps_wait=wait)
    assert env.events == []


@pytest.fixture
def config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    task = TASK_BY_NUMBER[1]
    path = raw / task.filename
    entries = []
    with h5py.File(path, "w") as handle:
        for number in range(1, 25):
            group = handle.create_group(f"data/demo_{number}")
            actions = np.array(
                [[0, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, -1]], np.float64
            )
            initial = np.array([number, 0.1, 0.2, 0.3], np.float64)
            group.create_dataset("actions", data=actions)
            group.create_dataset("init_state", data=initial)
            group.attrs["model_file"] = "<mujoco/>"
            entries.append(
                {
                    "demo_id": f"demo_{number}",
                    "frames": 3,
                    "actions_sha256": vector_fingerprint(actions),
                    "initial_state_sha256": vector_fingerprint(initial),
                    "model_xml_sha256": hashlib.sha256(b"<mujoco/>").hexdigest(),
                }
            )
    stat = path.stat()
    record = {
        "number": 1,
        "name": task.name,
        "instruction": task.instruction,
        "raw_file": {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "raw_demos": entries,
        "full_demos": entries[:2],
    }
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"audit_version": 1, "status": "complete", "tasks": [record]}))
    repo = tmp_path / "libero-mem"
    bddl = repo / "libero/libero/bddl_files/libero_mem" / f"{task.name}.bddl"
    bddl.parent.mkdir(parents=True)
    bddl.write_text("test BDDL identity")
    return replay.ReplayConfig(
        raw_root=raw, audit=audit, libero_repo=repo, output_dir=tmp_path / "result", task_ids=(1,)
    )


def test_seen_and_unseen_initial_state_pools(config):
    assert replay.replay_plan(config)["jobs"][0]["demo"]["demo_id"] == "demo_24"
    jobs = replay.replay_plan(replace(config, num_episodes=2))["jobs"]
    assert [job["demo"]["demo_id"] for job in jobs] == ["demo_23", "demo_24"]
    assert replay.replay_plan(replace(config, selection="seen"))["jobs"][0]["demo"]["demo_id"] == "demo_1"
    assert replay.replay_plan(replace(config, demo_ids=(5,)))["jobs"][0]["demo"]["demo_id"] == "demo_5"
    for cfg in (
        replace(config, demo_ids=(1,)),
        replace(config, demo_ids=(3,)),
        replace(config, selection="seen", demo_ids=(24,)),
    ):
        with pytest.raises(ValueError, match="outside the"):
            replay.replay_plan(cfg)


def test_unseen_tail_overlap_and_duplicate_inventory_are_rejected(config):
    report = json.loads(config.audit.read_text())
    report["tasks"][0]["raw_demos"][-1]["initial_state_sha256"] = report["tasks"][0]["full_demos"][0][
        "initial_state_sha256"
    ]
    config.audit.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="overlaps Full"):
        replay.replay_plan(config)


def test_raw_expert_integrity_and_exact_program(config):
    job = replay.replay_plan(config)["jobs"][0]
    initial, program, start = replay.load_expert(job, 2)
    assert initial.tolist() == [24, 0.1, 0.2, 0.3]
    assert start == 1 and program.shape == (4, 7)
    np.testing.assert_array_equal(program[:2, 0], [0.1, 0.2])
    np.testing.assert_array_equal(program[2:], np.tile([0, 0, 0, 0, 0, 0, -1], (2, 1)))
    path = Path(job["raw_file"])
    stat = path.stat()
    with h5py.File(path, "r+") as handle:
        handle["data/demo_24/actions"][1, 0] = 0.5
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="vectors differ"):
        replay.load_expert(job, 2)


@pytest.mark.parametrize("overshoot", [False, True])
def test_replay_never_stops_on_intermediate_done_and_closes(config, overshoot):
    environments = []

    def factory(task, **kwargs):
        env = Env(overshoot=overshoot)
        environments.append((env, kwargs))
        return env

    result = replay.run_replay(replace(config, tail_steps=3), env_factory=factory)
    env, args = environments[0]
    assert result["status"] == "completed"
    assert result["environment_reset"] == RESET_PROTOCOL
    episode = result["episodes"][0]
    assert episode["action_steps"] == 5 and episode["first_success_step"] == 2
    assert episode["success"] is (not overshoot)
    assert result["expert_successes"] == int(not overshoot)
    assert env.calls == 25 and env.env.tick_calls == 5
    assert args["horizon"] == 26 and env.closed
    rows = [json.loads(line) for line in next(config.output_dir.glob("*.jsonl")).read_text().splitlines()]
    assert len(rows) == 5 and rows[0]["environment_done"] and not rows[0]["success"]
    assert json.loads((config.output_dir / "summary.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("dim, fail", [(4, True), (5, False)])
def test_errors_fail_run_instead_of_counting_policy_failures(config, dim, fail):
    env = Env(dim=dim, fail=fail)
    with pytest.raises((RuntimeError, ValueError)):
        replay.run_replay(config, env_factory=lambda *args, **kwargs: env)
    summary = json.loads((config.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["total_episodes"] == 0 and env.closed


def test_truncated_smoke_is_labeled(config):
    result = replay.run_replay(replace(config, max_steps=1), env_factory=lambda *args, **kwargs: Env())
    episode = result["episodes"][0]
    assert episode["truncated"] and episode["action_steps"] == 1
    assert not episode["success"]


def test_dry_run_imports_no_simulator_and_does_not_create_output(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run imported the simulator")

    monkeypatch.setattr(replay, "configure_runtime", forbidden)
    result = replay.run_replay(replace(config, dry_run=True))
    assert result["status"] == "dry_run" and result["episodes"] == 1
    assert not config.output_dir.exists()
    config.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        replay.run_replay(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_ids": (1, 1)},
        {"demo_ids": (1, 1)},
        {"num_episodes": 0},
        {"num_steps_wait": -1},
        {"max_steps": 0},
        {"tail_steps": -1},
    ],
)
def test_invalid_configuration(config, kwargs):
    with pytest.raises(ValueError):
        replay.replay_plan(replace(config, **kwargs))


def test_existing_libero_environment_helper_preserves_defaults_and_accepts_options(tmp_path, monkeypatch):
    from examples.Libero.eval import utils

    calls = []

    class RenderEnv:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def seed(self, seed):
            self.seed_value = seed

    package, module, envs = ModuleType("libero"), ModuleType("libero.libero"), ModuleType("libero.libero.envs")
    module.get_libero_path = lambda key: str(tmp_path)
    envs.OffScreenRenderEnv = RenderEnv
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    task = SimpleNamespace(language="instruction", problem_folder="libero_mem", bddl_file="task.bddl")
    env, language = utils.get_libero_env(task)
    assert env.seed_value == 0 and language == "instruction"
    assert calls[0] == {
        "bddl_file_name": str(tmp_path / "libero_mem/task.bddl"),
        "camera_heights": 256,
        "camera_widths": 256,
        "horizon": 1000,
    }
    utils.get_libero_env(task, horizon=123, use_camera_obs=False, camera_depths=False)
    assert calls[1]["horizon"] == 123 and calls[1]["use_camera_obs"] is False
    assert calls[1]["camera_depths"] is False


def test_existing_video_helper_keeps_default_fps_and_allows_20fps(tmp_path, monkeypatch):
    from examples.Libero.eval import utils

    rates = []

    class Writer:
        def append_data(self, image):
            assert image.shape == (2, 4, 3)

        def close(self):
            pass

    def writer(path, fps):
        rates.append(fps)
        return Writer()

    monkeypatch.setattr(utils.imageio, "get_writer", writer)
    image = np.zeros((2, 2, 3), np.uint8)
    utils.save_rollout_video([image], [image], 0, True, "test", output_dir=tmp_path)
    utils.save_rollout_video([image], [image], 1, True, "test", output_dir=tmp_path, fps=20)
    assert rates == [30, 20]
