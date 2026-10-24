"""Scientific invariants for VFD scoring, calibration splits, and saved plots."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.libero_vfd import (
    SnapshotVFDPolicy,
    calibrate,
    detection_metrics,
    flow_grid,
    get_threshold,
    integrate_vfd,
)


def test_velocity_difference_uses_shared_states_and_ot_weights():
    noise = torch.zeros(2, 3, 9)
    scorer_states = []

    def actor(x, time):
        return torch.ones_like(x)

    def scorer(x, time):
        scorer_states.append((time, x.clone()))
        return torch.zeros_like(x)

    terminal, scores = integrate_vfd(actor, scorer, noise, 4)
    torch.testing.assert_close(terminal, torch.ones_like(noise))
    expected = 3 * 7 * sum(flow_grid(4).values())
    torch.testing.assert_close(scores, torch.full((2,), expected))
    assert [t for t, _ in scorer_states] == [0.25, 0.5, 0.75]
    for time, state in scorer_states:
        torch.testing.assert_close(state, torch.full_like(noise, time))


def test_padding_disagreement_is_not_uncertainty():
    noise = torch.zeros(2, 4, 32)

    def scorer(x, time):
        result = torch.zeros_like(x)
        result[..., 7:] = 1000
        return result

    _, scores = integrate_vfd(lambda x, time: torch.zeros_like(x), scorer, noise, 8)
    assert torch.equal(scores, torch.zeros(2))


@pytest.mark.parametrize("times", [[1.0], [-0.1], [0.2], [], [0.0]])
def test_reject_invalid_or_uninformative_flow_times(times):
    with pytest.raises(ValueError):
        flow_grid(8, times)


def test_selected_flow_time_uses_spacing_to_endpoint():
    assert flow_grid(8, [0.5, 0.75]) == {4: 0.25, 6: 0.75}


@pytest.fixture
def tiny_policy():
    # The real inference method runs against a tiny deterministic head, avoiding
    # heavyweight checkpoints while checking positions, time buckets and slicing.
    from transformers.feature_extraction_utils import BatchFeature

    head = SimpleNamespace(
        action_dim=7,
        action_horizon=2,
        num_inference_timesteps=8,
        num_timestep_buckets=1000,
        config=SimpleNamespace(action_dim=7, action_horizon=2, add_pos_embed=True),
        future_tokens=torch.nn.Embedding(2, 7),
        position_embedding=torch.nn.Embedding(2, 7),
        state_encoder=lambda state, ids: state,
        action_encoder=lambda x, times, ids: x + times[:, None, None] / 1000,
        action_decoder=lambda x, ids: x,
        model=lambda hidden_states, encoder_hidden_states, timestep: hidden_states * 0.1
        + encoder_hidden_states.mean(),
        process_backbone_output=lambda value: value,
    )
    visual = BatchFeature(data={"backbone_features": torch.ones(1, 2, 7)})
    action_input = BatchFeature(data={"embodiment_id": torch.tensor([0]), "state": torch.ones(1, 1, 7)})
    policy = SimpleNamespace(
        model=SimpleNamespace(
            action_head=head, prepare_input=lambda inputs: (None, action_input), backbone=lambda inputs: visual
        )
    )
    return policy, visual, action_input


def test_actor_integration_matches_original_action_head(tiny_policy):
    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

    policy, visual, action_input = tiny_policy
    head = policy.model.action_head
    velocity, features = SnapshotVFDPolicy.velocity_function(policy, {})
    torch.manual_seed(17)
    reference = FlowmatchingActionHead.get_action(head, visual, action_input)["action_pred"]
    torch.manual_seed(17)
    noise = torch.randn(1, 2, 7)
    actual, scores = integrate_vfd(velocity, velocity, noise, 8)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.equal(scores, torch.zeros(1))


def test_extra_uncertainty_samples_preserve_actor_actions_and_rng(tiny_policy, monkeypatch):
    from contextlib import nullcontext

    policy, _, _ = tiny_policy
    policy._check_state_is_batched = lambda observation: False
    policy.apply_transforms = lambda observation: {}
    policy._get_unnormalized_action = lambda action: {"action.x": action.cpu().numpy()}
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monitor = SnapshotVFDPolicy.__new__(SnapshotVFDPolicy)
    monitor.actor = monitor.scorer = policy
    monitor.num_steps = 8
    monitor.flow_times = None
    chunks, rngs = [], []
    for count in [1, 4]:
        monitor.num_samples = count
        monitor.generator = torch.Generator().manual_seed(99)
        torch.manual_seed(17)
        chunk, score, samples = monitor.get_action_with_vfd({"state.x": np.zeros((1, 1))})
        chunks.append(chunk["action.x"])
        rngs.append(torch.get_rng_state().clone())
        assert score == 0 and samples.shape == (count,)
    np.testing.assert_array_equal(chunks[0], chunks[1])
    assert torch.equal(rngs[0], rngs[1])


def write_run(root, values, successes, initial_start=0, smoke=False):
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"score_signature": {"test": "same-pair"}, "smoke_only": smoke, "status": "completed"})
    )
    task = root / "task00"
    task.mkdir()
    for index, (value, success) in enumerate(zip(values, successes), start=initial_start):
        path = task / f"episode{index:03d}"
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "suite": "libero_spatial",
                    "task_id": 0,
                    "initial_state_index": index,
                    "success": success,
                    "complete": True,
                }
            )
        )
        np.savez(path.with_suffix(".npz"), uncertainty=np.array([0, value], dtype=float))
    return root


def test_calibration_uses_successful_rollout_maxima_and_refuses_overlap(tmp_path):
    root = write_run(tmp_path / "calibration", list(range(1, 21)) + [1000], [True] * 20 + [False])
    calibration = calibrate([root], tmp_path / "threshold.json")
    assert calibration["thresholds"]["libero_spatial/task00"]["threshold"] == pytest.approx(18.1)
    signature = {"test": "same-pair"}
    metadata = {"suite": "libero_spatial", "task_id": 0, "initial_state_index": 25}
    assert get_threshold(calibration, signature, metadata) == pytest.approx(18.1)
    metadata["initial_state_index"] = 20  # Even failed calibration candidates cannot leak into test.
    with pytest.raises(ValueError, match="overlap"):
        get_threshold(calibration, signature, metadata)
    with pytest.raises(ValueError, match="does not match"):
        get_threshold(calibration, {"test": "other-pair"}, metadata)


def test_no_calibration_from_too_few_successes_or_smoke_runs(tmp_path):
    root = write_run(tmp_path / "few", [1, 2], [True, False])
    with pytest.raises(ValueError, match="successful rollouts"):
        calibrate([root], tmp_path / "threshold.json")
    smoke = write_run(tmp_path / "smoke", [1, 2], [True, True], smoke=True)
    with pytest.raises(ValueError, match="smoke"):
        calibrate([smoke], tmp_path / "smoke_threshold.json", min_successes=2)
    assert not (tmp_path / "threshold.json").exists()


def test_one_class_metrics_are_undefined_and_threshold_crossing_is_strict():
    result = detection_metrics([(True, np.array([0, 1]), 1)])
    assert result["TNR"] == 1
    assert result["TPR"] is None and result["AUROC"] is None
    result = detection_metrics([(True, np.array([0, 1]), 1), (False, np.array([2, 0]), 1)])
    assert result["balanced_accuracy"] == 1
    assert result["TWA"] == 1
    assert result["mean_detection_time"] == 0


def test_auroc_ties_and_reversed_ranking():
    tied = detection_metrics([(True, np.array([1]), 0.5), (False, np.array([1]), 0.5)])
    assert tied["AUROC"] == 0.5
    reversed_result = detection_metrics([(True, np.array([2]), 1), (False, np.array([0]), 1)])
    assert reversed_result["AUROC"] == 0


def test_source_only_global_threshold_can_transfer_without_task_thresholds(tmp_path):
    root = write_run(tmp_path / "source", [1, 2], [True, True])
    calibration = calibrate([root], tmp_path / "global.json", min_successes=2, scope="global")
    target = {"suite": "libero_goal", "task_id": 3, "initial_state_index": 0}
    assert get_threshold(calibration, {"test": "same-pair"}, target) == pytest.approx(1.9)


def test_plot_and_metrics_from_saved_data_without_a_gpu(tmp_path):
    from scripts.eval_libero_vfd import plot_run

    calibration_run = write_run(tmp_path / "calibration", [1, 2], [True, True])
    calibration = calibrate([calibration_run], tmp_path / "threshold.json", min_successes=2)
    test = write_run(tmp_path / "test", [0.5, 5], [True, False], initial_start=10)
    result = plot_run(test, calibration, normalize=True)
    assert result["metrics_by_task"]["libero_spatial/task00"]["balanced_accuracy"] == 1
    assert len(list((test / "plots").glob("*.png"))) == 2
    assert len(list((test / "plots").glob("*.pdf"))) == 2
    assert result["episodes"][1]["first_alarm_step"] == 1


def test_record_dry_run_does_not_create_output(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    actor, scorer = tmp_path / "actor", tmp_path / "scorer"
    actor.mkdir()
    scorer.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/eval_libero_vfd.py"
    output = tmp_path / "not_created"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "record",
            "--actor",
            str(actor),
            "--scorer",
            str(scorer),
            "--suite",
            "spatial",
            "--task-ids",
            "0",
            "--output-dir",
            str(output),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["task_ids"] == [0]
    assert not output.exists()


@pytest.mark.parametrize("interrupt_second", [False, True])
def test_record_plots_each_episode_before_the_next_and_preserves_interrupted_data(
    tmp_path, monkeypatch, interrupt_second
):
    """Exercise the actual record loop with fake inference/simulation, no GPU."""
    import sys
    from types import ModuleType

    import matplotlib.figure
    import transformers

    from examples.Libero.eval import utils
    from scripts import eval_libero_vfd as evaluation

    actor, scorer = tmp_path / "actor", tmp_path / "scorer"
    actor.mkdir()
    scorer.mkdir()
    output = tmp_path / "recording"
    saved_figures = []
    original_savefig = matplotlib.figure.Figure.savefig

    def savefig(figure, path, *args, **kwargs):
        saved_figures.append(str(path))
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    monkeypatch.setattr(evaluation, "prepare_libero", lambda *args: None)
    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    class Monitor:
        signature = {"test": "same-pair"}

        def __init__(self, *args):
            pass

        def get_action_with_vfd(self, observation):
            return (
                {f"action.{key}": np.zeros((16, 1)) for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
                1.0,
                np.array([1.0]),
            )

    monkeypatch.setattr(evaluation, "SnapshotVFDPolicy", Monitor)

    class Env:
        resets = 0
        closed = False

        def seed(self, seed):
            pass

        def reset(self):
            if self.resets:
                assert (output / "plots/task00_episode000.png").is_file()
                assert (output / "plots/task00_episode000.pdf").is_file()
                summary = json.loads((output / "summary.json").read_text())
                assert summary["status"] == "running"
                assert len(summary["episodes"]) == 1
            self.resets += 1

        def set_init_state(self, state):
            return {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
                "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            }

        def step(self, action):
            if interrupt_second and self.resets == 2:
                raise KeyboardInterrupt("test interruption")
            return {}, 0, True, {}

        def close(self):
            self.closed = True

    env = Env()
    task = SimpleNamespace(language="test task")
    suite = SimpleNamespace(get_task=lambda index: task, get_task_init_states=lambda index: [0, 1])
    package = ModuleType("libero")
    module = ModuleType("libero.libero")
    module.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    monkeypatch.setattr(utils, "get_libero_env", lambda *args, **kwargs: (env, "test task"))
    args = SimpleNamespace(
        actor=actor,
        scorer=scorer,
        suite="spatial",
        trials=2,
        num_action_samples=1,
        snapshot_stride=20,
        initial_state_start=0,
        max_steps=1,
        wait_steps=0,
        flow_times=None,
        denoising_steps=8,
        task_ids="0",
        output_dir=output,
        calibration=None,
        seed=0,
        uncertainty_seed=1234,
        renderer="osmesa",
        data_config=evaluation.DATA_CONFIG,
        gpu_memory_limit_gb=0,
        verify_actor=False,
        dry_run=False,
        save_video=False,
    )
    if interrupt_second:
        with pytest.raises(KeyboardInterrupt, match="test interruption"):
            evaluation.record(args)
    else:
        evaluation.record(args)
    expected = 1 if interrupt_second else 2
    assert len(saved_figures) == expected * 2  # One PNG and one PDF per completed episode, no redraw.
    assert len(set(saved_figures)) == len(saved_figures)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["episodes"]) == expected
    assert summary["status"] == ("interrupted" if interrupt_second else "completed")
    assert env.closed
    if interrupt_second:
        partial = output / "task00/episode001.json"
        assert not json.loads(partial.read_text())["complete"]
        assert partial.with_suffix(".npz").is_file()
        assert not (output / "plots/task00_episode001.png").exists()
