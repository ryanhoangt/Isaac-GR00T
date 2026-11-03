"""Scientific invariants for Pilot 2; no 3B weights, GPU or simulator needed."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.Libero.pilot2.data import make_split, pair_goal, perturb_state, sample_mode
from examples.Libero.pilot2.runtime import RecoverySchedule, disturbed_command, sample_disturbance
from examples.Libero.pilot2.debug import clone_features, tiny_batch, tiny_head


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_reference_split_is_per_task_and_never_sampled_for_training():
    episodes = [
        {"episode_index": task * 6 + i, "task_id": task, "length": 100 + i} for task in range(2) for i in range(6)
    ]
    train, reference = make_split(episodes)
    assert len(train) == len(reference) == 6
    train_map = {item["episode_index"]: item for item in train}
    heldout = {item["episode_index"] for item in reference}
    rng = np.random.default_rng(5)
    for item in train:
        for cross_probability in (0, 1):
            target, step, cross = pair_goal(
                train_map, item["episode_index"], 100, rng, cross_probability=cross_probability
            )
            assert target not in heldout
            assert train_map[target]["task_id"] == item["task_id"]
            assert 0 <= step < train_map[target]["length"]
            assert cross == bool(cross_probability)
            assert (target != item["episode_index"]) == cross


def test_masking_frequencies_and_raw_state_noise_leave_actions_unchanged():
    rng = np.random.default_rng(11)
    modes = [sample_mode(rng) for _ in range(20000)]
    for name, rate in (("language", 0.4), ("both", 0.4), ("goal", 0.2)):
        assert abs(modes.count(name) / len(modes) - rate) < 0.015
    raw = {
        f"state.{key}": np.zeros((1, 2 if key == "gripper" else 1), dtype=np.float32)
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    }
    raw["action.x"] = np.ones((16, 1), dtype=np.float32)
    changed = perturb_state(raw, rng)
    np.testing.assert_array_equal(changed["action.x"], raw["action.x"])
    assert np.any(changed["state.gripper"] != 0)
    assert np.all(raw["state.x"] == 0)


def test_real_flow_head_backward_and_masked_goal_invariance():
    torch.manual_seed(2)
    head = tiny_head()
    visual, action = tiny_batch()
    loss = head(clone_features(visual), action)["loss"]
    loss.backward()
    gradient = head.goal_conditioner.subgoal_view.grad
    assert torch.isfinite(loss) and gradient is not None and gradient.abs().sum() > 0
    head.eval()
    torch.manual_seed(19)
    reference = head.get_action(clone_features(visual), action)["action_pred"]
    changed = clone_features(visual)
    changed["goal_features"][0] = torch.randn_like(changed["goal_features"][0]) * 100
    changed["goal_features"][1] = torch.randn_like(changed["goal_features"][1]) * 100
    torch.manual_seed(19)
    actual = head.get_action(changed, action)["action_pred"]
    torch.testing.assert_close(reference[0], actual[0], rtol=0, atol=0)
    assert not torch.allclose(reference[1], actual[1])
    assert reference.shape == (3, 16, 32)


def test_padding_in_current_and_goal_cannot_change_valid_predictions():
    torch.manual_seed(20)
    head = tiny_head().eval()
    visual, action = tiny_batch()
    changed = clone_features(visual)
    changed["backbone_features"][:, -1] = 1000
    changed["goal_features"][:, -1] = -1000
    torch.manual_seed(7)
    expected = head.get_action(clone_features(visual), action)["action_pred"]
    torch.manual_seed(7)
    actual = head.get_action(changed, action)["action_pred"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiny_goal_checkpoint_reload_preserves_actions(tmp_path):
    torch.manual_seed(3)
    head = tiny_head().eval()
    torch.save(head.state_dict(), tmp_path / "head.pt")
    restored = tiny_head().eval()
    from gr00t.model.action_head.flow_matching_action_head import (
        FlowmatchingActionHead,
        FlowmatchingActionHeadConfig,
    )

    restored = FlowmatchingActionHead(FlowmatchingActionHeadConfig.from_dict(head.config.to_dict())).eval()
    restored.load_state_dict(torch.load(tmp_path / "head.pt", weights_only=True))
    visual, action = tiny_batch()
    torch.manual_seed(9)
    expected = head.get_action(clone_features(visual), action)["action_pred"]
    torch.manual_seed(9)
    actual = restored.get_action(clone_features(visual), action)["action_pred"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_language_only_minibatch_has_no_unused_goal_parameters():
    head = tiny_head()
    visual, action = tiny_batch()
    visual["goal_mode_id"].zero_()
    visual.pop("goal_features")
    visual.pop("goal_attention_mask")
    head(visual, action)["loss"].backward()
    assert head.goal_conditioner.subgoal_view.grad is not None
    assert head.goal_conditioner.null_goal.grad is not None


@pytest.mark.parametrize("trigger", [25, 60])
def test_recovery_switches_at_exact_steps_even_inside_a_chunk(trigger):
    schedule = RecoverySchedule("C", trigger)
    assert [schedule.phase(t) for t in range(trigger, trigger + 4)] == ["perturb"] * 4
    assert schedule.invalidate_queue(trigger + 4)
    assert [schedule.phase(t) for t in range(trigger + 4, trigger + 36)] == ["both"] * 32
    assert schedule.phase(trigger + 36) == "language"
    assert schedule.invalidate_queue(trigger + 36)
    copy = RecoverySchedule("B", trigger)
    assert copy.phase(trigger + 19) == "copy"
    assert copy.phase(trigger + 20) == "language"


def test_perturbation_units_and_norm_are_physical_not_normalized_actions():
    controller = SimpleNamespace(
        input_min=np.full(6, -1),
        input_max=np.ones(6),
        output_min=np.full(6, -0.05),
        output_max=np.full(6, 0.05),
        use_delta=True,
    )
    rng = np.random.default_rng(12)
    for _ in range(100):
        offset = sample_disturbance(rng)
        assert 0.05 <= np.linalg.norm(offset) <= 0.08
        assert np.abs(offset).max() <= 0.07
        action = disturbed_command(np.zeros(7), offset, controller)
        np.testing.assert_allclose(action[:3] * 0.05 * 4, offset)
        np.testing.assert_array_equal(action[3:], np.zeros(4))


def test_scaled_disturbances_share_rng_and_direction_and_obey_controller_clipping():
    values, next_random = [], []
    for scale in (1, 2, 4):
        rng = np.random.default_rng(5000)
        values.append(sample_disturbance(rng, scale))
        next_random.append(rng.random())
    np.testing.assert_array_equal(values[1], values[0] * 2)
    np.testing.assert_array_equal(values[2], values[0] * 4)
    assert next_random[0] == next_random[1] == next_random[2]
    controller = SimpleNamespace(
        input_min=np.full(6, -1), input_max=np.ones(6), output_min=np.full(6, -0.05), output_max=np.full(6, 0.05)
    )
    command = disturbed_command(np.zeros(7), values[2], controller)
    assert np.max(np.abs(command[:3])) <= 1
    assert np.linalg.norm(command[:3] * 0.05 * 4) < np.linalg.norm(values[2])
    for scale in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            sample_disturbance(np.random.default_rng(0), scale)


def test_reference_retrieval_is_task_restricted_and_clamps_forward_goal(tmp_path, monkeypatch):
    import pandas as pd
    from examples.Libero.pilot2.runtime import ReferenceBuffer
    from examples.Libero.pilot2 import runtime

    root = tmp_path / "reference"
    (root / "meta").mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for task in (0, 1):
        path = root / f"data/episode_{task:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "task_index": [task] * 4,
                "observation.state": [np.full(8, i, np.float32) for i in range(4)],
                "action": [np.full(7, i, np.float32) for i in range(4)],
            }
        ).to_parquet(path)
    path = tmp_path / "index.npz"
    np.savez(path, features=np.array([[0.0, 1.0], [1.0, 0.0]]), episodes=[0, 1], frames=[1, 0], tasks=[0, 1])
    path.with_suffix(".json").write_text(json.dumps({"split_id": "same", "reference": str(root)}))
    monkeypatch.setattr(runtime, "read_video_frame", lambda path, index: np.full((4, 4, 3), index, np.uint8))
    reference = ReferenceBuffer(path, "same", lambda image: np.array([1.0, 0.0]))
    result = reference.retrieve(np.zeros((4, 4, 3)), task=0)
    assert result["episode"] == 0  # task 1 has a better feature match, but is forbidden
    assert result["frame"] == 1 and result["goal_frame"] == 3
    np.testing.assert_array_equal(result["actions"][-1], np.full(7, 3))
    assert reference.distance(np.full(3, 2), task=0) == 0  # uses all reference states, not index stride
    with pytest.raises(ValueError, match="another"):
        ReferenceBuffer(path, "different", lambda image: np.array([1.0, 0.0]))


def test_eagle_skips_goal_pass_for_language_only_batches():
    from gr00t.model.backbone.eagle_backbone import EagleBackbone
    from transformers.feature_extraction_utils import BatchFeature

    backbone = EagleBackbone.__new__(EagleBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.tune_visual = False
    backbone.set_frozen_modules_to_eval_mode = lambda: None
    calls = []

    def encode(inputs, prefix="eagle_"):
        calls.append(prefix)
        return torch.zeros(2, 3, 8), torch.ones(2, 3)

    backbone.forward_eagle = encode
    output = backbone(BatchFeature(data={"goal_mode_id": torch.zeros(2, dtype=torch.long)}))
    assert calls == ["eagle_"] and "goal_features" not in output
    calls.clear()
    output = backbone(BatchFeature(data={"goal_mode_id": torch.tensor([0, 1])}))
    assert calls == ["eagle_", "goal_eagle_"] and "goal_features" in output


def test_report_does_not_count_unperturbed_or_smoke_episodes_as_recovery(tmp_path):
    from examples.Libero.pilot2.diagnostics import summarize

    root = tmp_path / "run"
    root.mkdir()
    manifest = {
        "status": "completed",
        "split_id": "same",
        "settings": {"mode": "language", "strategy": "A", "trigger": 25},
        "episodes": ["trial0", "trial1"],
        "task_mapping_version": 1,
        "task_mapping": {"0": {"reference_task_id": 5}},
        "reference_episode_tasks": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for trial in range(2):
        (root / f"trial{trial}.json").write_text(
            json.dumps(
                {
                    "task_id": 0,
                    "reference_task_id": 5,
                    "initial_state_index": trial,
                    "smoke_only": False,
                    "success": trial == 0,
                    "recovery_eligible": trial == 1,
                    "subtask_predicate_fraction": 0.5,
                    "goal_ee_proxy_rate": None,
                    "predictions": [],
                    "contraction_m": {"0": 0.1} if trial else {},
                }
            )
        )
    result = summarize([root], tmp_path / "report")
    assert result[0]["success_pct"] == 50
    assert result[0]["recovery_pct"] == 0
    assert result[0]["recovery_eligible"] == 1
    assert (tmp_path / "report/contraction.png").is_file()
    row = json.loads((root / "trial0.json").read_text())
    row["smoke_only"] = True
    (root / "trial0.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Smoke"):
        summarize([root], tmp_path / "bad_report")


def test_prepare_recomputes_normalization_without_reference_leakage(tmp_path):
    import pandas as pd
    from examples.Libero.pilot2.data import prepare

    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.image": {"dtype": "video"}},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/modality.json").write_text("{}")
    (source / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": t, "task": f"task{t}"}) + "\n" for t in range(10))
    )
    episodes = []
    for task in range(10):
        for local_id in range(5):
            idx = task * 5 + local_id
            value = -1000 if local_id < 3 else task * 10 + local_id
            episodes.append({"episode_index": idx, "tasks": [f"task{task}"], "length": 2})
            path = source / f"data/chunk-000/episode_{idx:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "task_index": [task] * 2,
                    "episode_index": [idx] * 2,
                    "index": [idx * 2, idx * 2 + 1],
                    "observation.state": [np.full(8, value, np.float32)] * 2,
                    "action": [np.full(7, value, np.float32)] * 2,
                }
            ).to_parquet(path)
            video = source / f"videos/chunk-000/observation.images.image/episode_{idx:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"placeholder; this test does not decode video")
    (source / "meta/episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in episodes))
    result = prepare(source, tmp_path / "split")
    assert len(result["train"]) == 20 and len(result["reference"]) == 30
    train_ids = {item["original_episode_index"] for item in result["train"]}
    assert not train_ids.intersection(item["original_episode_index"] for item in result["reference"])
    stats = json.loads((tmp_path / "split/train/meta/stats.json").read_text())
    assert stats["observation.state"]["min"] == [3] * 8
    assert stats["action"]["min"] == [3] * 7


def test_rollout_executes_chunks_and_recovery_uses_post_disturbance_observation():
    from examples.Libero.pilot2.evaluate import rollout
    from examples.Libero.pilot2.data import LANGUAGE_KEY

    class FakeEnv:
        def __init__(self, terminal_step=85):
            self.terminal_step = terminal_step
            controller = SimpleNamespace(
                input_min=np.full(6, -1),
                input_max=np.ones(6),
                output_min=np.full(6, -0.05),
                output_max=np.full(6, 0.05),
            )
            self.robots = [SimpleNamespace(controller=controller)]
            self.env = SimpleNamespace(
                parsed_problem={"goal_state": [["test"]]},
                _eval_predicate=lambda _: self.steps >= self.terminal_step,
            )
            self.reset()

        def reset(self):
            self.steps = 0
            self.position = np.zeros(3)

        def obs(self):
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            return {
                "robot0_eef_pos": self.position.copy(),
                "agentview_image": image,
                "robot0_eye_in_hand_image": image,
            }

        def set_init_state(self, _):
            return self.obs()

        def get_sim_state(self):
            return np.r_[self.position, self.steps]

        def step(self, action):
            self.steps += 1
            self.position += np.asarray(action)[:3] * 0.05
            return self.obs(), 0, self.steps >= self.terminal_step, {}

    class FakePolicy:
        model = SimpleNamespace(device=torch.device("cpu"))

        def __init__(self):
            self.modes = []

        def get_action(self, obs):
            self.modes.append(int(obs["goal_mode_id"][0]))
            return {
                f"action.{key}": np.zeros((16, 1), np.float32)
                for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
            }

    reference_tasks = []
    reference = SimpleNamespace(
        distance=lambda pos, task: reference_tasks.append(task) or float(np.linalg.norm(pos)),
        retrieve=lambda image, task: reference_tasks.append(task)
        or {
            "goal_image": image,
            "goal_position": np.zeros(3),
            "episode": 0,
            "frame": 0,
            "goal_frame": 16,
            "similarity": 1.0,
            "actions": np.zeros((16, 7)),
        },
    )
    adapter = SimpleNamespace(
        _process_observation=lambda obs, lang: {"state.x": obs["robot0_eef_pos"][:1, None], LANGUAGE_KEY: [lang]}
    )
    all_positions = []
    for strategy in ("A", "B", "C"):
        policy = FakePolicy()
        result, arrays = rollout(
            FakeEnv(),
            None,
            SimpleNamespace(language="task"),
            0,
            adapter,
            policy,
            reference,
            strategy=strategy,
            trigger=25,
            seed=123,
            reference_task_id=7,
        )
        assert result["task_id"] == 0 and result["reference_task_id"] == 7
        assert result["success"] and result["recovery_eligible"]
        assert result["perturbation_applied_steps"] == 4
        assert set(result["contraction_m"]) == {"0", "8", "16", "32"}
        prediction_steps = [item["step"] for item in result["predictions"]]
        assert prediction_steps == [0, 16, 29, 45, 61]
        assert sum(mode == 1 for mode in policy.modes) == (2 if strategy == "C" else 0)
        all_positions.append(arrays["ee_positions"][:30])
    np.testing.assert_array_equal(all_positions[0], all_positions[1])
    np.testing.assert_array_equal(all_positions[0], all_positions[2])
    assert set(reference_tasks) == {7}
    hashes = []
    for strategy in ("A", "B", "C"):
        result, arrays = rollout(
            FakeEnv(300),
            np.zeros(8),
            SimpleNamespace(language="task"),
            0,
            adapter,
            FakePolicy(),
            reference,
            strategy=strategy,
            trigger=60,
            seed=123,
            max_steps=110,
            reference_task_id=7,
        )
        assert result["perturbation_applied_steps"] == 4
        np.testing.assert_allclose(result["perturbation_command_offset_m"], result["perturbation_target_m"])
        assert arrays["executed_actions"].shape == arrays["nominal_actions"].shape == (110, 7)
        assert [p["step"] for p in result["predictions"]] == [0, 16, 32, 48, 64, 80, 96]
        assert sum(p["phase"] == "both" for p in result["predictions"]) == (2 if strategy == "C" else 0)
        assert sum(p["phase"] == "copy" for p in result["predictions"]) == (1 if strategy == "B" else 0)
        hashes.append(
            (
                result["initial_state_sha256"],
                result["pre_recovery_sim_sha256"],
                result["pre_recovery_actions_sha256"],
            )
        )
    assert hashes[0] == hashes[1] == hashes[2]
    levels = []
    for scale in (1, 2, 4):
        result, arrays = rollout(
            FakeEnv(300),
            np.zeros(8),
            SimpleNamespace(language="task"),
            0,
            adapter,
            FakePolicy(),
            reference,
            strategy="C",
            trigger=60,
            seed=5000,
            max_steps=110,
            reference_task_id=7,
            perturb_scale=scale,
        )
        levels.append(result)
        assert result["perturbation_applied_steps"] == 4
        assert [p["step"] for p in result["predictions"] if p["phase"] == "both"] == [64, 80]
    assert len({r["pre_perturb_sim_sha256"] for r in levels}) == 1
    assert len({r["pre_perturb_actions_sha256"] for r in levels}) == 1
    assert len({r["pre_perturb_ee_sha256"] for r in levels}) == 1
    np.testing.assert_array_equal(
        levels[2]["perturbation_target_m"], np.array(levels[0]["perturbation_target_m"]) * 4
    )
    assert any(p["clipped"] for p in levels[2]["perturbation_commands"])
