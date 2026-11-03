from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gr00t.rag.diagnosis_protocol import (
    audit_recovery,
    disturbance,
    flush_queue,
    make_cases,
    perturb_action,
    phase,
    query_seed,
)
from gr00t.rag.dual_mode_diagnosis import DiagnosticPolicy, support_scores
from gr00t.rag.dual_mode import DualModeActionHead
from tests.test_dual_mode_ravla import config


def test_schedule_and_zero_sham_share_query_boundaries():
    for strategy in "ABC":
        case = dict(kind="recovery", strategy=strategy, trigger=60, scale=1)
        assert [step for step in range(110) if phase(case, step) == "perturb"] == [60, 61, 62, 63]
        assert flush_queue(case, 64)
        assert phase(case, 63) == "perturb"
        assert phase(case, 64) == {"A": "action", "B": "copy", "C": "goal"}[strategy]
        assert phase(case, 96) == "action"
        assert phase(case, 95) == ("goal" if strategy == "C" else "action")
    assert len(make_cases("standard")) == 3
    assert len(make_cases("recovery")) == 6
    assert len(make_cases("all")) * 10 * 5 == 450


def test_disturbance_scaling_and_sham():
    base, target = disturbance(123, 1)
    same, zero = disturbance(123, 0)
    np.testing.assert_array_equal(base, same)
    np.testing.assert_array_equal(target, base)
    assert 0.05 <= np.linalg.norm(base) <= 0.08
    assert np.all(np.abs(base) <= 0.07)
    assert not zero.any()
    assert query_seed(1, 2, 3, 64) == query_seed(1, 2, 3, 64)
    assert query_seed(1, 2, 3, 64) != query_seed(1, 2, 3, 80)


def test_command_perturbation_preserves_orientation_gripper_and_logs_clipping():
    controller = SimpleNamespace(
        use_delta=True,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-0.05 * np.ones(6),
        output_max=0.05 * np.ones(6),
    )
    nominal = np.array([0, 0, 0, 0.1, -0.2, 0.3, 1])
    target = np.array([0.06, 0, 0])
    command, log = perturb_action(nominal, target, controller)
    np.testing.assert_array_equal(command[3:], nominal[3:])
    np.testing.assert_allclose(np.asarray(log["delivered_offset_m"]) * 4, target)
    assert not log["clipped"]
    _, clipped = perturb_action(nominal, target * 10, controller)
    assert clipped["clipped"]
    sham, log = perturb_action(nominal, target * 0, controller)
    np.testing.assert_array_equal(sham, nominal)
    assert not log["clipped"]


def test_fp32_support_renormalizes_bf16_without_changing_selected_index():
    query = torch.tensor([0.3, 0.7, 0.2], dtype=torch.bfloat16)
    key = query.clone()
    result = support_scores(query, key, torch.tensor(0.99609375))
    assert result["support_fp32"] == pytest.approx(0, abs=1e-6)
    assert result["support_bf16"] == 0.00390625


def test_cached_goal_probe_is_repeatable_and_masks_actions():
    torch.manual_seed(4)
    head = DualModeActionHead(config()).eval()
    entry = {
        "backbone_features": torch.randn(3, 5, 32),
        "backbone_attention_mask": torch.ones(3, 5),
        "action": torch.randn(3, 16, 8),
        "state": torch.randn(3, 1, 8),
        "embodiment_id": torch.zeros(3, dtype=torch.long),
    }

    def unapply(value):
        keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        return {f"action.{key}": value["action"][:, i : i + 1].numpy() for i, key in enumerate(keys)}

    policy = SimpleNamespace(
        vla=SimpleNamespace(action_head=head), memory={"task": entry}, transforms=SimpleNamespace(unapply=unapply)
    )
    api = DiagnosticPolicy(policy)
    first = api.probe_cached("task", 0, 1, 42)
    entry["action"].mul_(1000)
    second = api.probe_cached("task", 0, 1, 42)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, api.probe_cached("task", 0, 2, 42))


def test_pairing_audit_rejects_different_physics_or_reference():
    common = dict(
        task_id=0,
        initial_index=5,
        initial_sha256="a",
        pre_perturb_sim_sha256="b",
        recovery_sim_sha256="c",
        prefix_actions_sha256="d",
        prefix_ee_sha256="e",
        recovery_eligible=True,
        recovery_context=dict(anchor_index=3, goal_index=7, retrieved_episode=2),
    )
    episodes = [dict(deepcopy(common), case=dict(kind="recovery", strategy=s, trigger=60, scale=1)) for s in "ABC"]
    assert audit_recovery(episodes)["complete_ABC_groups"] == 1
    episodes[2]["recovery_sim_sha256"] = "changed"
    with pytest.raises(ValueError, match="pairing mismatch"):
        audit_recovery(episodes)
    episodes[2]["recovery_sim_sha256"] = "c"
    episodes[2]["recovery_context"]["goal_index"] = 9
    with pytest.raises(ValueError, match="reference mismatch"):
        audit_recovery(episodes)


def test_report_uses_common_observed_contraction_cohort_and_checks_resume(tmp_path):
    import csv
    import json

    from gr00t.rag.diagnosis_protocol import digest
    from scripts.rag.diagnosis_report import read_episodes, summarize

    manifest = dict(signature="test_run", expected_episodes=6)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    paths = []
    for initial in (5, 6):
        for strategy in "ABC":
            case = dict(name=f"recovery_{strategy}", kind="recovery", strategy=strategy, trigger=60, scale=1)
            directory = tmp_path / case["name"] / f"task00_init{initial:03d}"
            directory.mkdir(parents=True)
            record = dict(
                complete=True,
                case=case,
                task_id=0,
                initial_index=initial,
                initial_sha256="a",
                pre_perturb_sim_sha256="b",
                recovery_sim_sha256="c",
                prefix_actions_sha256="d",
                prefix_ee_sha256="e",
                recovery_eligible=True,
                recovery_context=dict(anchor_index=3, goal_index=7, retrieved_episode=2),
                success=(strategy == "C"),
                smoke_only=False,
                queries=[
                    dict(
                        control_step=64,
                        phase="action",
                        query_ms=10,
                        rpc_ms=12,
                        support_fp32=0.02,
                        support_bf16=0.0234375,
                    )
                ],
                contraction={str(delta): 0.05 for delta in (0, 8, 16, 32)},
                signature=digest(dict(run=manifest["signature"], case=case, task=0, initial=initial)),
            )
            if initial == 6 and strategy == "C":
                record["contraction"]["32"] = None  # Early success: do not pad the missing timestamp.
            path = directory / "episode.json"
            path.write_text(json.dumps(record))
            paths.append(path)
    rows, audit = summarize(tmp_path)
    assert audit == dict(complete_ABC_groups=2, partial_groups=0)
    assert {r["condition"]: r["success_pct"] for r in rows}["recovery_C"] == 100
    contraction = list(csv.DictReader((tmp_path / "contraction.csv").open()))
    assert len(contraction) == 12
    assert {row["common_pairs"] for row in contraction} == {"1"}
    pairs = list(csv.DictReader((tmp_path / "paired_comparisons.csv").open()))
    assert all(row["pairs"] == "2" for row in pairs)
    assert (tmp_path / "support_traces.png").is_file()
    broken = json.loads(paths[0].read_text())
    broken["signature"] = "different_checkpoint_or_protocol"
    paths[0].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Invalid episode result"):
        read_episodes(tmp_path, manifest)


def test_reference_builder_diagnosis_flag_preserves_legacy_default():
    from scripts.rag.build_dual_mode_memory import Args

    assert not Args(output=__import__("pathlib").Path("unused")).diagnosis
