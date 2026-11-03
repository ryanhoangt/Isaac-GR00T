"""Scientific pairing, denominator and resume guards without a GPU checkpoint."""

import csv
import json
from argparse import Namespace

import numpy as np
import pytest

from examples.Libero.pilot2.recovery import build_stages, completed, preflight, recover
from examples.Libero.pilot2.evaluate import evaluation_settings
from examples.Libero.pilot2.recovery_report import audit_pairs, summarize_recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def recipe():
    return dict(
        variant="dual",
        split_id="split",
        base_model="base",
        seed=42,
        batch_size=32,
        num_gpus=1,
        max_steps=5000,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        scheduler="cosine",
        normalization="train_only",
        action_horizon=16,
        denoising_steps_eval=4,
    )


def args_fixture(tmp_path):
    checkpoint = tmp_path / "dual/checkpoint-5000"
    write(checkpoint / "experiment_cfg/pilot2.json", recipe())
    write(checkpoint / "trainer_state.json", {"global_step": 5000})
    write(checkpoint / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    split = tmp_path / "data/split.json"
    reference = [{"episode_index": i, "task_id": i // 3} for i in range(30)]
    write(split, {"split_id": "split", "reference": reference})
    index = tmp_path / "reference.npz"
    np.savez(index, episodes=np.arange(30), tasks=np.arange(30) // 3)
    write(
        index.with_suffix(".json"),
        {
            "split_id": "split",
            "encoder": "base",
            "frames": 30,
            "reference": str(split.parent / "reference"),
            "stride": 1,
        },
    )
    libero = tmp_path / "LIBERO"
    (libero / "libero/libero/bddl_files").mkdir(parents=True)
    return Namespace(
        checkpoint=checkpoint,
        index=index,
        split=split,
        output=tmp_path / "output",
        device="cuda",
        task_ids=list(range(10)),
        triggers=[25, 60],
        trials=20,
        initial_start=5,
        max_steps=1000,
        seed=0,
        video_trials=1,
        libero_repo=libero,
        smoke=False,
        dry_run=True,
        resume=False,
        wandb_mode="disabled",
        wandb_entity=None,
    )


def paired_fixture(tmp_path):
    inputs = []
    for strategy in "ABC":
        root = tmp_path / strategy
        inputs.append(root)
        settings = dict(
            checkpoint="/checkpoint",
            split="/split",
            index="/index",
            task_ids=[0],
            trials=2,
            initial_start=5,
            max_steps=1000,
            seed=0,
            mode="language",
            strategy=strategy,
            trigger=25,
            save_video=False,
            video_trials=None,
        )
        names = ["task00_init005", "task00_init006"]
        manifest = dict(
            status="completed",
            recovery_protocol_version=1,
            task_mapping_version=1,
            split_id="split",
            task_mapping={"0": {"reference_task_id": 5}},
            reference_episode_tasks={"15": 5},
            episodes=names,
            settings=settings,
            control={"horizon": 16, "actions_per_query": 16, "denoising_steps": 4},
            training_recipe=recipe(),
            perturbation_protocol={"duration_steps": 4},
        )
        write(root / "manifest.json", manifest)
        for initial, name in zip((5, 6), names):
            eligible = initial == 5
            predictions = []
            for step in [0, 16, 29, 45, 61] if eligible else [0]:
                mode = "both" if strategy == "C" and 29 <= step < 61 else "language"
                phase = "copy" if strategy == "B" and step == 29 else mode
                p = dict(
                    step=step,
                    mode=mode,
                    phase=phase,
                    latency_ms=None if phase == "copy" else 5,
                    retrieval_ms=2 if phase in ("copy", "both") else 0,
                )
                if phase in ("copy", "both"):
                    p.update(episode=15, frame=10, goal_frame=26)
                predictions.append(p)
            row = dict(
                task_id=0,
                initial_state_index=initial,
                reference_task_id=5,
                strategy=strategy,
                trigger=25,
                seed=initial * 1000,
                initial_state_sha256=f"initial{initial}",
                recovery_eligible=eligible,
                perturbation_applied_steps=4 if eligible else 0,
                perturbation_target_m=[0.06, 0, 0],
                perturbation_commands=[{"step": i, "added_command": [0.3, 0, 0]} for i in range(25, 29)]
                if eligible
                else [],
                pre_recovery_sim_sha256="same-sim" if eligible else None,
                pre_recovery_ee_sha256="same-ee" if eligible else None,
                pre_recovery_actions_sha256="same-commands" if eligible else None,
                action_steps=(100 if strategy == "C" else 1000) if eligible else 10,
                success=strategy == "C" or not eligible,
                complete=True,
                smoke_only=False,
                predictions=predictions,
                subtask_predicate_fraction=0.5,
                goal_ee_proxy_rate=None,
                contraction_m={"0": 0.1, "8": 0.08, "16": 0.06, "32": 0.03} if eligible else {},
            )
            write(root / f"{name}.json", row)
            (root / f"{name}.npz").write_bytes(b"fixture")
    return inputs


def test_large_plan_runs_only_six_recovery_conditions_and_is_read_only(tmp_path, capsys):
    args = args_fixture(tmp_path)
    recover(args)
    text = capsys.readouterr().out
    assert '"rollouts": 1200' in text
    assert text.count(" -m examples.Libero.pilot2 evaluate ") == 6
    assert "--initial-start 5" in text and "--trials 20" in text
    assert "pilot2 train " not in text and "pilot2 index " not in text
    assert not args.output.exists()


def test_resume_refuses_changed_settings_before_starting_a_model(tmp_path):
    args = args_fixture(tmp_path)
    args.resume, args.dry_run = True, False
    write(args.output / "recovery_plan.json", {"settings": {"changed": True}})
    with pytest.raises(ValueError, match="changed"):
        recover(args)


def test_interrupted_stage_is_marked_failed_and_resume_reuses_its_directory(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    args.dry_run = False
    stage, output, _, expected = build_stages(args)[0][0]
    module = __import__("examples.Libero.pilot2.recovery", fromlist=["_run"])

    def interrupt(command, log, on_line=None):
        write(
            output / "manifest.json",
            {"settings": expected, "recovery_protocol_version": 1, "status": "running", "episodes": []},
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        recover(args)
    plan = json.loads((args.output / "recovery_plan.json").read_text())
    assert plan["status"] == plan["stages"][stage]["status"] == "failed"
    args.resume = True

    def resumed(command, log, on_line=None):
        assert "--resume" in command
        raise RuntimeError("Reached partial stage resume")

    monkeypatch.setattr(module, "_run", resumed)
    with pytest.raises(RuntimeError, match="Reached partial stage resume"):
        recover(args)


def test_paired_audit_excludes_trials_finishing_before_full_perturbation(tmp_path):
    inputs = paired_fixture(tmp_path)
    _, counts = audit_pairs(inputs)
    assert counts["25"] == dict(
        paired_trials=2, fully_perturbed=1, excluded_before_full_perturbation=1, common_observed_32_step_cohort=1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pre_recovery_sim_sha256", "different-objects"),
        ("pre_recovery_actions_sha256", "different-controls"),
        ("perturbation_target_m", [0.05, 0.02, 0]),
    ],
)
def test_audit_refuses_unpaired_start_states_commands_or_disturbances(tmp_path, field, value):
    inputs = paired_fixture(tmp_path)
    path = inputs[1] / "task00_init005.json"
    row = json.loads(path.read_text())
    row[field] = value
    write(path, row)
    with pytest.raises(ValueError, match="different"):
        audit_pairs(inputs)


def test_b_and_c_must_retrieve_same_milestone_from_same_start(tmp_path):
    inputs = paired_fixture(tmp_path)
    path = inputs[2] / "task00_init005.json"
    row = json.loads(path.read_text())
    for p in row["predictions"]:
        if p["mode"] == "both":
            p["frame"], p["goal_frame"] = 20, 36
    write(path, row)
    with pytest.raises(ValueError, match="different milestones"):
        audit_pairs(inputs)


def test_report_uses_paired_recovery_denominator_and_no_terminal_distance_padding(tmp_path):
    inputs = paired_fixture(tmp_path)
    path = inputs[2] / "task00_init005.json"
    row = json.loads(path.read_text())
    row["contraction_m"]["32"] = None
    write(path, row)
    output = tmp_path / "report"
    records = summarize_recovery(inputs, output)
    overall = {row["condition"]: row for row in records if row["task_id"] == "all"}
    assert overall["strategy_A_t25"]["success_pct"] == 50
    assert overall["strategy_A_t25"]["recovery_pct"] == 0
    assert overall["strategy_C_t25"]["recovery_pct"] == 100
    assert json.loads((output / "pairing_audit.json").read_text())["25"]["common_observed_32_step_cohort"] == 0
    with (output / "paired_contraction.csv").open() as stream:
        assert all(row["paired_cohort_n"] == "0" for row in csv.DictReader(stream))
    assert (output / "recovery_success.png").is_file()


def test_completed_run_can_be_skipped_but_changed_settings_are_rejected(tmp_path):
    inputs = paired_fixture(tmp_path)
    settings = json.loads((inputs[0] / "manifest.json").read_text())["settings"]
    assert completed(inputs[0], settings)
    with pytest.raises(ValueError, match="different settings"):
        completed(inputs[0], {**settings, "seed": 7})


def test_1x_settings_remain_compatible_and_scaled_jobs_forward_the_level(tmp_path):
    args = args_fixture(tmp_path)
    legacy = preflight(args)
    assert "perturb_scale" not in legacy
    assert all("--perturb-scale" not in command for _, _, command, _ in build_stages(args)[0])
    args.perturb_scale = 4.0
    scaled = preflight(args)
    assert scaled["perturb_scale"] == 4.0
    assert {k: v for k, v in scaled.items() if k != "perturb_scale"} == legacy
    for _, _, command, expected in build_stages(args)[0]:
        assert command[command.index("--perturb-scale") + 1] == "4.0"
        assert expected["perturb_scale"] == 4.0


def test_evaluation_resume_canonicalizes_default_scale_only(tmp_path):
    args = Namespace(
        checkpoint=tmp_path / "checkpoint", split=tmp_path / "split", index=tmp_path / "index", resume=True
    )
    old = evaluation_settings(args)
    args.perturb_scale = 1.0
    assert evaluation_settings(args) == old
    args.perturb_scale = 2.0
    assert evaluation_settings(args)["perturb_scale"] == 2.0
    for value in (0, -1, float("nan"), float("inf")):
        args.perturb_scale = value
        with pytest.raises(ValueError, match="finite and positive"):
            evaluation_settings(args)


def test_recovery_audit_validates_scaled_bounds_and_rejects_mixed_levels(tmp_path):
    inputs = paired_fixture(tmp_path)
    for root in inputs:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["settings"]["perturb_scale"] = 4.0
        write(path, manifest)
        for name in manifest["episodes"]:
            path = root / f"{name}.json"
            row = json.loads(path.read_text())
            row["perturbation_scale"] = 4.0
            row["perturbation_base_target_m"] = row["perturbation_target_m"]
            row["perturbation_target_m"] = (np.array(row["perturbation_target_m"]) * 4).tolist()
            write(path, row)
    audit_pairs(inputs)
    report = tmp_path / "scaled_report"
    summarize_recovery(inputs, report)
    assert json.loads((report / "disturbance_metadata.json").read_text())["scale"] == 4.0
    with (report / "disturbance_delivery.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(np.isclose(float(r["requested_norm_cm"]), 24) for r in rows)
    path = inputs[2] / "task00_init005.json"
    row = json.loads(path.read_text())
    row["perturbation_scale"] = 2.0
    write(path, row)
    with pytest.raises(ValueError, match="scale disagrees"):
        audit_pairs(inputs)
