"""Tracking isolation and the requested 2K-step pilot defaults."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def tracking_args(variant):
    return SimpleNamespace(
        report_to="wandb",
        wandb_project="icl-vla-uncertainty",
        wandb_entity="ryan-nlp",
        wandb_group=None,
        run_name=None,
        wandb_mode="online",
        variant=variant,
        seed=42,
        max_steps=2,
    )


def test_variants_share_fixed_project_and_group_but_never_run_ids(tmp_path, monkeypatch):
    from examples.Libero.pilot2.tracking import start_tracking, finish_tracking

    ids = iter(["baseline-id", "dual-id"])
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id=kwargs["id"],
            name=kwargs["name"],
            project=kwargs["project"],
            entity=kwargs["entity"],
            url=f"https://wandb.ai/test/runs/{kwargs['id']}",
            finish=Mock(),
        )

    fake = SimpleNamespace(util=SimpleNamespace(generate_id=lambda: next(ids)), init=init)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for key in (
        "WANDB_PROJECT",
        "WANDB_MODE",
        "WANDB_DISABLED",
        "WANDB_LOG_MODEL",
        "WANDB_WATCH",
        "WANDB_DIR",
        "WANDB_RUN_GROUP",
        "WANDB_NAME",
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_ENTITY",
    ):
        monkeypatch.setenv(key, "old-inherited-value")
    for variant in ("baseline", "dual"):
        output = tmp_path / "pilot2_2k" / variant
        output.mkdir(parents=True)
        run = start_tracking(tracking_args(variant), {"max_steps": 2}, output)
        finish_tracking(run, output, exit_code=0)
        record = json.loads((output / "wandb_run.json").read_text())
        assert record["status"] == "finished" and record["mode"] == "online"
        run.finish.assert_called_once_with(exit_code=0)
    assert {call["project"] for call in calls} == {"icl-vla-uncertainty"}
    assert {call["group"] for call in calls} == {"pilot2_2k"}
    assert len({call["id"] for call in calls}) == 2
    assert {call["name"] for call in calls} == {"baseline_seed42", "dual_seed42"}
    assert all(call["resume"] == "never" and "smoke" in call["tags"] for call in calls)


def test_project_guard_and_tensorboard_do_not_create_other_wandb_runs(tmp_path):
    from examples.Libero.pilot2.tracking import start_tracking

    args = tracking_args("dual")
    args.wandb_project = "another-project"
    with pytest.raises(ValueError, match="fixed project"):
        start_tracking(args, {}, tmp_path)
    args.report_to = "tensorboard"
    assert start_tracking(args, {}, tmp_path) is None
    assert not (tmp_path / "wandb_run.json").exists()


def test_cli_defaults_to_2k_online_fixed_project_even_with_stale_env(tmp_path, monkeypatch):
    from examples.Libero.pilot2 import __main__, training

    seen = []
    monkeypatch.setenv("WANDB_PROJECT", "stale-other-project")
    monkeypatch.setattr(training, "train", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["pilot2", "train", "--variant", "baseline", "--output", str(tmp_path)])
    __main__.main()
    args = seen[0]
    assert args.max_steps == 2000 and args.save_steps == 1000
    assert args.batch_size * args.gradient_accumulation_steps == 128
    assert args.report_to == "wandb" and args.wandb_mode == "online"
    assert args.wandb_project == "icl-vla-uncertainty" and args.logging_steps == 10


def test_runner_respects_normalized_wandb_list(tmp_path, monkeypatch):
    from transformers import TrainingArguments
    from gr00t.experiment import runner
    from gr00t.data.dataset import LeRobotSingleDataset

    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset.tag = "new_embodiment"
    dataset._metadata = SimpleNamespace(model_dump=lambda mode: {})
    monkeypatch.setattr(runner, "DefaultDataCollator", lambda: None)
    monkeypatch.setattr(runner.TrainRunner, "create_trainer", lambda self, **kwargs: SimpleNamespace())
    monkeypatch.setenv("WANDB_PROJECT", "icl-vla-uncertainty")
    monkeypatch.setenv("WANDB_DIR", "")
    args = TrainingArguments(output_dir=str(tmp_path), report_to="wandb", use_cpu=True)
    assert args.report_to == ["wandb"]  # Transformers normalizes the string.
    run = runner.TrainRunner(model=None, training_args=args, train_dataset=dataset)
    assert run.training_args.report_to == ["wandb"]
    assert json.loads((tmp_path / "wandb_config.json").read_text())["project"] == "icl-vla-uncertainty"
    assert not (tmp_path / "runs").exists()
