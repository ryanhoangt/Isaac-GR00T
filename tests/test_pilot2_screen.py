"""CPU checks for the Pilot 2 screening launcher and resume guards."""

import json
from argparse import Namespace

import pytest

from examples.Libero.pilot2.screen import _completed, _preflight, screen


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _fixture(tmp_path):
    split = {
        "split_id": "fixture-split",
        "reference_per_task": 3,
        "reference": [
            {"task_id": task, "original_episode_index": task * 3 + episode}
            for task in range(10)
            for episode in range(3)
        ],
        "train": [{"original_episode_index": 100}],
    }
    split_path = tmp_path / "split.json"
    _write(split_path, split)
    root = tmp_path / "training"
    recipe = {
        "split_id": "fixture-split",
        "base_model": "base",
        "seed": 42,
        "batch_size": 32,
        "num_gpus": 1,
        "max_steps": 5000,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "scheduler": "cosine",
        "normalization": "train_only_min_max",
        "action_horizon": 16,
        "denoising_steps_eval": 4,
        "frozen_vision": True,
        "frozen_llm": True,
    }
    for variant in ("baseline", "dual"):
        checkpoint = root / variant / "checkpoint-5000"
        _write(checkpoint / "experiment_cfg/pilot2.json", {**recipe, "variant": variant})
        _write(checkpoint / "trainer_state.json", {"global_step": 5000})
        _write(checkpoint / "model.safetensors.index.json", {})
    libero = tmp_path / "LIBERO"
    (libero / "libero/libero/bddl_files").mkdir(parents=True)
    args = Namespace(
        split=split_path,
        training_root=root,
        checkpoint_step=5000,
        output=tmp_path / "screen",
        stride=1,
        states=1000,
        task_ids=list(range(10)),
        trials=5,
        initial_start=0,
        max_steps=1000,
        seed=0,
        video_trials=1,
        libero_repo=libero,
        smoke=False,
        recovery=False,
        reuse_screen=None,
        wandb_mode="disabled",
        wandb_entity=None,
        dry_run=True,
        resume=False,
    )
    return args, split


def test_screen_dry_run_is_read_only_and_uses_matched_checkpoints(tmp_path, capsys):
    args, _ = _fixture(tmp_path)
    screen(args)
    captured = capsys.readouterr().out
    assert not args.output.exists()
    assert "checkpoint-5000" in captured
    assert "--states 1000" in captured
    assert captured.count(" -m examples.Libero.pilot2 evaluate ") == 4


def test_preflight_rejects_mismatched_training_recipe(tmp_path):
    args, _ = _fixture(tmp_path)
    recipe_path = args.training_root / "dual/checkpoint-5000/experiment_cfg/pilot2.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["seed"] = 7
    _write(recipe_path, recipe)
    with pytest.raises(ValueError, match="seed"):
        _preflight(args)


def test_full_protocol_plan_adds_paired_recovery_conditions(tmp_path, capsys):
    args, _ = _fixture(tmp_path)
    args.recovery = True
    screen(args)
    captured = capsys.readouterr().out
    assert captured.count(" -m examples.Libero.pilot2 evaluate ") == 10
    for trigger in (25, 60):
        for strategy in "ABC":
            assert f"--strategy {strategy} --trigger {trigger}" in captured


def test_existing_index_requires_matching_split_encoder_and_stride(tmp_path):
    args, split = _fixture(tmp_path)
    index = tmp_path / "reference.npz"
    index.write_bytes(b"test")
    metadata = {"split_id": split["split_id"], "encoder": "base", "stride": 1}
    _write(index.with_suffix(".json"), metadata)
    assert _completed("index", index, split, {"base_model": "base"}, args)
    args.stride = 64
    with pytest.raises(ValueError, match="stride"):
        _completed("index", index, split, {"base_model": "base"}, args)


def test_reuse_links_valid_stages_and_starts_with_goal_eval(tmp_path, monkeypatch):
    args, split = _fixture(tmp_path)
    donor = tmp_path / "old_screen"
    donor.mkdir()
    (donor / "reference.npz").write_bytes(b"fixture")
    _write(donor / "reference.json", {"split_id": split["split_id"], "encoder": "base", "stride": 1})
    (donor / "sensitivity.csv").write_text("fixture")
    _write(donor / "sensitivity.json", {"split_id": split["split_id"], "requested_states": 1000})
    for mode in ("vanilla", "language"):
        names = [f"task{task:02d}_init{initial:03d}" for task in range(10) for initial in range(5)]
        manifest = {
            "status": "completed",
            "split_id": split["split_id"],
            "episodes": names,
            "settings": {
                "checkpoint": str(
                    args.training_root / ("baseline" if mode == "vanilla" else "dual") / "checkpoint-5000"
                ),
                "mode": mode,
                "strategy": "standard",
                "trigger": None,
                "trials": 5,
                "initial_start": 0,
                "max_steps": 1000,
                "seed": 0,
                "task_ids": list(range(10)),
            },
        }
        _write(donor / mode / "manifest.json", manifest)
        for name in names:
            _write(donor / mode / f"{name}.json", {})
            (donor / mode / f"{name}.npz").write_bytes(b"fixture")
    args.reuse_screen = donor
    args.dry_run = False
    module = __import__("examples.Libero.pilot2.screen", fromlist=["_run"])
    commands = []

    def stop_at_first_new_stage(command, log):
        commands.append(command)
        raise RuntimeError("Stop before loading a GPU model")

    monkeypatch.setattr(module, "_run", stop_at_first_new_stage)
    with pytest.raises(RuntimeError, match="Stop before"):
        screen(args)
    assert len(commands) == 1 and commands[0][commands[0].index("--mode") + 1] == "goal"
    assert (args.output / "reference.npz").is_symlink()
    assert (args.output / "vanilla").is_symlink()
    assert not (args.output / "goal").exists()
