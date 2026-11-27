"""Exact co-training batches, fold validation, seed/normalization manifests and failure handling."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import tyro

from examples.Libero_Mem import training
from examples.Libero_Mem.environment import _module_from_checkout
from examples.Libero_Mem.splits import extract_fold
from gr00t.data.dataset import LeRobotMixtureDataset
from scripts.gr00t_finetune import ArgsConfig
from test_libero_mem_splits import canonical_template as canonical_template


@pytest.mark.parametrize("kind", ["package", "namespace", "mixed_namespace", "unknown"])
def test_runtime_recognizes_loaded_namespace_but_rejects_other_checkouts(tmp_path, kind):
    root = tmp_path / "fork/libero"
    modules = {
        "package": SimpleNamespace(__file__=str(root / "__init__.py")),
        "namespace": SimpleNamespace(__file__=None, __path__=[str(root)]),
        "mixed_namespace": SimpleNamespace(__file__=None, __path__=[str(root), str(tmp_path / "other/libero")]),
        "unknown": SimpleNamespace(__file__=None),
    }
    assert _module_from_checkout(modules[kind], root) is (kind in ("package", "namespace"))


class ToyDataset:
    def __init__(self, number, length):
        self.number = number
        self.trajectory_ids = np.array([0])
        self.trajectory_lengths = np.array([length])

    def __len__(self):
        return int(self.trajectory_lengths.sum())

    def get_step_data(self, trajectory, step):
        return {"dataset": self.number, "step": step}

    def transforms(self, value):
        return value


def toy_mixture(monkeypatch, *, balanced=True, seed=42):
    monkeypatch.setattr(LeRobotMixtureDataset, "update_metadata", lambda *args: None)
    datasets = [ToyDataset(i, length) for i, length in enumerate((13, 3, 7, 17, 19))]
    return LeRobotMixtureDataset(
        [(dataset, 1.0 if i == 0 else 0.25) for i, dataset in enumerate(datasets)],
        mode="train",
        balance_dataset_weights=False,
        balanced_primary_batches=balanced,
        seed=seed,
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_every_real_dataloader_batch_is_half_mem_half_auxiliary(monkeypatch, workers):
    mixture = toy_mixture(monkeypatch)
    np.testing.assert_array_equal(mixture.dataset_sampling_weights, [0.5, 0.125, 0.125, 0.125, 0.125])
    loader = DataLoader(
        mixture, batch_size=4, shuffle=True, num_workers=workers, generator=torch.Generator().manual_seed(42)
    )
    count = 0
    for batch in loader:
        assert (batch["dataset"] == 0).sum().item() == len(batch["dataset"]) // 2
        count += len(batch["dataset"])
    assert count == len(mixture) == 26  # Includes an even partial last batch.


def test_balanced_sampling_reproducible_and_epoch_sensitive(monkeypatch):
    first = toy_mixture(monkeypatch)
    second = toy_mixture(monkeypatch)
    indices = list(range(8))
    assert first.__getitems__(indices) == second.__getitems__(indices)
    second.set_epoch(1)
    assert first.__getitems__(indices) != second.__getitems__(indices)
    with pytest.raises(ValueError, match="even"):
        first.__getitems__([0, 1, 2])
    first._dataset_sampling_weights = np.array([0.2] * 5)
    # The constructor rejects mismatched group probabilities.
    with pytest.raises(ValueError, match="probability 0.5"):
        LeRobotMixtureDataset(
            [(ToyDataset(i, 2), 1.0) for i in range(2)], mode="train", balanced_primary_batches=True
        )


def test_existing_unbalanced_sampling_is_preserved(monkeypatch):
    mixture = toy_mixture(monkeypatch, balanced=False)
    indices = list(range(7))
    assert mixture.__getitems__(indices) == [mixture[index] for index in indices]


@pytest.fixture(scope="module")
def training_template(canonical_template, tmp_path_factory):  # noqa: F811 -- imported fixture
    root = tmp_path_factory.mktemp("libero_mem_training")
    data = root / "data"
    data.mkdir()
    for fold in ("A", "B"):
        extract_fold(canonical_template, data / f"libero_mem_fold_{fold.lower()}_train", training_fold=fold)
    auxiliary = root / "aux"
    for suite in ("spatial", "object", "goal", "long"):
        destination = auxiliary / suite
        shutil.copytree(canonical_template, destination)
        info_path = destination / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"] = {key: value for key, value in info["features"].items() if not key.startswith("source_")}
        info_path.write_text(json.dumps(info))
    base = root / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"fixture weights")
    return root


@pytest.fixture
def config(training_template, tmp_path):
    root = tmp_path / "inputs"
    shutil.copytree(training_template, root)
    return training.TrainingConfig(
        output_root=tmp_path / "output with spaces",
        datasets_root=root / "data",
        auxiliary_root=root / "aux",
        base_model=root / "base",
        max_steps=2,
        save_steps=2,
        save_total_limit=1,
        batch_size=2,
        dataloader_num_workers=0,
    )


def test_plan_validates_both_folds_and_forwards_complete_recipe(config):
    plans = training.training_plan(config)
    assert [plan["training_fold"] for plan in plans] == ["A", "B"]
    for plan in plans:
        command = plan["command"]
        parsed = tyro.cli(ArgsConfig, args=command[2:])
        assert parsed.dataset_weights == [1.0, 0.25, 0.25, 0.25, 0.25]
        assert parsed.balanced_primary_batches and not parsed.balance_dataset_weights
        assert parsed.seed == config.seed and parsed.max_steps == 2 and parsed.batch_size == 2
        assert not parsed.tune_llm and not parsed.tune_visual
        assert parsed.tune_projector and parsed.tune_diffusion_model and not parsed.save_final_model
        assert parsed.data_config == training.DATA_CONFIG and len(parsed.dataset_path) == 5
        assert plan["normalization_scope"] == "training_fold_and_auxiliary_mixture"
        assert not set(plan["train_tasks"]) & set(plan["held_out_tasks"])
    single = training.training_plan(replace(config, mode="mem-only"))
    assert all(len(plan["datasets"]) == 1 and not plan["balanced_primary_batches"] for plan in single)
    assert not config.output_root.exists()


def test_dry_run_does_not_probe_gpu_or_create_output(config, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run started a process"))
    result = training.run_training(replace(config, dry_run=True))
    assert result["status"] == "dry_run" and len(result["folds"]) == 2
    assert not config.output_root.exists()


@pytest.mark.parametrize(
    "defect", ["wrong_fold", "held_out_instruction", "missing_video", "base", "auxiliary_mem"]
)
def test_malformed_data_rejected_before_training(config, defect):
    fold = config.datasets_root / "libero_mem_fold_b_train"
    if defect == "wrong_fold":
        path = fold / "manifest.json"
        value = json.loads(path.read_text())
        value["training_fold"] = "A"
        path.write_text(json.dumps(value))
    elif defect == "held_out_instruction":
        (fold / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"held out task"}\n')
    elif defect == "missing_video":
        next((fold / "videos").rglob("*.mp4")).unlink()
    elif defect == "base":
        (config.base_model / "model.safetensors").unlink()
    else:
        path = config.auxiliary_root / "goal/meta/info.json"
        value = json.loads(path.read_text())
        value["features"]["source_task_id"] = {"dtype": "int64"}
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        training.training_plan(config)
    assert not config.output_root.exists()


def fake_process(calls, *, fail=None, bad_metadata=False):
    def run(command, **kwargs):
        if command[1] == "-c":
            calls.append("cuda_probe")
            return subprocess.CompletedProcess(command, 0)
        output = Path(command[command.index("--output-dir") + 1])
        calls.append(output.name)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
        if fail == output.name:
            raise subprocess.CalledProcessError(7, command)
        root_metadata = output / "experiment_cfg/metadata.json"
        root_metadata.parent.mkdir()
        root_metadata.write_text('{"new_embodiment": {"training": "normalization"}}')
        checkpoint = output / "checkpoint-2"
        (checkpoint / "experiment_cfg").mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"saved fixture weights")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 2, "epoch": 0.1}))
        (checkpoint / "experiment_cfg/metadata.json").write_bytes(
            b"{}" if bad_metadata else root_metadata.read_bytes()
        )
        return subprocess.CompletedProcess(command, 0)

    return run


def test_sequential_launch_saves_manifest_and_standalone_checkpoints(config, monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(subprocess, "run", fake_process(calls))
    result = training.run_training(config)
    assert result["status"] == "completed" and calls == ["cuda_probe", "fold_a", "fold_b"]
    for fold in ("a", "b"):
        output = config.output_root / f"fold_{fold}"
        manifest = json.loads((output / "run_manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["seed"] == manifest["data_seed"] == 42
        assert manifest["effective_batch_size"] == 2 and manifest["compute_dtype"] == "bfloat16"
        assert manifest["dataset_probabilities"] == [0.5, 0.125, 0.125, 0.125, 0.125]
        assert len(manifest["datasets"]) == 5 and (output / "train.log").is_file()
        assert json.loads((output / "checkpoints.json").read_text())[0]["global_step"] == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    with pytest.raises(FileExistsError):
        training.run_training(config)


@pytest.mark.parametrize("failure,bad_metadata", [("fold_a", False), (None, True)])
def test_failure_stops_before_second_fold(config, monkeypatch, failure, bad_metadata):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(subprocess, "run", fake_process(calls, fail=failure, bad_metadata=bad_metadata))
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        training.run_training(config)
    assert calls == ["cuda_probe", "fold_a"]
    result = json.loads((config.output_root / "summary.json").read_text())
    assert result["status"] == result["folds"][0]["status"] == "failed"
    assert not (config.output_root / "fold_b").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folds": ("A", "A")},
        {"seed": -1},
        {"batch_size": 3},
        {"max_steps": 3},
        {"learning_rate": float("nan")},
        {"dataloader_num_workers": -1},
    ],
)
def test_invalid_recipe_rejected(config, kwargs):
    with pytest.raises(ValueError):
        training.training_plan(replace(config, **kwargs))


@pytest.mark.parametrize("weights", [[1.0], [0, 1], [float("nan"), 1], [0.5, 0.5]])
def test_invalid_generic_training_weights(weights):
    with pytest.raises(ValueError, match="dataset_weights"):
        ArgsConfig(dataset_path=["first", "second"], dataset_weights=weights)
