"""Exercise logging with a tiny CPU model and real optimizer, without loading assets."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def trainer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/rag/3_train_ravla.py"
    spec = importlib.util.spec_from_file_location("ravla_training_logging_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    models = []

    class TinyVLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.config = SimpleNamespace()
            self.calls = 0
            self.fail_on = None
            models.append(self)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def set_retrieval_components(self, *args):
            pass

        def init_new_weights(self):
            pass

        def forward(self, inputs):
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("injected training failure")
            mse = self.linear(inputs).square().mean()
            margin = mse * 0 + 0.25
            return SimpleNamespace(mse_loss=mse, margin_loss=margin, loss=mse + 0.1 * margin)

        def save_pretrained(self, path):
            Path(path, "tiny_model.pt").write_text("checkpoint")

    class Dataset:
        def __init__(self, *args, **kwargs):
            pass

    class Mixture(Dataset):
        def save_metadata(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "CachedLeRobotSingleDataset", Dataset)
    monkeypatch.setattr(module, "BoundedLeRobotVideoDataset", Dataset)
    monkeypatch.setattr(module, "FramePathsMixtureDataset", Mixture)
    monkeypatch.setattr(module, "RAVLADataCollator", lambda: None)
    monkeypatch.setattr(module, "DataLoader", lambda *a, **k: [torch.tensor([[float(i)]]) for i in (1, 2, 3)])
    monkeypatch.setattr(module, "RetrieverWrapperWithHead", SimpleNamespace(from_pretrained=lambda *a, **k: TinyVLA()))
    monkeypatch.setattr(module, "RAVLA", TinyVLA)
    monkeypatch.setattr(module.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    module.test_models = models
    return module


class RecordingRun:
    def __init__(self):
        self.config = {}
        self.summary = {}
        self.metrics = []
        self.definitions = []
        self.exit_codes = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, metrics):
        self.metrics.append(metrics)

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


def install_wandb(monkeypatch):
    run = RecordingRun()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        torch.randn(5)  # Logging setup must not change the seeded training trajectory.
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return run, calls


def config(trainer, path, **kwargs):
    return trainer.ArgsConfig(
        vla_path=str(path), dataset_paths=["source"], seed=42,
        num_epochs=1, num_workers=0, log_freq=2, **kwargs,
    )


def test_logs_steps_epochs_config_and_checkpoint(trainer, monkeypatch, tmp_path):
    run, calls = install_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "project-from-environment")
    trainer.main(config(trainer, tmp_path))
    assert calls[0]["project"] == "icl-vla-uncertainty"
    assert calls[0]["config"]["seed"] == 42
    assert calls[0]["dir"] == str(tmp_path.resolve())
    assert run.config["total_steps"] == 3
    assert run.config["trainable_parameters"] == 2
    steps = [entry for entry in run.metrics if "train/loss" in entry]
    assert [entry["global_step"] for entry in steps] == [1, 2, 3]
    assert steps[0]["train/learning_rate"] == pytest.approx(2e-4)
    assert steps[-1]["train/learning_rate"] < steps[0]["train/learning_rate"]
    assert all(isinstance(entry["train/grad_norm"], float) for entry in steps)
    epoch = next(entry for entry in run.metrics if "epoch/loss" in entry)
    assert epoch["epoch/loss"] == pytest.approx(sum(e["train/loss"] for e in steps) / 3)
    assert run.summary["last_checkpoint"] == str((tmp_path / "checkpoint_1").resolve())
    assert run.summary["completed_epochs"] == 1
    assert run.exit_codes == [0]
    assert json.loads((tmp_path / "training_config.json").read_text())["seed"] == 42


def test_disabled_logging_preserves_seeded_training(trainer, monkeypatch, tmp_path):
    run, calls = install_wandb(monkeypatch)
    trainer.main(config(trainer, tmp_path / "enabled"))
    logged_weights = trainer.test_models[-1].linear.weight.detach().clone()
    monkeypatch.setitem(sys.modules, "wandb", None)
    trainer.main(config(trainer, tmp_path / "disabled", wandb_enable=False))
    torch.testing.assert_close(trainer.test_models[-1].linear.weight, logged_weights, rtol=0, atol=0)
    assert len(calls) == 1


def test_training_failure_finishes_run_with_failure_status(trainer, monkeypatch, tmp_path):
    run, _ = install_wandb(monkeypatch)
    original_init = trainer.RAVLA.init_new_weights

    def fail_on_second_batch(model):
        original_init(model)
        model.fail_on = 2

    monkeypatch.setattr(trainer.RAVLA, "init_new_weights", fail_on_second_batch)
    with pytest.raises(RuntimeError, match="injected training failure"):
        trainer.main(config(trainer, tmp_path))
    assert run.exit_codes == [1]


def test_invalid_log_frequency_fails_before_initializing_wandb(trainer, monkeypatch, tmp_path):
    _, calls = install_wandb(monkeypatch)
    cfg = config(trainer, tmp_path)
    cfg.log_freq = 0
    with pytest.raises(ValueError, match="log_freq must be positive"):
        trainer.main(cfg)
    assert not calls


def test_sparse_logs_flush_at_each_epoch_end(trainer, monkeypatch, tmp_path):
    run, calls = install_wandb(monkeypatch)
    cfg = config(trainer, tmp_path, wandb_project="explicit-project", wandb_name="gos_seed42")
    cfg.num_epochs = 2
    cfg.log_freq = 100
    trainer.main(cfg)
    assert calls[0]["project"] == "explicit-project"
    assert calls[0]["name"] == "gos_seed42"
    assert [e["global_step"] for e in run.metrics if "train/loss" in e] == [1, 3, 6]
    assert [e["epoch"] for e in run.metrics if "epoch/loss" in e] == [1, 2]


@pytest.mark.parametrize("cache_video", [False, True])
def test_video_cache_is_opt_in(trainer, monkeypatch, tmp_path, cache_video):
    selected = []

    def on_demand(**kwargs):
        selected.append("on_demand")
        return object()

    def cached(**kwargs):
        selected.append("cached")
        return object()

    monkeypatch.setattr(trainer, "BoundedLeRobotVideoDataset", on_demand)
    monkeypatch.setattr(trainer, "CachedLeRobotSingleDataset", cached)
    cfg = config(trainer, tmp_path, wandb_enable=False)
    assert cfg.cache_video is False
    cfg.cache_video = cache_video
    trainer.main(cfg)
    assert selected == ["cached" if cache_video else "on_demand"]
