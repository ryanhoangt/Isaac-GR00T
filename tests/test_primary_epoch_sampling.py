"""Primary frames are shuffled once per epoch; auxiliary data remains sampled."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from accelerate import skip_first_batches

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.experiment.trainer import BaseSampler, DualBrainTrainer


class IndexedDataset:
    def __init__(self, number, lengths):
        self.number = number
        self.trajectory_ids = np.arange(len(lengths))
        self.trajectory_lengths = np.asarray(lengths)
        self.all_steps = [
            (trajectory, step) for trajectory, length in enumerate(lengths) for step in range(length)
        ]
        self.epoch = 0

    def __len__(self):
        return len(self.all_steps)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_step_data(self, trajectory, step):
        anchor = int(self.trajectory_lengths[:trajectory].sum()) + int(step)
        return {"dataset": self.number, "anchor": anchor, "sample_epoch": self.epoch}

    def transforms(self, sample):
        return sample

    def __getitem__(self, index):
        return self.transforms(self.get_step_data(*self.all_steps[index]))


def mixture(monkeypatch, *, seed=42):
    monkeypatch.setattr(LeRobotMixtureDataset, "update_metadata", lambda *args: None)
    datasets = [IndexedDataset(0, [5, 8]), IndexedDataset(1, [3, 7]), IndexedDataset(2, [9])]
    return LeRobotMixtureDataset(
        [(datasets[0], 1.0), (datasets[1], 0.5), (datasets[2], 0.5)],
        mode="train",
        seed=seed,
        balance_dataset_weights=False,
        balanced_primary_batches=True,
        primary_sampling="epoch",
    )


def anchors(batches):
    return [
        anchor
        for batch in batches
        for dataset, anchor in zip(batch["dataset"].tolist(), batch["anchor"].tolist())
        if dataset == 0
    ]


@pytest.mark.parametrize("workers", [0, 2])
def test_exact_primary_coverage_and_worker_epoch_propagation(monkeypatch, workers):
    data = mixture(monkeypatch)
    sampler = BaseSampler(data, shuffle=True, seed=42)
    loader = DataLoader(data, batch_size=4, sampler=sampler, num_workers=workers, persistent_workers=workers > 0)
    first = list(loader)
    assert sorted(anchors(first)) == list(range(13)) and len(anchors(first)) == 13
    for batch in first:
        assert (batch["dataset"] == 0).sum() == len(batch["dataset"]) // 2
    assert len(first[-1]["dataset"]) == 2  # Odd primary length: one primary + one auxiliary.
    sampler.set_epoch(1)
    second = list(loader)
    assert sorted(anchors(second)) == list(range(13))
    assert anchors(first) != anchors(second)
    assert all((batch["sample_epoch"] == 1).all() for batch in second)

    sampler.set_epoch(0)
    skipped = skip_first_batches(loader, 2)
    assert anchors(list(skipped)) == anchors(first)[4:]
    sampler.set_epoch(0)
    repeated = list(loader)
    for expected, actual in zip(first, repeated):
        for key in expected:
            assert torch.equal(expected[key], actual[key])


def test_sampler_resume_suffix_is_deterministic(monkeypatch):
    data = mixture(monkeypatch)
    sampler = BaseSampler(data, shuffle=True, seed=42)
    sampler.set_epoch(3)
    indices = list(sampler)
    expected = data.__getitems__(indices[8:])
    restarted = BaseSampler(data, shuffle=True, seed=42)
    restarted.set_epoch(3)
    assert data.__getitems__(list(restarted)[8:]) == expected
    with pytest.raises(ValueError, match="paired indices"):
        data.__getitems__([(3, 0), (3, 3)])


@pytest.mark.parametrize("workers", [0, 2])
def test_real_trainer_and_accelerate_do_not_pad_or_drop_primary_frames(monkeypatch, tmp_path, workers):
    data = mixture(monkeypatch)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        dataloader_num_workers=workers,
        dataloader_persistent_workers=workers > 0,
        dataloader_prefetch_factor=2 if workers else None,
    )
    trainer = DualBrainTrainer(
        model=torch.nn.Linear(1, 1), args=args, train_dataset=data, compute_dtype=torch.float32
    )
    loader = trainer.get_train_dataloader()
    assert not trainer.accelerator.even_batches
    first = list(loader)
    assert sorted(anchors(first)) == list(range(13))
    assert sum(len(batch["dataset"]) for batch in first) == 26
    loader.set_epoch(1)
    second = list(loader)
    assert sorted(anchors(second)) == list(range(13))
    assert anchors(first) != anchors(second)
    assert all((batch["sample_epoch"] == 1).all() for batch in second)


def test_legacy_base_sampler_stays_a_normal_permutation():
    sampler = BaseSampler(IndexedDataset(0, [13]), shuffle=True, seed=42)
    assert sorted(sampler) == list(range(13))
