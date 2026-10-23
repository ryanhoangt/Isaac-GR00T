"""Exercise epoch checkpoints and optional final export on a tiny CPU model."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

from gr00t.experiment.runner import TrainRunner
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.utils.experiment import CheckpointFormatCallback


class TinyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs):
        return {"loss": (self.linear(inputs["x"]) - inputs["labels"]).square().mean()}


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        x = torch.tensor([index / 8], dtype=torch.float32)
        return {"x": x, "labels": 2 * x}


@pytest.mark.parametrize("save_final_model", [False, True])
def test_two_epoch_checkpoints(tmp_path, save_final_model):
    output = tmp_path / "training"
    metadata = output / "experiment_cfg"
    metadata.mkdir(parents=True)
    (metadata / "metadata.json").write_text('{"test": true}')
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        per_device_train_batch_size=3,  # Include an incomplete final batch.
        num_train_epochs=2,
        max_steps=-1,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = DualBrainTrainer(
        model=TinyModel(PretrainedConfig()),
        args=args,
        train_dataset=TinyDataset(),
        compute_dtype=torch.float32,
    )
    trainer.add_callback(CheckpointFormatCallback("test", metadata))
    runner = SimpleNamespace(trainer=trainer, training_args=args, resume_from_checkpoint=False)
    TrainRunner.train(runner, save_final_model=save_final_model)
    checkpoints = sorted(output.glob("checkpoint-*"))
    assert len(checkpoints) == 2
    for epoch, checkpoint in enumerate(checkpoints, start=1):
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["epoch"] == epoch
        assert (checkpoint / "model.safetensors").is_file()
        assert (checkpoint / "optimizer.pt").is_file()
        assert (checkpoint / "experiment_cfg/metadata.json").is_file()
    assert (output / "model.safetensors").is_file() == save_final_model
