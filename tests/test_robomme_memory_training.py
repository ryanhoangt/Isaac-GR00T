"""CLI cache/vision checks and actual trainer checkpoint/resume integration."""

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import TrainerCallback, TrainingArguments
from transformers.feature_extraction_utils import BatchFeature
import yaml

import test_robomme_memory_dataset as dataset_tests

from examples.RoboMME.feature_cache import build_feature_cache, cache_plan, json_fingerprint
from examples.RoboMME.memory_dataset import RoboMMEMemoryDataset
from examples.RoboMME.memory_training import (
    MEMORY_DATA_CONFIG,
    effective_data_config,
    memory_run_record,
    prepare_memory_run,
    resolve_memory_checkpoint,
    verify_memory_cache,
)
from examples.RoboMME.visual_features import FrozenMemoryVisionEncoder
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from scripts import gr00t_finetune as finetune


# Reuse the local converted-data fixture while exercising a real vision cache.
prepared = dataset_tests.prepared


def tiny_eagle():
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    from gr00t.model.backbone.eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

    config = Eagle2_5_VLConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 224,
            "patch_size": 14,
        },
        text_config={
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
        },
        use_pixel_shuffle=False,
        mlp_connector_layers=1,
        force_image_size=224,
        select_layer=-1,
        _attn_implementation="sdpa",
    )
    config.vision_config._attn_implementation = "sdpa"
    return Eagle2_5_VLForConditionalGeneration(
        config, vision_model=SiglipVisionModel(config.vision_config), language_model=nn.Identity()
    )


class TinyTrainingBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eagle_model = tiny_eagle()
        self.set_trainable_parameters(**kwargs)

    def set_trainable_parameters(self, tune_visual=False, tune_llm=False):
        self.eagle_model.vision_model.requires_grad_(tune_visual)
        self.eagle_model.mlp1.requires_grad_(tune_visual)

    def prepare_input(self, batch):
        return BatchFeature(data=batch)

    def forward(self, batch):
        self.eagle_model.eval()
        tokens = self.eagle_model.extract_feature(batch["eagle_pixel_values"])
        batch_size = batch["state"].shape[0]
        tokens = tokens.reshape(batch_size, -1, 4)
        return BatchFeature(
            data={
                "backbone_features": tokens,
                "backbone_attention_mask": torch.ones(tokens.shape[:2], dtype=torch.bool),
            }
        )


def tiny_head_config():
    return {
        "hidden_size": 8,
        "input_embedding_dim": 8,
        "backbone_embedding_dim": 4,
        "max_state_dim": 64,
        "action_dim": 32,
        "action_horizon": 16,
        "max_num_embodiments": 32,
        "num_target_vision_tokens": 2,
        "num_inference_timesteps": 2,
        "max_seq_len": 32,
        "use_vlln": False,
        "add_pos_embed": True,
        "diffusion_model_cfg": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "output_dim": 8,
            "num_layers": 2,
            "cross_attention_dim": 4,
            "dropout": 0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
        },
    }


@pytest.fixture
def setup_training(prepared, tmp_path, monkeypatch):
    import gr00t.model.gr00t_n1 as model_module

    monkeypatch.setattr(model_module, "EagleBackbone", TinyTrainingBackbone)
    torch.manual_seed(11)
    model = GR00T_N1_5(
        GR00T_N1_5_Config(backbone_cfg={}, action_head_cfg=tiny_head_config(), action_horizon=16, action_dim=32),
        local_model_path=str(tmp_path),
    )
    base = tmp_path / "base"
    model.save_pretrained(base)
    memory = copy.deepcopy(prepared[3])
    memory.update(budget=32, memory_token_dim=8)
    memory["attention"].update(num_heads=2, head_dim=4)
    memory["memory_feature"]["pos"]["hidden_dim"] = 6
    config_path = tmp_path / "memory.yaml"
    config_path.write_text(yaml.safe_dump(memory))
    encoder = FrozenMemoryVisionEncoder(model.backbone.eagle_model, memory, device="cpu", compute_dtype="float32")
    cache = tmp_path / "real_cache"
    build_feature_cache(cache_plan(prepared[0], memory), cache, encoder, batch_size=2)
    args = finetune.ArgsConfig(
        dataset_path=[str(prepared[1])],
        memory_config=str(config_path),
        memory_cache=str(cache),
        base_model_path=str(base),
        output_dir=str(tmp_path / "train"),
        batch_size=1,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=1,
        learning_rate=1e-3,
        warmup_ratio=0,
        max_steps=2,
        save_steps=1,
        save_total_limit=None,
        save_final_model=False,
        video_backend="torchvision_av",
    )
    return args, model, memory


@pytest.mark.parametrize(
    "change",
    [
        {"memory_cache": "cache"},
        {"memory_config": "config.yaml"},
        {"memory_cache": "cache", "memory_config": "config.yaml", "tune_visual": True},
        {"memory_cache": "cache", "memory_config": "config.yaml", "lora_rank": 8},
        {"memory_cache": "cache", "memory_config": "config.yaml", "tune_diffusion_model": False},
    ],
)
def test_invalid_memory_cli_rejected_early(change):
    args = finetune.ArgsConfig(dataset_path=["missing"], **change)
    with pytest.raises(ValueError):
        effective_data_config(args)


def test_cli_selects_memory_config_and_preserves_vanilla_defaults():
    args = finetune.ArgsConfig(dataset_path=["unused"])
    assert effective_data_config(args) == args.data_config
    assert args.dataloader_multiprocessing_context is None
    args.memory_cache, args.memory_config = "cache", "config.yaml"
    assert effective_data_config(args) == MEMORY_DATA_CONFIG


def make_dataset(args, memory):
    return RoboMMEMemoryDataset(
        args.dataset_path[0], memory_cache=args.memory_cache, memory_config=memory, video_backend="torchvision_av"
    )


def test_live_cache_signature_matches_and_detects_weights_or_implementation(setup_training):
    args, model, memory = setup_training
    dataset = make_dataset(args, memory)
    verify_memory_cache(model, dataset)
    with torch.no_grad():
        model.backbone.eagle_model.mlp1[0].bias.add_(1)
    with pytest.raises(ValueError, match="vision_weights_sha256"):
        verify_memory_cache(model, dataset)
    model.backbone.eagle_model.mlp1.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        verify_memory_cache(model, dataset)
    dataset.close()


def test_cache_preprocessing_mismatch_is_rejected_without_moving_model(setup_training):
    args, model, memory = setup_training
    dataset = make_dataset(args, memory)
    parameters = [(parameter.device, parameter.dtype) for parameter in model.parameters()]
    dataset.memory_encoder_signature["preprocessing"]["antialias"] = False
    with pytest.raises(ValueError, match="preprocessing"):
        verify_memory_cache(model, dataset)
    assert [(parameter.device, parameter.dtype) for parameter in model.parameters()] == parameters
    dataset.close()


def test_resume_identity_rejects_recipe_changes_before_overwriting_sidecars(setup_training):
    args, _, memory = setup_training
    dataset = make_dataset(args, memory)
    record = prepare_memory_run(args, dataset, memory, MEMORY_DATA_CONFIG, "spawn", None)
    checkpoint = Path(args.output_dir) / "checkpoint-1/experiment_cfg"
    checkpoint.mkdir(parents=True)
    (checkpoint / "memory_training.json").write_text(json.dumps(record))
    before = (Path(args.output_dir) / "experiment_cfg/memory_training.json").read_text()
    changed = replace(args, resume=True, seed=43)
    with pytest.raises(ValueError, match="configuration/data/cache differ"):
        prepare_memory_run(changed, dataset, memory, MEMORY_DATA_CONFIG, "spawn", str(checkpoint.parent))
    assert (Path(args.output_dir) / "experiment_cfg/memory_training.json").read_text() == before
    operational = replace(args, resume=True, max_steps=5, save_steps=5, dataloader_num_workers=1)
    assert (
        memory_run_record(operational, dataset, memory, MEMORY_DATA_CONFIG, "spawn")["identity_sha256"]
        == record["identity_sha256"]
    )
    dataset.close()


def test_fresh_output_and_missing_resume_checkpoint_rejected(tmp_path):
    args = finetune.ArgsConfig(dataset_path=["unused"], output_dir=str(tmp_path / "run"))
    assert resolve_memory_checkpoint(args) is None
    with pytest.raises(ValueError, match="existing"):
        resolve_memory_checkpoint(replace(args, resume=True))
    sidecar = Path(args.output_dir) / "experiment_cfg/memory_training.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}")
    with pytest.raises(FileExistsError):
        resolve_memory_checkpoint(args)


class StopAfterFirstStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
            control.should_save = True
        return control


def test_real_main_trains_saves_and_resumes_optimizer_with_spawn_workers(setup_training, monkeypatch):
    args, _, memory = setup_training
    original_runner = finetune.TrainRunner
    observations = []

    class RecordingRunner(original_runner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert self.trainer.dataloader_multiprocessing_context == "spawn"
            optimizer = self.trainer.create_optimizer()
            groups = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            for name, parameter in self.trainer.model.named_parameters():
                if "memory" in name:
                    assert parameter.requires_grad and id(parameter) in groups
            if not kwargs["resume_from_checkpoint"]:
                self.trainer.add_callback(StopAfterFirstStep())
            observations.append(self)

    def cpu_training_args(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, tf32=False, report_to=[], disable_tqdm=True)
        return TrainingArguments(**kwargs)

    monkeypatch.setattr(finetune, "TrainRunner", RecordingRunner)
    monkeypatch.setattr(finetune, "TrainingArguments", cpu_training_args)
    finetune.main(args)
    output = Path(args.output_dir)
    first = output / "checkpoint-1"
    assert first.is_dir() and (first / "optimizer.pt").is_file()
    assert (first / "experiment_cfg/memory_config.yaml").is_file()
    saved = json.loads((first / "experiment_cfg/memory_training.json").read_text())
    assert saved["identity_sha256"] == json_fingerprint(saved["identity"])
    assert json.loads((first / "config.json").read_text())["action_head_cfg"]["memory_config"] == memory
    assert observations[0].trainer.state.global_step == 1
    finetune.main(replace(args, resume=True))
    assert observations[1].trainer.state.global_step == 2
    second = output / "checkpoint-2"
    assert (second / "optimizer.pt").is_file()
    optimizer = torch.load(second / "optimizer.pt", map_location="cpu", weights_only=False)
    assert all(int(value["step"]) == 2 for value in optimizer["state"].values())
    first_optimizer = torch.load(first / "optimizer.pt", map_location="cpu", weights_only=False)
    assert all(int(value["step"]) == 1 for value in first_optimizer["state"].values())
    assert (
        saved["identity_sha256"]
        == json.loads((second / "experiment_cfg/memory_training.json").read_text())["identity_sha256"]
    )
    assert resolve_memory_checkpoint(replace(args, resume=True)) == str(second.resolve())


@pytest.mark.parametrize("with_memory", [False, True])
def test_zero_worker_main_and_vanilla_training_path(setup_training, monkeypatch, with_memory):
    args, _, _ = setup_training

    def cpu_training_args(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, tf32=False, report_to=[], disable_tqdm=True)
        return TrainingArguments(**kwargs)

    monkeypatch.setattr(finetune, "TrainingArguments", cpu_training_args)
    args = replace(args, max_steps=1, dataloader_num_workers=0)
    if not with_memory:
        args = replace(
            args,
            memory_config=None,
            memory_cache=None,
            data_config="examples.RoboMME.custom_data_config:RoboMMEDataConfig",
        )
    finetune.main(args)
    checkpoint = Path(args.output_dir) / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    config = json.loads((checkpoint / "config.json").read_text())
    assert bool(config["action_head_cfg"].get("memory_config")) == with_memory
    assert (checkpoint / "experiment_cfg/memory_training.json").exists() == with_memory
