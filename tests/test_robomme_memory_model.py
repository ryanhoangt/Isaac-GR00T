"""Memory modulation math, train/inference wiring and pretrained compatibility."""

import copy
import math

import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from examples.RoboMME.memory_config import load_memory_config
from gr00t.model.action_head.cross_attention_dit import DiT
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from gr00t.model.action_head.memory import MemoryFeatureEncoder, MemoryModulation, apply_rope


def memory_config():
    config = load_memory_config()
    config["budget"] = 32
    config["memory_token_dim"] = 8
    config["memory_feature"]["img"]["input_dim"] = 5
    config["memory_feature"]["pos"] = {"input_dim": 12, "hidden_dim": 6}
    config["attention"].update(num_heads=2, head_dim=4)
    return config


def head_config(memory=None, expand_batch=None):
    return FlowmatchingActionHeadConfig(
        hidden_size=8,
        input_embedding_dim=8,
        backbone_embedding_dim=6,
        max_state_dim=4,
        action_dim=3,
        action_horizon=4,
        max_num_embodiments=1,
        num_target_vision_tokens=2,
        num_inference_timesteps=3,
        max_seq_len=16,
        use_vlln=False,
        add_pos_embed=True,
        expand_batch=expand_batch,
        memory_config=memory,
        diffusion_model_cfg={
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "output_dim": 8,
            "num_layers": 2,
            "cross_attention_dim": 6,
            "dropout": 0,
            "final_dropout": False,
            "positional_embeddings": None,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
        },
    )


def inputs(memory=True, batch_size=2):
    data = {
        "state": torch.randn(batch_size, 1, 4),
        "action": torch.randn(batch_size, 4, 3),
        "action_mask": torch.ones(batch_size, 4, 3, dtype=torch.bool),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
    }
    if memory:
        data.update(
            static_image_emb=torch.randn(batch_size, 32, 5),
            static_pos_emb=torch.randn(batch_size, 32, 12),
            static_mask=torch.arange(32)[None].expand(batch_size, -1) < 16,
        )
    backbone = BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, 5, 6),
            "backbone_attention_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        }
    )
    return backbone, BatchFeature(data=data)


def reference_rope(x, positions, base=10000):
    result = torch.empty_like(x)
    half = x.shape[-1] // 2
    for index, position in enumerate(positions):
        for channel in range(half):
            phase = float(position) / base ** (2 * channel / x.shape[-1])
            left, right = x[:, index, :, channel], x[:, index, :, channel + half]
            result[:, index, :, channel] = left * math.cos(phase) - right * math.sin(phase)
            result[:, index, :, channel + half] = right * math.cos(phase) + left * math.sin(phase)
    return result


def test_rope_matches_gemma_split_half_convention():
    x = torch.arange(2 * 3 * 2 * 4).float().reshape(2, 3, 2, 4) / 10
    positions = torch.tensor([0, 32, 97])
    torch.testing.assert_close(apply_rope(x, positions, 10000), reference_rope(x, positions), atol=1e-6, rtol=1e-6)


def test_modulation_matches_explicit_one_kv_head_attention_and_rmsnorm():
    torch.manual_seed(4)
    module = MemoryModulation(8, memory_config()).eval()
    x, memory = torch.randn(2, 3, 8), torch.randn(2, 32, 8)
    mask = torch.arange(32)[None].expand(2, -1) < 16

    def rms(value):
        return value / torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6)

    q = torch.nn.functional.linear(rms(x) * (1 + module.norm_scale), module.q.weight).reshape(2, 3, 2, 4)
    kv = torch.nn.functional.linear(rms(memory) * (1 + module.norm_scale), module.kv.weight).reshape(2, 32, 2, 4)
    q = reference_rope(q, range(32, 35))
    k = reference_rope(kv[:, :, :1], range(32))
    logits = torch.einsum("bthd,bshd->bhts", q, k.expand(-1, -1, 2, -1)) / 2
    probabilities = logits.masked_fill(~mask[:, None, None], float("-inf")).softmax(-1)
    attended = torch.einsum("bhts,bsd->bthd", probabilities, kv[:, :, 1]).reshape(2, 3, 8)
    attended = torch.nn.functional.linear(attended, module.out.weight)
    scale, shift = torch.nn.functional.linear(attended, module.modulation.weight, module.modulation.bias).chunk(
        2, -1
    )
    expected = rms(x) * (1 + scale) + shift
    torch.testing.assert_close(module(x, memory, mask), expected, atol=1e-6, rtol=1e-6)
    assert module.kv.out_features == 2 * module.head_dim


def test_masked_tokens_have_no_effect_or_gradient():
    module = MemoryModulation(8, memory_config()).eval()
    x, memory = torch.randn(2, 3, 8), torch.randn(2, 32, 8, requires_grad=True)
    mask = torch.arange(32)[None].expand(2, -1) < 16
    expected = module(x, memory, mask)
    corrupted = memory.detach().clone()
    corrupted[~mask] = float("nan")
    torch.testing.assert_close(module(x, corrupted, mask), expected)
    expected.square().mean().backward()
    assert not memory.grad[~mask].any()
    assert memory.grad[mask].abs().sum() > 0


def test_feature_encoder_positions_can_be_disabled():
    config = memory_config()
    config["use_pos_emb"] = False
    encoder = MemoryFeatureEncoder(config)
    images = torch.randn(2, 32, 5)
    torch.testing.assert_close(encoder(images, torch.randn(2, 32, 12)), encoder(images, torch.zeros(2, 32, 12)))
    assert not hasattr(encoder, "pos_proj")


def test_enable_memory_preserves_pretrained_parameters_and_is_idempotent():
    head = FlowmatchingActionHead(head_config())
    parameters = {name: (parameter, parameter.detach().clone()) for name, parameter in head.named_parameters()}
    config = memory_config()
    head.enable_memory(config)
    for name, (original, value) in parameters.items():
        assert dict(head.named_parameters())[name] is original
        torch.testing.assert_close(original, value, rtol=0, atol=0)
    encoder = head.memory_encoder
    head.enable_memory(config)
    assert head.memory_encoder is encoder
    config["budget"] = 64
    assert head.config.memory_config["budget"] == 32
    with pytest.raises(ValueError, match="different configuration"):
        head.enable_memory(config)


def test_flow_loss_backpropagates_into_every_memory_module():
    torch.manual_seed(7)
    head = FlowmatchingActionHead(head_config(memory_config())).train()
    backbone, data = inputs()
    loss = head(backbone, data)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if name.startswith("memory_encoder.") or ".memory_modulation." in name
    ]
    assert parameters
    for name, parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert all(block.memory_modulation is not None for block in head.model.transformer_blocks)


def test_inference_encodes_memory_once_and_uses_it_at_every_layer_and_step():
    head = FlowmatchingActionHead(head_config(memory_config())).eval()
    counts = {"encoder": 0, "blocks": [0, 0]}

    def encoded(*args):
        counts["encoder"] += 1

    def modulated(index):
        def hook(*args):
            counts["blocks"][index] += 1

        return hook

    head.memory_encoder.register_forward_hook(encoded)
    for index, block in enumerate(head.model.transformer_blocks):
        block.memory_modulation.register_forward_hook(modulated(index))
    backbone, data = inputs()
    prediction = head.get_action(backbone, data)["action_pred"]
    assert prediction.shape == (2, 4, 3) and torch.isfinite(prediction).all()
    assert counts == {"encoder": 1, "blocks": [3, 3]}


def test_memory_changes_inference_under_fixed_sampling_noise():
    head = FlowmatchingActionHead(head_config(memory_config())).eval()
    backbone, data = inputs()
    torch.manual_seed(42)
    first = head.get_action(copy.deepcopy(backbone), data)["action_pred"]
    other = copy.deepcopy(data)
    other["static_image_emb"] = torch.randn_like(other["static_image_emb"]) * 3
    torch.manual_seed(42)
    second = head.get_action(copy.deepcopy(backbone), other)["action_pred"]
    assert (first - second).abs().max() > 1e-6


def test_expand_batch_repeats_memory_and_masks_with_training_samples():
    head = FlowmatchingActionHead(head_config(memory_config(), expand_batch=2))
    seen = []
    head.memory_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[0]))
    backbone, data = inputs()
    assert torch.isfinite(head(backbone, data)["loss"])
    assert seen == [4]


@pytest.mark.parametrize("mode", ["forward", "get_action"])
@pytest.mark.parametrize("invalid", ["missing", "float_mask", "empty_mask", "shape", "nan"])
def test_invalid_memory_is_rejected_in_train_and_inference(mode, invalid):
    head = FlowmatchingActionHead(head_config(memory_config())).eval()
    backbone, data = inputs()
    if invalid == "missing":
        data.pop("static_pos_emb")
    elif invalid == "float_mask":
        data["static_mask"] = data["static_mask"].float()
    elif invalid == "empty_mask":
        data["static_mask"][0] = False
    elif invalid == "shape":
        data["static_image_emb"] = data["static_image_emb"][:, :16]
    else:
        data["static_image_emb"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        getattr(head, mode)(backbone, data)


def test_padded_nonfinite_raw_features_are_ignored():
    head = FlowmatchingActionHead(head_config(memory_config())).eval()
    backbone, data = inputs()
    expected = head._prepare_memory(data, 2)["memory_tokens"]
    changed = copy.deepcopy(data)
    changed["static_image_emb"][~changed["static_mask"]] = float("nan")
    changed["static_pos_emb"][~changed["static_mask"]] = float("inf")
    actual = head._prepare_memory(changed, 2)["memory_tokens"]
    torch.testing.assert_close(actual, expected)


def test_disabled_memory_has_no_extra_parameters_and_does_not_ignore_memory_input():
    head = FlowmatchingActionHead(head_config())
    assert not any("memory" in name for name in head.state_dict())
    backbone, data = inputs(memory=False)
    assert torch.isfinite(head(copy.deepcopy(backbone), data)["loss"])
    assert head.get_action(copy.deepcopy(backbone), data)["action_pred"].shape == (2, 4, 3)
    _, memory = inputs()
    with pytest.raises(ValueError, match="memory disabled"):
        head.get_action(backbone, memory)


def test_memory_parameters_follow_diffusion_training_flag():
    head = FlowmatchingActionHead(head_config(memory_config()))
    head.set_trainable_parameters(True, False)
    assert not any(parameter.requires_grad for parameter in head.memory_encoder.parameters())
    assert not any(
        parameter.requires_grad
        for block in head.model.transformer_blocks
        for parameter in block.memory_modulation.parameters()
    )
    head.set_trainable_parameters(True, True)
    assert all(parameter.requires_grad for parameter in head.memory_encoder.parameters())


def test_dit_and_head_config_weights_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    head = FlowmatchingActionHead(head_config(memory_config())).eval()
    head.config.save_pretrained(tmp_path)
    save_file(head.state_dict(), tmp_path / "model.safetensors")
    restored = FlowmatchingActionHead(FlowmatchingActionHeadConfig.from_pretrained(tmp_path)).eval()
    restored.load_state_dict(load_file(tmp_path / "model.safetensors"), strict=True)
    backbone, data = inputs()
    torch.manual_seed(42)
    before = head.get_action(copy.deepcopy(backbone), data)["action_pred"]
    torch.manual_seed(42)
    after = restored.get_action(copy.deepcopy(backbone), data)["action_pred"]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.config.memory_config == memory_config()
    dit = DiT.from_config(head.model.config)
    dit.load_state_dict(head.model.state_dict(), strict=True)
    assert all(block.memory_modulation is not None for block in dit.transformer_blocks)


class FakeBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(6, 6)

    def prepare_input(self, inputs):
        return BatchFeature(data=inputs)

    def forward(self, inputs):
        features = self.projection(inputs["test_vlm_features"])
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": torch.ones(features.shape[:2], dtype=torch.bool),
            }
        )

    def set_trainable_parameters(self, **kwargs):
        self.requires_grad_(False)


def test_full_gr00t_save_load_and_training_input_path(tmp_path, monkeypatch):
    import gr00t.model.gr00t_n1 as model_module

    monkeypatch.setattr(model_module, "EagleBackbone", FakeBackbone)
    config = model_module.GR00T_N1_5_Config(
        backbone_cfg={},
        action_head_cfg=head_config().to_dict(),
        action_horizon=4,
        action_dim=3,
        compute_dtype="float32",
    )
    model = model_module.GR00T_N1_5(config, local_model_path=str(tmp_path))
    model.enable_memory(memory_config())
    backbone, data = inputs()
    data["test_vlm_features"] = backbone["backbone_features"]
    loss = model(data)["loss"]
    loss.backward()
    assert model.action_head.memory_encoder.encoder_static.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    restored = model_module.GR00T_N1_5.from_pretrained(
        str(tmp_path), local_files_only=True, tune_visual=False
    ).eval()
    model.eval()
    torch.manual_seed(42)
    before = model.get_action(data)["action_pred"]
    torch.manual_seed(42)
    after = restored.get_action(data)["action_pred"]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.config.action_head_cfg["memory_config"] == memory_config()


def test_wrong_dit_width_rejected_before_enabling():
    head = FlowmatchingActionHead(head_config())
    with pytest.raises(ValueError, match="DiT inner width"):
        head.enable_memory(load_memory_config())
    assert head.memory_encoder is None
