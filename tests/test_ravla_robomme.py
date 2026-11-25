"""Verify canonical routing and padding invariance without loading GR00T."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rag import data
from gr00t.rag.ravla import RAVLAActionHead, RAVLADataCollator, RAVLADiT
from gr00t.model.action_head.cross_attention_dit import SelfAttentionTransformer
from gr00t.model.transforms import DefaultDataCollator


def collator(monkeypatch, required=False):
    instance = RAVLADataCollator.__new__(RAVLADataCollator)
    instance.require_memory_key = required
    monkeypatch.setattr(DefaultDataCollator, "__call__", lambda self, rows: {"rows": rows})
    return instance


def test_dataset_routes_same_instruction_to_different_banks(monkeypatch, tmp_path):
    prompt = "prefix<image-1><image-2>shared instruction<|im_end|>"
    dataset = SimpleNamespace(
        task_groups={"shared instruction": [0, 1]},
        trajectory_ids=np.arange(2),
        all_steps=[(0, 0), (1, 0)],
        dataset_path=tmp_path,
        get_step_data=lambda episode, frame: {"eagle_content": {"text_list": [prompt]}},
        transforms=lambda value: value,
    )

    def initialize(mixture, data_mixture, **kwargs):
        mixture.datasets = [item for item, _ in data_mixture]

    monkeypatch.setattr(data.LeRobotMixtureDataset, "__init__", initialize)
    monkeypatch.setattr(
        data, "_alignment_task_names", lambda *args: ({0: "VideoUnmask", 1: "VideoUnmaskSwap"}, None)
    )
    mixture = data.FramePathsMixtureDataset(data_mixture=[(dataset, 1)], memory_task_keys=True)
    assert mixture.required_memory_keys == {"VideoUnmask", "VideoUnmaskSwap"}
    rows = [mixture[0], mixture[1]]
    batch = collator(monkeypatch, required=True)(rows)
    assert batch["tasks"] == ["VideoUnmask", "VideoUnmaskSwap"]
    assert all(row["eagle_content"]["text_list"] == [prompt] for row in batch["rows"])
    assert all("memory_key" not in row for row in batch["rows"])
    assert all("memory_key" in row for row in rows)  # Collation does not mutate source features.
    legacy = data.FramePathsMixtureDataset(data_mixture=[(dataset, 1)])
    assert legacy.memory_key_type == "instruction" and "memory_key" not in legacy[0]
    assert collator(monkeypatch)([legacy[0]])["tasks"] == ["shared instruction"]


def test_canonical_collator_rejects_missing_or_mixed_keys(monkeypatch):
    feature = {"eagle_content": {"text_list": ["<image-1><image-2>instruction<|im_end|>"]}}
    with pytest.raises(ValueError, match="requires memory_key"):
        collator(monkeypatch, required=True)([feature])
    with pytest.raises(ValueError, match="Mixed"):
        collator(monkeypatch)([feature, {**feature, "memory_key": "task"}])


def tiny_head():
    head = RAVLAActionHead.__new__(RAVLAActionHead)
    torch.nn.Module.__init__(head)
    head.model = SimpleNamespace(mask_padding=True)
    head.vlln = torch.nn.LayerNorm(8)
    head.vl_self_attention = SelfAttentionTransformer(
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        max_num_positional_embeddings=32,
    )
    return head.eval()


def test_backbone_projection_ignores_padding_and_retains_gradients():
    torch.manual_seed(17)
    head = tiny_head()
    values = torch.randn(2, 4, 8)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    first = head.process_backbone_output(
        BatchFeature(
            {
                "backbone_features": values.clone(),
                "backbone_attention_mask": mask,
            }
        )
    )["backbone_features"]
    extended = torch.cat([torch.full((2, 3, 8), 1000.0), values], dim=1)
    extended[:, 3:][~mask.bool()] = -2000.0
    second = head.process_backbone_output(
        BatchFeature(
            {
                "backbone_features": extended,
                "backbone_attention_mask": torch.cat([torch.zeros(2, 3), mask], dim=1),
            }
        )
    )["backbone_features"]
    torch.testing.assert_close(first, second[:, 3:], rtol=0, atol=0)
    assert torch.count_nonzero(first[~mask.bool()]) == 0
    first.square().sum().backward()
    assert head.vlln.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in head.parameters())


@pytest.mark.parametrize("random_context", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_dit_context_padding_is_ignored_and_random_masks_stay_aligned(k, random_context):
    torch.manual_seed(23)
    model = RAVLADiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        cross_attention_dim=8,
        interleave_self_attention=True,
    ).eval()
    model.mask_padding = True
    inputs = dict(
        hidden_states=torch.randn(2, 3, 8),
        encoder_hidden_states=torch.randn(2, 3, 8),
        encoder_attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
        ret_encoder_hidden_states=torch.randn(2 * k, 4, 8),
        ret_encoder_attention_mask=torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]).repeat(k, 1),
        ret_action_features=torch.randn(2 * k, 2, 8),
        timestep=torch.tensor([5, 9]),
        ret_random=random_context,
    )
    first = model(**inputs)
    changed = dict(inputs)
    for feature_key, mask_key in (
        ("encoder_hidden_states", "encoder_attention_mask"),
        ("ret_encoder_hidden_states", "ret_encoder_attention_mask"),
    ):
        values, mask = inputs[feature_key].clone(), inputs[mask_key]
        values[~mask.bool()] = 1000.0
        changed[feature_key] = torch.cat([torch.full((len(values), 2, 8), -1000.0), values], dim=1)
        changed[mask_key] = torch.cat([torch.zeros(len(mask), 2), mask], dim=1)
    second = model(**changed)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    first.sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_legacy_projection_keeps_original_behavior():
    head = tiny_head()
    head.model.mask_padding = False
    features = torch.randn(2, 4, 8)
    expected = head.vl_self_attention(head.vlln(features))
    actual = head.process_backbone_output(
        BatchFeature(
            {
                "backbone_features": features,
                "backbone_attention_mask": torch.zeros(2, 4),
            }
        )
    )["backbone_features"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
