"""Check pair ordering and stable BF16 training for the retriever's SimCLR loss."""

import math

import pytest
import torch
from torch.nn.functional import normalize

from gr00t.rag.retriever import RetrieverWrapperWithHead


def loss(embeddings, temperature=.1):
    return RetrieverWrapperWithHead.compute_simclr_loss(None, embeddings, temperature)


def test_positives_are_corresponding_rows_in_opposite_halves():
    vectors = torch.eye(2).repeat(2, 1)
    expected = math.log1p(2 * math.exp(-10))
    assert loss(vectors).item() == pytest.approx(expected, abs=1e-7)
    # Swapping the second half makes the known positives point to orthogonal vectors.
    wrong_order = vectors[[0, 1, 3, 2]]
    assert loss(wrong_order) > loss(vectors) + 9


def test_bfloat16_embeddings_produce_float32_loss_and_finite_gradients():
    generator = torch.Generator().manual_seed(42)
    vectors = normalize(torch.randn(8, 16, generator=generator), dim=-1).to(torch.bfloat16).requires_grad_()
    value = loss(vectors)
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    assert torch.isfinite(vectors.grad).all()
    assert vectors.grad.abs().sum() > 0


@pytest.mark.parametrize("count", [0, 2, 3, 5])
def test_missing_negatives_or_unequal_halves_are_rejected(count):
    with pytest.raises(ValueError, match="two equal halves"):
        loss(torch.ones(count, 4))


def test_zero_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        loss(torch.eye(4), temperature=0)
