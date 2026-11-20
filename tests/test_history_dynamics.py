import numpy as np
import pytest
import torch

from gr00t.rag.history_dynamics import HistoryDeltaMLP, HistoryEnsemble, history_inputs, history_windows
from gr00t.rag.local_dynamics import DeltaMLP
from test_local_dynamics import codec


def test_history_is_strictly_past_and_common_cohort_has_actual_endpoints():
    state = np.arange(27 * 8, dtype=np.float32).reshape(27, 8)
    action = np.arange(27 * 7, dtype=np.float32).reshape(27, 7)
    values, times = history_windows(state, action)
    s, a, delta, hs, ha = values
    np.testing.assert_array_equal(times, [8, 9, 10])
    np.testing.assert_array_equal(hs[0], state[:8])
    np.testing.assert_array_equal(ha[0], action[:8])
    np.testing.assert_array_equal(a[0], action[8:24])
    np.testing.assert_array_equal(delta[-1], state[26, :3] - state[10, :3])
    np.testing.assert_array_equal(s[-1], state[10])
    with pytest.raises(ValueError, match="too short"):
        history_windows(state[:24], action[:24])


def test_history_does_not_read_future_observations_or_actions():
    state, action = np.zeros((27, 8)), np.zeros((27, 7))
    before, _ = history_windows(state, action)
    state[9:] = 9
    action[8:] = 7
    after, _ = history_windows(state, action)
    for index in (0, 3, 4):
        np.testing.assert_array_equal(before[index][0], after[index][0])
    assert not np.array_equal(before[1][0], after[1][0])
    assert not np.array_equal(before[2][0], after[2][0])


def test_history_feature_dimensions_and_current_relative_encoding():
    c = codec()
    state, action = torch.ones(2, 8) * 0.5, torch.zeros(2, 16, 7)
    past, previous = torch.ones(2, 8, 8) * 0.25, torch.zeros(2, 8, 7)
    for h in (0, 4, 8):
        features = history_inputs(c, state, action, past, previous, h)
        assert features.shape == (2, 120 + 15 * h)
        torch.testing.assert_close(features[:, :120], DeltaMLP.inputs(c.state(state), c.actions(action)))
        if h:
            torch.testing.assert_close(features[:, 120:].reshape(2, h, 15)[..., :8], torch.full((2, h, 8), -0.5))
    with pytest.raises(ValueError, match="Insufficient"):
        history_inputs(c, state, action, past, previous, 9)


def test_zero_history_model_is_exactly_baseline_at_same_seed():
    torch.manual_seed(8)
    original = DeltaMLP()
    torch.manual_seed(8)
    control = HistoryDeltaMLP(0)
    for name, value in original.state_dict().items():
        assert torch.equal(value, control.state_dict()[name])


def test_history_ensemble_roundtrip_and_frozen_action_gradients(tmp_path):
    torch.manual_seed(4)
    model = HistoryEnsemble(4, count=3, hidden=16).freeze()
    state, action = torch.ones(2, 8) * 0.5, torch.zeros(2, 16, 7, requires_grad=True)
    past, previous = torch.ones(2, 8, 8) * 0.5, torch.zeros(2, 8, 7)
    x = history_inputs(codec(), state, action, past, previous, 4)
    prediction, spread = model(x)
    prediction.square().sum().backward()
    assert action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = HistoryEnsemble(4, count=3, hidden=16)
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    torch.testing.assert_close(restored(x.detach())[0], prediction)
    assert (spread >= 0).all()
