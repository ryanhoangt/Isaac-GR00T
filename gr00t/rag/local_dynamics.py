"""Offline proprio/action chunk dynamics and differentiable goal-head rollouts.

Inspired by CCIL's normalized residual model. This is an endpoint-loss pilot,
not corrective-label generation or a certificate of real-plant continuity.
"""

from copy import deepcopy

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class PolicyCodec(nn.Module):
    def __init__(self, state_min, state_max, action_min, action_max):
        super().__init__()
        for name, value in locals().copy().items():
            if name in ("state_min", "state_max", "action_min", "action_max"):
                self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        if self.state_min.shape != (8,) or self.action_min.shape != (7,):
            raise ValueError("Expected eight proprio and seven action coordinates")

    @classmethod
    def from_metadata(cls, metadata):
        stats = metadata["new_embodiment"].statistics
        if hasattr(stats, "model_dump"):
            stats = stats.model_dump()

        def vector(kind, keys, name):
            result = []
            for key in keys:
                value = stats[kind][key]
                value = value.model_dump() if hasattr(value, "model_dump") else value
                result.extend(torch.as_tensor(value[name]).reshape(-1).tolist())
            return result

        return cls(
            vector("state", STATE_KEYS, "min"),
            vector("state", STATE_KEYS, "max"),
            vector("action", ACTION_KEYS, "min"),
            vector("action", ACTION_KEYS, "max"),
        )

    @staticmethod
    def normalize(value, low, high):
        span = high - low
        return torch.where(span > 0, 2 * (value - low) / span.clamp_min(1e-12) - 1, torch.zeros_like(value))

    def state(self, raw):
        return self.normalize(raw.float(), self.state_min, self.state_max)

    def actions(self, value, normalized=False):
        if normalized:
            raw = (value[..., :7].float() + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        else:
            raw = value.float()
        # Match deployed command saturation and binary gripper conversion.
        arm = raw[..., :6].clamp(-1, 1)
        middle = (self.action_min[6] + self.action_max[6]) / 2
        grip = torch.where(raw[..., 6:7] >= middle, self.action_max[6], self.action_min[6])
        raw = torch.cat((arm, grip), dim=-1)
        return self.normalize(raw, self.action_min, self.action_max)


class DeltaMLP(nn.Module):
    def __init__(self, hidden=128, horizon=16, spectral=False):
        super().__init__()
        self.hidden, self.horizon, self.spectral = hidden, horizon, spectral
        self.net = nn.Sequential(
            nn.Linear(8 + 7 * horizon, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        if spectral:
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.utils.parametrizations.spectral_norm(layer)
        for name, size in (
            ("input_mean", 8 + 7 * horizon),
            ("input_scale", 8 + 7 * horizon),
            ("output_mean", 3),
            ("output_scale", 3),
        ):
            self.register_buffer(name, torch.ones(size) if name.endswith("scale") else torch.zeros(size))

    @staticmethod
    def inputs(state, actions):
        if state.shape[-1] != 8 or actions.shape[-2:] != (16, 7):
            raise ValueError("Expected Bx8 state and Bx16x7 actions")
        return torch.cat((state.float(), actions.float().flatten(-2)), dim=-1)

    @torch.no_grad()
    def fit_normalization(self, inputs, targets):
        self.input_mean.copy_(inputs.mean(0))
        self.input_scale.copy_((inputs - self.input_mean).abs().mean(0).clamp_min(1e-3))
        self.output_mean.copy_(targets.mean(0))
        self.output_scale.copy_((targets - self.output_mean).abs().mean(0).clamp_min(1e-3))

    def forward(self, inputs):
        normalized = (inputs.float() - self.input_mean) / self.input_scale
        return self.net(normalized) * self.output_scale + self.output_mean


class DynamicsEnsemble(nn.Module):
    def __init__(self, count=3, hidden=128, spectral=False):
        super().__init__()
        self.members = nn.ModuleList([DeltaMLP(hidden=hidden, spectral=spectral) for _ in range(count)])

    def forward(self, state, actions):
        inputs = DeltaMLP.inputs(state, actions)
        predictions = torch.stack([member(inputs) for member in self.members])
        return predictions.mean(0), predictions.std(0, unbiased=False).norm(dim=-1)

    def freeze(self):
        self.eval().requires_grad_(False)
        return self


def differentiable_goal_action(head, visual, state, embodiment_id, context, noise):
    """Actual four-step inference path, with autograd and inference-time dropout."""
    previous = head.training
    head.eval()
    try:
        current = head.process_backbone_output(BatchFeature(data=dict(visual)))
        modes = torch.ones(len(state), device=state.device, dtype=torch.bool)
        dummy = torch.zeros_like(noise)
        prepared = head._context(context, dummy, modes)
        actions = noise
        for step in range(4):
            times = torch.full((len(state),), int(step / 4 * head.num_timestep_buckets), device=state.device)
            velocity = head._velocity(
                current, state, embodiment_id, actions, times, context, dummy, modes, prepared_context=prepared
            )
            actions = actions + velocity / 4
        return actions
    finally:
        head.train(previous)


def endpoint_loss(dynamics, codec, state_raw, predicted_actions, target_delta, std_limit, error_scale=0.05):
    state = codec.state(state_raw)
    actions = codec.actions(predicted_actions, normalized=True)
    prediction, spread = dynamics(state, actions)
    inputs = DeltaMLP.inputs(state, actions)
    # Empirical support heuristic, not a calibrated coverage guarantee.
    standardized = (inputs - dynamics.members[0].input_mean) / dynamics.members[0].input_scale
    supported = (standardized.abs().mean(-1) < 5) & (spread.detach() <= std_limit)
    error = ((prediction - target_delta.float()) / error_scale).square().mean(-1)
    loss = (error * supported).sum() / supported.sum().clamp_min(1)
    return loss, dict(
        endpoint_mse_m2=float((prediction.detach() - target_delta).square().mean()),
        endpoint_supported=float(supported.float().mean()),
        ensemble_std_m=float(spread.detach().mean()),
    )


def clone_frozen(module):
    return deepcopy(module).eval().requires_grad_(False)
