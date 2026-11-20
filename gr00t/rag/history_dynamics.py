"""Causal proprio/action-history dynamics; no privileged environment state."""

import numpy as np
import torch
from torch import nn

from gr00t.rag.endpoint_data import read_episode, windows
from gr00t.rag.local_dynamics import DeltaMLP


def history_windows(states, actions, context=8, horizon=16):
    if context < 1 or horizon != 16:
        raise ValueError("Use positive history and horizon 16")
    # Validate raw episode shapes/finiteness and require observed endpoints.
    current, future, delta = windows(states, actions, horizon)
    states, actions = np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)
    times = np.arange(context, len(current))
    if not len(times):
        raise ValueError("Episode too short for observed history and endpoint")
    past_state = np.stack([states[t - context : t] for t in times])
    past_action = np.stack([actions[t - context : t] for t in times])
    return (current[times], future[times], delta[times], past_state, past_action), times


def collect_history(recipe, split, context=8):
    collected = [[] for _ in range(5)]
    provenance = []
    for item in recipe["split"]:
        for episode in item[split + "_episode_ids"]:
            states, actions = read_episode(recipe["args"]["dataset_paths"][item["dataset"]], episode)
            values, times = history_windows(states, actions, context)
            for parts, value in zip(collected, values):
                parts.append(value)
            provenance.extend((item["dataset"], episode, int(t)) for t in times)
    return tuple(torch.from_numpy(np.concatenate(parts)) for parts in collected), provenance


def history_inputs(codec, state, actions, past_state, past_action, steps):
    if steps < 0 or steps > past_state.shape[-2] or past_action.shape[-2] != past_state.shape[-2]:
        raise ValueError("Insufficient or inconsistent causal history")
    if past_state.shape[-1] != 8 or past_action.shape[-1] != 7:
        raise ValueError("Expected history proprio=8, action=7")
    state = codec.state(state)
    base = DeltaMLP.inputs(state, codec.actions(actions))
    if steps == 0:
        return base
    relative = codec.state(past_state[:, -steps:]) - state[:, None]
    previous = codec.actions(past_action[:, -steps:])
    return torch.cat((base, torch.cat((relative, previous), -1).flatten(1)), -1)


class HistoryDeltaMLP(DeltaMLP):
    """Same depth/width/normalization as the baseline, with extra causal inputs."""

    def __init__(self, steps, hidden=128):
        nn.Module.__init__(self)
        if steps < 0:
            raise ValueError("Negative history")
        self.steps = steps
        dim = 120 + 15 * steps
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )
        for name, size in (("input_mean", dim), ("input_scale", dim), ("output_mean", 3), ("output_scale", 3)):
            self.register_buffer(name, torch.ones(size) if name.endswith("scale") else torch.zeros(size))


class HistoryEnsemble(nn.Module):
    def __init__(self, steps, count=3, hidden=128):
        super().__init__()
        self.members = nn.ModuleList([HistoryDeltaMLP(steps, hidden) for _ in range(count)])

    def forward(self, inputs):
        predictions = torch.stack([member(inputs) for member in self.members])
        return predictions.mean(0), predictions.std(0, unbiased=False).norm(dim=-1)

    def freeze(self):
        return self.eval().requires_grad_(False)
