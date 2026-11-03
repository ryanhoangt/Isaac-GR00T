"""Isolated goal-controller positive control; the original action head is frozen."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
from gr00t.rag.dual_mode import DualModeActionHead, DualModeRAVLA, validate_modes


class GoalMSEHead(DualModeActionHead):
    """One positive flow-matching pass, without an adherence/negative pass."""

    def __init__(self, config, margin=0.0, lambda_=0.0):
        super().__init__(config, margin, lambda_)
        # HF constructs new submodules under the checkpoint's BF16 default.
        # Beta/Dirichlet sampling requires FP32, then sample_time casts the result.
        self.beta_dist = torch.distributions.Beta(
            torch.tensor(config.noise_beta_alpha, dtype=torch.float32),
            torch.tensor(config.noise_beta_beta, dtype=torch.float32),
        )

    def forward(self, backbone_output, action_input):
        self.set_frozen_modules_to_eval_mode()
        target = action_input.action
        modes = validate_modes(action_input.context_mode, len(target), target.device)
        if not modes.all():
            raise ValueError("Goal positive-control training requires goal-only batches")
        current = self.process_backbone_output(BatchFeature(data=dict(backbone_output)))
        source = self._source(action_input.ret_actions, modes)
        time = self.sample_time(len(target), dtype=target.dtype, device=target.device)
        x = time[:, None, None] * target + (1 - time[:, None, None]) * source
        velocity = target - source
        predicted = self._velocity(
            current,
            action_input.state,
            action_input.embodiment_id,
            x,
            (time * self.num_timestep_buckets).long(),
            action_input.ret_backbone_output,
            action_input.ret_actions,
            modes,
        )
        error = (predicted - velocity).float().square() * action_input.action_mask
        loss = error.sum() / action_input.action_mask.sum().clamp_min(1)
        return BatchFeature(data=dict(loss=loss, mse_loss=loss, goal_mse=loss, margin_loss=loss.new_zeros(())))


class IsolatedGoalHead(DualModeActionHead):
    def __init__(self, config, margin=0.0, lambda_=0.0):
        super().__init__(config, margin, lambda_)
        self.goal_branch = GoalMSEHead(deepcopy(config), margin=0.0, lambda_=0.0)
        self.config.isolated_goal_branch = True
        self.set_trainable_parameters(True, True)

    def set_trainable_parameters(self, tune_projector, tune_diffusion_model):
        super().set_trainable_parameters(tune_projector, tune_diffusion_model)
        if hasattr(self, "goal_branch"):
            self.requires_grad_(False)
            self.goal_branch.set_trainable_parameters(tune_projector, tune_diffusion_model)

    def train(self, mode=True):
        # Owner dropout always stays in eval, even while the goal branch trains.
        nn.Module.train(self, False)
        if hasattr(self, "goal_branch"):
            self.goal_branch.train(mode)
        return self

    def initialize_goal_from_owner(self):
        # These two new owner parameters are absent in one-mode checkpoints.
        # HF empty initialization does not initialize bare nn.Parameters: a NaN
        # goal-view value would contaminate action mode through 0 * embedding.
        nn.init.zeros_(self.goal_mode_token)
        nn.init.zeros_(self.goal_view_embedding)
        state = {key: value for key, value in self.state_dict().items() if not key.startswith("goal_branch.")}
        self.goal_branch.load_state_dict(state, strict=True)
        nn.init.normal_(self.goal_branch.goal_mode_token, std=0.02)
        nn.init.normal_(self.goal_branch.goal_view_embedding, std=0.02)

    def forward(self, backbone_output, action_input):
        return self.goal_branch(backbone_output, action_input)

    @torch.no_grad()
    def get_action(self, backbone_output, action_input):
        modes = validate_modes(
            action_input.get("context_mode", torch.zeros(len(action_input.state))),
            len(action_input.state),
            action_input.state.device,
        )
        if modes.all():
            self.goal_branch.num_inference_timesteps = self.num_inference_timesteps
            return self.goal_branch.get_action(backbone_output, action_input)
        if modes.any():
            raise ValueError("Use homogeneous modes per request for the isolated control")
        return super().get_action(backbone_output, action_input)


class GoalControlRAVLA(DualModeRAVLA):
    def __init__(self, config, local_model_path, margin=0.0, lambda_=0.0):
        super().__init__(config, local_model_path, margin, lambda_)
        self.action_head = IsolatedGoalHead(FlowmatchingActionHeadConfig(**config.action_head_cfg))
        config.action_head_cfg["isolated_goal_branch"] = True


def owner_digest(head):
    digest = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        if name.startswith("goal_branch."):
            continue
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(data))
    return digest.hexdigest()


def assert_warm_owner(head, checkpoint):
    """Compare every original action-head tensor against the source checkpoint."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    weights = json.loads(index.read_text())["weight_map"] if index.exists() else None
    groups = {}
    for name, tensor in head.state_dict().items():
        if name.startswith("goal_branch.") or name in ("goal_mode_token", "goal_view_embedding"):
            continue
        key = "action_head." + name
        if weights is not None and key not in weights:
            raise ValueError("Original action-head weight absent in warm checkpoint: " + key)
        filename = weights[key] if weights is not None else "model.safetensors"
        groups.setdefault(filename, []).append((key, tensor))
    count = 0
    for filename, entries in groups.items():
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as source:
            for key, tensor in entries:
                expected = source.get_tensor(key).to(dtype=tensor.dtype)
                actual = tensor.detach().cpu()
                if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                    raise ValueError("Warm-start Mode-1 tensor differs from checkpoint: " + key)
                count += 1
    return count
