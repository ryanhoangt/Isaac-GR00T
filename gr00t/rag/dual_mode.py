"""Opt-in dual-mode RA-VLA. Mode 0 = action context; mode 1 = visual goal.

No uncertainty gate or claim of geometric recovery is encoded in this module.
The original RAVLA classes and checkpoint behavior are unchanged.
"""

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
from gr00t.rag.ravla import RAVLA, RAVLAActionHead


def validate_modes(modes, batch, device):
    modes = torch.as_tensor(modes, device=device).reshape(-1)
    if modes.shape != (batch,) or not torch.all((modes == 0) | (modes == 1)):
        raise ValueError("context_mode must contain one 0/1 value per observation")
    return modes.bool()


class DualModeActionHead(RAVLAActionHead):
    def __init__(self, config, margin=0.01, lambda_=0.1):
        super().__init__(config, margin, lambda_)
        dim = config.diffusion_model_cfg["cross_attention_dim"]
        self.goal_mode_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.goal_view_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.config.dual_mode = True

    def _context(self, raw, actions, modes):
        # Copy the container: process_backbone_output mutates its input.
        visual = self.process_backbone_output(BatchFeature(data=dict(raw))).backbone_features
        k = len(actions) // len(modes)
        active = modes.repeat_interleave(k)[:, None, None]
        # Remove actions BEFORE the encoder as well as replacing its output.
        safe_actions = torch.where(active, torch.zeros_like(actions), actions)
        encoded = self.ret_action_encoder(safe_actions)
        encoded = torch.where(active, self.goal_mode_token.expand_as(encoded), encoded)
        encoded = encoded + self.ret_position_embedding.weight.unsqueeze(0)
        visual = visual + active.to(visual.dtype) * self.goal_view_embedding
        return visual, encoded

    def _source(self, actions, modes, gaussian=None):
        batch = len(modes)
        mean = actions.reshape(batch, -1, *actions.shape[1:]).mean(1)
        if not modes.any():
            return mean
        if gaussian is None:
            gaussian = torch.randn_like(mean)
        if gaussian.shape != mean.shape:
            raise ValueError("Gaussian source shape differs from action shape")
        return torch.where(modes[:, None, None], gaussian, mean)

    def _velocity(self, current, state, embodiment, x, times, context, actions, modes, prepared_context=None):
        visual, action_context = (
            prepared_context if prepared_context is not None else self._context(context, actions, modes)
        )
        encoded = self.action_encoder(x, times, embodiment)
        if self.config.add_pos_embed:
            positions = torch.arange(x.shape[1], device=x.device)
            encoded = encoded + self.position_embedding(positions).unsqueeze(0)
        state_features = self.state_encoder(state, embodiment)
        future = self.future_tokens.weight.unsqueeze(0).expand(len(x), -1, -1)
        hidden = torch.cat((state_features, future, encoded), dim=1)
        output = self.model(
            hidden_states=hidden,
            encoder_hidden_states=current.backbone_features,
            encoder_attention_mask=current.backbone_attention_mask,
            ret_encoder_hidden_states=visual,
            ret_encoder_attention_mask=context["backbone_attention_mask"],
            ret_action_features=action_context,
            timestep=times,
        )
        return self.action_decoder(output, embodiment)[:, -x.shape[1] :]

    def forward(self, backbone_output, action_input):
        self.set_frozen_modules_to_eval_mode()
        target = action_input.action
        modes = validate_modes(action_input.context_mode, len(target), target.device)
        current = self.process_backbone_output(BatchFeature(data=dict(backbone_output)))
        source = self._source(action_input.ret_actions, modes)
        time = self.sample_time(len(target), dtype=target.dtype, device=target.device)
        x = time[:, None, None] * target + (1 - time[:, None, None]) * source
        velocity = target - source
        times = (time * self.num_timestep_buckets).long()

        def predict(context, actions):
            return self._velocity(
                current,
                action_input.state,
                action_input.embodiment_id,
                x,
                times,
                context,
                actions,
                modes,
            )

        # Common random numbers: both contexts see identical dropout as well as
        # identical x/time/target. Restore post-positive RNG state afterwards.
        devices = [target.device.index] if target.is_cuda else []
        cpu_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state(target.device) if target.is_cuda else None
        positive = predict(action_input.ret_backbone_output, action_input.ret_actions)
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_before)
            if target.is_cuda:
                torch.cuda.set_rng_state(cuda_before, target.device)
            negative = predict(action_input.neg_backbone_output, action_input.neg_ret_actions)

        mask = action_input.action_mask
        denominator = mask.sum((1, 2)).clamp_min(1)
        mse = ((positive - velocity).float().square() * mask).sum((1, 2)) / denominator
        mse_neg = ((negative - velocity).float().square() * mask).sum((1, 2)) / denominator
        valid = action_input.negative_valid.to(mse.dtype).reshape(-1)
        hinge = (self.margin + mse - mse_neg).clamp_min(0)
        margin_loss = (hinge * valid).sum() / valid.sum().clamp_min(1)
        mse_loss = ((positive - velocity).float().square() * mask).sum() / mask.sum()
        result = {
            "loss": mse_loss + self.lambda_ * margin_loss,
            "mse_loss": mse_loss,
            "margin_loss": margin_loss,
            "goal_fraction": modes.float().mean(),
            "negative_valid_fraction": valid.mean(),
        }
        for name, select in (("action", ~modes), ("goal", modes)):
            count = select.sum().clamp_min(1)
            result[f"{name}_mse"] = (mse * select).sum() / count
            result[f"{name}_negative_mse"] = (mse_neg * select).sum() / count
            result[f"{name}_margin"] = (hinge * valid * select).sum() / (valid * select).sum().clamp_min(1)
            result[f"{name}_count"] = select.sum()
        return BatchFeature(data=result)

    @torch.no_grad()
    def get_action(self, backbone_output, action_input):
        state = action_input.state
        modes = validate_modes(action_input.get("context_mode", torch.zeros(len(state))), len(state), state.device)
        current = self.process_backbone_output(BatchFeature(data=dict(backbone_output)))
        actions = self._source(action_input.ret_actions, modes, action_input.get("flow_source"))
        prepared_context = self._context(action_input.ret_backbone_output, action_input.ret_actions, modes)
        for step in range(self.num_inference_timesteps):
            times = torch.full(
                (len(state),),
                int(step / self.num_inference_timesteps * self.num_timestep_buckets),
                device=state.device,
            )
            velocity = self._velocity(
                current,
                state,
                action_input.embodiment_id,
                actions,
                times,
                action_input.ret_backbone_output,
                action_input.ret_actions,
                modes,
                prepared_context=prepared_context,
            )
            actions = actions + (1.0 / self.num_inference_timesteps) * velocity
        return BatchFeature(data={"action_pred": actions})


class DualModeRAVLA(RAVLA):
    def __init__(self, config, local_model_path, margin=0.01, lambda_=0.1):
        super().__init__(config, local_model_path, margin, lambda_)
        head_config = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = DualModeActionHead(head_config, margin, lambda_)
        config.action_head_cfg["dual_mode"] = True

    def augment_inputs(self, inputs):
        if "ret_actions" in inputs:
            return inputs
        raise ValueError("DualModeRAVLA requires explicit context; use prepare_training_batch or DualModePolicy")
