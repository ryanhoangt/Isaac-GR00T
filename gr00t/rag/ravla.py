import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_, normal_
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.transforms import DefaultDataCollator
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHeadConfig,
    FlowmatchingActionHead,
)
from gr00t.model.action_head.cross_attention_dit import DiT
from gr00t.rag.retriever import retrieve_from_memory


class RAVLADataCollator(DefaultDataCollator):
    def __init__(self, *args, require_memory_key=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_memory_key = require_memory_key

    def __call__(self, features):
        supplied = ["memory_key" in feature for feature in features]
        if any(supplied) and not all(supplied):
            raise ValueError("Mixed canonical and instruction memory keys in one batch")
        if self.require_memory_key and not all(supplied):
            raise ValueError("Canonical memory lookup requires memory_key from the dataset")
        tasks = []
        for feature in features:
            if all(supplied):
                task = feature["memory_key"]
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("Invalid canonical memory key")
            else:
                text = feature["eagle_content"]["text_list"][0]
                task = text.split("<image-1><image-2>")[1].split("<|im_end|>")[0]
            tasks.append(task)

        clean = [{key: value for key, value in feature.items() if key != "memory_key"} for feature in features]
        batch = super().__call__(clean)
        batch["tasks"] = tasks

        return batch


class RAVLA(GR00T_N1_5):
    def __init__(self, config, local_model_path, margin=0.0, lambda_=0.0, mask_padding=None):
        super().__init__(config, local_model_path)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = RAVLAActionHead(action_head_cfg, margin, lambda_)
        if mask_padding is not None:
            config.ravla_mask_padding = mask_padding
        self.action_head.model.mask_padding = getattr(config, "ravla_mask_padding", False)

    def set_retrieval_components(self, retriever, memory, k):
        self.retriever = [retriever.eval().requires_grad_(False)]
        self.memory = memory
        self.k = k

    def init_new_weights(self):
        xavier_normal_(self.action_head.ret_action_encoder.weight)
        constant_(self.action_head.ret_action_encoder.bias, 0)
        normal_(self.action_head.ret_position_embedding.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def augment_inputs(self, inputs):
        tasks = inputs.pop("tasks")
        queries = self.retriever[0](self, inputs)

        (_, _, ret_actions), (ret_backbone_features, ret_backbone_attention_masks) = retrieve_from_memory(
            self.memory, self.k, tasks, queries
        )
        # inputs["action"]: (    B, 16, 32)
        #      ret_actions: (B * k, 16, 32)

        inputs["ret_actions"] = ret_actions
        inputs["ret_backbone_output"] = BatchFeature(
            data={
                "backbone_features": ret_backbone_features,
                "backbone_attention_mask": ret_backbone_attention_masks,
            }
        )
        return inputs

    def forward(self, inputs):
        inputs = self.augment_inputs(inputs)
        return super().forward(inputs)

    @torch.no_grad()
    def get_action(self, inputs):
        inputs = self.augment_inputs(inputs)
        return super().get_action(inputs)


class RAVLAActionHead(FlowmatchingActionHead):
    def __init__(self, config, margin=0.0, lambda_=0.0):
        super().__init__(config)
        self.model = RAVLADiT(**config.diffusion_model_cfg)

        cross_attention_dim = config.diffusion_model_cfg["cross_attention_dim"]
        self.ret_action_encoder = nn.Linear(config.action_dim, cross_attention_dim)
        self.ret_position_embedding = nn.Embedding(config.action_horizon, cross_attention_dim)

        self.margin = margin
        self.lambda_ = lambda_

    def process_backbone_output(self, backbone_output):
        if not getattr(self.model, "mask_padding", False):
            return super().process_backbone_output(backbone_output)
        # The VL self-attention has absolute positions and no padding-mask API.
        # Process equal-length, unpadded sequences together, then restore layout.
        features = backbone_output["backbone_features"]
        valid = backbone_output["backbone_attention_mask"].bool()
        lengths = valid.sum(-1)
        if not lengths.all():
            raise ValueError("Backbone context must contain valid tokens")
        output = torch.zeros_like(features)
        for length in lengths.unique().tolist():
            rows = (lengths == length).nonzero(as_tuple=True)[0]
            packed = features[rows][valid[rows]].reshape(len(rows), length, features.shape[-1])
            processed = self.vl_self_attention(self.vlln(packed))
            padded = torch.zeros_like(features[rows]).masked_scatter(
                valid[rows].unsqueeze(-1).expand_as(features[rows]), processed.reshape(-1)
            )
            output = output.index_copy(0, rows, padded)
        backbone_output["backbone_features"] = output
        return backbone_output

    def forward(self, backbone_output, action_input):
        self.set_frozen_modules_to_eval_mode()

        ret_actions = action_input.pop("ret_actions")
        ret_backbone_output = action_input.pop("ret_backbone_output")

        backbone_output = self.process_backbone_output(backbone_output)
        ret_backbone_output = self.process_backbone_output(ret_backbone_output)

        ret_action_features = self.ret_action_encoder(ret_actions)
        ret_action_features += self.ret_position_embedding.weight.unsqueeze(0)

        states, actions = action_input.state, action_input.action
        embodiment_ids = action_input.embodiment_id

        B, L, D = actions.shape
        k = ret_actions.shape[0] // B
        noise = ret_actions.reshape(B, k, L, D).mean(1)

        t = self.sample_time(B, dtype=self.dtype, device=self.device)
        noisy_actions = t[:, None, None] * actions + (1 - t[:, None, None]) * noise
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_actions, t_discretized, embodiment_ids)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(L, device=self.device)
            pos_embeds = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embeds

        state_features = self.state_encoder(states, embodiment_ids)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        input_embeds = torch.cat([state_features, future_tokens, action_features], dim=1)

        hidden_states = self.model(
            hidden_states=input_embeds,
            encoder_hidden_states=backbone_output.backbone_features,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            ret_encoder_hidden_states=ret_backbone_output.backbone_features,
            ret_encoder_attention_mask=ret_backbone_output.backbone_attention_mask,
            ret_action_features=ret_action_features,
            timestep=t_discretized,
        )
        preds = self.action_decoder(hidden_states, embodiment_ids)
        pred_velocity = preds[:, -L:]

        hidden_states_r = self.model(
            hidden_states=input_embeds,
            encoder_hidden_states=backbone_output.backbone_features,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            ret_encoder_hidden_states=ret_backbone_output.backbone_features,
            ret_encoder_attention_mask=ret_backbone_output.backbone_attention_mask,
            ret_action_features=ret_action_features,
            timestep=t_discretized,
            ret_random=True,
        )
        preds_r = self.action_decoder(hidden_states_r, embodiment_ids)
        pred_velocity_r = preds_r[:, -L:]

        action_mask = action_input.action_mask
        mse = F.mse_loss(pred_velocity.float(), velocity.float(), reduction="none") * action_mask
        mse_r = F.mse_loss(pred_velocity_r.float(), velocity.float(), reduction="none") * action_mask
        mse_loss = mse.sum() / action_mask.sum()

        mse = mse.sum((1, 2)) / action_mask.sum((1, 2))
        mse_r = mse_r.sum((1, 2)) / action_mask.sum((1, 2))
        margin_loss = (self.margin - (mse_r - mse)).clamp(min=0).mean()

        loss = mse_loss + self.lambda_ * margin_loss
        return BatchFeature(
            data={
                "mse_loss": mse_loss,
                "margin_loss": margin_loss,
                "loss": loss,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output, action_input):
        ret_actions = action_input.pop("ret_actions")
        ret_backbone_output = action_input.pop("ret_backbone_output")

        backbone_output = self.process_backbone_output(backbone_output)
        ret_backbone_output = self.process_backbone_output(ret_backbone_output)

        ret_action_features = self.ret_action_encoder(ret_actions)
        ret_action_features += self.ret_position_embedding.weight.unsqueeze(0)

        states = action_input.state
        embodiment_ids = action_input.embodiment_id

        B, L, D = states.shape[0], self.config.action_horizon, self.config.action_dim
        k = ret_actions.shape[0] // B
        actions = ret_actions.reshape(B, k, L, D).mean(1)

        state_features = self.state_encoder(states, embodiment_ids)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_discretized = int(t * self.num_timestep_buckets)
            t_discretized = torch.full((B,), t_discretized, device=self.device)
            action_features = self.action_encoder(actions, t_discretized, embodiment_ids)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(L, device=self.device)
                pos_embeds = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embeds

            input_embeds = torch.cat([state_features, future_tokens, action_features], dim=1)

            hidden_states = self.model(
                hidden_states=input_embeds,
                encoder_hidden_states=backbone_output.backbone_features,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                ret_encoder_hidden_states=ret_backbone_output.backbone_features,
                ret_encoder_attention_mask=ret_backbone_output.backbone_attention_mask,
                ret_action_features=ret_action_features,
                timestep=t_discretized,
            )
            preds = self.action_decoder(hidden_states, embodiment_ids)
            pred_velocity = preds[:, -L:]

            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})


class RAVLADiT(DiT):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        return_all_hidden_states=False,
        ret_encoder_hidden_states=None,
        ret_encoder_attention_mask=None,
        ret_action_features=None,
        ret_random=False,
    ):
        t_embeds = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        ret_encoder_hidden_states = ret_encoder_hidden_states.contiguous()
        ret_action_features = ret_action_features.contiguous()

        all_hidden_states = [hidden_states]

        ret_encoder_hidden_states = torch.cat([ret_encoder_hidden_states, ret_action_features], dim=1)

        k = len(ret_encoder_hidden_states) // len(encoder_hidden_states)
        ret_masks = None
        if getattr(self, "mask_padding", False):
            if encoder_attention_mask is None or ret_encoder_attention_mask is None:
                raise ValueError("Padding-aware RA-VLA requires query and memory masks")
            encoder_attention_mask = encoder_attention_mask.bool()
            ret_masks = torch.cat(
                [
                    ret_encoder_attention_mask.bool(),
                    torch.ones(ret_action_features.shape[:2], dtype=torch.bool, device=ret_action_features.device),
                ],
                dim=1,
            )
        if ret_random:
            ret_encoder_hidden_states = ret_encoder_hidden_states.roll(k, dims=0)
            if ret_masks is not None:
                ret_masks = ret_masks.roll(k, dims=0)

        ret_encoder_hidden_states = [ret_encoder_hidden_states[i::k] for i in range(k)]
        if ret_masks is not None:
            ret_masks = [ret_masks[i::k] for i in range(k)]
        # NOTE: hard-coded indices, num_layers(=16)
        ret_encoder_hidden_states_indices = {
            1: [0, 0, 0, 0, 0, 0, 0, 0],
            2: [1, 1, 1, 1, 0, 0, 0, 0],
            3: [2, 2, 1, 1, 1, 0, 0, 0],
            4: [3, 3, 2, 2, 1, 1, 0, 0],
            5: [4, 3, 2, 2, 1, 1, 0, 0],
            6: [5, 4, 3, 2, 1, 1, 0, 0],
            7: [6, 5, 4, 3, 2, 1, 0, 0],
            8: [7, 6, 5, 4, 3, 2, 1, 0],
        }[k]

        for block_index, block in enumerate(self.transformer_blocks):
            if block_index % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(hidden_states, temb=t_embeds)
            else:
                index = ret_encoder_hidden_states_indices[block_index // 2]
                cat_encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, ret_encoder_hidden_states[index]], dim=1
                )
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=cat_encoder_hidden_states,
                    attention_mask=(
                        torch.cat([encoder_attention_mask, ret_masks[index]], dim=1)[:, None, :]
                        if ret_masks is not None
                        else None
                    ),
                    temb=t_embeds,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(t_embeds)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
