"""PyTorch port of RoboMME FeatureEncoder, MemoryAttention and MemoryRMSNorm.

Reference: robomme_policy_learning/src/mme_vla_suite/models/{representation/
mem_encoder.py,integration/history_gemma.py}. GR00T retains its pretrained FFN
LayerNorm and timestep conditioning; the added memory RMSNorm precedes that LN.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F


class MemoryFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["use_state_emb"]:
            raise ValueError("This baseline encodes visual memory without historical state")
        self.use_pos_emb = config["use_pos_emb"]
        width = config["memory_feature"]["img"]["input_dim"]
        if self.use_pos_emb:
            pos = config["memory_feature"]["pos"]
            self.pos_proj = nn.Linear(pos["input_dim"], pos["hidden_dim"])
            width += pos["hidden_dim"]
        self.encoder_static = nn.Linear(width, config["memory_token_dim"])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, image_emb, pos_emb):
        if self.use_pos_emb:
            image_emb = torch.cat([image_emb, F.silu(self.pos_proj(pos_emb))], dim=-1)
        return self.encoder_static(image_emb)


def rms_normalize(x):
    """Keep the normalization arithmetic in float32, as in MemoryRMSNorm."""
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return normalized


def apply_rope(x, positions, base):
    # Match openpi.models.gemma._apply_rope: split-half rotation, FP32 phases.
    frequency = base ** (torch.arange(x.shape[-1] // 2, device=x.device).float() * (2.0 / x.shape[-1]))
    phases = positions.float()[:, None] / frequency[None]
    cos, sin = phases.cos()[None, :, None], phases.sin()[None, :, None]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class MemoryModulation(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        if dim != config["memory_token_dim"]:
            raise ValueError(f"Memory width {config['memory_token_dim']} must match DiT inner width {dim}")
        attention = config["attention"]
        self.heads, self.head_dim = attention["num_heads"], attention["head_dim"]
        if self.heads < 1 or self.head_dim < 2 or self.head_dim % 2 or self.heads * self.head_dim != dim:
            raise ValueError("Memory query heads must span DiT width with an even head dimension")
        self.rope_base = attention["rope_base"]
        self.norm_scale = nn.Parameter(torch.zeros(dim))
        self.q = nn.Linear(dim, self.heads * self.head_dim, bias=False)
        self.kv = nn.Linear(dim, 2 * self.head_dim, bias=False)  # one KV head
        self.out = nn.Linear(self.heads * self.head_dim, dim, bias=False)
        self.modulation = nn.Linear(dim, 2 * dim)
        for module in (self.q, self.kv, self.out):
            nn.init.normal_(module.weight, std=dim**-0.5)
        nn.init.normal_(self.modulation.weight, std=attention["output_init_std"])
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, memory, mask):
        # DiT validates nonempty masks once per pass, not once per layer.
        if mask.dtype != torch.bool or mask.shape != memory.shape[:2]:
            raise ValueError("Memory mask must be boolean [batch,memory_tokens]")
        batch, length, _ = x.shape
        mem_len = memory.shape[1]
        query = (rms_normalize(x) * (1 + self.norm_scale.float())).to(x.dtype)
        safe_memory = memory.masked_fill(~mask[..., None], 0)
        context = (rms_normalize(safe_memory) * (1 + self.norm_scale.float())).to(memory.dtype)
        q = self.q(query).reshape(batch, length, self.heads, self.head_dim)
        k, v = self.kv(context).reshape(batch, mem_len, 2, self.head_dim).unbind(2)
        q = apply_rope(q, torch.arange(mem_len, mem_len + length, device=x.device), self.rope_base)
        k = apply_rope(k[:, :, None], torch.arange(mem_len, device=x.device), self.rope_base)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2).expand(-1, self.heads, -1, -1),
            v[:, None].expand(-1, self.heads, -1, -1),
            attn_mask=mask[:, None, None, :],
        )
        attended = self.out(attended.transpose(1, 2).reshape(batch, length, -1))
        scale, shift = self.modulation(attended).chunk(2, dim=-1)
        return (rms_normalize(x) * (1 + scale.float()) + shift.float()).to(x.dtype)


def validate_memory_config(config, dim):
    """Validate architecture fields without depending on benchmark data code."""
    config = copy.deepcopy(config)
    if config["integration_type"] != "modulation" or config["representation_type"] != "perceptual":
        raise ValueError("Only perceptual memory-as-modulator is supported")
    if config["num_views"] != 1 or config["use_state_emb"]:
        raise ValueError("This baseline uses front-view visual memory without historical state")
    if config["memory_token_dim"] != dim:
        raise ValueError(f"Memory width {config['memory_token_dim']} must match DiT inner width {dim}")
    return config
