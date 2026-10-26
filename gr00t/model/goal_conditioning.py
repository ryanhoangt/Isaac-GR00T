"""Optional Pilot 2 conditioning; vanilla checkpoints have no added parameters."""

import torch
from torch import nn


class GoalConditioning(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mode_embedding = nn.Embedding(2, feature_dim)
        self.subgoal_view = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.null_goal = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

    def forward(self, current, current_mask, mode, goal=None, goal_mask=None):
        batch = current.shape[0]
        mode = mode.reshape(batch).long()
        if not torch.all((mode == 0) | (mode == 1)):
            raise ValueError("goal_mode_id must contain only 0 or 1")
        active = mode.bool()
        if goal is None:
            if active.any():
                raise ValueError("Active goal conditioning requires goal features")
            # Keep both learned view/null parameters in the graph even if an
            # entire (small, possibly distributed) minibatch is language-only.
            goal = (self.null_goal + 0.0 * self.subgoal_view).expand(batch, 1, -1)
            goal_mask = torch.ones(batch, 1, dtype=torch.bool, device=current.device)
        else:
            goal_mask = goal_mask.bool().clone()
            # Replace the whole inactive sequence before attention, rather than
            # merely feeding black pixels through a biased vision encoder.
            goal = torch.where(active[:, None, None], goal + self.subgoal_view, self.null_goal)
            goal_mask[~active] = False
            goal_mask[~active, 0] = True
        mode_token = self.mode_embedding(mode)[:, None]
        mask = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=current.device), current_mask.bool(), goal_mask),
            dim=1,
        )
        return torch.cat((mode_token, current, goal), dim=1), mask
