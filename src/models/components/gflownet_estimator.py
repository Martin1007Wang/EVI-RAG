from __future__ import annotations

import torch
from torch import nn


class GFlowNetEstimator(nn.Module):
    """估计 logF(s) / logZ：MLP([state_emb || question_emb]) -> scalar."""

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.log_z_head = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        last_linear = self.log_z_head[-1]
        if not isinstance(last_linear, nn.Linear):
            raise TypeError(f"Expected final layer to be nn.Linear, got {type(last_linear)}")
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.0)

    def build_context(self, state_emb: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        if state_emb.shape != question_tokens.shape:
            raise ValueError(
                "GFlowNetEstimator.build_context expects matching shapes for state_emb and question_tokens, "
                f"got state_emb={tuple(state_emb.shape)}, question_tokens={tuple(question_tokens.shape)}"
            )
        if state_emb.size(-1) != self.hidden_dim:
            raise ValueError(f"hidden_dim mismatch: state_emb.size(-1)={state_emb.size(-1)} vs hidden_dim={self.hidden_dim}")
        return torch.cat([state_emb, question_tokens], dim=-1)

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        if context.size(-1) != 2 * self.hidden_dim:
            raise ValueError(
                f"context last dim must be 2*hidden_dim={2 * self.hidden_dim}, got {context.size(-1)}"
            )
        return self.log_z_head(context).squeeze(-1)


__all__ = ["GFlowNetEstimator"]
