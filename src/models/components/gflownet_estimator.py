from __future__ import annotations

import torch
from torch import nn


class GFlowNetEstimator(nn.Module):
    """估计 logF(s) / logZ：MLP([state_emb || question_emb]) -> scalar."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.log_z_head = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if zero_init_last:
            last_linear = self.log_z_head[-1]
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def build_context(self, state_emb: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        if state_emb.shape != question_tokens.shape:
            question_tokens = self._broadcast_question(state_emb, question_tokens)
        return torch.cat([state_emb, question_tokens], dim=-1)

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        return self.log_z_head(context).squeeze(-1)

    def forward(self, state_emb: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        context = self.build_context(state_emb, question_tokens)
        return self.log_z(context)

    @staticmethod
    def _broadcast_question(state_emb: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        if question_tokens.dim() > state_emb.dim():
            return question_tokens
        if question_tokens.dim() < state_emb.dim():
            if (
                question_tokens.dim() >= 2
                and question_tokens.shape[0] == state_emb.shape[0]
                and question_tokens.shape[-1] == state_emb.shape[-1]
            ):
                mid_dims = (1,) * (state_emb.dim() - question_tokens.dim())
                shape = (question_tokens.shape[0],) + mid_dims + (question_tokens.shape[-1],)
                question_tokens = question_tokens.view(shape)
            else:
                leading = (1,) * (state_emb.dim() - question_tokens.dim())
                question_tokens = question_tokens.view(leading + tuple(question_tokens.shape))
        return question_tokens.expand_as(state_emb)


__all__ = ["GFlowNetEstimator"]
