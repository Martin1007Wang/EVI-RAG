from __future__ import annotations

import torch
from torch import nn


class GraphLogZHead(nn.Module):
    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        question_features: torch.Tensor,
        start_summary: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.mlp(
            torch.cat((question_features, start_summary), dim=-1)
        ).squeeze(-1)
        return torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
