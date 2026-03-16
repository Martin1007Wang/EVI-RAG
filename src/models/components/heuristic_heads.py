from __future__ import annotations

import torch
from torch import nn


class LearnedHeuristicHead(nn.Module):
    def __init__(self, *, hidden_dim: int, dropout: float, feature_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        state_features: torch.Tensor,
        question_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.mlp(
            torch.cat((state_features, question_features), dim=-1)
        ).squeeze(-1)
        return logits.to(dtype=torch.float32)


__all__ = ["LearnedHeuristicHead"]
