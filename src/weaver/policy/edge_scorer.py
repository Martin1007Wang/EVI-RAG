from __future__ import annotations

import math

import torch
from torch import nn

Tensor = torch.Tensor


class QuestionConditionedEdgeScorer(nn.Module):
    """
    Edge action energy scorer.

    Exposes two terms:

      1. score_alignment(q, e):
           Static query-edge alignment, used exclusively by frontier pruning.
           NOT used in edge scoring to avoid double-biasing.

      2. score_state(z, e):
           State-conditioned edge compatibility (the sole scoring signal).
           c_theta(z, e) = MLP([h_z || h_e || h_z * h_e])
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_hidden_dim: int | None = None,
        dropout: float = 0.1,
        interaction_dim: int = 64,
    ) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.hidden_dim = hidden_dim

        inner_dim = int(state_hidden_dim or min(hidden_dim, 256))
        self.interaction_dim = interaction_dim
        self.state_proj = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.edge_proj = nn.Linear(hidden_dim, interaction_dim, bias=False)
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + interaction_dim, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner_dim, 1, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def score_state(self, *, state_h, edge_h):
        interaction = self.state_proj(state_h) * self.edge_proj(edge_h)  # [F, d]
        x = torch.cat([state_h, edge_h, interaction], dim=-1)
        return self.state_mlp(x).squeeze(-1)
