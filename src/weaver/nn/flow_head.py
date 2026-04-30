from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import build_mlp, init_xavier, require_finite, zero_last_linear


class FlowHead(nn.Module):
    """
    Estimate state log-flow log F(s | q).

    The root state is not handled by a separate Z head. For root state s0:

        log Z(q) := log F(s0 | q)

    This keeps a single source of truth for GFlowNet state flow.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        self.score_scale = self.hidden_dim**-0.5

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.s_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.residual = build_mlp(
            input_dim=self.hidden_dim * 3,
            output_dim=1,
            hidden_dim=max(self.hidden_dim // 2, 1),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

        init_xavier(self.q_proj)
        init_xavier(self.s_proj)

        if zero_init:
            zero_last_linear(self.residual)

    def forward(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        query_h = require_finite(query_h, name="query_h")
        state_h = require_finite(state_h, name="state_h")

        if query_h.ndim != 2 or state_h.ndim != 2:
            raise ValueError(
                "query_h and state_h must have shape [B, H], "
                f"got query_h={tuple(query_h.shape)}, state_h={tuple(state_h.shape)}."
            )
        if query_h.shape != state_h.shape:
            raise ValueError(
                f"query_h and state_h shape mismatch: {tuple(query_h.shape)} != {tuple(state_h.shape)}."
            )
        if query_h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {query_h.size(-1)}."
            )

        query_h = self.query_norm(query_h)
        state_h = self.state_norm(state_h)

        q = self.q_proj(query_h)
        s = self.s_proj(state_h)

        bilinear = (q * s).sum(dim=-1) * self.score_scale
        residual = self.residual(
            torch.cat([query_h, state_h, query_h * state_h], dim=-1)
        ).squeeze(-1)

        return bilinear + residual


__all__ = ["FlowHead"]
