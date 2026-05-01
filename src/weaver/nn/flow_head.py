from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import build_mlp, require_finite, zero_last_linear


class FlowHead(nn.Module):
    """
    Estimate state log-flow from query-conditioned state representation.

        log F_theta(s | q) = f_theta(h_s)

    The root state defines the partition value:

        log Z(q) := log F_theta(s_0 | q)

    No separate Z head is used.
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
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        num_layers = int(num_layers)
        if num_layers not in {1, 2}:
            raise ValueError(f"flow_head.num_layers must be 1 or 2, got {num_layers}.")

        if num_layers == 1:
            self.net = nn.Linear(self.hidden_dim, 1)
        else:
            self.net = build_mlp(
                input_dim=self.hidden_dim,
                output_dim=1,
                hidden_dim=max(self.hidden_dim // 2, 1),
                num_layers=2,
                dropout=float(dropout),
            )

        if zero_init:
            zero_last_linear(self.net)

    def forward(self, *, state_h: torch.Tensor) -> torch.Tensor:
        state_h = require_finite(state_h, name="state_h")

        if state_h.ndim != 2:
            raise ValueError(
                f"state_h must have shape [B, H], got {tuple(state_h.shape)}."
            )
        if state_h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected state_h hidden_dim={self.hidden_dim}, got {state_h.size(-1)}."
            )

        return self.net(state_h).squeeze(-1)


__all__ = ["FlowHead"]
