from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import init_xavier


class EdgeEncoder(nn.Module):
    """
    Model-space edge encoder.

    For an edge e = (u, r, v):

        phi_E(e) = W_E [h_u, h_r, h_v]

    StateReadout uses the same encoder for active-edge evidence and frontier
    summaries, so each graph edge has one neural representation.
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        init_xavier(self.proj)

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> torch.Tensor:
        if src_h.shape != rel_h.shape or src_h.shape != dst_h.shape:
            raise ValueError(
                "src_h, rel_h, and dst_h must have identical shapes, got "
                f"{tuple(src_h.shape)}, {tuple(rel_h.shape)}, {tuple(dst_h.shape)}."
            )

        if src_h.ndim != 2:
            raise ValueError(
                f"edge inputs must have shape [E, H], got {tuple(src_h.shape)}."
            )

        if src_h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected edge hidden_dim={self.hidden_dim}, got {src_h.size(-1)}."
            )

        return self.proj(torch.cat([src_h, rel_h, dst_h], dim=-1))


__all__ = ["EdgeEncoder"]
