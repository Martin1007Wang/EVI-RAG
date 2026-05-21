from __future__ import annotations

import torch
from torch import nn


class EdgeEncoder(nn.Module):
    """
    Build a role-preserving KG edge token for e=(u,r,v).

        h_e = concat(h_u, h_r, h_v)

    Contract:
    - src_h, rel_h, dst_h are already produced by FeatureEncoder.
    - FeatureEncoder owns all semantic/model-space projection decisions.
    - EdgeEncoder does not normalize, project, detach, cast, reshape, or move tensors.
    - The output is a structured model-space edge token with dimension 3H.
    - Consumers decide how to compress or score it.
    """

    output_multiplier: int = 3

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.output_multiplier * self.hidden_dim

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([src_h, rel_h, dst_h], dim=-1)


__all__ = ["EdgeEncoder"]
