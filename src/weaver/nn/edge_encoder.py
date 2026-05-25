from __future__ import annotations

import torch
from torch import nn

from src.utils.nn_utils import init_xavier


class EdgeEncoder(nn.Module):
    """
    Encode directed KG edges from model-space endpoint and relation tokens.

    src_h, rel_h, dst_h: [E, H]
    output: W_src src_h + W_rel rel_h + W_dst dst_h, shape [E, H]
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.src_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.rel_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dst_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.src_proj(src_h) + self.rel_proj(rel_h) + self.dst_proj(dst_h)

    def reset_parameters(self) -> None:
        init_xavier(self.src_proj)
        init_xavier(self.rel_proj)
        init_xavier(self.dst_proj)


__all__ = ["EdgeEncoder"]
