# src/models/components/positional_encoding.py
"""
[系统实体] 正弦位置编码
"""
from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0.")
        half_dim = self.dim // 2
        inv_freq = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / max(half_dim, 1))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._has_odd = bool(self.dim % 2)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        steps = steps.to(device=self.inv_freq.device, dtype=torch.float32).view(-1, 1)
        freqs = steps * self.inv_freq.view(1, -1)
        emb = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1)
        if self._has_odd:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


__all__ = ["SinusoidalPositionalEncoding"]
