from __future__ import annotations

from torch import nn


def init_linear_xavier(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


__all__ = ["init_linear_xavier"]
