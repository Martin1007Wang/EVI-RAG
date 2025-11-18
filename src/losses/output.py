from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class LossOutput:
    """Minimal container with scalar metrics split into components and generic metrics."""

    loss: torch.Tensor
    components: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


__all__ = ["LossOutput"]
