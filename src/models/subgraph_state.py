from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SubgraphState:
    """Immutable snapshot of the current MDP subgraph state."""

    active_nodes: torch.Tensor
    active_edges: torch.Tensor

    def __post_init__(self) -> None:
        if self.active_nodes.dtype != torch.bool:
            raise TypeError("active_nodes must be a torch.bool tensor.")
        if self.active_edges.dtype != torch.bool:
            raise TypeError("active_edges must be a torch.bool tensor.")
        if self.active_nodes.dim() != 1:
            raise ValueError("active_nodes must be a 1D tensor.")
        if self.active_edges.dim() != 1:
            raise ValueError("active_edges must be a 1D tensor.")
        if self.active_nodes.device != self.active_edges.device:
            raise ValueError(
                "active_nodes and active_edges must live on the same device."
            )

    @classmethod
    def from_tensors(
        cls,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> "SubgraphState":
        # clone() breaks the version-counter alias with rollout's mutable state.
        return cls(
            active_nodes=active_nodes.detach().clone(),
            active_edges=active_edges.detach().clone(),
        )


__all__ = ["SubgraphState"]
