from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .topology import GraphTopology


@dataclass(frozen=True)
class GroupedLocalNodeIndex:
    """Grouped graph-local node indices with hidden ptr bookkeeping."""

    local_indices: torch.Tensor
    _group_ptr: torch.Tensor

    @classmethod
    def from_group_ptr(
        cls,
        *,
        local_indices: torch.Tensor,
        group_ptr: torch.Tensor,
        num_groups: int,
        field_name: str,
    ) -> "GroupedLocalNodeIndex":
        grouped_index = cls(local_indices=local_indices, _group_ptr=group_ptr)
        grouped_index.validate(num_groups=num_groups, field_name=field_name)
        return grouped_index

    def validate(self, *, num_groups: int, field_name: str) -> None:
        if self.local_indices.dtype != torch.long or self.local_indices.dim() != 1:
            raise ValueError(
                f"{field_name} local_indices must be 1D torch.long, got "
                f"{self.local_indices.dtype} {tuple(self.local_indices.shape)}."
            )
        if self._group_ptr.dtype != torch.long or self._group_ptr.dim() != 1:
            raise ValueError(
                f"{field_name} group_ptr must be 1D torch.long, got "
                f"{self._group_ptr.dtype} {tuple(self._group_ptr.shape)}."
            )
        if int(self._group_ptr.numel()) != int(num_groups) + 1:
            raise ValueError(
                f"{field_name} group_ptr must have length num_groups + 1: "
                f"group_ptr={int(self._group_ptr.numel())}, num_groups={int(num_groups)}."
            )
        if int(self._group_ptr[0].item()) != 0:
            raise ValueError(f"{field_name} group_ptr must start at 0.")
        if bool((self._group_ptr[1:] < self._group_ptr[:-1]).any().item()):
            raise ValueError(f"{field_name} group_ptr must be non-decreasing.")
        if int(self._group_ptr[-1].item()) != int(self.local_indices.numel()):
            raise ValueError(
                f"{field_name} group_ptr mismatch with local_indices length."
            )

    def counts(
        self,
        *,
        clamp_negative: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        counts = self._group_ptr[1:] - self._group_ptr[:-1]
        if clamp_negative:
            counts = counts.clamp(min=0)
        if device is None:
            return counts
        return counts.to(device=device)

    def graph_index(
        self,
        *,
        device: torch.device | None = None,
        clamp_negative: bool = False,
    ) -> torch.Tensor:
        target_device = self.local_indices.device if device is None else device
        counts = self.counts(clamp_negative=clamp_negative, device=target_device)
        if int(counts.numel()) == 0:
            return torch.empty((0,), device=target_device, dtype=torch.long)
        return torch.arange(
            int(counts.numel()), device=target_device, dtype=torch.long
        ).repeat_interleave(counts)


@dataclass(frozen=True)
class GraphObservation:
    """Static exogenous observations for a graph-search episode."""

    node_features: torch.Tensor
    relation_features: torch.Tensor
    node_ids: torch.Tensor
    question_embedding: torch.Tensor
    question_context: torch.Tensor
    question_valid_mask: torch.Tensor
    q_local_indices: GroupedLocalNodeIndex
    sample_ids: tuple[str, ...]

    @property
    def num_graphs(self) -> int:
        return int(self.question_embedding.size(0))

    def validate(self, *, topology: "GraphTopology") -> None:
        if self.node_features.dim() != 2 or not torch.is_floating_point(
            self.node_features
        ):
            raise ValueError(
                "node_features must be 2D floating point in graph observation."
            )
        if int(self.node_features.size(0)) != int(topology.num_nodes):
            raise ValueError(
                "node_features row count mismatch with topology.num_nodes in graph observation: "
                f"node_features={int(self.node_features.size(0))}, num_nodes={int(topology.num_nodes)}."
            )
        if self.relation_features.dim() != 2 or not torch.is_floating_point(
            self.relation_features
        ):
            raise ValueError(
                "relation_features must be 2D floating point in graph observation."
            )
        if self.node_ids.dtype != torch.long or self.node_ids.dim() != 1:
            raise ValueError("node_ids must be 1D torch.long in graph observation.")
        if int(self.node_ids.numel()) != int(topology.num_nodes):
            raise ValueError(
                "node_ids length mismatch with topology.num_nodes in graph observation: "
                f"node_ids={int(self.node_ids.numel())}, num_nodes={int(topology.num_nodes)}."
            )
        if self.question_embedding.dim() != 2 or not torch.is_floating_point(
            self.question_embedding
        ):
            raise ValueError(
                "question_embedding must be 2D floating point in graph observation."
            )
        if int(self.question_embedding.size(0)) != int(topology.num_graphs):
            raise ValueError(
                "question_embedding batch mismatch with topology.num_graphs in graph observation: "
                f"question_embedding={int(self.question_embedding.size(0))}, num_graphs={int(topology.num_graphs)}."
            )
        if self.question_context.dim() != 3 or not torch.is_floating_point(
            self.question_context
        ):
            raise ValueError(
                "question_context must be 3D floating point in graph observation."
            )
        if int(self.question_context.size(0)) != int(topology.num_graphs):
            raise ValueError(
                "question_context batch mismatch with topology.num_graphs in graph observation: "
                f"question_context={int(self.question_context.size(0))}, num_graphs={int(topology.num_graphs)}."
            )
        if (
            self.question_valid_mask.dtype != torch.bool
            or self.question_valid_mask.dim() != 2
        ):
            raise ValueError(
                "question_valid_mask must be 2D bool in graph observation."
            )
        if tuple(self.question_valid_mask.shape) != tuple(
            self.question_context.shape[:2]
        ):
            raise ValueError(
                "question_valid_mask shape mismatch with question_context in graph observation."
            )
        if bool((~self.question_valid_mask).all(dim=1).any().item()):
            raise ValueError(
                "question_valid_mask contains rows without valid tokens in graph observation."
            )
        if len(self.sample_ids) != int(topology.num_graphs):
            raise ValueError(
                "sample_ids length mismatch with topology.num_graphs in graph observation: "
                f"sample_ids={len(self.sample_ids)}, num_graphs={int(topology.num_graphs)}."
            )
        self.q_local_indices.validate(
            num_groups=topology.num_graphs,
            field_name="q_local_indices",
        )
        if int(topology.edge_type.numel()) > 0 and int(
            self.relation_features.size(0)
        ) <= int(topology.edge_type.max().item()):
            raise ValueError(
                "relation_features do not cover all topology edge types in graph observation."
            )


@dataclass(frozen=True)
class SearchObservation:
    """Lightweight observation metadata needed after encoding finishes."""

    node_ids: torch.Tensor
    q_local_indices: GroupedLocalNodeIndex
    sample_ids: tuple[str, ...]

    @classmethod
    def from_graph_observation(
        cls, observation: GraphObservation
    ) -> "SearchObservation":
        return cls(
            node_ids=observation.node_ids,
            q_local_indices=observation.q_local_indices,
            sample_ids=tuple(str(sample_id) for sample_id in observation.sample_ids),
        )


__all__ = ["GraphObservation", "GroupedLocalNodeIndex", "SearchObservation"]
