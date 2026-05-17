from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class GraphContext:
    """
    Label-free static graph context for rollout and policy evaluation.

    This object intentionally excludes target and oracle shortest-path fields.
    Inference code should be able to operate with only this context.
    """

    edge_index: torch.Tensor
    node_to_graph: torch.Tensor
    edge_to_graph: torch.Tensor
    anchor_mask: torch.Tensor
    num_nodes: int
    num_edges: int
    num_graphs: int
    device: torch.device

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> GraphContext:
        device = batch.edge_index.device if device is None else device
        edge_index = batch.edge_index.to(device=device)
        node_to_graph = batch.batch.to(device=device).view(-1)
        edge_to_graph = node_to_graph.index_select(0, edge_index[0])
        anchor_mask = torch.zeros(
            int(batch.num_nodes_total),
            dtype=torch.bool,
            device=device,
        )
        anchors = batch.anchor_node_ids.to(device=device).view(-1)
        if anchors.numel() > 0:
            anchor_mask[anchors] = True
        return cls(
            edge_index=edge_index,
            node_to_graph=node_to_graph,
            edge_to_graph=edge_to_graph,
            anchor_mask=anchor_mask,
            num_nodes=int(batch.num_nodes_total),
            num_edges=int(batch.num_edges_total),
            num_graphs=int(batch.num_graphs_total),
            device=device,
        )


@dataclass(frozen=True, slots=True)
class RewardContext:
    """
    Target-bearing supervision context used only during training/evaluation.
    """

    target_mask: torch.Tensor
    target_count_by_graph: torch.Tensor
    edge_index: torch.Tensor
    node_to_graph: torch.Tensor
    anchor_mask: torch.Tensor
    expand_budget: int

    def to(
        self,
        *,
        device: torch.device,
    ) -> RewardContext:
        return RewardContext(
            target_mask=self.target_mask.to(device=device),
            target_count_by_graph=self.target_count_by_graph.to(device=device),
            edge_index=self.edge_index.to(device=device),
            node_to_graph=self.node_to_graph.to(device=device),
            anchor_mask=self.anchor_mask.to(device=device),
            expand_budget=self.expand_budget,
        )

__all__ = ["GraphContext", "RewardContext"]
