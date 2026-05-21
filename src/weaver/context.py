from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class DirectedAdjacencyIndex:
    """
    CSR-style indices over physical directed KG edges.

    out_ptr / edge_ids_by_src:
        edge ids grouped by source node.

    in_ptr / edge_ids_by_dst:
        edge ids grouped by destination node.

    This does not create inverse edges. Incoming edges are still original
    physical KG edge ids.
    """

    out_ptr: torch.Tensor
    edge_ids_by_src: torch.Tensor
    in_ptr: torch.Tensor
    edge_ids_by_dst: torch.Tensor


@dataclass(frozen=True, slots=True)
class GraphContext:
    """
    Static label-free graph context.

    This object contains only graph structure and question anchors.
    It is valid for rollout, policy evaluation, frontier construction,
    and inference.

    It intentionally excludes target labels, reward values, oracle paths,
    rollout trajectories, and training transitions.

    Assumption:
        edge_to_graph is inferred from the source node graph.
        Therefore preprocessing must guarantee no cross-graph edges.
    """

    edge_index: torch.Tensor  # [2, E]
    node_to_graph: torch.Tensor  # [N]
    edge_to_graph: torch.Tensor  # [E]
    anchor_mask: torch.Tensor  # [N]
    adjacency: DirectedAdjacencyIndex

    num_nodes: int
    num_edges: int
    num_graphs: int

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
    ) -> GraphContext:
        edge_index = batch.edge_index
        node_to_graph = batch.batch

        edge_to_graph = node_to_graph.index_select(
            0,
            edge_index[0],
        )

        anchor_mask = torch.zeros(
            int(batch.num_nodes_total),
            dtype=torch.bool,
            device=edge_index.device,
        )
        anchor_node_ids = batch.anchor_node_ids
        if anchor_node_ids.numel() > 0:
            anchor_mask[anchor_node_ids] = True

        adjacency = build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=int(batch.num_nodes_total),
        )

        return cls(
            edge_index=edge_index,
            node_to_graph=node_to_graph,
            edge_to_graph=edge_to_graph,
            anchor_mask=anchor_mask,
            adjacency=adjacency,
            num_nodes=int(batch.num_nodes_total),
            num_edges=int(batch.num_edges_total),
            num_graphs=int(batch.num_graphs_total),
        )

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    @property
    def outgoing_ptr(self) -> torch.Tensor:
        return self.adjacency.out_ptr

    @property
    def edge_ids_by_src(self) -> torch.Tensor:
        return self.adjacency.edge_ids_by_src


@dataclass(frozen=True, slots=True)
class TargetContext:
    """
    Target-label context for supervision, reward computation, and evaluation.

    This object contains only reachable answer labels.
    Graph structure belongs to GraphContext.
    """

    target_mask: torch.Tensor  # [N]
    target_count_by_graph: torch.Tensor  # [G]

    @classmethod
    def from_batch(
        cls,
        *,
        batch: RetrievalBatch,
        graph_context: GraphContext,
    ) -> TargetContext:
        target_mask = torch.zeros(
            int(graph_context.num_nodes),
            dtype=torch.bool,
            device=graph_context.device,
        )

        target_node_ids = batch.reachable_target_node_ids
        if target_node_ids.numel() > 0:
            target_mask[target_node_ids] = True

        target_graph_ids = graph_context.node_to_graph.index_select(
            0,
            target_node_ids,
        )
        target_count_by_graph = torch.bincount(
            target_graph_ids,
            minlength=int(graph_context.num_graphs),
        )

        return cls(
            target_mask=target_mask,
            target_count_by_graph=target_count_by_graph,
        )

    @property
    def device(self) -> torch.device:
        return self.target_mask.device

    @property
    def valid_graph_mask(self) -> torch.Tensor:
        return self.target_count_by_graph.gt(0)


def build_directed_adjacency_index(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> DirectedAdjacencyIndex:
    edge_ids = torch.arange(
        int(edge_index.size(1)),
        dtype=torch.long,
        device=edge_index.device,
    )

    src_node_ids = edge_index[0]
    dst_node_ids = edge_index[1]

    out_ptr, edge_ids_by_src = build_node_to_edge_csr(
        node_ids=src_node_ids,
        edge_ids=edge_ids,
        num_nodes=int(num_nodes),
    )
    in_ptr, edge_ids_by_dst = build_node_to_edge_csr(
        node_ids=dst_node_ids,
        edge_ids=edge_ids,
        num_nodes=int(num_nodes),
    )

    return DirectedAdjacencyIndex(
        out_ptr=out_ptr,
        edge_ids_by_src=edge_ids_by_src,
        in_ptr=in_ptr,
        edge_ids_by_dst=edge_ids_by_dst,
    )


def build_node_to_edge_csr(
    *,
    node_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a CSR-style node -> edge-id index.

    Returns:
        ptr:
            [num_nodes + 1], where edges for node i are stored in
            edge_ids_by_node[ptr[i]:ptr[i + 1]].

        edge_ids_by_node:
            Edge ids sorted/grouped by their associated node id.
    """

    order = torch.argsort(node_ids)
    sorted_node_ids = node_ids.index_select(0, order)
    edge_ids_by_node = edge_ids.index_select(0, order)

    edge_count_by_node = torch.bincount(
        sorted_node_ids,
        minlength=int(num_nodes),
    )

    ptr = torch.empty(
        int(num_nodes) + 1,
        dtype=torch.long,
        device=node_ids.device,
    )
    ptr[0] = 0
    ptr[1:] = torch.cumsum(edge_count_by_node, dim=0)

    return ptr, edge_ids_by_node


def build_outgoing_index(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_node_to_edge_csr(
        node_ids=edge_index[0],
        edge_ids=torch.arange(
            int(edge_index.size(1)),
            dtype=torch.long,
            device=edge_index.device,
        ),
        num_nodes=int(num_nodes),
    )


__all__ = [
    "DirectedAdjacencyIndex",
    "GraphContext",
    "TargetContext",
    "build_directed_adjacency_index",
    "build_outgoing_index",
    "build_node_to_edge_csr",
]
