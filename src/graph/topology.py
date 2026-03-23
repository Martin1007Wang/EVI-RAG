from __future__ import annotations

from dataclasses import dataclass

import torch

from .observation import GroupedLocalNodeIndex


@dataclass(frozen=True)
class CsrAdjacency:
    """Compact CSR adjacency used by graph-search policies."""

    crow: torch.Tensor
    col: torch.Tensor
    edge_ids: torch.Tensor

    def crow_indices(self) -> torch.Tensor:
        return self.crow

    def col_indices(self) -> torch.Tensor:
        return self.col

    def values(self) -> torch.Tensor:
        return self.edge_ids


def _gather_actions_from_csr(
    *,
    adjacency: CsrAdjacency,
    nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crow = adjacency.crow_indices()
    col = adjacency.col_indices()
    edge_ids = adjacency.values()

    start_ptr = crow[nodes]
    end_ptr = crow[nodes + 1]
    out_degrees = end_ptr - start_ptr
    total_edges = int(out_degrees.sum().item())
    if total_edges == 0:
        empty = torch.empty((0,), dtype=torch.long, device=nodes.device)
        return empty, empty, out_degrees

    base_index = start_ptr.repeat_interleave(out_degrees)
    segment_starts = out_degrees.cumsum(0) - out_degrees
    flat_offsets = torch.arange(total_edges, device=nodes.device, dtype=torch.long)
    gather_index = base_index + (
        flat_offsets - segment_starts.repeat_interleave(out_degrees)
    )
    return edge_ids[gather_index], col[gather_index], out_degrees


def _gather_active_actions_from_csr(
    *,
    adjacency: CsrAdjacency,
    current_nodes: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if current_nodes.dim() != 1:
        raise ValueError("current_nodes must be a 1D tensor.")
    if active_mask.dtype != torch.bool or active_mask.dim() != 1:
        raise ValueError("active_mask must be a 1D bool tensor.")
    if tuple(current_nodes.shape) != tuple(active_mask.shape):
        raise ValueError("current_nodes and active_mask must have the same shape.")

    total_agents = int(current_nodes.numel())
    filtered_degrees = torch.zeros(
        (total_agents,),
        device=current_nodes.device,
        dtype=torch.long,
    )
    empty = torch.empty((0,), device=current_nodes.device, dtype=torch.long)
    active_agent_index = torch.nonzero(active_mask, as_tuple=False).view(-1)
    if int(active_agent_index.numel()) == 0:
        return empty, empty, empty, filtered_degrees

    active_nodes = current_nodes.index_select(0, active_agent_index)
    edge_ids, neighbor_nodes, active_degrees = _gather_actions_from_csr(
        adjacency=adjacency,
        nodes=active_nodes,
    )
    filtered_degrees.index_copy_(0, active_agent_index, active_degrees)
    if int(edge_ids.numel()) == 0:
        return edge_ids, neighbor_nodes, empty, filtered_degrees

    edge_agent_index = active_agent_index.repeat_interleave(active_degrees)
    return edge_ids, neighbor_nodes, edge_agent_index, filtered_degrees


@dataclass(frozen=True)
class GraphTopology:
    """Static disconnected graph topology shared across a search episode."""

    num_graphs: int
    num_nodes: int
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    _graph_node_offsets: torch.Tensor
    adjacency: CsrAdjacency
    reverse_adjacency: CsrAdjacency

    @property
    def graph_node_offsets(self) -> torch.Tensor:
        return self._graph_node_offsets

    def validate(self) -> None:
        if int(self.num_graphs) < 1:
            raise ValueError("GraphTopology.num_graphs must be >= 1.")
        if int(self.num_nodes) < 0:
            raise ValueError("GraphTopology.num_nodes must be >= 0.")
        if (
            self.edge_index.dtype != torch.long
            or self.edge_index.dim() != 2
            or int(self.edge_index.size(0)) != 2
        ):
            raise ValueError("GraphTopology.edge_index must be [2, E] torch.long.")
        if self.edge_type.dtype != torch.long or self.edge_type.dim() != 1:
            raise ValueError("GraphTopology.edge_type must be 1D torch.long.")
        if int(self.edge_type.numel()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.edge_type length mismatch with edge_index column count."
            )
        if (
            self._graph_node_offsets.dtype != torch.long
            or self._graph_node_offsets.dim() != 1
        ):
            raise ValueError("GraphTopology graph node offsets must be 1D torch.long.")
        if int(self._graph_node_offsets.numel()) != int(self.num_graphs) + 1:
            raise ValueError(
                "GraphTopology graph node offsets length mismatch with num_graphs."
            )
        if int(self._graph_node_offsets[0].item()) != 0:
            raise ValueError("GraphTopology graph node offsets must start at 0.")
        if bool(
            (self._graph_node_offsets[1:] < self._graph_node_offsets[:-1]).any().item()
        ):
            raise ValueError("GraphTopology graph node offsets must be non-decreasing.")
        if int(self._graph_node_offsets[-1].item()) != int(self.num_nodes):
            raise ValueError(
                "GraphTopology graph node offsets do not end at num_nodes."
            )
        if int(self.edge_index.numel()) > 0:
            if bool((self.edge_index < 0).any().item()) or bool(
                (self.edge_index >= int(self.num_nodes)).any().item()
            ):
                raise ValueError(
                    "GraphTopology.edge_index contains out-of-range nodes."
                )
            source_graph = self.graph_index_from_nodes(self.edge_index[0])
            target_graph = self.graph_index_from_nodes(self.edge_index[1])
            if bool((source_graph != target_graph).any().item()):
                raise ValueError(
                    "GraphTopology.edge_index crosses graph boundaries in a disconnected batch."
                )
        if self.adjacency.crow.dtype != torch.long or self.adjacency.crow.dim() != 1:
            raise ValueError("GraphTopology.adjacency.crow must be 1D torch.long.")
        if int(self.adjacency.crow.numel()) != int(self.num_nodes) + 1:
            raise ValueError(
                "GraphTopology.adjacency.crow length mismatch with num_nodes."
            )
        if int(self.adjacency.crow[0].item()) != 0:
            raise ValueError("GraphTopology.adjacency.crow must start at 0.")
        if bool((self.adjacency.crow[1:] < self.adjacency.crow[:-1]).any().item()):
            raise ValueError("GraphTopology.adjacency.crow must be non-decreasing.")
        if int(self.adjacency.crow[-1].item()) != int(self.edge_index.size(1)):
            raise ValueError("GraphTopology.adjacency.crow does not end at edge count.")
        if self.adjacency.col.dtype != torch.long or self.adjacency.col.dim() != 1:
            raise ValueError("GraphTopology.adjacency.col must be 1D torch.long.")
        if (
            self.adjacency.edge_ids.dtype != torch.long
            or self.adjacency.edge_ids.dim() != 1
        ):
            raise ValueError("GraphTopology.adjacency.edge_ids must be 1D torch.long.")
        if int(self.adjacency.col.numel()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.adjacency.col length mismatch with edge count."
            )
        if int(self.adjacency.edge_ids.numel()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.adjacency.edge_ids length mismatch with edge count."
            )
        if int(self.adjacency.col.numel()) > 0 and (
            bool((self.adjacency.col < 0).any().item())
            or bool((self.adjacency.col >= int(self.num_nodes)).any().item())
        ):
            raise ValueError("GraphTopology.adjacency.col contains out-of-range nodes.")
        if (
            self.reverse_adjacency.crow.dtype != torch.long
            or self.reverse_adjacency.crow.dim() != 1
        ):
            raise ValueError(
                "GraphTopology.reverse_adjacency.crow must be 1D torch.long."
            )
        if int(self.reverse_adjacency.crow.numel()) != int(self.num_nodes) + 1:
            raise ValueError(
                "GraphTopology.reverse_adjacency.crow length mismatch with num_nodes."
            )
        if int(self.reverse_adjacency.crow[0].item()) != 0:
            raise ValueError("GraphTopology.reverse_adjacency.crow must start at 0.")
        if bool(
            (self.reverse_adjacency.crow[1:] < self.reverse_adjacency.crow[:-1])
            .any()
            .item()
        ):
            raise ValueError(
                "GraphTopology.reverse_adjacency.crow must be non-decreasing."
            )
        if int(self.reverse_adjacency.crow[-1].item()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.reverse_adjacency.crow does not end at edge count."
            )
        if (
            self.reverse_adjacency.col.dtype != torch.long
            or self.reverse_adjacency.col.dim() != 1
        ):
            raise ValueError(
                "GraphTopology.reverse_adjacency.col must be 1D torch.long."
            )
        if (
            self.reverse_adjacency.edge_ids.dtype != torch.long
            or self.reverse_adjacency.edge_ids.dim() != 1
        ):
            raise ValueError(
                "GraphTopology.reverse_adjacency.edge_ids must be 1D torch.long."
            )
        if int(self.reverse_adjacency.col.numel()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.reverse_adjacency.col length mismatch with edge count."
            )
        if int(self.reverse_adjacency.edge_ids.numel()) != int(self.edge_index.size(1)):
            raise ValueError(
                "GraphTopology.reverse_adjacency.edge_ids length mismatch with edge count."
            )
        if int(self.reverse_adjacency.col.numel()) > 0 and (
            bool((self.reverse_adjacency.col < 0).any().item())
            or bool((self.reverse_adjacency.col >= int(self.num_nodes)).any().item())
        ):
            raise ValueError(
                "GraphTopology.reverse_adjacency.col contains out-of-range nodes."
            )

    def graph_index_from_nodes(self, node_index: torch.Tensor) -> torch.Tensor:
        offsets = self._graph_node_offsets[1:].to(
            device=node_index.device, dtype=torch.long
        )
        safe_nodes = node_index.clamp(min=0, max=max(self.num_nodes - 1, 0))
        graph_index = torch.searchsorted(offsets, safe_nodes, right=True)
        valid = node_index >= 0
        return torch.where(valid, graph_index, torch.zeros_like(graph_index))

    def all_node_graph_index(
        self, *, device: torch.device | None = None
    ) -> torch.Tensor:
        target_device = self.edge_index.device if device is None else device
        node_index = torch.arange(
            self.num_nodes, device=target_device, dtype=torch.long
        )
        return self.graph_index_from_nodes(node_index)

    def resolve_local_node_indices(
        self,
        local_node_index: GroupedLocalNodeIndex,
        *,
        field_name: str,
        validate_grouping: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if validate_grouping:
            local_node_index.validate(num_groups=self.num_graphs, field_name=field_name)
        graph_index = local_node_index.graph_index(
            device=local_node_index.local_indices.device,
        )
        local_indices = local_node_index.local_indices.to(dtype=torch.long)
        if int(local_indices.numel()) == 0:
            return local_indices, graph_index

        graph_offsets = self._graph_node_offsets[:-1].to(
            device=local_indices.device,
            dtype=torch.long,
        )
        graph_sizes = (self._graph_node_offsets[1:] - self._graph_node_offsets[:-1]).to(
            device=local_indices.device,
            dtype=torch.long,
        )
        max_local = graph_sizes.index_select(0, graph_index)
        invalid = (local_indices < 0) | (local_indices >= max_local)
        if bool(invalid.any().item()):
            invalid_graphs = torch.unique(graph_index[invalid]).tolist()
            raise ValueError(
                f"{field_name} contains out-of-range local node indices for graphs "
                f"{invalid_graphs}."
            )
        absolute_indices = local_indices + graph_offsets.index_select(0, graph_index)
        return absolute_indices, graph_index

    def build_node_membership_mask(
        self,
        local_node_index: GroupedLocalNodeIndex,
        *,
        field_name: str,
        device: torch.device | None = None,
        debug_checks: bool = False,
    ) -> torch.Tensor:
        target_device = self.edge_index.device if device is None else device
        if debug_checks:
            self.validate()
            local_node_index.validate(num_groups=self.num_graphs, field_name=field_name)
        mask = torch.zeros((self.num_nodes,), dtype=torch.bool, device=target_device)
        if int(local_node_index.local_indices.numel()) == 0:
            return mask

        counts = local_node_index.counts(
            clamp_negative=not debug_checks,
            device=target_device,
        )
        if int(counts.sum().item()) != int(local_node_index.local_indices.numel()):
            raise ValueError(f"{field_name} ptr mismatch with index length.")
        graph_index = torch.arange(
            int(counts.numel()), device=target_device, dtype=torch.long
        ).repeat_interleave(counts)
        local_indices = local_node_index.local_indices.to(
            device=target_device,
            dtype=torch.long,
        )
        graph_offsets = self._graph_node_offsets[:-1].to(
            device=target_device,
            dtype=torch.long,
        )
        graph_sizes = (self._graph_node_offsets[1:] - self._graph_node_offsets[:-1]).to(
            device=target_device,
            dtype=torch.long,
        )
        max_local = graph_sizes.index_select(0, graph_index)
        invalid = (local_indices < 0) | (local_indices >= max_local)
        if bool(invalid.any().item()):
            raise ValueError(
                f"{field_name} out of range for membership mask construction."
            )
        absolute_indices = local_indices + graph_offsets.index_select(0, graph_index)
        mask.scatter_(0, absolute_indices, True)
        return mask

    def gather_outgoing_edges(
        self,
        *,
        current_nodes: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _gather_active_actions_from_csr(
            adjacency=self.adjacency,
            current_nodes=current_nodes,
            active_mask=active_mask,
        )

    def gather_incoming_edges(
        self,
        *,
        current_nodes: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _gather_active_actions_from_csr(
            adjacency=self.reverse_adjacency,
            current_nodes=current_nodes,
            active_mask=active_mask,
        )

    def graph_index_from_edges(self, edge_ids: torch.Tensor) -> torch.Tensor:
        safe_edge_ids = edge_ids.clamp(
            min=0, max=max(int(self.edge_index.size(1)) - 1, 0)
        )
        source_nodes = self.edge_index[0].index_select(0, safe_edge_ids)
        graph_index = self.graph_index_from_nodes(source_nodes)
        valid = edge_ids >= 0
        return torch.where(valid, graph_index, torch.zeros_like(graph_index))

    def has_super_source_layout(
        self,
        *,
        node_entity_ids: torch.Tensor,
        device: torch.device | None = None,
    ) -> bool:
        try:
            self.infer_super_source_indices(
                node_entity_ids=node_entity_ids,
                device=device,
            )
        except ValueError:
            return False
        return True

    def infer_super_source_indices(
        self,
        *,
        node_entity_ids: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = node_entity_ids.device if device is None else device
        graph_node_offsets = self._graph_node_offsets.to(
            device=target_device,
            dtype=torch.long,
        )
        if int(graph_node_offsets.numel()) < 2:
            raise ValueError(
                "graph node offsets must describe at least one graph for super source inference."
            )
        if int(graph_node_offsets[-1].item()) != int(self.num_nodes):
            raise ValueError(
                "graph node offsets do not end at num_nodes for super source inference."
            )
        node_entity_ids_long = node_entity_ids.to(
            device=target_device, dtype=torch.long
        )
        if int(node_entity_ids_long.numel()) != int(self.num_nodes):
            raise ValueError(
                "node_entity_ids length mismatch with num_nodes when inferring super source nodes: "
                f"node_entity_ids={int(node_entity_ids_long.numel())}, num_nodes={int(self.num_nodes)}."
            )
        counts = graph_node_offsets[1:] - graph_node_offsets[:-1]
        if bool((counts < 2).any().item()):
            raise ValueError(
                "Each graph must contain at least two nodes when super source is enabled."
            )
        question_super_abs = graph_node_offsets[1:] - 2
        answer_super_abs = graph_node_offsets[1:] - 1
        super_mask = node_entity_ids_long < 0
        if int(super_mask.sum().item()) != 2 * int(self.num_graphs):
            raise ValueError(
                "Super-source layout invariant violated: expected exactly two virtual nodes per graph "
                f"(got {int(super_mask.sum().item())} negatives for {int(self.num_graphs)} graphs)."
            )
        question_ok = bool(super_mask.index_select(0, question_super_abs).all().item())
        answer_ok = bool(super_mask.index_select(0, answer_super_abs).all().item())
        if not (question_ok and answer_ok):
            raise ValueError(
                "Super-source layout invariant violated: trailing per-graph nodes are not both virtual."
            )
        return question_super_abs, answer_super_abs


__all__ = ["CsrAdjacency", "GraphTopology"]
