from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.weaver.state import RolloutState, State

from .feature_encoder import FeatureBank
from .frontier_context import FrontierContext


def build_frontier(
    *,
    fb: FeatureBank,
    batch: RetrievalBatch,
    state: State | RolloutState,
    frontier_mode: str = "boundary",
) -> FrontierContext:
    device = fb.node_h.device
    frontier_mode = str(frontier_mode)
    if frontier_mode != "boundary":
        raise ValueError(
            "frontier_mode must be 'boundary', "
            f"got {frontier_mode!r}."
        )

    if isinstance(state, RolloutState):
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        rollout_ids, edge_ids = _rollout_frontier_edge_ids(
            state=state,
            edge_index=edge_index,
            node_incident_edge_ids=fb.node_incident_edge_ids,
            node_incident_ptr=fb.node_incident_ptr,
            frontier_mode=frontier_mode,
        )
        return _frontier_context_from_rollout_ids(
            batch=batch,
            state=state,
            rollout_ids=rollout_ids,
            edge_ids=edge_ids,
            device=device,
        )

    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    src, dst = edge_index
    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=device, dtype=torch.bool)
    if state.boundary_nodes is None:
        raise RuntimeError("Boundary frontier requires State.boundary_nodes.")
    boundary_nodes = state.boundary_nodes.to(device=device, dtype=torch.bool)
    frontier_mask = (
        boundary_nodes.index_select(0, src)
        & ~active_nodes.index_select(0, dst)
        & ~active_edges
    )
    # REMOVED: all-active frontier expansion — see methodology.md §3.3
    edge_ids = frontier_mask.nonzero(as_tuple=False).flatten()
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return FrontierContext(
            edge_ids=empty,
            src=empty,
            dst=empty,
            graph_id=empty,
            src_active=torch.empty((0,), dtype=torch.bool, device=device),
            dst_active=torch.empty((0,), dtype=torch.bool, device=device),
            static_graph_id=empty,
        )

    graph_id = batch.edge_batch.to(device=device, dtype=torch.long).index_select(0, edge_ids)
    return FrontierContext(
        edge_ids=edge_ids,
        src=src.index_select(0, edge_ids),
        dst=dst.index_select(0, edge_ids),
        graph_id=graph_id,
        src_active=active_nodes.index_select(0, src.index_select(0, edge_ids)),
        dst_active=active_nodes.index_select(0, dst.index_select(0, edge_ids)),
        static_graph_id=graph_id,
    )


def _rollout_frontier_edge_ids(
    *,
    state: RolloutState,
    edge_index: torch.Tensor,
    node_incident_edge_ids: torch.Tensor | None,
    node_incident_ptr: torch.Tensor | None,
    frontier_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    if node_incident_edge_ids is None or node_incident_ptr is None:
        raise RuntimeError(
            "Rollout frontier construction requires FeatureBank node incidence "
            "tensors. FeatureEncoder.prepare_rollout_context must provide them."
        )

    active_node_rows, active_node_ids = state.boundary_node_trace_rows(
        edge_index=edge_index.to(device=device, dtype=torch.long),
    )
    # REMOVED: all-active frontier expansion — see methodology.md §3.3
    active_node_rows = active_node_rows.to(device=device, dtype=torch.long).view(-1)
    active_node_ids = active_node_ids.to(device=device, dtype=torch.long).view(-1)
    if active_node_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    incident_edges = node_incident_edge_ids.to(device=device, dtype=torch.long).view(-1)
    incident_ptr = node_incident_ptr.to(device=device, dtype=torch.long).view(-1)
    degrees = (
        incident_ptr.index_select(0, active_node_ids + 1)
        - incident_ptr.index_select(0, active_node_ids)
    )
    if degrees.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    rollout_ids = torch.repeat_interleave(active_node_rows, degrees)
    edge_offsets = torch.arange(
        degrees.sum(),
        dtype=torch.long,
        device=device,
    ) - torch.repeat_interleave(torch.cumsum(degrees, dim=0) - degrees, degrees)
    starts = incident_ptr.index_select(0, active_node_ids)
    edge_ids = incident_edges.index_select(
        0,
        torch.repeat_interleave(starts, degrees) + edge_offsets,
    )
    if edge_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    frontier = ~state.contains_active_edges(
        rollout_ids=rollout_ids,
        edge_ids=edge_ids,
    )
    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)
    outgoing = src.eq(active_node_ids.repeat_interleave(degrees))
    dst_active = state.contains_active_nodes(
        rollout_ids=rollout_ids,
        node_ids=dst,
        edge_index=edge_index,
    )
    frontier = frontier & outgoing & ~dst_active
    rollout_ids = rollout_ids[frontier]
    edge_ids = edge_ids[frontier]
    if edge_ids.numel() == 0:
        return rollout_ids, edge_ids

    num_edges = int(state.num_edges)
    key = rollout_ids * num_edges + edge_ids
    key = torch.unique(key, sorted=True)
    return key.div(num_edges, rounding_mode="floor"), key.remainder(num_edges)


def _frontier_context_from_rollout_ids(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    rollout_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    device: torch.device,
) -> FrontierContext:
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return FrontierContext(
            edge_ids=empty,
            src=empty,
            dst=empty,
            graph_id=empty,
            src_active=torch.empty((0,), dtype=torch.bool, device=device),
            dst_active=torch.empty((0,), dtype=torch.bool, device=device),
            static_graph_id=empty,
        )

    rollout_ids = rollout_ids.to(device=device, dtype=torch.long)
    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)
    static_graph_id = state.rollout_to_graph.index_select(0, rollout_ids)
    src_active = state.contains_active_nodes(
        rollout_ids=rollout_ids,
        node_ids=src,
        edge_index=edge_index,
    )
    dst_active = state.contains_active_nodes(
        rollout_ids=rollout_ids,
        node_ids=dst,
        edge_index=edge_index,
    )
    return FrontierContext(
        edge_ids=edge_ids,
        src=src,
        dst=dst,
        graph_id=rollout_ids,
        src_active=src_active,
        dst_active=dst_active,
        static_graph_id=static_graph_id,
    )
