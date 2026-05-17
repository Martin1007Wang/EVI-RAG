from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch


def target_node_values(
    *,
    batch: RetrievalBatch,
    flat: torch.Tensor,
    target_pos: int,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    target_offset, graph_id = _target_flat_offset(batch=batch, target_pos=target_pos, unit="node")
    node_ptr = batch.ptr.to(device=node_ids.device, dtype=torch.long)
    local_node_ids = node_ids.to(dtype=torch.long) - node_ptr[graph_id]
    idx = target_offset.to(device=node_ids.device) + local_node_ids
    return flat.to(device=node_ids.device).index_select(0, idx).to(dtype=torch.long)


def target_edge_values(
    *,
    batch: RetrievalBatch,
    flat: torch.Tensor,
    target_pos: int,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    target_offset, graph_id = _target_flat_offset(batch=batch, target_pos=target_pos, unit="edge")
    edge_ptr = _edge_ptr(batch).to(device=edge_ids.device)
    local_edge_ids = edge_ids.to(dtype=torch.long) - edge_ptr[graph_id]
    idx = target_offset.to(device=edge_ids.device) + local_edge_ids
    return flat.to(device=edge_ids.device).index_select(0, idx).to(dtype=torch.float32)


def _target_flat_offset(
    *,
    batch: RetrievalBatch,
    target_pos: int,
    unit: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch.reachable_target_node_ids.view(-1)
    if int(target_pos) < 0 or int(target_pos) >= int(targets.numel()):
        raise IndexError(f"target_pos {target_pos} is outside [0, {int(targets.numel())}).")

    device = targets.device
    node_batch = batch.batch.to(device=device, dtype=torch.long)
    graph_id = node_batch.index_select(0, targets.to(dtype=torch.long))[int(target_pos)]
    target_batch = node_batch.index_select(0, targets.to(dtype=torch.long))
    target_counts = torch.bincount(
        target_batch,
        minlength=int(batch.num_graphs_total),
    ).to(device=device, dtype=torch.long)

    same_graph_positions = target_batch.eq(graph_id).nonzero(as_tuple=False).view(-1)
    local_target_pos = same_graph_positions.eq(int(target_pos)).nonzero(as_tuple=False).view(-1)[0]

    if unit == "node":
        unit_counts = torch.diff(batch.ptr.to(device=device, dtype=torch.long))
    elif unit == "edge":
        unit_counts = torch.diff(_edge_ptr(batch).to(device=device, dtype=torch.long))
    else:
        raise ValueError(f"unknown unit: {unit}")

    graph_offsets = torch.cat(
        [
            unit_counts.new_zeros(1),
            (target_counts * unit_counts).cumsum(dim=0)[:-1],
        ],
        dim=0,
    )
    target_offset = graph_offsets[graph_id] + local_target_pos * unit_counts[graph_id]
    return target_offset, graph_id


def _edge_ptr(batch: RetrievalBatch) -> torch.Tensor:
    edge_batch = batch.edge_batch.to(dtype=torch.long)
    edge_counts = torch.bincount(edge_batch, minlength=int(batch.num_graphs_total))
    return torch.cat([edge_counts.new_zeros(1), edge_counts.cumsum(dim=0)], dim=0)
