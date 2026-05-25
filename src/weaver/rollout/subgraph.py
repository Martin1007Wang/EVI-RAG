from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.graph.masks import anchor_node_mask
from src.weaver.context import GraphContext
from src.weaver.rollout.trajectory import TrajectoryBatch


def union_subgraph_masks(
    trajectories: TrajectoryBatch,
    context: GraphContext,
    batch: RetrievalBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if trajectories.num_trajectories == 0:
        return (
            anchor_node_mask(batch, device=context.device),
            torch.zeros(int(context.num_edges), dtype=torch.bool, device=context.device),
        )

    edge_ids = trajectories.edge_ids[trajectories.valid_edge_mask()]
    edge_mask = torch.zeros(int(context.num_edges), dtype=torch.bool, device=context.device)
    node_mask = anchor_node_mask(batch, device=context.device)
    if edge_ids.numel() > 0:
        edge_ids = edge_ids.to(device=context.device, dtype=torch.long)
        edge_mask[edge_ids] = True
        node_mask[context.edge_src.index_select(0, edge_ids)] = True
        node_mask[context.edge_dst.index_select(0, edge_ids)] = True
    return node_mask, edge_mask


def stacked_subgraph_masks(
    trajectories: TrajectoryBatch,
    context: GraphContext,
    batch: RetrievalBatch | None = None,
    *,
    sample_ids: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del batch
    out_device = device or context.device
    num_nodes = int(context.num_nodes)
    num_edges = int(context.num_edges)
    if trajectories.num_trajectories == 0:
        return (
            torch.zeros((0, num_nodes), dtype=torch.bool, device=out_device),
            torch.zeros((0, num_edges), dtype=torch.bool, device=out_device),
        )

    if sample_ids is None:
        sample_ids = _grouped_sample_ids(trajectories, num_graphs=int(context.num_graphs))
    sample_ids = sample_ids.to(device=context.device, dtype=torch.long).view(-1)
    if int(sample_ids.numel()) != trajectories.num_trajectories:
        raise ValueError("sample_ids must match trajectory count.")

    num_samples = int(sample_ids.max().item()) + 1 if sample_ids.numel() > 0 else 0
    node_masks = torch.zeros((num_samples, num_nodes), dtype=torch.bool, device=context.device)
    edge_masks = torch.zeros((num_samples, num_edges), dtype=torch.bool, device=context.device)

    anchor_rows, anchor_nodes = _anchor_sample_node_pairs(
        trajectories=trajectories,
        context=context,
        sample_ids=sample_ids,
    )
    if anchor_nodes.numel() > 0:
        node_masks[anchor_rows, anchor_nodes] = True

    valid = trajectories.valid_edge_mask()
    if bool(valid.any()):
        traj_rows = valid.nonzero(as_tuple=True)[0]
        edge_ids = trajectories.edge_ids[valid].to(device=context.device, dtype=torch.long)
        edge_sample_ids = sample_ids.index_select(0, traj_rows.to(device=context.device))
        edge_masks[edge_sample_ids, edge_ids] = True
        node_masks[edge_sample_ids, context.edge_src.index_select(0, edge_ids)] = True
        node_masks[edge_sample_ids, context.edge_dst.index_select(0, edge_ids)] = True

    return (
        node_masks.to(device=out_device, dtype=torch.bool),
        edge_masks.to(device=out_device, dtype=torch.bool),
    )


def _anchor_sample_node_pairs(
    *,
    trajectories: TrajectoryBatch,
    context: GraphContext,
    sample_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph_ids = trajectories.graph_ids.to(device=context.device, dtype=torch.long)
    starts = context.anchor_ptr.index_select(0, graph_ids)
    ends = context.anchor_ptr.index_select(0, graph_ids + 1)
    lengths = ends - starts
    if not bool(lengths.gt(0).any()):
        empty = torch.empty(0, dtype=torch.long, device=context.device)
        return empty, empty
    rows = torch.repeat_interleave(sample_ids, lengths)
    offsets = torch.cumsum(lengths, dim=0) - lengths
    positions = torch.arange(
        int(lengths.sum().item()),
        dtype=torch.long,
        device=context.device,
    ) - torch.repeat_interleave(offsets, lengths) + torch.repeat_interleave(starts, lengths)
    return rows, context.anchor_node_ids.index_select(0, positions)


def _grouped_sample_ids(trajectories: TrajectoryBatch, *, num_graphs: int) -> torch.Tensor:
    graph_ids = trajectories.graph_ids.to(dtype=torch.long)
    counts = torch.bincount(graph_ids, minlength=int(num_graphs))
    expected = torch.repeat_interleave(
        torch.arange(int(num_graphs), dtype=torch.long, device=trajectories.device),
        counts,
    )
    if expected.shape != graph_ids.shape or not bool(expected.eq(graph_ids).all()):
        raise ValueError("trajectory graph_ids must be grouped by graph; pass explicit sample_ids for mixed trajectories.")
    starts = torch.cumsum(counts, dim=0) - counts
    return torch.arange(
        trajectories.num_trajectories,
        dtype=torch.long,
        device=trajectories.device,
    ) - starts.index_select(0, graph_ids)


__all__ = [
    "stacked_subgraph_masks",
    "union_subgraph_masks",
]
