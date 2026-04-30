from __future__ import annotations

import torch
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.weaver.nn.backbone import FeatureBank
from src.weaver.state import State


def frontier_edges(
    *,
    batch: RetrievalBatch,
    state: State,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward legal expansion edges.

        C(s) = {e=(u,v) not in E_s : u in V_s or v in V_s}

    Returns:
        edge_ids:
            Physical edge ids in the current batched graph.
        edge_batch:
            Physical graph id for each returned edge.
    """
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=device, dtype=torch.bool)

    src, dst = edge_index
    incident = active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
    frontier = incident & ~active_edges

    edge_ids = frontier.nonzero(as_tuple=False).view(-1)
    return edge_ids, edge_batch.index_select(0, edge_ids)


def graph_state_features(
    *,
    batch: RetrievalBatch,
    state: State,
    fb: FeatureBank,
    feature_dim: int,
) -> torch.Tensor | None:
    """
    Per-graph state features.

    Current convention:
        [active_node_ratio, selected_nonroot_edge_ratio, remaining_budget_ratio]

    These are derived from the subgraph state. They do not use rollout time.
    """
    if feature_dim == 0:
        return None
    if feature_dim != 3:
        raise ValueError(
            f"graph_state_features currently supports feature_dim=3, got {feature_dim}."
        )

    device = fb.node_h.device
    dtype = fb.query_h.dtype
    num_graphs = int(batch.num_graphs)

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

    active_nodes = state.active_nodes.to(device=device, dtype=dtype)
    selected_nonroot_edges = state.selected_nonroot_edges().to(
        device=device, dtype=dtype
    )

    node_total = (
        torch.bincount(node_batch, minlength=num_graphs)
        .to(device=device, dtype=dtype)
        .clamp_min(1)
    )
    edge_total = (
        torch.bincount(edge_batch, minlength=num_graphs)
        .to(device=device, dtype=dtype)
        .clamp_min(1)
    )

    node_count = scatter_sum(active_nodes, node_batch, dim=0, dim_size=num_graphs)
    nonroot_edge_count = scatter_sum(
        selected_nonroot_edges, edge_batch, dim=0, dim_size=num_graphs
    )

    remaining_budget = state.remaining_budget_per_graph(
        edge_batch=edge_batch,
        num_graphs=num_graphs,
    ).to(device=device, dtype=dtype)

    remaining_ratio = remaining_budget / float(max(1, state.expand_budget))

    return torch.stack(
        [
            node_count / node_total,
            nonroot_edge_count / edge_total,
            remaining_ratio,
        ],
        dim=-1,
    )


def masked_mean(
    *,
    values: torch.Tensor,
    mask: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if values.numel() == 0 or not bool(mask.any()):
        return values.new_zeros((int(num_graphs), values.size(-1)))

    return scatter_mean(
        values=values[mask],
        batch_index=batch_index[mask],
        num_graphs=num_graphs,
    )


def scatter_mean(
    *,
    values: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((int(num_graphs), values.size(-1)))

    batch_index = batch_index.to(device=values.device, dtype=torch.long)

    total = scatter_sum(values, batch_index, dim=0, dim_size=int(num_graphs))
    count = torch.bincount(batch_index, minlength=int(num_graphs)).to(
        device=values.device,
        dtype=values.dtype,
    )

    return total / count.clamp_min(1).unsqueeze(-1)


__all__ = [
    "frontier_edges",
    "graph_state_features",
    "masked_mean",
    "scatter_mean",
]
