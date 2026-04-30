# src/weaver/rollout/terminal_subgraph.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.graph.ops import build_anchor_induced_edge_mask
from src.weaver.rollout.schema import RolloutBatch


@dataclass(frozen=True)
class UnionSubgraphMasks:
    """
    Union of terminal subgraphs over rollout samples.

    node_mask:
        Boolean mask over all nodes in the current batched graph, shape [N].

    edge_mask:
        Boolean mask over all edges in the current batched graph, shape [E].
    """

    node_mask: torch.Tensor
    edge_mask: torch.Tensor


def default_eval_device() -> torch.device:
    return torch.device("cpu")


def batch_num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs)


def batch_num_nodes(batch: RetrievalBatch) -> int:
    return int(batch.num_nodes_total)


def batch_num_edges(batch: RetrievalBatch) -> int:
    return int(batch.num_edges_total)


def node_mask_from_ids(
    ids: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = ids.to(device=device, dtype=torch.long).view(-1)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    if ids.numel() == 0:
        return mask

    _check_id_range(ids, upper=num_nodes, name=name)
    mask[ids] = True
    return mask


def anchor_node_mask(
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    return node_mask_from_ids(
        batch.anchor_node_ids,
        num_nodes=batch_num_nodes(batch),
        device=device,
        name="anchor_node_ids",
    )


def eval_target_node_mask(
    batch: RetrievalBatch,
    *,
    device: torch.device,
    use_reachable_targets: bool = True,
) -> torch.Tensor:
    """
    Build target-node mask for retrieval evaluation.

    If use_reachable_targets=True and reachable_target_node_ids exists, metrics
    are computed over reachable/teachable targets. If False, metrics use all
    target_node_ids present in the graph.
    """
    if use_reachable_targets:
        reachable = getattr(batch, "reachable_target_node_ids", None)
        if isinstance(reachable, torch.Tensor) and reachable.numel() > 0:
            return node_mask_from_ids(
                reachable,
                num_nodes=batch_num_nodes(batch),
                device=device,
                name="reachable_target_node_ids",
            )

    return node_mask_from_ids(
        batch.target_node_ids,
        num_nodes=batch_num_nodes(batch),
        device=device,
        name="target_node_ids",
    )


def root_edge_mask(
    batch: RetrievalBatch,
    *,
    anchors: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    mask = build_anchor_induced_edge_mask(edge_index, anchors)

    expected_shape = (batch_num_edges(batch),)
    if mask.shape != expected_shape:
        raise ValueError(f"root_edge_mask must have shape {expected_shape}, " f"got {tuple(mask.shape)}.")

    return mask


def terminal_subgraph_mask(
    batch: RetrievalBatch,
    rollout: RolloutBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct the terminal subgraph represented by one RolloutBatch.

    Returned masks are over the full physical batched graph:
    - node_mask: [N]
    - edge_mask: [E]

    Initial active state:
    - anchor nodes are active;
    - anchor-induced root edges are active.

    Then every valid expanded selected edge is added, together with its endpoints.

    Coordinate convention:
    - rollout.traces.selected_edge_ids are physical edge ids in this batch.
    - STOP keeps selected_edge_ids = -1.
    """

    num_graphs = batch_num_graphs(batch)
    num_nodes = batch_num_nodes(batch)
    num_edges = batch_num_edges(batch)

    edge_index = batch.edge_index.to(device=device, dtype=torch.long)

    node_mask = anchor_node_mask(batch, device=device)
    edge_mask = root_edge_mask(batch, anchors=node_mask, device=device)

    selected_edge_ids = rollout.traces.selected_edge_ids.to(
        device=device,
        dtype=torch.long,
    )
    continue_mask = rollout.traces.continue_mask.to(device=device, dtype=torch.bool)

    if selected_edge_ids.ndim != 2:
        raise ValueError(f"selected_edge_ids must have shape [B, T], " f"got {tuple(selected_edge_ids.shape)}.")

    if continue_mask.shape != selected_edge_ids.shape:
        raise ValueError(
            "continue_mask must have the same shape as selected_edge_ids: " f"{tuple(continue_mask.shape)} != {tuple(selected_edge_ids.shape)}."
        )

    batch_size, horizon = selected_edge_ids.shape
    if batch_size != num_graphs:
        raise ValueError(f"rollout batch size must match batch.num_graphs: " f"{batch_size} != {num_graphs}.")

    trajectory_length = rollout.stats.trajectory_length.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    if trajectory_length.shape != (num_graphs,):
        raise ValueError(f"trajectory_length must have shape [{num_graphs}], " f"got {tuple(trajectory_length.shape)}.")

    step_ids = torch.arange(horizon, device=device).unsqueeze(0)
    valid_steps = step_ids < trajectory_length.unsqueeze(1)
    valid_expands = valid_steps & continue_mask & selected_edge_ids.ge(0)

    if not bool(valid_expands.any()):
        return node_mask, edge_mask

    edge_ids = selected_edge_ids[valid_expands].view(-1)
    _check_id_range(edge_ids, upper=num_edges, name="selected_edge_ids")

    edge_mask[edge_ids] = True

    endpoints = edge_index[:, edge_ids].reshape(-1)
    _check_id_range(endpoints, upper=num_nodes, name="selected edge endpoints")
    node_mask[endpoints] = True

    return node_mask, edge_mask


def stack_terminal_subgraph_masks(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack terminal subgraph masks for rollout samples.

    Returns:
    - node_masks: [R, N]
    - edge_masks: [R, E]
    """
    if device is None:
        device = default_eval_device()

    num_nodes = batch_num_nodes(batch)
    num_edges = batch_num_edges(batch)

    if not rollouts:
        return (
            torch.zeros((0, num_nodes), dtype=torch.bool, device=device),
            torch.zeros((0, num_edges), dtype=torch.bool, device=device),
        )

    node_masks: list[torch.Tensor] = []
    edge_masks: list[torch.Tensor] = []

    for rollout in rollouts:
        nodes, edges = terminal_subgraph_mask(
            batch=batch,
            rollout=rollout,
            device=device,
        )
        node_masks.append(nodes)
        edge_masks.append(edges)

    return torch.stack(node_masks, dim=0), torch.stack(edge_masks, dim=0)


def compute_union_subgraph_masks(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device | None = None,
) -> UnionSubgraphMasks:
    """
    Compute node/edge union over terminal subgraphs from rollout samples.
    """
    if device is None:
        device = default_eval_device()

    node_masks, edge_masks = stack_terminal_subgraph_masks(
        rollouts,
        batch,
        device=device,
    )

    if node_masks.numel() == 0:
        return UnionSubgraphMasks(
            node_mask=torch.zeros(
                batch_num_nodes(batch),
                dtype=torch.bool,
                device=device,
            ),
            edge_mask=torch.zeros(
                batch_num_edges(batch),
                dtype=torch.bool,
                device=device,
            ),
        )

    return UnionSubgraphMasks(
        node_mask=node_masks.any(dim=0),
        edge_mask=edge_masks.any(dim=0),
    )


def _check_id_range(
    ids: torch.Tensor,
    *,
    upper: int,
    name: str,
) -> None:
    if ids.numel() == 0:
        return

    min_id = int(ids.min())
    max_id = int(ids.max())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(f"{name} contains ids outside range [0, {upper}): " f"min={min_id}, max={max_id}.")


__all__ = [
    "UnionSubgraphMasks",
    "anchor_node_mask",
    "batch_num_edges",
    "batch_num_graphs",
    "batch_num_nodes",
    "compute_union_subgraph_masks",
    "default_eval_device",
    "eval_target_node_mask",
    "node_mask_from_ids",
    "root_edge_mask",
    "stack_terminal_subgraph_masks",
    "terminal_subgraph_mask",
]
