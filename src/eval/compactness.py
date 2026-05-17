from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.schema import RetrievalBatch
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask
from src.graph.ops import prune_to_protected_core
from src.utils.scatter import scatter_sum
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor


def default_eval_device() -> torch.device:
    return torch.device("cpu")


def batch_num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs)


def compute_compactness_expectations(
    rollouts: Sequence[RolloutResult],
    batch: RetrievalBatch,
    *,
    include_dangling: bool = False,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Expected terminal-subgraph size/cost over rollout samples.

    Metrics:
    - expected_nodes: mean active nodes per graph.
    - expected_edges: mean active edges per graph.
    - selected_edge_count: mean selected edges per graph.
    - dangling_edge_ratio: optional ratio of selected edges pruned away from the
      protected anchor/target core.
    """
    device = device or default_eval_device()

    node_masks, edge_masks = SubgraphReconstructor(batch, device=device).stack(rollouts)

    metrics = compactness_from_masks(
        node_masks=node_masks,
        edge_masks=edge_masks,
        batch=batch,
        device=device,
    )

    if include_dangling:
        metrics["dangling_edge_ratio"] = dangling_edge_ratio_from_masks(
            node_masks=node_masks,
            edge_masks=edge_masks,
            batch=batch,
            device=device,
        )

    return metrics


def compactness_from_masks(
    *,
    node_masks: torch.Tensor,
    edge_masks: torch.Tensor,
    batch: RetrievalBatch,
    device: torch.device,
) -> dict[str, float]:
    num_graphs = batch_num_graphs(batch)

    if node_masks.numel() == 0:
        return {
            "expected_nodes": 0.0,
            "expected_edges": 0.0,
            "selected_edge_count": 0.0,
        }

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

    node_counts = per_graph_counts(
        node_masks,
        node_batch,
        num_graphs=num_graphs,
    )
    edge_counts = per_graph_counts(
        edge_masks,
        edge_batch,
        num_graphs=num_graphs,
    )

    return {
        "expected_nodes": float(node_counts.mean().item()),
        "expected_edges": float(edge_counts.mean().item()),
        "selected_edge_count": float(edge_counts.mean().item()),
    }


def dangling_edge_ratio_from_masks(
    *,
    node_masks: torch.Tensor,
    edge_masks: torch.Tensor,
    batch: RetrievalBatch,
    device: torch.device,
) -> float:
    """
    Mean dangling-edge ratio over rollout/graph pairs with at least one selected edge.

    protected nodes:
        anchors plus active target nodes.

    dangling edges:
        selected active edges removed by prune_to_protected_core.
    """
    if node_masks.numel() == 0:
        return 0.0

    num_graphs = batch_num_graphs(batch)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)

    anchors = anchor_node_mask(batch, device=device)
    targets = eval_target_node_mask(batch, device=device, use_reachable_targets=True)

    ratios: list[torch.Tensor] = []

    for nodes, edges in zip(node_masks, edge_masks, strict=True):
        protected_nodes = anchors | (nodes & targets)

        _, core_edges = prune_to_protected_core(
            active_nodes=nodes,
            active_edges=edges,
            edge_index=edge_index,
            protected_nodes=protected_nodes,
        )

        selected_edges = edges
        dangling_edges = selected_edges & ~core_edges

        selected_count = per_graph_counts(
            selected_edges.unsqueeze(0),
            edge_batch,
            num_graphs=num_graphs,
        ).squeeze(0)
        dangling_count = per_graph_counts(
            dangling_edges.unsqueeze(0),
            edge_batch,
            num_graphs=num_graphs,
        ).squeeze(0)

        valid = selected_count.gt(0.0)
        if bool(valid.any()):
            ratios.append(dangling_count[valid] / selected_count[valid])

    if not ratios:
        return 0.0

    return float(torch.cat(ratios).mean().item())


def per_graph_counts(
    masks: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    num_graphs: int,
) -> torch.Tensor:
    """
    Count true items per graph.

    masks:
        [R, N] or [R, E]

    batch_index:
        [N] or [E]

    Returns:
        [R, B]
    """
    if masks.ndim != 2:
        raise ValueError(f"masks must have shape [R, M], got {tuple(masks.shape)}.")

    batch_index = batch_index.to(device=masks.device, dtype=torch.long)
    num_rollouts = int(masks.size(0))
    row_offsets = torch.arange(num_rollouts, device=masks.device).unsqueeze(1) * int(
        num_graphs
    )
    flat_index = (batch_index.unsqueeze(0) + row_offsets).reshape(-1)

    counts = scatter_sum(
        masks.to(dtype=torch.float32).reshape(-1),
        flat_index,
        dim=0,
        dim_size=num_rollouts * int(num_graphs),
    )
    return counts.view(num_rollouts, int(num_graphs))


__all__ = [
    "compactness_from_masks",
    "compute_compactness_expectations",
    "dangling_edge_ratio_from_masks",
    "per_graph_counts",
]
