from __future__ import annotations

import torch
from torch_scatter import scatter_sum
from dataclasses import dataclass
from src.data.schema import RetrievalBatch
from src.models.rollout import RolloutBatch, StateReconstructor
from src.utils.reward_utils import (
    build_anchor_induced_edge_mask,
    per_graph_mask_count,
    prune_to_protected_core,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@dataclass
class UnionSubgraphMasks:
    """Terminal subgraph union across all rollouts, aligned to the full batch graph."""

    union_nodes: torch.Tensor  # BoolTensor [num_nodes]
    union_edges: torch.Tensor  # BoolTensor [num_edges]


def _metric_device(batch: RetrievalBatch) -> torch.device:
    del batch
    return torch.device("cpu")


def _stack_terminal_states(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rollouts:
        return (
            torch.zeros((0, batch.num_nodes), dtype=torch.bool, device=device),
            torch.zeros((0, int(batch.edge_index.size(1))), dtype=torch.bool, device=device),
        )

    nodes: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    for rollout in rollouts:
        active_nodes, active_edges = StateReconstructor.terminal_subgraph_masks(
            batch,
            rollout,
            device=device,
        )
        nodes.append(active_nodes)
        edges.append(active_edges)
    return torch.stack(nodes, dim=0), torch.stack(edges, dim=0)


def _compute_recall_matrix(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return a (N, B) recall matrix and a length-B valid-graph mask.

    Recall is undefined for graphs whose materialized candidate graph contains
    no in-graph gold target nodes (|Y*| = 0 after preprocessing/materialization).
    Those graphs are excluded from recall averages rather than silently forcing
    the denominator to 1 and treating them as zero-recall.

    Note: ``batch.train_target_mask`` must already reflect the materialized
    training target universe so that ``gold_per_graph`` correctly captures |Y*|.
    """
    N = len(rollouts)
    B = int(batch.ptr.numel()) - 1
    batch_index = batch.batch.to(device)
    target_mask = _get_train_target_mask(batch).to(device)

    stacked_nodes, _ = _stack_terminal_states(rollouts, batch, device=device)  # [N, V]
    hit_nodes = stacked_nodes & target_mask.unsqueeze(0)  # [N, V]

    expanded_batch_idx = batch_index.unsqueeze(0).expand(N, -1)  # [N, V]
    hits_per_graph = scatter_sum(
        hit_nodes.float(), expanded_batch_idx, dim=1, dim_size=B
    )  # [N, B]
    gold_per_graph = scatter_sum(
        target_mask.float(), batch_index, dim=0, dim_size=B
    )  # [B]

    valid_graph_mask = gold_per_graph > 0  # [B]

    recall_matrix = torch.zeros((N, B), dtype=torch.float32, device=device)
    if valid_graph_mask.any():
        recall_matrix[:, valid_graph_mask] = hits_per_graph[
            :, valid_graph_mask
        ] / gold_per_graph[valid_graph_mask].unsqueeze(0)
    return recall_matrix, valid_graph_mask


def _mean_over_valid_graphs(
    values: torch.Tensor, valid_graph_mask: torch.Tensor
) -> float:
    """
    Average ``values`` restricted to graphs where recall is well-defined.

    Accepts both 1-D (per-graph) and 2-D (rollout × graph) tensors.
    Returns 0.0 when there are no valid graphs to avoid division-by-zero.
    """
    if values.numel() == 0 or not valid_graph_mask.any():
        return 0.0
    if values.dim() == 1:
        return float(values[valid_graph_mask].mean().item())
    return float(values[:, valid_graph_mask].mean().item())


def _per_rollout_mean_nodes(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> float:
    """
    Average number of active nodes per graph, averaged over rollouts.

    This includes *all* graphs (no valid-graph filtering) because node count
    is always well-defined regardless of gold availability.
    """
    B = int(batch.ptr.numel()) - 1
    batch_index = batch.batch.to(device)
    stacked_nodes, _ = _stack_terminal_states(rollouts, batch, device=device)
    total = 0.0
    for active_nodes in stacked_nodes:
        total += (
            scatter_sum(active_nodes.float(), batch_index, dim=0, dim_size=B)
            .mean()
            .item()
        )
    return total / max(len(rollouts), 1)


def _expected_dangling_ratio(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device,
) -> float:
    """
    Estimate E[dangling_ratio | added_edges > 0] over graph-rollout pairs.

    The "protected core" is conservative by design: only gold nodes that the
    current rollout actually reached (``active_gold``) are protected, so
    partial trajectories that approach but never touch a gold node are still
    counted as dangling. This matches reward semantics and is stricter than
    the idealized E*_b used in some paper notation.

    The denominator counts (graph, rollout) *pairs* where ``added_edges > 0``
    rather than whole rollouts, so a single rollout that adds edges in only
    some graphs contributes proportionally.
    """
    B = int(batch.ptr.numel()) - 1
    edge_batch = batch.edge_batch.to(device)
    edge_index = batch.edge_index.to(device)
    target_mask = _get_train_target_mask(batch).to(device)
    anchor_mask = batch.is_anchor_mask.to(device)

    root_edges = build_anchor_induced_edge_mask(edge_index, anchor_mask)
    total_ratio = 0.0
    valid_pairs = 0  # (graph, rollout) pairs where added_edges > 0

    stacked_nodes, stacked_edges = _stack_terminal_states(rollouts, batch, device=device)
    for active_nodes, active_edges in zip(stacked_nodes, stacked_edges):

        active_gold = active_nodes & target_mask
        protected_nodes = anchor_mask | active_gold
        _, core_edges = prune_to_protected_core(
            active_nodes=active_nodes,
            active_edges=active_edges,
            edge_index=edge_index,
            protected_nodes=protected_nodes,
        )

        added_edges = active_edges & ~root_edges
        dangling_edges = added_edges & ~core_edges

        dangling_count = per_graph_mask_count(
            dangling_edges, edge_batch, B, dtype=torch.float32
        )
        added_count = per_graph_mask_count(
            added_edges, edge_batch, B, dtype=torch.float32
        )

        valid_mask = added_count > 0
        if valid_mask.any():
            total_ratio += (
                (dangling_count[valid_mask] / added_count[valid_mask]).sum().item()
            )
            valid_pairs += int(valid_mask.sum().item())

    if valid_pairs == 0:
        return 0.0
    return total_ratio / valid_pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_union_subgraph_masks(
    rollouts: list[RolloutBatch], batch: RetrievalBatch
) -> UnionSubgraphMasks:
    """
    Merge terminal rollout subgraphs into union node/edge boolean masks.

    Tensor sizes are derived from ``batch`` to stay aligned with the full
    candidate graph regardless of rollout coverage.

    This helper is part of the public inference surface: downstream RAG
    callers may want the union graph even when the main evaluation metrics
    operate on the raw rollout list.
    """
    device = _metric_device(batch)

    # Use batch dimensions as the authoritative size source so that
    # union tensors are correctly shaped even when rollouts is empty or when
    # no rollout is available.
    num_nodes = batch.num_nodes
    num_edges = int(batch.edge_index.size(1))

    union_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    union_edges = torch.zeros(num_edges, dtype=torch.bool, device=device)

    stacked_nodes, stacked_edges = _stack_terminal_states(rollouts, batch, device=device)
    if stacked_nodes.numel() > 0:
        union_nodes |= stacked_nodes.any(dim=0)
    if stacked_edges.numel() > 0:
        union_edges |= stacked_edges.any(dim=0)

    return UnionSubgraphMasks(union_nodes=union_nodes, union_edges=union_edges)


def compute_distribution_expectations(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
) -> dict[str, float]:
    """
    Monte-Carlo estimates of the policy distribution's natural expectations.

    Answers: "if we sample one trajectory from π_θ without any oracle
    selection, how good is the average terminal subgraph?"

    Returns
    -------
    expected_recall : float
        Mean per-graph recall averaged over rollouts and valid graphs.
    expected_nodes : float
        Mean active-node count per graph, averaged over rollouts.
    expected_dangling_ratio : float
        Mean fraction of added edges that are dangling, over pairs where
        at least one edge was added beyond the anchor-induced root graph.
    """
    device = _metric_device(batch)
    if not rollouts:
        return {
            "expected_recall": 0.0,
            "expected_nodes": 0.0,
            "expected_dangling_ratio": 0.0,
        }

    recall_matrix, valid_graph_mask = _compute_recall_matrix(
        rollouts, batch, device=device
    )
    return {
        "expected_recall": _mean_over_valid_graphs(recall_matrix, valid_graph_mask),
        "expected_nodes": _per_rollout_mean_nodes(rollouts, batch, device=device),
        "expected_dangling_ratio": _expected_dangling_ratio(
            rollouts, batch, device=device
        ),
    }


def compute_high_reward_discovery(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
    *,
    ks: list[int] | tuple[int, ...] | None = None,
) -> dict[str, float]:
    """
    Best-of-K discovery metrics under a fixed sampling budget.

    ``oracle_max_recall_at_K`` is the best recall found within the first K
    independent samples. ``success_at_K`` is the perfect-recall hit rate,
    analogous to Pass@K reporting.

    Parameters
    ----------
    ks : list[int] | None
        Values of K to evaluate. Defaults to ``[1]`` so that the returned
        keys are always stable (``oracle_max_recall_at_1``, ``success_at_1``).
        Values larger than ``len(rollouts)`` are silently clamped away.

    Notes
    -----
    Previously the default was ``[N]`` (i.e. the full rollout count), which
    caused the empty-rollout fallback branch and the non-empty branch to
    return different key names, breaking callers that expected fixed keys.
    The default is now ``[1]`` for consistency.
    """
    N = len(rollouts)
    device = _metric_device(batch)

    # FIX: default to k=1 so key names are stable regardless of rollout count.
    requested_ks = list(ks) if ks is not None else [1]
    effective_ks = sorted({int(k) for k in requested_ks if 1 <= int(k) <= max(N, 1)})
    if not effective_ks:
        effective_ks = [1]

    if N == 0:
        return {f"oracle_max_recall_at_{k}": 0.0 for k in effective_ks} | {
            f"success_at_{k}": 0.0 for k in effective_ks
        }

    recall_matrix, valid_graph_mask = _compute_recall_matrix(
        rollouts, batch, device=device
    )

    metrics: dict[str, float] = {}
    for k in effective_ks:
        best_recall = recall_matrix[:k].max(dim=0).values  # [B]
        metrics[f"oracle_max_recall_at_{k}"] = _mean_over_valid_graphs(
            best_recall, valid_graph_mask
        )
        if valid_graph_mask.any():
            metrics[f"success_at_{k}"] = float(
                (best_recall[valid_graph_mask] == 1.0).float().mean().item()
            )
        else:
            metrics[f"success_at_{k}"] = 0.0
    return metrics


def compute_exploration_diversity(
    rollouts: list[RolloutBatch],
    batch: RetrievalBatch,
) -> dict[str, float]:
    """
    Edge-level Jaccard diversity across N terminal subgraphs.

    Diversity is measured on edge topology rather than node sets because
    identical node sets can still encode different reasoning paths. Graphs
    with zero candidate edges contribute 0.0 to the batch average (rather
    than being dropped) to avoid optimistic upward bias.

    For a pair of rollouts (i, j) where both subgraphs are empty,
    Jaccard similarity is defined as 1.0 (identical empty sets), giving
    Jaccard distance 0.0 — no diversity between two identical non-trajectories.
    """
    N = len(rollouts)
    device = _metric_device(batch)
    if N <= 1:
        return {"edge_jaccard_diversity": 0.0}

    B = int(batch.ptr.numel()) - 1
    edge_batch = batch.edge_batch.to(device)
    _, stacked_edges = _stack_terminal_states(rollouts, batch, device=device)  # [N, E]

    # Upper-triangular mask selects each unordered pair exactly once.
    triu_mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1
    )

    graph_diversities: list[float] = []
    for b in range(B):
        edge_mask = edge_batch == b
        if not bool(edge_mask.any().item()):
            graph_diversities.append(0.0)
            continue

        graphs = stacked_edges[:, edge_mask].float()  # [N, E_b]
        inter = torch.mm(graphs, graphs.T)  # [N, N]  intersection sizes
        sizes = graphs.sum(dim=1)  # [N]
        union = sizes.unsqueeze(0) + sizes.unsqueeze(1) - inter  # [N, N]

        # empty ∩ empty = empty ∪ empty → similarity = 1, distance = 0
        jaccard = inter / union.clamp(min=1e-8)
        jaccard[union == 0] = 1.0

        jaccard_dist = 1.0 - jaccard
        graph_diversities.append(float(jaccard_dist[triu_mask].mean().item()))

    return {
        "edge_jaccard_diversity": sum(graph_diversities)
        / max(len(graph_diversities), 1),
    }


def _get_train_target_mask(batch: RetrievalBatch) -> torch.Tensor:
    return batch.train_target_mask
