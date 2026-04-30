from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import nn
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.weaver.state import State


@dataclass(frozen=True)
class AnswerStats:
    hits: torch.Tensor
    gold: torch.Tensor
    retrieved: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


@dataclass(frozen=True)
class SupportStats:
    support: torch.Tensor
    supported_targets: torch.Tensor
    reachable_targets: torch.Tensor


@dataclass(frozen=True)
class MinimalityStats:
    minimal_edge_count: torch.Tensor
    minimality_gap: torch.Tensor
    expanded_edge_count: torch.Tensor


@dataclass(frozen=True)
class TerminalRewardOutput:
    log_reward: torch.Tensor

    utility: torch.Tensor
    base_log_reward: torch.Tensor

    answer_f1: torch.Tensor
    answer_precision: torch.Tensor
    answer_recall: torch.Tensor
    answer_hits: torch.Tensor
    answer_gold: torch.Tensor
    retrieved_node_count: torch.Tensor

    answer_support: torch.Tensor
    supported_targets: torch.Tensor
    reachable_targets: torch.Tensor

    expanded_edge_count: torch.Tensor
    minimal_edge_count: torch.Tensor
    minimality_gap: torch.Tensor

    edge_penalty: torch.Tensor
    minimality_penalty: torch.Tensor


class RewardModel(nn.Module):
    """
    Verified Minimal Sufficient Evidence reward.

    Terminal subgraph x is rewarded by verifiable answer utility and penalized
    by avoidable redundancy.

        U(x)
            = answer_weight * F1_A(x)
            + (1 - answer_weight) * Support_reach(x)

        m_delta(x)
            = min |E(x') \\ E0|
              subject to x' subset x and U(x') >= U(x) - delta

        MinGap_delta(x)
            = |E(x) \\ E0| - m_delta(x)

        log R(x)
            = log(eps + U(x))
              - edge_cost * |E(x) \\ E0|
              - minimality_gap_cost * MinGap_delta(x)

    The reward is terminal-only. It does not expose target distances or shortest
    path labels to the policy. Gold-derived information is used only as a
    verifier for terminal rollout quality.
    """

    def __init__(
        self,
        *,
        utility_epsilon: float = 1.0e-4,
        log_reward_clip_min: float = -30.0,
        answer_weight: float = 0.7,
        minimality_tolerance: float = 0.02,
        edge_cost: float = 0.03,
        minimality_gap_cost: float = 0.15,
        zero_utility_minimality: bool = True,
        debug_checks: bool = False,
    ) -> None:
        super().__init__()

        self.utility_epsilon = float(utility_epsilon)
        self.log_reward_clip_min = float(log_reward_clip_min)
        self.answer_weight = float(answer_weight)
        self.minimality_tolerance = float(minimality_tolerance)
        self.edge_cost = float(edge_cost)
        self.minimality_gap_cost = float(minimality_gap_cost)
        self.zero_utility_minimality = bool(zero_utility_minimality)
        self.debug_checks = bool(debug_checks)

        if self.utility_epsilon <= 0.0:
            raise ValueError(
                f"utility_epsilon must be > 0, got {self.utility_epsilon}."
            )
        if self.log_reward_clip_min >= 0.0:
            raise ValueError(
                f"log_reward_clip_min must be < 0, got {self.log_reward_clip_min}."
            )
        if not 0.0 <= self.answer_weight <= 1.0:
            raise ValueError(
                f"answer_weight must be in [0, 1], got {self.answer_weight}."
            )
        if self.minimality_tolerance < 0.0:
            raise ValueError(
                f"minimality_tolerance must be >= 0, got {self.minimality_tolerance}."
            )
        if self.edge_cost < 0.0:
            raise ValueError(f"edge_cost must be >= 0, got {self.edge_cost}.")
        if self.minimality_gap_cost < 0.0:
            raise ValueError(
                f"minimality_gap_cost must be >= 0, got {self.minimality_gap_cost}."
            )

    @torch.no_grad()
    def forward(
        self,
        retrieval_batch: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        state: State | None = None,
    ) -> torch.Tensor:
        return self.evaluate_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=active_nodes,
            active_edges=active_edges,
            state=state,
        ).log_reward

    @torch.no_grad()
    def evaluate_terminal_state(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        state: State | None = None,
    ) -> TerminalRewardOutput:
        if active_nodes.dtype != torch.bool:
            raise TypeError(f"active_nodes must be bool, got {active_nodes.dtype}.")
        if active_edges.dtype != torch.bool:
            raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")

        device = active_nodes.device
        dtype = torch.float32

        num_nodes = int(retrieval_batch.num_nodes_total)
        num_edges = int(retrieval_batch.num_edges_total)
        num_graphs = int(retrieval_batch.num_graphs)

        if active_nodes.numel() != num_nodes:
            raise ValueError(
                f"active_nodes length mismatch: {active_nodes.numel()} != {num_nodes}."
            )
        if active_edges.numel() != num_edges:
            raise ValueError(
                f"active_edges length mismatch: {active_edges.numel()} != {num_edges}."
            )

        edge_index = retrieval_batch.edge_index.to(device=device, dtype=torch.long)
        node_batch = retrieval_batch.batch.to(device=device, dtype=torch.long)
        edge_batch = retrieval_batch.edge_batch.to(device=device, dtype=torch.long)

        anchor_mask = node_mask(
            retrieval_batch.anchor_node_ids,
            num_nodes=num_nodes,
            device=device,
            debug_checks=self.debug_checks,
            name="anchor_node_ids",
        )

        target_mask = node_mask(
            target_ids(retrieval_batch),
            num_nodes=num_nodes,
            device=device,
            debug_checks=self.debug_checks,
            name="target_node_ids",
        )

        if state is None:
            root_edges = torch.zeros(num_edges, dtype=torch.bool, device=device)
        else:
            root_edges = state.root_active_edges.to(device=device, dtype=torch.bool)

        if root_edges.numel() != num_edges:
            raise ValueError(
                f"root edge mask length mismatch: {root_edges.numel()} != {num_edges}."
            )

        answer = answer_stats(
            active_nodes=active_nodes,
            anchor_mask=anchor_mask,
            target_mask=target_mask,
            node_batch=node_batch,
            num_graphs=num_graphs,
            dtype=dtype,
        )

        support = anchor_answer_support(
            edge_index=edge_index,
            active_nodes=active_nodes,
            active_edges=active_edges,
            anchor_mask=anchor_mask,
            target_mask=target_mask,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            dtype=dtype,
        )

        utility = self.utility(answer.f1, support.support)

        minimality = minimality_stats(
            edge_index=edge_index,
            active_edges=active_edges,
            root_edges=root_edges,
            anchor_mask=anchor_mask,
            target_mask=target_mask,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            answer_weight=self.answer_weight,
            tolerance=self.minimality_tolerance,
            zero_utility_minimality=self.zero_utility_minimality,
            dtype=dtype,
        )

        base_log_reward = (utility + self.utility_epsilon).log()
        edge_penalty = self.edge_cost * minimality.expanded_edge_count
        minimality_penalty = self.minimality_gap_cost * minimality.minimality_gap

        log_reward = (base_log_reward - edge_penalty - minimality_penalty).clamp_min(
            self.log_reward_clip_min
        )

        return TerminalRewardOutput(
            log_reward=log_reward,
            utility=utility,
            base_log_reward=base_log_reward,
            answer_f1=answer.f1,
            answer_precision=answer.precision,
            answer_recall=answer.recall,
            answer_hits=answer.hits,
            answer_gold=answer.gold,
            retrieved_node_count=answer.retrieved,
            answer_support=support.support,
            supported_targets=support.supported_targets,
            reachable_targets=support.reachable_targets,
            expanded_edge_count=minimality.expanded_edge_count,
            minimal_edge_count=minimality.minimal_edge_count,
            minimality_gap=minimality.minimality_gap,
            edge_penalty=edge_penalty,
            minimality_penalty=minimality_penalty,
        )

    def utility(self, answer_f1: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return self.answer_weight * answer_f1 + (1.0 - self.answer_weight) * support


def answer_stats(
    *,
    active_nodes: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> AnswerStats:
    """
    Answer retrieval quality.

    Non-answer anchors are excluded from the retrieved denominator. Anchors are
    query conditions, not retrieved answers. If an anchor is also a target, it is
    still counted as retrieved.
    """
    device = active_nodes.device

    active_gold = active_nodes & target_mask
    retrieved = active_nodes & (~anchor_mask | target_mask)

    hits = count_by_graph(active_gold, node_batch, num_graphs, dtype=dtype)
    gold = count_by_graph(target_mask, node_batch, num_graphs, dtype=dtype)
    retrieved_count = count_by_graph(retrieved, node_batch, num_graphs, dtype=dtype)

    precision = torch.zeros(num_graphs, dtype=dtype, device=device)
    recall = torch.zeros(num_graphs, dtype=dtype, device=device)

    has_retrieved = retrieved_count > 0.0
    has_gold = gold > 0.0

    precision[has_retrieved] = hits[has_retrieved] / retrieved_count[has_retrieved]
    recall[has_gold] = hits[has_gold] / gold[has_gold]

    denom = precision + recall
    f1 = torch.zeros(num_graphs, dtype=dtype, device=device)
    valid = denom > 0.0
    f1[valid] = 2.0 * precision[valid] * recall[valid] / denom[valid]

    return AnswerStats(
        hits=hits,
        gold=gold,
        retrieved=retrieved_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def anchor_answer_support(
    *,
    edge_index: torch.Tensor,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> SupportStats:
    """
    Fraction of reachable target nodes connected to at least one active anchor
    inside the selected terminal subgraph.

    Connectivity is undirected by design. This judges whether the generated
    evidence subgraph supports an answer entity from the anchor context.
    """
    device = active_nodes.device

    support = torch.zeros(num_graphs, dtype=dtype, device=device)
    supported_targets = torch.zeros(num_graphs, dtype=dtype, device=device)
    reachable_targets = count_by_graph(
        target_mask,
        node_batch,
        num_graphs,
        dtype=dtype,
    )

    for graph_id in range(num_graphs):
        graph_nodes = node_batch.eq(graph_id)
        graph_targets = (target_mask & graph_nodes).nonzero(as_tuple=False).view(-1)
        if graph_targets.numel() == 0:
            continue

        graph_anchors = (
            (anchor_mask & active_nodes & graph_nodes).nonzero(as_tuple=False).view(-1)
        )
        if graph_anchors.numel() == 0:
            continue

        graph_edges = (
            (active_edges & edge_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )

        reached = connected_nodes_from_anchors(
            edge_index=edge_index,
            edge_ids=graph_edges,
            anchors=graph_anchors,
        )

        if not reached:
            continue

        hit = sum(int(node_id) in reached for node_id in graph_targets.tolist())
        supported_targets[graph_id] = float(hit)
        support[graph_id] = float(hit) / float(max(1, int(graph_targets.numel())))

    return SupportStats(
        support=support,
        supported_targets=supported_targets,
        reachable_targets=reachable_targets,
    )


def minimality_stats(
    *,
    edge_index: torch.Tensor,
    active_edges: torch.Tensor,
    root_edges: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    answer_weight: float,
    tolerance: float,
    zero_utility_minimality: bool,
    dtype: torch.dtype,
) -> MinimalityStats:
    """
    Exact minimal sufficient subset search over selected non-root edges.

    For each terminal subgraph x, enumerate all subsets of selected expanded
    edges and find the smallest subset preserving utility within tolerance.

    This is exact and cheap when expand_budget is small. If budget becomes large,
    this function should be replaced by beam/rejection subset search.
    """
    device = active_edges.device

    expanded_edges = active_edges & ~root_edges
    expanded_edge_count = count_by_graph(
        expanded_edges,
        edge_batch,
        num_graphs,
        dtype=dtype,
    )

    minimal_edge_count = expanded_edge_count.clone()

    for graph_id in range(num_graphs):
        graph_expanded = (
            (expanded_edges & edge_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )

        if graph_expanded.numel() == 0:
            minimal_edge_count[graph_id] = 0.0
            continue

        full_edges = (
            (active_edges & edge_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )

        full_utility = graph_utility(
            edge_index=edge_index,
            selected_edges=full_edges,
            root_edges=root_edges,
            candidate_extra_edges=graph_expanded,
            kept_extra_edge_ids=graph_expanded,
            anchor_mask=anchor_mask,
            target_mask=target_mask,
            node_batch=node_batch,
            graph_id=graph_id,
            answer_weight=answer_weight,
            dtype=dtype,
        )

        if zero_utility_minimality and full_utility <= 0.0:
            minimal_edge_count[graph_id] = float(graph_expanded.numel())
            continue

        threshold = max(0.0, full_utility - float(tolerance))
        best_size = int(graph_expanded.numel())

        expanded_list = [int(edge_id) for edge_id in graph_expanded.tolist()]

        for subset_size in range(best_size + 1):
            found = False
            for subset in combinations(expanded_list, subset_size):
                kept = torch.tensor(subset, dtype=torch.long, device=device)
                utility = graph_utility(
                    edge_index=edge_index,
                    selected_edges=full_edges,
                    root_edges=root_edges,
                    candidate_extra_edges=graph_expanded,
                    kept_extra_edge_ids=kept,
                    anchor_mask=anchor_mask,
                    target_mask=target_mask,
                    node_batch=node_batch,
                    graph_id=graph_id,
                    answer_weight=answer_weight,
                    dtype=dtype,
                )

                if utility >= threshold:
                    best_size = subset_size
                    found = True
                    break

            if found:
                break

        minimal_edge_count[graph_id] = float(best_size)

    minimality_gap = (expanded_edge_count - minimal_edge_count).clamp_min(0.0)

    return MinimalityStats(
        minimal_edge_count=minimal_edge_count,
        minimality_gap=minimality_gap,
        expanded_edge_count=expanded_edge_count,
    )


def graph_utility(
    *,
    edge_index: torch.Tensor,
    selected_edges: torch.Tensor,
    root_edges: torch.Tensor,
    candidate_extra_edges: torch.Tensor,
    kept_extra_edge_ids: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    graph_id: int,
    answer_weight: float,
    dtype: torch.dtype,
) -> float:
    """
    Utility of a graph-local subset.

    The subset always keeps root edges from the original terminal state and
    keeps only the proposed non-root edge subset. Active nodes are reconstructed
    from graph anchors and selected edge endpoints.
    """
    device = edge_index.device

    graph_nodes = node_batch.eq(int(graph_id))
    graph_anchors = (anchor_mask & graph_nodes).nonzero(as_tuple=False).view(-1)
    graph_targets = (target_mask & graph_nodes).nonzero(as_tuple=False).view(-1)

    if graph_targets.numel() == 0:
        return 0.0

    graph_root_edges = (
        (root_edges & edge_ids_to_mask(selected_edges, int(edge_index.size(1)), device))
        .nonzero(as_tuple=False)
        .view(-1)
    )

    if kept_extra_edge_ids.numel() == 0:
        graph_edges = graph_root_edges
    elif graph_root_edges.numel() == 0:
        graph_edges = kept_extra_edge_ids
    else:
        graph_edges = torch.cat([graph_root_edges, kept_extra_edge_ids], dim=0)

    active_node_set = {int(node_id) for node_id in graph_anchors.tolist()}

    if graph_edges.numel() > 0:
        src = edge_index[0].index_select(0, graph_edges).tolist()
        dst = edge_index[1].index_select(0, graph_edges).tolist()
        for left, right in zip(src, dst):
            active_node_set.add(int(left))
            active_node_set.add(int(right))

    hits = sum(int(node_id) in active_node_set for node_id in graph_targets.tolist())

    retrieved = [
        node_id
        for node_id in active_node_set
        if (not bool(anchor_mask[node_id].item())) or bool(target_mask[node_id].item())
    ]

    precision = float(hits) / float(len(retrieved)) if retrieved else 0.0
    recall = float(hits) / float(max(1, int(graph_targets.numel())))

    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    reached = connected_nodes_from_anchors(
        edge_index=edge_index,
        edge_ids=graph_edges,
        anchors=graph_anchors,
    )

    support_hits = sum(int(node_id) in reached for node_id in graph_targets.tolist())
    support = float(support_hits) / float(max(1, int(graph_targets.numel())))

    return float(answer_weight) * f1 + (1.0 - float(answer_weight)) * support


def connected_nodes_from_anchors(
    *,
    edge_index: torch.Tensor,
    edge_ids: torch.Tensor,
    anchors: torch.Tensor,
) -> set[int]:
    visited = {int(node_id) for node_id in anchors.tolist()}
    frontier = list(visited)

    adjacency: dict[int, list[int]] = {}

    if edge_ids.numel() > 0:
        src = edge_index[0].index_select(0, edge_ids).tolist()
        dst = edge_index[1].index_select(0, edge_ids).tolist()

        for left, right in zip(src, dst):
            left_id = int(left)
            right_id = int(right)
            adjacency.setdefault(left_id, []).append(right_id)
            adjacency.setdefault(right_id, []).append(left_id)

    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)

    return visited


def target_ids(batch: RetrievalBatch) -> torch.Tensor:
    reachable = getattr(batch, "reachable_target_node_ids", None)
    if isinstance(reachable, torch.Tensor) and reachable.numel() > 0:
        return reachable

    return batch.target_node_ids


def node_mask(
    ids: torch.Tensor,
    *,
    num_nodes: int,
    device: torch.device,
    debug_checks: bool,
    name: str,
) -> torch.Tensor:
    ids = ids.to(device=device, dtype=torch.long).view(-1)

    if debug_checks:
        check_ids_in_range(ids, upper=int(num_nodes), name=name)

    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if ids.numel() > 0:
        mask[ids] = True

    return mask


def edge_ids_to_mask(
    edge_ids: torch.Tensor,
    num_edges: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(int(num_edges), dtype=torch.bool, device=device)
    if edge_ids.numel() > 0:
        mask[edge_ids.to(device=device, dtype=torch.long)] = True
    return mask


def count_by_graph(
    mask: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return scatter_sum(
        mask.to(dtype=dtype),
        batch_index,
        dim=0,
        dim_size=int(num_graphs),
    )


def check_ids_in_range(
    ids: torch.Tensor,
    *,
    upper: int,
    name: str,
) -> None:
    """
    Debug-only id range check.

    This uses .item(), so it synchronizes GPU execution.
    Keep debug_checks=False in normal training.
    """
    if ids.numel() == 0:
        return

    min_id = int(ids.amin().item())
    max_id = int(ids.amax().item())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(
            f"{name} contains ids outside range [0, {upper}): "
            f"min={min_id}, max={max_id}."
        )


__all__ = [
    "AnswerStats",
    "SupportStats",
    "MinimalityStats",
    "TerminalRewardOutput",
    "RewardModel",
    "answer_stats",
    "anchor_answer_support",
    "minimality_stats",
    "count_by_graph",
]
