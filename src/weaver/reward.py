from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.graph.ops import build_anchor_induced_edge_mask
from src.weaver.state import State


@dataclass(frozen=True, slots=True)
class AnswerStats:
    hits: torch.Tensor
    gold: torch.Tensor
    retrieved: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


@dataclass(frozen=True, slots=True)
class SupportStats:
    supported_answer_recall: torch.Tensor
    supported_answer_count: torch.Tensor
    reward_answer_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class CompactnessStats:
    expanded_edge_count: torch.Tensor
    answer_degree_excess: torch.Tensor


@dataclass(frozen=True, slots=True)
class TerminalRewardOutput:
    log_reward: torch.Tensor

    utility: torch.Tensor
    base_log_reward: torch.Tensor

    supported_answer_recall: torch.Tensor
    supported_answer_count: torch.Tensor
    reward_answer_count: torch.Tensor

    expanded_edge_count: torch.Tensor
    complexity_penalty: torch.Tensor

    # Diagnostics only. These fields do not enter terminal reward.
    answer_f1: torch.Tensor
    answer_precision: torch.Tensor
    answer_recall: torch.Tensor
    answer_hits: torch.Tensor
    answer_gold: torch.Tensor
    retrieved_node_count: torch.Tensor
    answer_degree_excess: torch.Tensor


class RewardModel(nn.Module):
    """
    Anchor-supported answer evidence reward.

    Terminal subgraph x is scored by:

        U(x, q)
            = fraction of reward answer nodes that are present and connected
              to at least one active anchor inside x.

        B(x)
            = |E_x \\ E_0|, the number of learned expanded non-root edges.

    Reward:

        log R(x)
            = log(eps + U(x, q)) - edge_cost * B(x)

    The reward is a terminal verifier. It does not expose target distances,
    shortest-path labels, teacher paths, or rollout-time decisions to the policy.
    Answer degree and F1-style statistics are diagnostics only.

    Recommended first setting:
        edge_cost = 0.08 ~ 0.12
    """

    def __init__(
        self,
        *,
        utility_epsilon: float = 1.0e-4,
        log_reward_clip_min: float = -30.0,
        edge_cost: float = 0.10,
        debug_checks: bool = False,
    ) -> None:
        super().__init__()

        self.utility_epsilon = float(utility_epsilon)
        self.log_reward_clip_min = float(log_reward_clip_min)
        self.edge_cost = float(edge_cost)
        self.debug_checks = bool(debug_checks)

        if self.utility_epsilon <= 0.0:
            raise ValueError(
                f"utility_epsilon must be > 0, got {self.utility_epsilon}."
            )
        if self.log_reward_clip_min >= 0.0:
            raise ValueError(
                f"log_reward_clip_min must be < 0, got {self.log_reward_clip_min}."
            )
        if self.edge_cost < 0.0:
            raise ValueError(f"edge_cost must be >= 0, got {self.edge_cost}.")

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
        if active_nodes.ndim == 2 or active_edges.ndim == 2:
            return self.evaluate_rollout_terminal_states(
                retrieval_batch=retrieval_batch,
                active_nodes=active_nodes,
                active_edges=active_edges,
                state=state,
            )

        return self._evaluate_single_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=active_nodes,
            active_edges=active_edges,
            state=state,
        )

    @torch.no_grad()
    def evaluate_rollout_terminal_states(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        state: object | None = None,
    ) -> TerminalRewardOutput:
        if active_nodes.dtype != torch.bool:
            raise TypeError(f"active_nodes must be bool, got {active_nodes.dtype}.")
        if active_edges.dtype != torch.bool:
            raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")
        if active_nodes.ndim != 2:
            raise ValueError(
                f"active_nodes must have shape [R, N], got {tuple(active_nodes.shape)}."
            )
        if active_edges.ndim != 2:
            raise ValueError(
                f"active_edges must have shape [R, E], got {tuple(active_edges.shape)}."
            )
        if active_nodes.size(0) != active_edges.size(0):
            raise ValueError(
                "active_nodes and active_edges must have the same rollout dimension: "
                f"{active_nodes.size(0)} != {active_edges.size(0)}."
            )

        rollout_to_graph = getattr(state, "rollout_to_graph", None)
        if not isinstance(rollout_to_graph, torch.Tensor):
            raise TypeError(
                "2D rollout reward evaluation requires state.rollout_to_graph."
            )

        device = active_nodes.device
        rollout_to_graph = rollout_to_graph.to(device=device, dtype=torch.long).view(-1)
        num_rollouts = int(active_nodes.size(0))
        if rollout_to_graph.shape != (num_rollouts,):
            raise ValueError(
                f"rollout_to_graph must have shape [{num_rollouts}], "
                f"got {tuple(rollout_to_graph.shape)}."
            )

        fields: dict[str, list[torch.Tensor]] = {
            name: [] for name in TerminalRewardOutput.__dataclass_fields__
        }

        for rollout_id in range(num_rollouts):
            graph_id = int(rollout_to_graph[rollout_id].item())
            reward = self._evaluate_single_terminal_state(
                retrieval_batch=retrieval_batch,
                active_nodes=active_nodes[rollout_id],
                active_edges=active_edges[rollout_id],
                state=None,
            )
            for name in fields:
                value = getattr(reward, name).index_select(
                    0,
                    torch.tensor([graph_id], dtype=torch.long, device=device),
                )[0]
                fields[name].append(value)

        stacked = {
            name: torch.stack(values, dim=0) if values else active_nodes.new_empty((0,))
            for name, values in fields.items()
        }

        return TerminalRewardOutput(**stacked)

    def _evaluate_single_terminal_state(
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

        edge_index = retrieval_batch.edge_index.to(device=device, dtype=torch.long)
        node_batch = retrieval_batch.batch.to(device=device, dtype=torch.long)
        edge_batch = retrieval_batch.edge_batch.to(device=device, dtype=torch.long)

        num_nodes = int(retrieval_batch.num_nodes_total)
        num_edges = int(edge_index.size(1))
        num_graphs = int(retrieval_batch.num_graphs)

        _validate_shapes(
            active_nodes=active_nodes,
            active_edges=active_edges,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

        anchor_mask = node_mask(
            retrieval_batch.anchor_node_ids,
            num_nodes=num_nodes,
            device=device,
            debug_checks=self.debug_checks,
            name="anchor_node_ids",
        )

        target_mask = node_mask(
            reward_target_ids(retrieval_batch),
            num_nodes=num_nodes,
            device=device,
            debug_checks=self.debug_checks,
            name="reward_target_node_ids",
        )

        root_edges = root_edge_mask(
            edge_index=edge_index,
            anchor_mask=anchor_mask,
            state=state,
            num_edges=num_edges,
            device=device,
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

        compactness = compactness_stats(
            edge_index=edge_index,
            active_edges=active_edges,
            root_edges=root_edges,
            target_mask=target_mask,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            dtype=dtype,
        )

        utility = support.supported_answer_recall
        base_log_reward = (utility + self.utility_epsilon).log()

        complexity_penalty = self.edge_cost * compactness.expanded_edge_count

        log_reward = (base_log_reward - complexity_penalty).clamp_min(
            self.log_reward_clip_min
        )

        return TerminalRewardOutput(
            log_reward=log_reward,
            utility=utility,
            base_log_reward=base_log_reward,
            supported_answer_recall=support.supported_answer_recall,
            supported_answer_count=support.supported_answer_count,
            reward_answer_count=support.reward_answer_count,
            expanded_edge_count=compactness.expanded_edge_count,
            complexity_penalty=complexity_penalty,
            answer_f1=answer.f1,
            answer_precision=answer.precision,
            answer_recall=answer.recall,
            answer_hits=answer.hits,
            answer_gold=answer.gold,
            retrieved_node_count=answer.retrieved,
            answer_degree_excess=compactness.answer_degree_excess,
        )


def root_edge_mask(
    *,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
    state: State | None,
    num_edges: int,
    device: torch.device,
) -> torch.Tensor:
    if state is not None:
        root_edges = state.root_edges.to(device=device, dtype=torch.bool)
    else:
        root_edges = build_anchor_induced_edge_mask(
            edge_index=edge_index,
            anchor_mask=anchor_mask.to(device=device, dtype=torch.bool),
        )

    if root_edges.shape != (int(num_edges),):
        raise ValueError(
            f"root_edges must have shape [{int(num_edges)}], "
            f"got {tuple(root_edges.shape)}."
        )

    return root_edges


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
    Diagnostic answer retrieval quality.

    This is no longer the reward utility. It is retained for metrics.
    Non-answer anchors are excluded from the retrieved denominator.
    """
    device = active_nodes.device
    num_graphs = int(num_graphs)

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
    Anchor-supported answer coverage.

        supported_answer_recall[g]
            = # target nodes connected to an active anchor inside selected subgraph
              / # target nodes

    Connectivity is undirected. That is intentional: this is evidence support,
    not directed logical entailment.
    """
    device = active_nodes.device
    num_graphs = int(num_graphs)

    supported_answer_recall = torch.zeros(num_graphs, dtype=dtype, device=device)
    supported_answer_count = torch.zeros(num_graphs, dtype=dtype, device=device)
    reward_answer_count = count_by_graph(
        target_mask,
        node_batch,
        num_graphs,
        dtype=dtype,
    )

    for graph_id in range(num_graphs):
        graph_nodes = node_batch.eq(graph_id)

        graph_targets = (target_mask & graph_nodes).nonzero(as_tuple=False).flatten()
        if graph_targets.numel() == 0:
            continue

        graph_anchors = (
            (anchor_mask & active_nodes & graph_nodes).nonzero(as_tuple=False).flatten()
        )
        if graph_anchors.numel() == 0:
            continue

        graph_edges = (
            (active_edges & edge_batch.eq(graph_id)).nonzero(as_tuple=False).flatten()
        )

        reached = connected_nodes_from_anchors(
            edge_index=edge_index,
            edge_ids=graph_edges,
            anchors=graph_anchors,
        )

        if not reached:
            continue

        hits = sum(int(node_id) in reached for node_id in graph_targets.tolist())
        supported_answer_count[graph_id] = float(hits)
        supported_answer_recall[graph_id] = float(hits) / float(
            max(1, int(graph_targets.numel()))
        )

    return SupportStats(
        supported_answer_recall=supported_answer_recall,
        supported_answer_count=supported_answer_count,
        reward_answer_count=reward_answer_count,
    )


def compactness_stats(
    *,
    edge_index: torch.Tensor,
    active_edges: torch.Tensor,
    root_edges: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> CompactnessStats:
    """
    Compactness prior inputs and diagnostics.

    expanded_edge_count:
        |E_s \\ E_0| per graph.

    answer_degree_excess:
        mean_y max(0, deg_s(y) - 1) over gold target nodes per graph.
        Diagnostic only; it does not enter the reward.
    """
    num_graphs = int(num_graphs)
    num_edges = int(edge_index.size(1))

    if root_edges.shape != (num_edges,):
        raise ValueError(
            f"root_edges must have shape [{num_edges}], got {tuple(root_edges.shape)}."
        )

    expanded_edges = active_edges & ~root_edges
    expanded_edge_count = count_by_graph(
        expanded_edges,
        edge_batch,
        num_graphs,
        dtype=dtype,
    )

    answer_degree_excess = answer_degree_excess_by_graph(
        edge_index=edge_index,
        active_edges=active_edges,
        target_mask=target_mask,
        node_batch=node_batch,
        num_graphs=num_graphs,
        dtype=dtype,
    )

    return CompactnessStats(
        expanded_edge_count=expanded_edge_count,
        answer_degree_excess=answer_degree_excess,
    )


def answer_degree_excess_by_graph(
    *,
    edge_index: torch.Tensor,
    active_edges: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Average excess active degree around target nodes.

        D_Y = mean_y max(0, deg_s(y) - 1)

    This is diagnostic only. It is intentionally not part of the terminal
    reward, which is limited to supported recall times the complexity prior.
    """
    device = active_edges.device
    num_nodes = int(node_batch.numel())
    num_graphs = int(num_graphs)

    active_degree = torch.zeros(num_nodes, dtype=dtype, device=device)

    if bool(active_edges.any()):
        src = edge_index[0, active_edges]
        dst = edge_index[1, active_edges]

        ones = torch.ones(src.numel(), dtype=dtype, device=device)
        active_degree.index_add_(0, src, ones)
        active_degree.index_add_(0, dst, ones)

    target_excess = (active_degree - 1.0).clamp_min(0.0) * target_mask.to(dtype=dtype)

    excess_sum = scatter_sum(
        target_excess,
        node_batch.to(device=device, dtype=torch.long),
        dim=0,
        dim_size=num_graphs,
    )

    target_count = count_by_graph(
        target_mask,
        node_batch,
        num_graphs,
        dtype=dtype,
    ).clamp_min(1.0)

    return excess_sum / target_count


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


def reward_target_ids(batch: RetrievalBatch) -> torch.Tensor:
    """
    Answer targets used by the terminal reward.

    Reachable targets are preferred and are not replaced by all answers when
    the tensor exists but is empty. That keeps reward aligned with retriever
    responsibility: unreachable answers should not become training penalties.
    """
    reachable = getattr(batch, "reachable_target_node_ids", None)
    if isinstance(reachable, torch.Tensor):
        return reachable

    return batch.target_node_ids


def target_ids(batch: RetrievalBatch) -> torch.Tensor:
    """
    Backward-compatible alias for reward target ids.
    """
    return reward_target_ids(batch)


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

    mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)

    if ids.numel() > 0:
        mask[ids] = True

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
        batch_index.to(device=mask.device, dtype=torch.long),
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
    Keep debug_checks=False during normal training.
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


def _validate_shapes(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_nodes: int,
    num_edges: int,
) -> None:
    if active_nodes.shape != (int(num_nodes),):
        raise ValueError(
            f"active_nodes must have shape [{int(num_nodes)}], "
            f"got {tuple(active_nodes.shape)}."
        )

    if active_edges.shape != (int(num_edges),):
        raise ValueError(
            f"active_edges must have shape [{int(num_edges)}], "
            f"got {tuple(active_edges.shape)}."
        )

    if node_batch.numel() != int(num_nodes):
        raise ValueError(
            f"batch node vector length mismatch: {node_batch.numel()} != {num_nodes}."
        )

    if edge_batch.numel() != int(num_edges):
        raise ValueError(
            f"edge_batch length mismatch: {edge_batch.numel()} != {num_edges}."
        )


__all__ = [
    "AnswerStats",
    "CompactnessStats",
    "RewardModel",
    "SupportStats",
    "TerminalRewardOutput",
    "anchor_answer_support",
    "answer_degree_excess_by_graph",
    "answer_stats",
    "compactness_stats",
    "connected_nodes_from_anchors",
    "count_by_graph",
    "reward_target_ids",
    "root_edge_mask",
    "target_ids",
]
