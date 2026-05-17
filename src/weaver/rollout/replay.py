from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.backward import UniformSubgraphBackwardKernel
from src.weaver.context import GraphContext
from src.weaver.rollout.engine import RolloutContext
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import FrontierBuilder, State
from src.weaver.transitions import TransitionBatch


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    transitions: TransitionBatch

    @property
    def num_transitions(self) -> int:
        return int(self.transitions.num_transitions)

    @property
    def device(self) -> torch.device:
        return self.transitions.device


@dataclass(frozen=True, slots=True)
class ReplaySampleBudget:
    policy_rollout: int
    replay_expand: int

    @property
    def total(self) -> int:
        return self.policy_rollout + self.replay_expand


class ShortestPathReplaySource:
    def __init__(
        self,
        *,
        expand_budget: int,
    ) -> None:
        self.expand_budget = int(expand_budget)
        if self.expand_budget < 0:
            raise ValueError(f"expand_budget must be non-negative, got {self.expand_budget}.")
        self.backward_kernel = UniformSubgraphBackwardKernel()

    @torch.no_grad()
    def sample_from_rollouts(
        self,
        *,
        batch: RetrievalBatch,
        rollouts: Sequence[RolloutResult],
        num_transitions: int,
        device: torch.device | None = None,
    ) -> ReplayBatch | None:
        num_transitions = int(num_transitions)
        if num_transitions <= 0:
            return None
        device = batch.edge_index.device if device is None else device
        graph_context = GraphContext.from_batch(batch, device=device)
        rollout_context = RolloutContext(
            graph_context=graph_context,
            features=_empty_feature_bank(device=device),
            frontier_builder=FrontierBuilder.from_graph_context(graph_context),
        )
        batches = [
            transitions_from_rollouts(
                batch=batch,
                rollouts=rollouts,
                budget=self.expand_budget,
                rollout_context=rollout_context,
                backward_kernel=self.backward_kernel,
                device=device,
            ),
            oracle_prefix_transitions(
                batch=batch,
                budget=self.expand_budget,
                rollout_context=rollout_context,
                backward_kernel=self.backward_kernel,
                device=device,
            ),
        ]
        present = [x for x in batches if x is not None and x.num_transitions > 0]
        if not present:
            return None
        merged = TransitionBatch.concat(present)
        deduped = dedupe_transitions(merged)
        if deduped.num_transitions > num_transitions:
            order = torch.randperm(deduped.num_transitions, device=device)[:num_transitions]
            deduped = deduped.select_rows(order)
        return ReplayBatch(transitions=deduped)


def transitions_from_rollouts(
    *,
    batch: RetrievalBatch,
    rollouts: Sequence[RolloutResult],
    budget: int,
    rollout_context: RolloutContext,
    backward_kernel: UniformSubgraphBackwardKernel,
    device: torch.device,
) -> TransitionBatch | None:
    all_batches: list[TransitionBatch] = []
    graph_context = GraphContext.from_batch(batch, device=device)
    for rollout in rollouts:
        graph_ids = rollout.source_graph_id.to(device=device, dtype=torch.long).view(-1)
        current = State.initial_from_graph_ids(
            batch,
            graph_ids=graph_ids,
            budget=budget,
            device=device,
        )
        for step in range(rollout.max_steps):
            expand_rows = rollout.expand_mask[:, step].to(device=device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
            if expand_rows.numel() == 0:
                continue
            parent = current.select_rows(expand_rows).clone()
            edge_ids = rollout.selected_edge_ids[:, step].to(device=device, dtype=torch.long).index_select(0, expand_rows)
            child = parent.clone()
            child.apply_edges_(
                edge_index=graph_context.edge_index,
                rows=torch.arange(parent.num_rollouts, device=device, dtype=torch.long),
                edge_ids=edge_ids,
            )
            log_backward_prob = backward_kernel.log_prob(
                parent_state=parent,
                child_state=child,
                action_edge_ids=edge_ids,
                context=rollout_context,
            )
            all_batches.append(
                TransitionBatch(
                    parent_state=parent,
                    child_state=child,
                    action_edge_ids=edge_ids,
                    log_backward_prob=log_backward_prob.to(device=device, dtype=torch.float32),
                )
            )
            current.apply_edges_(
                edge_index=graph_context.edge_index,
                rows=expand_rows,
                edge_ids=edge_ids,
            )
    if not all_batches:
        return None
    return TransitionBatch.concat(all_batches)


def oracle_prefix_transitions(
    *,
    batch: RetrievalBatch,
    budget: int,
    rollout_context: RolloutContext,
    backward_kernel: UniformSubgraphBackwardKernel,
    device: torch.device,
) -> TransitionBatch | None:
    graph_context = GraphContext.from_batch(batch, device=device)
    batches: list[TransitionBatch] = []
    targets = batch.reachable_target_node_ids.to(device=device, dtype=torch.long).view(-1)
    for graph_id in range(graph_context.num_graphs):
        graph_targets = targets[graph_context.node_to_graph.index_select(0, targets).eq(graph_id)]
        for target in graph_targets.tolist():
            path = _incident_shortest_edge_path(
                graph_context=graph_context,
                graph_id=graph_id,
                target=int(target),
                budget=int(budget),
            )
            if not path:
                continue
            current = State.initial_from_graph_ids(
                batch,
                graph_ids=torch.tensor([graph_id], dtype=torch.long, device=device),
                budget=budget,
                device=device,
            )
            for edge_id in path[:budget]:
                parent = current.clone()
                child = current.clone()
                edge_tensor = torch.tensor([edge_id], dtype=torch.long, device=device)
                child.apply_edges_(
                    edge_index=graph_context.edge_index,
                    rows=torch.zeros(1, dtype=torch.long, device=device),
                    edge_ids=edge_tensor,
                )
                log_backward_prob = backward_kernel.log_prob(
                    parent_state=parent,
                    child_state=child,
                    action_edge_ids=edge_tensor,
                    context=rollout_context,
                )
                batches.append(
                    TransitionBatch(
                        parent_state=parent,
                        child_state=child,
                        action_edge_ids=edge_tensor,
                        log_backward_prob=log_backward_prob.to(device=device, dtype=torch.float32),
                    )
                )
                current = child
    if not batches:
        return None
    return TransitionBatch.concat(batches)


def dedupe_transitions(batch: TransitionBatch) -> TransitionBatch:
    seen: set[tuple[int, tuple[int, ...], int]] = set()
    keep: list[int] = []
    edge_masks = batch.parent_state.edge_mask.detach().cpu()
    graphs = batch.parent_state.row_to_graph.detach().cpu()
    action_edge_ids = batch.action_edge_ids.detach().cpu()
    for row in range(batch.num_transitions):
        edges = tuple(edge_masks[row].nonzero(as_tuple=False).view(-1).tolist())
        key = (int(graphs[row]), edges, int(action_edge_ids[row]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(row)
    if not keep:
        return batch.select_rows(torch.empty(0, dtype=torch.long, device=batch.device))
    return batch.select_rows(torch.tensor(keep, dtype=torch.long, device=batch.device))


def _incident_shortest_edge_path(
    *,
    graph_context: GraphContext,
    graph_id: int,
    target: int,
    budget: int,
) -> list[int]:
    anchors = (graph_context.anchor_mask & graph_context.node_to_graph.eq(graph_id)).nonzero(as_tuple=False).view(-1)
    if anchors.numel() == 0:
        return []
    if bool(torch.any(anchors.eq(int(target)))):
        return []

    frontier = [int(x) for x in anchors.tolist()]
    parent_node: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    seen = set(frontier)
    src_all = graph_context.edge_index[0]
    dst_all = graph_context.edge_index[1]
    edge_graph = graph_context.edge_to_graph

    for _ in range(int(budget)):
        next_frontier: list[int] = []
        for node in frontier:
            incident = ((src_all.eq(node) | dst_all.eq(node)) & edge_graph.eq(graph_id)).nonzero(as_tuple=False).view(-1)
            for edge_id_tensor in incident.tolist():
                edge_id = int(edge_id_tensor)
                src = int(src_all[edge_id])
                dst = int(dst_all[edge_id])
                other = dst if src == node else src
                if other in seen:
                    continue
                seen.add(other)
                parent_node[other] = node
                parent_edge[other] = edge_id
                if other == int(target):
                    path: list[int] = []
                    cur = other
                    while cur not in set(int(x) for x in anchors.tolist()):
                        path.append(parent_edge[cur])
                        cur = parent_node[cur]
                    path.reverse()
                    return path
                next_frontier.append(other)
        frontier = next_frontier
        if not frontier:
            break
    return []


def _empty_feature_bank(*, device: torch.device):
    from src.weaver.nn.feature_encoder import FeatureBank

    empty_bool = torch.empty(0, dtype=torch.bool, device=device)
    empty_float = torch.empty((0, 1), dtype=torch.float32, device=device)
    return FeatureBank(
        node_h=empty_float,
        edge_h=empty_float,
        query_h=empty_float,
        node_is_non_text=empty_bool,
        node_sem_h=empty_float,
        rel_sem_h=empty_float,
        query_sem_h=empty_float,
        rel_h=empty_float,
    )


__all__ = [
    "ReplayBatch",
    "ReplaySampleBudget",
    "ShortestPathReplaySource",
    "dedupe_transitions",
    "oracle_prefix_transitions",
    "transitions_from_rollouts",
]
