from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.objectives.transition_batch import (
    NonterminalTransitionBatch,
    TransitionSource,
)
from src.weaver.objectives.transition_builder import ReplayTransitionStats
from src.weaver.state import ActionSpace, ExpansionBatch, StateBatch, cat_state_batches

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class WeakTransitionSourceOutput:
    nonterminal: NonterminalTransitionBatch | None
    stats: ReplayTransitionStats


@dataclass(frozen=True, slots=True)
class WeakTransitionSource:
    """
    Build bounded weak-supervision nonterminal transitions.
    """

    max_depth: int
    mode: str = "positive_frontier"
    max_transitions_per_graph: int = 128
    max_states_per_graph: int = 64
    sample_by: str = "weak_weight"
    max_positive_edges_per_state: int | None = 8

    @torch.no_grad()
    def collect(
        self,
        *,
        graph_context: GraphContext,
        target_context: TargetContext,
        initial_state: StateBatch,
    ) -> WeakTransitionSourceOutput:
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.mode != "positive_frontier":
            raise ValueError(f"Unsupported replay mode: {self.mode!r}.")
        if int(self.max_transitions_per_graph) < 0:
            raise ValueError("max_transitions_per_graph must be nonnegative.")
        if int(self.max_states_per_graph) <= 0:
            raise ValueError("max_states_per_graph must be positive.")
        if int(self.max_transitions_per_graph) == 0:
            return WeakTransitionSourceOutput(
                nonterminal=None,
                stats=ReplayTransitionStats(
                    prefix_count=0,
                    positive_transition_count=0,
                    prefix_with_positive_rate=0.0,
                    mean_positive_edges_per_prefix=0.0,
                ),
            )

        frontier = initial_state
        emitted_parent_states: list[StateBatch] = []
        emitted_parent_ids: list[Tensor] = []
        emitted_edge_ids: list[Tensor] = []
        emitted_child_states: list[StateBatch] = []
        prefix_count = 0
        prefix_with_positive = 0
        positive_transition_count = 0
        emitted_by_graph: dict[int, int] = {}

        for _ in range(int(self.max_depth)):
            if int(frontier.num_states) == 0:
                break

            action_space = frontier.action_space(graph_context)
            positive = target_context.shortest_path_edge_mask.index_select(
                0,
                action_space.expand_edge_ids,
            )

            if int(action_space.num_states) > 0:
                prefix_count += int(action_space.num_states)

            if not bool(positive.any()):
                break

            selected_state_ids, selected_edge_ids = _select_positive_expansions(
                action_space=action_space,
                positive_mask=positive,
                edge_weight=target_context.shortest_path_edge_weight,
                max_positive_edges_per_state=self.max_positive_edges_per_state,
                sample_by=self.sample_by,
            )

            if int(selected_edge_ids.numel()) == 0:
                break

            selected_state_ids, selected_edge_ids = _cap_positive_transitions_per_graph(
                frontier=frontier,
                state_ids=selected_state_ids,
                edge_ids=selected_edge_ids,
                remaining_budget_per_graph=_remaining_transition_budget(
                    frontier=frontier,
                    emitted_by_graph=emitted_by_graph,
                    max_transitions_per_graph=int(self.max_transitions_per_graph),
                ),
            )

            if int(selected_edge_ids.numel()) == 0:
                break

            local_positive_states = torch.unique(selected_state_ids)
            prefix_with_positive += int(local_positive_states.numel())
            positive_transition_count += int(selected_edge_ids.numel())
            for graph_id in frontier.graph_ids.index_select(0, selected_state_ids).tolist():
                emitted_by_graph[int(graph_id)] = emitted_by_graph.get(int(graph_id), 0) + 1

            child = frontier.branch(
                ExpansionBatch(
                    state_ids=selected_state_ids,
                    edge_ids=selected_edge_ids,
                )
            )

            emitted_parent_states.append(frontier)
            emitted_parent_ids.append(selected_state_ids)
            emitted_edge_ids.append(selected_edge_ids)
            emitted_child_states.append(child)

            frontier = _prune_positive_frontier(
                child=child,
                max_states_per_graph=int(self.max_states_per_graph),
                edge_weight=target_context.shortest_path_edge_weight,
            )

        if not emitted_parent_states:
            return WeakTransitionSourceOutput(
                nonterminal=None,
                stats=ReplayTransitionStats(
                    prefix_count=prefix_count,
                    positive_transition_count=0,
                    prefix_with_positive_rate=0.0,
                    mean_positive_edges_per_prefix=0.0,
                ),
            )

        parent_state = cat_state_batches(emitted_parent_states)
        parent_state_ids = []
        offset = 0
        for state_batch, ids in zip(emitted_parent_states, emitted_parent_ids, strict=True):
            parent_state_ids.append(ids + int(offset))
            offset += int(state_batch.num_states)
        edge_ids = torch.cat(emitted_edge_ids, dim=0)
        child_state = cat_state_batches(emitted_child_states)
        source = torch.full(
            (int(edge_ids.numel()),),
            int(TransitionSource.WEAK_REPLAY),
            dtype=torch.long,
            device=parent_state.device,
        )

        return WeakTransitionSourceOutput(
            nonterminal=NonterminalTransitionBatch(
                parent_state=parent_state,
                parent_state_ids=torch.cat(parent_state_ids, dim=0),
                edge_ids=edge_ids,
                child_state=child_state,
                source=source,
            ),
            stats=ReplayTransitionStats(
                prefix_count=prefix_count,
                positive_transition_count=positive_transition_count,
                prefix_with_positive_rate=(
                    float(prefix_with_positive) / float(prefix_count)
                    if prefix_count > 0
                    else 0.0
                ),
                mean_positive_edges_per_prefix=(
                    float(positive_transition_count) / float(prefix_with_positive)
                    if prefix_with_positive > 0
                    else 0.0
                ),
            ),
        )


def initial_replay_state_batch(
    *,
    graph_context: GraphContext,
    target_context: TargetContext,
    budget: int,
) -> StateBatch:
    valid_graph_ids = target_context.valid_graph_mask.nonzero(as_tuple=False).flatten()
    return StateBatch.initial(
        graph_ids=valid_graph_ids.to(
            device=graph_context.device,
            dtype=torch.long,
        ),
        budget=int(budget),
    )


def _select_positive_expansions(
    *,
    action_space: ActionSpace,
    positive_mask: Tensor,
    edge_weight: Tensor,
    max_positive_edges_per_state: int | None,
    sample_by: str,
) -> tuple[Tensor, Tensor]:
    candidate_rows = positive_mask.nonzero(as_tuple=False).flatten()
    if int(candidate_rows.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=positive_mask.device)
        return empty, empty

    state_ids = action_space.expand_state_ids.index_select(0, candidate_rows)
    edge_ids = action_space.expand_edge_ids.index_select(0, candidate_rows)
    if max_positive_edges_per_state is None:
        return state_ids, edge_ids

    max_edges = int(max_positive_edges_per_state)
    if max_edges <= 0:
        empty = torch.empty(0, dtype=torch.long, device=positive_mask.device)
        return empty, empty

    kept_state_ids: list[Tensor] = []
    kept_edge_ids: list[Tensor] = []
    num_states = int(action_space.num_states)
    for state_id in range(num_states):
        rows = state_ids.eq(state_id).nonzero(as_tuple=False).flatten()
        if int(rows.numel()) == 0:
            continue
        state_edge_ids = edge_ids.index_select(0, rows)
        weights = edge_weight.index_select(0, state_edge_ids)
        keep = _sample_positive_rows(
            rows=rows,
            weights=weights,
            max_edges=max_edges,
            sample_by=sample_by,
        )
        kept_state_ids.append(state_ids.index_select(0, keep))
        kept_edge_ids.append(edge_ids.index_select(0, keep))

    if not kept_state_ids:
        empty = torch.empty(0, dtype=torch.long, device=positive_mask.device)
        return empty, empty
    return torch.cat(kept_state_ids, dim=0), torch.cat(kept_edge_ids, dim=0)


def _sample_positive_rows(
    *,
    rows: Tensor,
    weights: Tensor,
    max_edges: int,
    sample_by: str,
) -> Tensor:
    if sample_by not in {"weak_weight", "uniform"}:
        raise ValueError(f"Unsupported replay sampling rule: {sample_by!r}.")
    if int(rows.numel()) <= max_edges:
        return rows
    if sample_by == "uniform":
        order = torch.randperm(int(rows.numel()), device=rows.device)
        return rows.index_select(0, order[:max_edges])

    _, order = torch.sort(weights, descending=True, stable=True)
    selected_rows = rows.index_select(0, order[:max_edges])
    return torch.sort(selected_rows).values


def _remaining_transition_budget(
    *,
    frontier: StateBatch,
    emitted_by_graph: dict[int, int],
    max_transitions_per_graph: int,
) -> dict[int, int]:
    remaining: dict[int, int] = {}
    for graph_id in torch.unique(frontier.graph_ids).tolist():
        remaining[int(graph_id)] = max(
            0,
            int(max_transitions_per_graph) - int(emitted_by_graph.get(int(graph_id), 0)),
        )
    return remaining


def _cap_positive_transitions_per_graph(
    *,
    frontier: StateBatch,
    state_ids: Tensor,
    edge_ids: Tensor,
    remaining_budget_per_graph: dict[int, int],
) -> tuple[Tensor, Tensor]:
    if int(state_ids.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=frontier.device)
        return empty, empty

    keep_rows: list[int] = []
    for row in range(int(state_ids.numel())):
        state_id = int(state_ids[row].item())
        graph_id = int(frontier.graph_ids[state_id].item())
        remaining = remaining_budget_per_graph.get(graph_id, 0)
        if remaining <= 0:
            continue
        keep_rows.append(row)
        remaining_budget_per_graph[graph_id] = remaining - 1

    if not keep_rows:
        empty = torch.empty(0, dtype=torch.long, device=frontier.device)
        return empty, empty

    keep = torch.tensor(keep_rows, dtype=torch.long, device=frontier.device)
    return (
        state_ids.index_select(0, keep),
        edge_ids.index_select(0, keep),
    )


def _prune_positive_frontier(
    *,
    child: StateBatch,
    max_states_per_graph: int,
    edge_weight: Tensor,
) -> StateBatch:
    keep_rows: list[int] = []
    rows_by_graph: dict[int, list[tuple[float, int]]] = {}

    for row in range(int(child.num_states)):
        graph_id = int(child.graph_ids[row].item())
        edge_count = int(child.edge_count[row].item())
        selected = child.edge_ids[row, :edge_count]
        score = float(edge_weight.index_select(0, selected).sum().item()) if edge_count > 0 else 0.0
        rows_by_graph.setdefault(graph_id, []).append((-score, row))

    for graph_id, scored_rows in rows_by_graph.items():
        scored_rows.sort()
        seen: set[tuple[int, ...]] = set()
        kept = 0
        for _, row in scored_rows:
            if kept >= max_states_per_graph:
                break
            edge_count = int(child.edge_count[row].item())
            key = tuple(int(v) for v in child.edge_ids[row, :edge_count].tolist())
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(row)
            kept += 1
        del graph_id

    if not keep_rows:
        return StateBatch.initial(
            graph_ids=torch.empty(0, dtype=torch.long, device=child.device),
            budget=int(child.budget),
        )

    return child.take(torch.tensor(keep_rows, dtype=torch.long, device=child.device))


__all__ = [
    "WeakTransitionSource",
    "WeakTransitionSourceOutput",
    "initial_replay_state_batch",
]
