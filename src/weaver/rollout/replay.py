from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.objectives.transition_batch import NonterminalTransitionBatch, TransitionSource
from src.weaver.objectives.transition_builder import ReplayTransitionStats
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplaySourceOutput:
    nonterminal: NonterminalTransitionBatch | None
    stats: ReplayTransitionStats


@dataclass(frozen=True, slots=True)
class ReplaySource:
    max_transitions_per_graph: int = 128
    top_k_per_state: int = 2

    @torch.no_grad()
    def collect(
        self,
        *,
        graph_context: GraphContext,
        target_context: TargetContext,
        replay_context: ReplayContext,
        initial_state: StateBatch,
    ) -> ReplaySourceOutput:
        if int(self.max_transitions_per_graph) < 0:
            raise ValueError("max_transitions_per_graph must be nonnegative.")
        if int(self.top_k_per_state) <= 0:
            raise ValueError("top_k_per_state must be positive.")
        if int(self.max_transitions_per_graph) == 0:
            return ReplaySourceOutput(
                nonterminal=None,
                stats=ReplayTransitionStats(
                    prefix_count=int(initial_state.num_states),
                    positive_transition_count=0,
                    prefix_with_positive_rate=0.0,
                    mean_positive_edges_per_prefix=0.0,
                ),
            )

        replay_program = _decode_replay_program(
            replay_context=replay_context,
            num_edges=int(graph_context.num_edges),
        )
        prefix_count = int(initial_state.num_states)
        if prefix_count == 0:
            return ReplaySourceOutput(
                nonterminal=None,
                stats=ReplayTransitionStats(
                    prefix_count=0,
                    positive_transition_count=0,
                    prefix_with_positive_rate=0.0,
                    mean_positive_edges_per_prefix=0.0,
                ),
            )

        action_space = initial_state.action_space(graph_context)
        selected_state_ids, selected_edge_ids = _select_replay_expansions(
            frontier=initial_state,
            graph_context=graph_context,
            target_context=target_context,
            replay_program=replay_program,
            action_space=action_space,
            top_k_per_state=int(self.top_k_per_state),
        )
        prefix_with_positive = int(torch.unique(selected_state_ids).numel()) if int(selected_state_ids.numel()) > 0 else 0
        selected_state_ids, selected_edge_ids = _cap_positive_transitions_per_graph(
            frontier=initial_state,
            state_ids=selected_state_ids,
            edge_ids=selected_edge_ids,
            remaining_budget_per_graph=_remaining_transition_budget(
                frontier=initial_state,
                emitted_by_graph={},
                max_transitions_per_graph=int(self.max_transitions_per_graph),
            ),
        )
        if int(selected_edge_ids.numel()) == 0:
            return ReplaySourceOutput(
                nonterminal=None,
                stats=ReplayTransitionStats(
                    prefix_count=prefix_count,
                    positive_transition_count=0,
                    prefix_with_positive_rate=0.0,
                    mean_positive_edges_per_prefix=0.0,
                ),
            )

        prefix_with_positive = int(torch.unique(selected_state_ids).numel())
        positive_transition_count = int(selected_edge_ids.numel())
        child_state = initial_state.branch(
            ExpansionBatch(state_ids=selected_state_ids, edge_ids=selected_edge_ids),
            graph_context=graph_context,
        )
        source = torch.full(
            (positive_transition_count,),
            int(TransitionSource.WEAK_REPLAY),
            dtype=torch.long,
            device=initial_state.device,
        )
        return ReplaySourceOutput(
            nonterminal=NonterminalTransitionBatch(
                parent_state=initial_state,
                parent_state_ids=selected_state_ids,
                edge_ids=selected_edge_ids,
                child_state=child_state,
                graph_context=graph_context,
                source=source,
            ),
            stats=ReplayTransitionStats(
                prefix_count=prefix_count,
                positive_transition_count=positive_transition_count,
                prefix_with_positive_rate=(float(prefix_with_positive) / float(prefix_count) if prefix_count > 0 else 0.0),
                mean_positive_edges_per_prefix=(float(positive_transition_count) / float(prefix_with_positive) if prefix_with_positive > 0 else 0.0),
            ),
        )


@dataclass(frozen=True, slots=True)
class _DecodedReplayProgram:
    candidate_edges: tuple[tuple[int, ...], ...]
    candidate_target_positions: tuple[tuple[int, ...], ...]
    edge_to_candidate_ids: tuple[tuple[int, ...], ...]


def initial_replay_state_batch(
    *,
    graph_context: GraphContext,
    target_context: TargetContext,
    budget: int,
) -> StateBatch:
    valid_graph_ids = target_context.valid_graph_mask.nonzero(as_tuple=False).flatten()
    return StateBatch.initial(
        graph_ids=valid_graph_ids.to(device=graph_context.device, dtype=torch.long),
        budget=int(budget),
        graph_context=graph_context,
    )


def _decode_replay_program(
    *,
    replay_context: ReplayContext,
    num_edges: int,
) -> _DecodedReplayProgram:
    candidate_edges: list[tuple[int, ...]] = []
    candidate_ptr = replay_context.candidate_ptr.tolist()
    candidate_edge_ids = replay_context.candidate_edge_ids.tolist()
    candidate_target_positions: list[tuple[int, ...]] = []
    candidate_target_ptr = replay_context.candidate_target_ptr.tolist()
    flat_candidate_target_positions = replay_context.candidate_target_positions.tolist()
    for candidate_id in range(max(len(candidate_ptr) - 1, 0)):
        start = int(candidate_ptr[candidate_id])
        end = int(candidate_ptr[candidate_id + 1])
        candidate_edges.append(tuple(int(edge_id) for edge_id in candidate_edge_ids[start:end]))
        target_start = int(candidate_target_ptr[candidate_id])
        target_end = int(candidate_target_ptr[candidate_id + 1])
        candidate_target_positions.append(
            tuple(int(pos) for pos in flat_candidate_target_positions[target_start:target_end])
        )

    edge_to_candidate_ids: list[tuple[int, ...]] = []
    edge_candidate_ptr = replay_context.edge_to_candidate_ptr.tolist()
    edge_candidate_ids = replay_context.edge_to_candidate_ids.tolist()
    if len(edge_candidate_ptr) == 0:
        edge_candidate_ptr = [0] * (int(num_edges) + 1)
    for edge_id in range(int(num_edges)):
        start = int(edge_candidate_ptr[edge_id])
        end = int(edge_candidate_ptr[edge_id + 1])
        edge_to_candidate_ids.append(tuple(int(candidate_id) for candidate_id in edge_candidate_ids[start:end]))

    return _DecodedReplayProgram(
        candidate_edges=tuple(candidate_edges),
        candidate_target_positions=tuple(candidate_target_positions),
        edge_to_candidate_ids=tuple(edge_to_candidate_ids),
    )


def _select_replay_expansions(
    *,
    frontier: StateBatch,
    graph_context: GraphContext,
    target_context: TargetContext,
    replay_program: _DecodedReplayProgram,
    action_space,
    top_k_per_state: int,
) -> tuple[Tensor, Tensor]:
    kept_state_ids: list[int] = []
    kept_edge_ids: list[int] = []
    for state_id in range(int(action_space.num_states)):
        start = int(action_space.expand_ptr[state_id].item())
        end = int(action_space.expand_ptr[state_id + 1].item())
        if start >= end:
            continue
        legal_edges = action_space.expand_edge_ids[start:end]
        ranked_edges = _rank_replay_edges_for_state(
            frontier=frontier,
            graph_context=graph_context,
            state_id=state_id,
            legal_edge_ids=legal_edges,
            target_context=target_context,
            replay_program=replay_program,
        )
        if not ranked_edges:
            continue
        ranked_edges = ranked_edges[: int(top_k_per_state)]
        kept_state_ids.extend([state_id] * len(ranked_edges))
        kept_edge_ids.extend(ranked_edges)

    if not kept_edge_ids:
        empty = torch.empty(0, dtype=torch.long, device=frontier.device)
        return empty, empty

    return (
        torch.tensor(kept_state_ids, dtype=torch.long, device=frontier.device),
        torch.tensor(kept_edge_ids, dtype=torch.long, device=frontier.device),
    )


def _rank_replay_edges_for_state(
    *,
    frontier: StateBatch,
    graph_context: GraphContext,
    state_id: int,
    legal_edge_ids: Tensor,
    target_context: TargetContext,
    replay_program: _DecodedReplayProgram,
) -> list[int]:
    budget_left = int(frontier.budget_left[state_id].item())
    if budget_left <= 0:
        return []

    selected_count = int(frontier.edge_count[state_id].item())
    selected_edges = {
        int(edge_id)
        for edge_id in frontier.selected_edge_ids[state_id, :selected_count].tolist()
        if int(edge_id) >= 0
    }
    covered_positions = _covered_target_positions_for_state(
        frontier=frontier,
        graph_context=graph_context,
        state_id=state_id,
        target_context=target_context,
    )

    scored: list[tuple[tuple[int, int, int, int], int]] = []
    for edge_id in legal_edge_ids.tolist():
        edge_id = int(edge_id)
        candidate_ids = replay_program.edge_to_candidate_ids[edge_id] if edge_id < len(replay_program.edge_to_candidate_ids) else tuple()
        if not candidate_ids:
            continue
        best_score: tuple[int, int, int, int] | None = None
        for candidate_id in candidate_ids:
            if candidate_id >= len(replay_program.candidate_edges):
                continue
            candidate_edges = replay_program.candidate_edges[candidate_id]
            if edge_id not in candidate_edges:
                continue
            residual_edges = [candidate_edge for candidate_edge in candidate_edges if candidate_edge not in selected_edges]
            if len(residual_edges) > budget_left:
                continue
            cover_gain = _cover_gain(
                covered_positions=covered_positions,
                candidate_target_positions=replay_program.candidate_target_positions[candidate_id],
            )
            if cover_gain <= 0 and edge_id in selected_edges:
                continue
            shared_trunk = -candidate_edges.index(edge_id)
            residual_len = len(residual_edges)
            score = (cover_gain, shared_trunk, -residual_len, -edge_id)
            if best_score is None or score > best_score:
                best_score = score
        if best_score is not None:
            scored.append((best_score, edge_id))

    scored.sort(reverse=True)
    return [edge_id for _, edge_id in scored]


def _covered_target_positions_for_state(
    *,
    frontier: StateBatch,
    graph_context: GraphContext,
    state_id: int,
    target_context: TargetContext,
) -> frozenset[int]:
    active = frontier.active_node_index(graph_context)
    active_rows = active.row_ids.eq(int(state_id))
    covered: set[int] = set()
    if not bool(active_rows.any()):
        return frozenset()
    active_nodes = active.node_ids[active_rows]
    for pos, node_id in enumerate(target_context.reachable_target_node_ids.tolist()):
        if bool(active_nodes.eq(int(node_id)).any()):
            covered.add(int(pos))
    return frozenset(covered)


def _cover_gain(
    *,
    covered_positions: frozenset[int],
    candidate_target_positions: tuple[int, ...],
) -> int:
    gain = 0
    for pos in candidate_target_positions:
        if int(pos) not in covered_positions:
            gain += 1
    return gain


def _remaining_transition_budget(
    *,
    frontier: StateBatch,
    emitted_by_graph: dict[int, int],
    max_transitions_per_graph: int,
) -> dict[int, int]:
    remaining: dict[int, int] = {}
    for graph_id in torch.unique(frontier.graph_ids).tolist():
        remaining[int(graph_id)] = max(0, int(max_transitions_per_graph) - int(emitted_by_graph.get(int(graph_id), 0)))
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
    return state_ids.index_select(0, keep), edge_ids.index_select(0, keep)


__all__ = [
    "ReplaySource",
    "ReplaySourceOutput",
    "initial_replay_state_batch",
]
