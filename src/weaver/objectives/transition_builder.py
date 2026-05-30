from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.objectives.transition_batch import (
    EdgeFlowMatchingBatch,
    NonterminalTransitionBatch,
    TerminalTransitionBatch,
    TransitionSource,
)
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayTransitionStats:
    prefix_count: int
    positive_transition_count: int
    prefix_with_positive_rate: float
    mean_positive_edges_per_prefix: float


def build_edge_flow_matching_batches_from_trajectories(
    *,
    trajectories: TrajectoryBatch,
    graph_context: GraphContext,
) -> tuple[NonterminalTransitionBatch | None, TerminalTransitionBatch | None]:
    batch = build_edge_flow_matching_batch(
        policy_trajectories=trajectories,
        replay_transitions=None,
        graph_context=graph_context,
    )
    return batch.nonterminal, batch.terminal


def build_edge_flow_matching_batch(
    *,
    policy_trajectories: TrajectoryBatch,
    replay_transitions: NonterminalTransitionBatch | None,
    graph_context: GraphContext,
) -> EdgeFlowMatchingBatch:
    policy_nonterminal = build_policy_nonterminal_transitions(
        trajectories=policy_trajectories,
        graph_context=graph_context,
    )
    policy_terminal = build_policy_terminal_transitions(
        trajectories=policy_trajectories,
        graph_context=graph_context,
    )
    replay_terminal = build_replay_terminal_transitions(
        nonterminal=replay_transitions,
    )

    return EdgeFlowMatchingBatch(
        nonterminal=concat_nonterminal_transition_batches(
            [policy_nonterminal, replay_transitions]
        ),
        terminal=concat_terminal_transition_batches(
            [policy_terminal, replay_terminal]
        ),
    )


def build_policy_nonterminal_transitions(
    *,
    trajectories: TrajectoryBatch,
    graph_context: GraphContext,
) -> NonterminalTransitionBatch | None:
    if int(trajectories.num_trajectories) == 0:
        return None

    budget = int(trajectories.budget)
    counts = trajectories.edge_count.to(dtype=torch.long)
    total = int(counts.sum().item())
    if total == 0:
        return None
    row_ids = torch.repeat_interleave(
        torch.arange(int(trajectories.num_trajectories), device=trajectories.device, dtype=torch.long),
        counts,
        output_size=total,
    )
    prefix_lens = torch.cat(
        [
            torch.arange(int(count.item()), device=trajectories.device, dtype=torch.long)
            for count in counts
            if int(count.item()) > 0
        ],
        dim=0,
    )
    parent_edge_ids = torch.full(
        (total, budget),
        -1,
        dtype=torch.long,
        device=trajectories.device,
    )
    edge_tensor = trajectories.edge_ids[trajectories.valid_edge_mask()]
    for slot in range(budget):
        mask = prefix_lens.gt(slot)
        if not bool(mask.any()):
            continue
        parent_edge_ids[mask, slot] = trajectories.edge_ids.index_select(0, row_ids[mask])[:, slot]
    parent_state = StateBatch.from_selected_edges(
        graph_ids=trajectories.graph_ids.index_select(0, row_ids),
        edge_ids=parent_edge_ids,
        edge_count=prefix_lens,
        budget=budget,
        graph_context=graph_context,
    )
    source = torch.full(
        (int(edge_tensor.numel()),),
        int(TransitionSource.POLICY),
        dtype=torch.long,
        device=trajectories.device,
    )
    return NonterminalTransitionBatch(
        parent_state=parent_state,
        parent_state_ids=torch.arange(
            int(edge_tensor.numel()),
            dtype=torch.long,
            device=trajectories.device,
        ),
        edge_ids=edge_tensor,
        graph_context=graph_context,
        source=source,
    )


def build_policy_terminal_transitions(
    *,
    trajectories: TrajectoryBatch,
    graph_context: GraphContext,
) -> TerminalTransitionBatch | None:
    if int(trajectories.num_trajectories) == 0:
        return None

    budget = int(trajectories.budget)
    prefix_counts = trajectories.edge_count.to(dtype=torch.long) + 1
    total = int(prefix_counts.sum().item())
    row_ids = torch.repeat_interleave(
        torch.arange(int(trajectories.num_trajectories), device=trajectories.device, dtype=torch.long),
        prefix_counts,
        output_size=total,
    )
    prefix_lens = torch.cat(
        [
            torch.arange(int(count.item()) + 1, device=trajectories.device, dtype=torch.long)
            for count in trajectories.edge_count
        ],
        dim=0,
    )
    state_edge_ids = torch.full(
        (total, budget),
        -1,
        dtype=torch.long,
        device=trajectories.device,
    )
    for slot in range(budget):
        mask = prefix_lens.gt(slot)
        if not bool(mask.any()):
            continue
        state_edge_ids[mask, slot] = trajectories.edge_ids.index_select(0, row_ids[mask])[:, slot]
    state = StateBatch.from_selected_edges(
        graph_ids=trajectories.graph_ids.index_select(0, row_ids),
        edge_ids=state_edge_ids,
        edge_count=prefix_lens,
        budget=budget,
        graph_context=graph_context,
    )
    source = torch.full(
        (int(state.num_states),),
        int(TransitionSource.POLICY),
        dtype=torch.long,
        device=trajectories.device,
    )
    return TerminalTransitionBatch(state=state, source=source)


def build_replay_terminal_transitions(
    *,
    nonterminal: NonterminalTransitionBatch | None,
) -> TerminalTransitionBatch | None:
    if nonterminal is None or int(nonterminal.num_transitions) == 0:
        return None

    device = nonterminal.device
    child_state = nonterminal.materialize_child_state()
    parent_rows = nonterminal.parent_state_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)
    parent_state = nonterminal.parent_state.take(parent_rows)
    states = cat_state_batches([parent_state, child_state])
    state, inverse = deduplicate_state_batch(states=states)
    del inverse
    source = torch.full(
        (int(state.num_states),),
        int(TransitionSource.WEAK_REPLAY),
        dtype=torch.long,
        device=device,
    )
    return TerminalTransitionBatch(state=state, source=source)


def concat_nonterminal_transition_batches(
    batches: list[NonterminalTransitionBatch | None],
) -> NonterminalTransitionBatch | None:
    non_empty = [batch for batch in batches if batch is not None and int(batch.num_transitions) > 0]
    if not non_empty:
        return None

    first = non_empty[0]
    parent_states: list[StateBatch] = []
    parent_state_ids: list[Tensor] = []
    edge_ids: list[Tensor] = []
    child_states: list[StateBatch] = []
    log_backward: list[Tensor] = []
    sources: list[Tensor] = []
    offset = 0

    has_child_state = all(batch.child_state is not None for batch in non_empty)
    has_log_backward = all(batch.log_backward is not None for batch in non_empty)
    has_source = all(batch.source is not None for batch in non_empty)

    for batch in non_empty:
        if batch.parent_state.device != first.parent_state.device:
            raise ValueError("Cannot concatenate NonterminalTransitionBatch on different devices.")
        if int(batch.parent_state.budget) != int(first.parent_state.budget):
            raise ValueError("Cannot concatenate NonterminalTransitionBatch with different budgets.")
        parent_states.append(batch.parent_state)
        parent_state_ids.append(
            batch.parent_state_ids.to(device=first.device, dtype=torch.long) + int(offset)
        )
        edge_ids.append(batch.edge_ids.to(device=first.device, dtype=torch.long))
        if has_child_state:
            child = batch.child_state
            assert child is not None
            child_states.append(child)
        if has_log_backward:
            value = batch.log_backward
            assert value is not None
            log_backward.append(value.to(device=first.device))
        if has_source:
            source = batch.source
            assert source is not None
            sources.append(source.to(device=first.device, dtype=torch.long))
        offset += int(batch.parent_state.num_states)

    return NonterminalTransitionBatch(
        parent_state=cat_state_batches(parent_states),
        parent_state_ids=torch.cat(parent_state_ids, dim=0),
        edge_ids=torch.cat(edge_ids, dim=0),
        child_state=cat_state_batches(child_states) if has_child_state else None,
        graph_context=first.graph_context if hasattr(first, "graph_context") else None,
        log_backward=torch.cat(log_backward, dim=0) if has_log_backward else None,
        source=torch.cat(sources, dim=0) if has_source else None,
    )


def concat_terminal_transition_batches(
    batches: list[TerminalTransitionBatch | None],
) -> TerminalTransitionBatch | None:
    non_empty = [batch for batch in batches if batch is not None and int(batch.num_transitions) > 0]
    if not non_empty:
        return None

    first = non_empty[0]
    has_source = all(batch.source is not None for batch in non_empty)

    return TerminalTransitionBatch(
        state=cat_state_batches([batch.state for batch in non_empty]),
        source=(
            torch.cat(
                [batch.source.to(device=first.device, dtype=torch.long) for batch in non_empty if batch.source is not None],
                dim=0,
            )
            if has_source
            else None
        ),
    )


def deduplicate_state_batch(
    *,
    states: StateBatch,
) -> tuple[StateBatch, Tensor]:
    if int(states.num_states) == 0:
        return states, torch.empty(0, dtype=torch.long, device=states.device)

    key_to_new: dict[tuple[int, ...], int] = {}
    keep_rows: list[int] = []
    inverse_ids: list[int] = []

    for row in range(int(states.num_states)):
        edge_count = int(states.edge_count[row].item())
        key = (
            int(states.graph_ids[row].item()),
            edge_count,
            *[int(v) for v in states.edge_ids[row, :edge_count].tolist()],
        )
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(keep_rows)
            key_to_new[key] = new_id
            keep_rows.append(row)
        inverse_ids.append(new_id)

    keep = torch.tensor(keep_rows, dtype=torch.long, device=states.device)
    inverse = torch.tensor(inverse_ids, dtype=torch.long, device=states.device)
    return states.take(keep), inverse


def transition_source_counts(
    *,
    batch: EdgeFlowMatchingBatch,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    nt_source = batch.nonterminal.source if batch.nonterminal is not None else None
    tt_source = batch.terminal.source if batch.terminal is not None else None
    metrics["efm/nonterminal_policy_count"] = _count_source(nt_source, TransitionSource.POLICY)
    metrics["efm/nonterminal_replay_count"] = _count_source(nt_source, TransitionSource.WEAK_REPLAY)
    metrics["efm/terminal_policy_state_count"] = _count_source(tt_source, TransitionSource.POLICY)
    metrics["efm/terminal_replay_state_count"] = _count_source(tt_source, TransitionSource.WEAK_REPLAY)
    return metrics


def _count_source(source: Tensor | None, which: TransitionSource) -> float:
    if source is None:
        return 0.0
    return float(source.eq(int(which)).sum().item())


__all__ = [
    "ReplayTransitionStats",
    "build_edge_flow_matching_batch",
    "build_edge_flow_matching_batches_from_trajectories",
    "build_policy_nonterminal_transitions",
    "build_policy_terminal_transitions",
    "build_replay_terminal_transitions",
    "concat_nonterminal_transition_batches",
    "concat_terminal_transition_batches",
    "deduplicate_state_batch",
    "transition_source_counts",
]
