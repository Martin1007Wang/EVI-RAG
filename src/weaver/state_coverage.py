from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.rollout.engine import RolloutContext
from src.weaver.rollout.replay import ReplayBatch, dedupe_states
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import State


@dataclass(frozen=True, slots=True)
class StateCoverageBatch:
    state: State

    @property
    def num_states(self) -> int:
        return int(self.state.num_rollouts)


class StateCoverageBuilder:
    """
    Build canonical state-coverage batches.

    Rollout and replay sources provide states only. The loss evaluates every
    exact frontier transition induced by these states.
    """

    def build(
        self,
        *,
        context: RolloutContext,
        rollouts: Sequence[RolloutResult],
        replay: ReplayBatch | None,
    ) -> StateCoverageBatch | None:
        states = [
            policy_visited_states(
                context=context,
                rollouts=rollouts,
            )
        ]
        if replay is not None and replay.num_transitions > 0:
            states.append(replay.state)
        present = [x for x in states if x is not None and x.num_rollouts > 0]
        if not present:
            return None
        return StateCoverageBatch(state=dedupe_states(State.concat(present)))


def policy_visited_states(
    *,
    context: RolloutContext,
    rollouts: Sequence[RolloutResult],
) -> State | None:
    device = context.device
    visited: list[State] = []
    for rollout in rollouts:
        graph_ids = rollout.source_graph_id.to(device=device, dtype=torch.long).view(-1)
        current = State.initial_from_flow_context(
            context.flow_context,
            budget=int(rollout.expand_budget),
            rollouts_per_graph=1,
        ).select_rows(graph_ids)
        for step in range(rollout.max_steps):
            rows = rollout.valid_mask[:, step].to(device=device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
            if rows.numel() > 0:
                visited.append(current.select_rows(rows).clone())
            expand_rows = rollout.expand_mask[:, step].to(device=device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
            if expand_rows.numel() == 0:
                continue
            edge_ids = rollout.selected_edge_ids[:, step].to(device=device, dtype=torch.long).index_select(0, expand_rows)
            current.apply_edges_(
                edge_index=context.flow_context.edge_index,
                rows=expand_rows,
                edge_ids=edge_ids,
            )
    if not visited:
        return None
    return dedupe_states(State.concat(visited))


__all__ = [
    "StateCoverageBatch",
    "StateCoverageBuilder",
    "policy_visited_states",
]
