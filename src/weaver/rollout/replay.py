from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class WeakReplayBatch:
    """
    Prefix states used for weak shortest-path edge supervision.

    Each state should have at least one positive weak replay edge in its current
    legal frontier. Loss code re-checks this and skips empty rows.
    """

    state: StateBatch

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def num_states(self) -> int:
        return int(self.state.num_states)

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device,
        budget: int,
    ) -> WeakReplayBatch:
        return cls(
            state=StateBatch.initial(
                graph_ids=torch.empty(0, dtype=torch.long, device=device),
                budget=int(budget),
            )
        )


@dataclass(frozen=True, slots=True)
class WeakReplaySource:
    """
    Build weak-supervision prefix states from shortest-path edge labels.

    No reward model is consulted here. Expansion follows only weak-label edges
    that are legal in the current frontier.
    """

    budget: int
    states_per_graph: int = 8
    branch_per_state: int = 2

    @torch.no_grad()
    def sample(
        self,
        *,
        graph: GraphContext,
        target: TargetContext,
    ) -> WeakReplayBatch:
        return weak_replay_state_batch(
            graph=graph,
            target=target,
            budget=int(self.budget),
            states_per_graph=int(self.states_per_graph),
            branch_per_state=int(self.branch_per_state),
        )


def weak_replay_state_batch(
    *,
    graph: GraphContext,
    target: TargetContext,
    budget: int,
    states_per_graph: int,
    branch_per_state: int,
) -> WeakReplayBatch:
    budget = int(budget)
    states_per_graph = int(states_per_graph)
    branch_per_state = int(branch_per_state)
    if budget < 0:
        raise ValueError("budget must be nonnegative.")
    if states_per_graph <= 0 or branch_per_state <= 0:
        return WeakReplayBatch.empty(device=graph.device, budget=budget)

    all_states: list[StateBatch] = []
    for graph_id in range(int(graph.num_graphs)):
        if not bool(target.valid_graph_mask[graph_id].item()):
            continue
        states = _weak_replay_states_for_graph(
            graph=graph,
            target=target,
            graph_id=int(graph_id),
            budget=budget,
            states_per_graph=states_per_graph,
            branch_per_state=branch_per_state,
        )
        all_states.extend(states)

    if not all_states:
        return WeakReplayBatch.empty(device=graph.device, budget=budget)

    return WeakReplayBatch(state=cat_state_batches(all_states))


def _weak_replay_states_for_graph(
    *,
    graph: GraphContext,
    target: TargetContext,
    graph_id: int,
    budget: int,
    states_per_graph: int,
    branch_per_state: int,
) -> list[StateBatch]:
    root = StateBatch.initial(
        graph_ids=torch.tensor([int(graph_id)], dtype=torch.long, device=graph.device),
        budget=int(budget),
    )
    beam = [root]
    kept: list[StateBatch] = []
    seen: set[tuple[int, ...]] = {()}

    for _ in range(int(budget)):
        next_beam: list[StateBatch] = []
        for parent in beam:
            action_space = parent.action_space(graph)
            positive = target.shortest_path_edge_mask.index_select(0, action_space.expand_edge_ids)
            if not bool(positive.any()):
                continue
            kept.append(parent)
            if len(kept) >= int(states_per_graph):
                return kept

            edge_rows = positive.nonzero(as_tuple=False).flatten()
            edge_ids = action_space.expand_edge_ids.index_select(0, edge_rows)
            edge_weight = target.shortest_path_edge_weight.index_select(0, edge_ids)
            order = torch.argsort(-edge_weight, stable=True)
            chosen_edges = edge_ids.index_select(0, order[: int(branch_per_state)])
            for edge_id in chosen_edges.tolist():
                edge_id = int(edge_id)
                key = _state_key_after_edge(parent, edge_id)
                if key in seen:
                    continue
                seen.add(key)
                child = parent.advance(
                    ExpansionBatch(
                        state_ids=torch.zeros(1, dtype=torch.long, device=graph.device),
                        edge_ids=torch.tensor([edge_id], dtype=torch.long, device=graph.device),
                    )
                )
                next_beam.append(child)
                if len(kept) + len(next_beam) >= int(states_per_graph):
                    break
            if len(kept) + len(next_beam) >= int(states_per_graph):
                break
        if not next_beam:
            break
        beam = next_beam

    return kept


def _state_key_after_edge(state: StateBatch, edge_id: int) -> tuple[int, ...]:
    selected = state.edge_ids[0, : int(state.edge_count[0].item())].tolist()
    selected.append(int(edge_id))
    return tuple(sorted(int(x) for x in selected))


__all__ = [
    "WeakReplayBatch",
    "WeakReplaySource",
    "weak_replay_state_batch",
]
