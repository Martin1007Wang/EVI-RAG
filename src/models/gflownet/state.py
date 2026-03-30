from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .prepared_batch import SubgraphPreparedBatch


@dataclass(frozen=True)
class SubgraphState:
    # The dynamic trace keeps only selected edge ids. The policy reconstructs the
    # full semantic subgraph state on demand from this trace plus PreparedBatch.
    edge_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(int(edge_id) for edge_id in self.edge_ids)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("SubgraphState.edge_ids must be sorted and unique.")

    @property
    def num_edges(self) -> int:
        return int(len(self.edge_ids))

    def key(self) -> tuple[int, ...]:
        return self.edge_ids

    def contains_edge(self, edge_id: int) -> bool:
        return int(edge_id) in self.edge_ids

    def with_edge(self, edge_id: int) -> SubgraphState:
        return SubgraphState(edge_ids=tuple(sorted({*self.edge_ids, int(edge_id)})))


@dataclass(frozen=True)
class SubgraphAnalysis:
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    component_labels: dict[int, int]
    anchor_component_count: int
    num_selected_edges: int


@dataclass(frozen=True)
class SubgraphAction:
    kind: str
    edge_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"add_edge", "stop"}:
            raise ValueError("SubgraphAction.kind must be 'add_edge' or 'stop'.")
        if self.kind == "add_edge" and self.edge_id is None:
            raise ValueError("add_edge actions require an edge_id.")
        if self.kind == "stop" and self.edge_id is not None:
            raise ValueError("stop actions must not carry an edge_id.")

    @property
    def is_stop(self) -> bool:
        return self.kind == "stop"

    @staticmethod
    def add_edge(edge_id: int) -> SubgraphAction:
        return SubgraphAction(kind="add_edge", edge_id=int(edge_id))

    @staticmethod
    def stop() -> SubgraphAction:
        return SubgraphAction(kind="stop")


@dataclass(frozen=True)
class SubgraphRolloutBatch:
    graph_ids: torch.Tensor
    states: tuple[SubgraphState, ...]
    done_mask: torch.Tensor
    view_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.states) != int(self.graph_ids.numel()):
            raise ValueError("states must align with graph_ids.")
        if tuple(self.done_mask.shape) != tuple(self.graph_ids.shape):
            raise ValueError("done_mask must align with graph_ids.")

    @property
    def num_states(self) -> int:
        return int(self.graph_ids.numel())

    def active_state_indices(self) -> list[int]:
        active = torch.nonzero(~self.done_mask, as_tuple=False).view(-1)
        return [int(idx) for idx in active.detach().cpu().tolist()]

    def state_key(self, index: int) -> tuple[int, ...]:
        return self.states[int(index)].key()


def _sorted_edge_records(
    *,
    topology: Any,
    edge_ids: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    records: list[tuple[int, int, int, int]] = []
    for edge_id in edge_ids:
        edge_idx = int(edge_id)
        records.append(
            (
                edge_idx,
                int(topology.edge_index[0, edge_idx].item()),
                int(topology.edge_type[edge_idx].item()),
                int(topology.edge_index[1, edge_idx].item()),
            )
        )
    records.sort(key=lambda item: item[0])
    return tuple(records)


def initial_subgraph_state() -> SubgraphState:
    return SubgraphState()


def initialize_subgraph_rollout_batch(
    *,
    prepared_batch: SubgraphPreparedBatch,
    num_rollouts: int,
) -> SubgraphRolloutBatch:
    if int(num_rollouts) < 1:
        raise ValueError("num_rollouts must be >= 1 for subgraph state initialization.")
    graph_ids = torch.arange(
        prepared_batch.num_graphs,
        device=prepared_batch.device,
        dtype=torch.long,
    ).repeat_interleave(int(num_rollouts))
    states = tuple(
        initial_subgraph_state()
        for _ in range(prepared_batch.num_graphs * int(num_rollouts))
    )
    return SubgraphRolloutBatch(
        graph_ids=graph_ids,
        states=states,
        done_mask=torch.zeros_like(graph_ids, dtype=torch.bool),
        view_shape=(prepared_batch.num_graphs, int(num_rollouts)),
    )


def analyze_subgraph_state(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
) -> SubgraphAnalysis:
    # Theory talks about a full subgraph state. Implementation factorizes it into
    # static context plus this lightweight edge trace, then rebuilds node-level
    # semantics here before scoring actions or rewards.
    anchors = prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)]
    selected_nodes = set(int(anchor) for anchor in anchors)
    directed_adj: dict[int, list[int]] = {}
    undirected_adj: dict[int, list[int]] = {}
    edge_records = _sorted_edge_records(
        topology=prepared_batch.topology,
        edge_ids=state.edge_ids,
    )
    for _, src, _, dst in edge_records:
        selected_nodes.add(int(src))
        selected_nodes.add(int(dst))
        directed_adj.setdefault(int(src), []).append(int(dst))
        undirected_adj.setdefault(int(src), []).append(int(dst))
        undirected_adj.setdefault(int(dst), []).append(int(src))
    ordered_nodes = tuple(sorted(int(node_id) for node_id in selected_nodes))
    reachability_bits = {int(node_id): 0 for node_id in ordered_nodes}
    queue: deque[int] = deque()
    for bit_idx, anchor in enumerate(anchors):
        anchor_node = int(anchor)
        anchor_bits = reachability_bits.get(anchor_node, 0) | (1 << int(bit_idx))
        reachability_bits[anchor_node] = anchor_bits
        queue.append(anchor_node)
    while queue:
        current = int(queue.popleft())
        current_bits = int(reachability_bits.get(current, 0))
        for neighbor in directed_adj.get(current, []):
            updated_bits = int(reachability_bits.get(neighbor, 0)) | current_bits
            if updated_bits == int(reachability_bits.get(neighbor, 0)):
                continue
            reachability_bits[int(neighbor)] = updated_bits
            queue.append(int(neighbor))
    component_labels: dict[int, int] = {}
    next_component = 0
    for node_id in ordered_nodes:
        node = int(node_id)
        if node in component_labels:
            continue
        stack = [node]
        component_labels[node] = int(next_component)
        while stack:
            current = int(stack.pop())
            for neighbor in undirected_adj.get(current, []):
                neighbor = int(neighbor)
                if neighbor in component_labels:
                    continue
                component_labels[neighbor] = int(next_component)
                stack.append(neighbor)
        next_component += 1
    anchor_components = {
        int(component_labels[int(anchor)])
        for anchor in anchors
        if int(anchor) in component_labels
    }
    return SubgraphAnalysis(
        selected_node_ids=ordered_nodes,
        reachability_bits=reachability_bits,
        component_labels=component_labels,
        anchor_component_count=int(len(anchor_components)),
        num_selected_edges=int(state.num_edges),
    )


def analyze_subgraph_rollout_batch(
    *,
    prepared_batch: SubgraphPreparedBatch,
    rollout_batch: SubgraphRolloutBatch,
) -> tuple[SubgraphAnalysis, ...]:
    return tuple(
        analyze_subgraph_state(
            prepared_batch=prepared_batch,
            graph_idx=int(rollout_batch.graph_ids[state_idx].item()),
            state=rollout_batch.states[state_idx],
        )
        for state_idx in range(rollout_batch.num_states)
    )


def transition_subgraph_rollout_batch(
    *,
    rollout_batch: SubgraphRolloutBatch,
    chosen_actions: tuple[SubgraphAction, ...],
) -> SubgraphRolloutBatch:
    if len(chosen_actions) != rollout_batch.num_states:
        raise ValueError("chosen_actions must align with rollout states.")
    next_states: list[SubgraphState] = []
    next_done_mask = rollout_batch.done_mask.clone()
    for state_idx, action in enumerate(chosen_actions):
        state = rollout_batch.states[state_idx]
        if bool(next_done_mask[state_idx].item()):
            next_states.append(state)
            continue
        if action.is_stop:
            # We do not materialize a literal bottom node; done_mask is the batched
            # absorbing-state marker that enforces the same stop semantics.
            next_done_mask[state_idx] = True
            next_states.append(state)
            continue
        if action.edge_id is None:
            raise RuntimeError("Expand actions must carry an edge_id.")
        next_states.append(state.with_edge(int(action.edge_id)))
    return SubgraphRolloutBatch(
        graph_ids=rollout_batch.graph_ids,
        states=tuple(next_states),
        done_mask=next_done_mask,
        view_shape=rollout_batch.view_shape,
    )


__all__ = [
    "SubgraphAction",
    "SubgraphAnalysis",
    "SubgraphRolloutBatch",
    "SubgraphState",
    "analyze_subgraph_rollout_batch",
    "analyze_subgraph_state",
    "initial_subgraph_state",
    "initialize_subgraph_rollout_batch",
    "transition_subgraph_rollout_batch",
]
