from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from .prepared_batch import SubgraphPreparedBatch


def _bit_count(bits: int) -> int:
    return int(int(bits).bit_count())


def _dedup_preserve_order(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


@dataclass(frozen=True)
class SubgraphState:
    # The semantic state is the current evidence subgraph itself, represented by the
    # selected KG edge ids. Node membership is derived from anchors plus edge endpoints.
    edge_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(int(edge_id) for edge_id in self.edge_ids))
        if len(set(normalized)) != len(normalized):
            raise ValueError("SubgraphState.edge_ids must not contain duplicates.")
        object.__setattr__(self, "edge_ids", normalized)

    @property
    def num_edges(self) -> int:
        return int(len(self.edge_ids))

    def key(self) -> tuple[Any, ...]:
        return tuple(int(edge_id) for edge_id in self.edge_ids)

    def contains_edge(self, edge_id: int) -> bool:
        return int(edge_id) in self.edge_ids

    def with_edge(self, edge_id: int) -> SubgraphState:
        edge_id = int(edge_id)
        if self.contains_edge(edge_id):
            raise ValueError(f"SubgraphState already contains edge_id={edge_id}.")
        return SubgraphState(edge_ids=tuple(self.edge_ids) + (edge_id,))

    def without_edge_id(self, edge_id: int) -> SubgraphState:
        edge_id = int(edge_id)
        if not self.contains_edge(edge_id):
            raise ValueError(
                f"Cannot remove missing edge_id={edge_id} from SubgraphState."
            )
        return SubgraphState(
            edge_ids=tuple(
                int(existing_edge_id)
                for existing_edge_id in self.edge_ids
                if int(existing_edge_id) != edge_id
            )
        )


@dataclass(frozen=True)
class SubgraphAnalysis:
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    component_labels: dict[int, int]
    anchor_component_count: int
    num_selected_edges: int
    num_state_nodes: int = 0
    state_node_graph_ids: tuple[int, ...] = ()
    state_node_entity_ids: tuple[int, ...] = ()
    state_node_reachability_bits: dict[int, int] = field(default_factory=dict)
    state_node_component_labels: dict[int, int] = field(default_factory=dict)
    entity_reachability_bits: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgraphAction:
    kind: str
    edge_id: int | None = None
    source_graph_node: int | None = None
    relation_id: int | None = None
    target_graph_node: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"add_edge", "stop"}:
            raise ValueError("SubgraphAction.kind must be 'add_edge' or 'stop'.")
        if self.kind == "add_edge" and self.edge_id is None:
            raise ValueError("add_edge actions require edge_id.")
        if self.kind == "stop" and any(
            value is not None
            for value in (
                self.edge_id,
                self.source_graph_node,
                self.relation_id,
                self.target_graph_node,
            )
        ):
            raise ValueError("stop actions must not carry edge metadata.")

    @property
    def is_stop(self) -> bool:
        return self.kind == "stop"

    @staticmethod
    def add_edge(
        edge_id: int,
        *,
        source_graph_node: int | None = None,
        relation_id: int | None = None,
        target_graph_node: int | None = None,
    ) -> SubgraphAction:
        return SubgraphAction(
            kind="add_edge",
            edge_id=int(edge_id),
            source_graph_node=(
                None if source_graph_node is None else int(source_graph_node)
            ),
            relation_id=None if relation_id is None else int(relation_id),
            target_graph_node=(
                None if target_graph_node is None else int(target_graph_node)
            ),
        )

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

    def state_key(self, index: int) -> tuple[Any, ...]:
        return self.states[int(index)].key()


def _state_anchor_nodes(
    *, prepared_batch: SubgraphPreparedBatch, graph_idx: int
) -> tuple[int, ...]:
    return tuple(
        int(node_id)
        for node_id in prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)]
    )


def initial_subgraph_state() -> SubgraphState:
    return SubgraphState()


def initialize_subgraph_rollout_batch(
    *, prepared_batch: SubgraphPreparedBatch, num_rollouts: int
) -> SubgraphRolloutBatch:
    if int(num_rollouts) < 1:
        raise ValueError("num_rollouts must be >= 1 for subgraph state initialization.")
    graph_ids = torch.arange(
        prepared_batch.num_graphs,
        device=prepared_batch.device,
        dtype=torch.long,
    ).repeat_interleave(int(num_rollouts))
    states = tuple(
        SubgraphState()
        for _ in range(int(prepared_batch.num_graphs) * int(num_rollouts))
    )
    return SubgraphRolloutBatch(
        graph_ids=graph_ids,
        states=states,
        done_mask=torch.zeros_like(graph_ids, dtype=torch.bool),
        view_shape=(prepared_batch.num_graphs, int(num_rollouts)),
    )


def _selected_node_ids(
    *, prepared_batch: SubgraphPreparedBatch, graph_idx: int, state: SubgraphState
) -> tuple[int, ...]:
    ordered_nodes = list(
        _state_anchor_nodes(prepared_batch=prepared_batch, graph_idx=graph_idx)
    )
    seen = set(int(node_id) for node_id in ordered_nodes)
    for edge_id in state.edge_ids:
        source = int(prepared_batch.topology.edge_index[0, int(edge_id)].item())
        target = int(prepared_batch.topology.edge_index[1, int(edge_id)].item())
        if source not in seen:
            seen.add(source)
            ordered_nodes.append(source)
        if target not in seen:
            seen.add(target)
            ordered_nodes.append(target)
    return tuple(int(node_id) for node_id in ordered_nodes)


def _selected_edges(
    *, prepared_batch: SubgraphPreparedBatch, state: SubgraphState
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(prepared_batch.topology.edge_index[0, int(edge_id)].item()),
            int(prepared_batch.topology.edge_index[1, int(edge_id)].item()),
        )
        for edge_id in state.edge_ids
    )


def _selected_node_entity_ids(
    *, prepared_batch: SubgraphPreparedBatch, selected_node_ids: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        int(prepared_batch.node_entity_ids[int(node_id)].item())
        for node_id in selected_node_ids
    )


def _propagate_reachability_bits(
    *,
    anchor_node_ids: tuple[int, ...],
    selected_node_ids: tuple[int, ...],
    selected_edges: tuple[tuple[int, int], ...],
) -> dict[int, int]:
    bits_by_node = {int(node_id): 0 for node_id in selected_node_ids}
    queue: deque[int] = deque()
    for anchor_idx, anchor_node_id in enumerate(anchor_node_ids):
        anchor_node_id = int(anchor_node_id)
        bits_by_node[anchor_node_id] = 1 << int(anchor_idx)
        queue.append(anchor_node_id)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in selected_edges:
        outgoing[int(source)].append(int(target))
    while queue:
        current = int(queue.popleft())
        current_bits = int(bits_by_node.get(int(current), 0))
        for neighbor in outgoing.get(int(current), ()):
            updated = int(bits_by_node.get(int(neighbor), 0)) | current_bits
            if updated == int(bits_by_node.get(int(neighbor), 0)):
                continue
            bits_by_node[int(neighbor)] = updated
            queue.append(int(neighbor))
    return bits_by_node


def _compute_components(
    *, selected_node_ids: tuple[int, ...], selected_edges: tuple[tuple[int, int], ...]
) -> dict[int, int]:
    undirected: dict[int, list[int]] = defaultdict(list)
    for source, target in selected_edges:
        source = int(source)
        target = int(target)
        undirected[source].append(target)
        undirected[target].append(source)
    labels: dict[int, int] = {}
    next_component = 0
    for node_id in selected_node_ids:
        node_id = int(node_id)
        if node_id in labels:
            continue
        queue: deque[int] = deque([node_id])
        labels[node_id] = int(next_component)
        while queue:
            current = int(queue.popleft())
            for neighbor in undirected.get(int(current), ()):
                neighbor = int(neighbor)
                if neighbor in labels:
                    continue
                labels[neighbor] = int(next_component)
                queue.append(neighbor)
        next_component += 1
    return labels


def analyze_subgraph_state(
    *, prepared_batch: SubgraphPreparedBatch, graph_idx: int, state: SubgraphState
) -> SubgraphAnalysis:
    anchor_node_ids = _state_anchor_nodes(
        prepared_batch=prepared_batch, graph_idx=graph_idx
    )
    selected_node_ids = _selected_node_ids(
        prepared_batch=prepared_batch,
        graph_idx=graph_idx,
        state=state,
    )
    selected_edges = _selected_edges(prepared_batch=prepared_batch, state=state)
    node_entity_ids = _selected_node_entity_ids(
        prepared_batch=prepared_batch,
        selected_node_ids=selected_node_ids,
    )
    reachability_bits = _propagate_reachability_bits(
        anchor_node_ids=anchor_node_ids,
        selected_node_ids=selected_node_ids,
        selected_edges=selected_edges,
    )
    component_labels = _compute_components(
        selected_node_ids=selected_node_ids,
        selected_edges=selected_edges,
    )
    anchor_components = {
        int(component_labels.get(int(anchor_node_id), -1))
        for anchor_node_id in anchor_node_ids
    }
    anchor_components.discard(-1)
    entity_bits: dict[int, int] = {}
    for node_id, entity_id in zip(selected_node_ids, node_entity_ids):
        entity_bits[int(entity_id)] = int(entity_bits.get(int(entity_id), 0)) | int(
            reachability_bits.get(int(node_id), 0)
        )
    return SubgraphAnalysis(
        selected_node_ids=selected_node_ids,
        reachability_bits={
            int(node_id): int(reachability_bits.get(int(node_id), 0))
            for node_id in selected_node_ids
        },
        component_labels={
            int(node_id): int(component_labels.get(int(node_id), -1))
            for node_id in selected_node_ids
        },
        anchor_component_count=int(len(anchor_components)),
        num_selected_edges=int(state.num_edges),
        num_state_nodes=int(len(selected_node_ids)),
        state_node_graph_ids=tuple(int(node_id) for node_id in selected_node_ids),
        state_node_entity_ids=tuple(int(entity_id) for entity_id in node_entity_ids),
        state_node_reachability_bits={
            int(node_id): int(reachability_bits.get(int(node_id), 0))
            for node_id in selected_node_ids
        },
        state_node_component_labels={
            int(node_id): int(component_labels.get(int(node_id), -1))
            for node_id in selected_node_ids
        },
        entity_reachability_bits=dict(entity_bits),
    )


def is_forward_valid_subgraph_state(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis: SubgraphAnalysis | None = None,
) -> bool:
    if analysis is None:
        analysis = analyze_subgraph_state(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
        )
    anchor_node_ids = set(
        int(node_id)
        for node_id in _state_anchor_nodes(
            prepared_batch=prepared_batch, graph_idx=graph_idx
        )
    )
    for node_id in analysis.selected_node_ids:
        node_id = int(node_id)
        if node_id in anchor_node_ids:
            continue
        if int(analysis.reachability_bits.get(int(node_id), 0)) <= 0:
            return False
    return True


def forward_valid_removable_edge_ids(
    *, prepared_batch: SubgraphPreparedBatch, graph_idx: int, state: SubgraphState
) -> tuple[int, ...]:
    if state.num_edges <= 0:
        return ()
    removable_edge_ids: list[int] = []
    for edge_id in state.edge_ids:
        parent_state = state.without_edge_id(int(edge_id))
        parent_analysis = analyze_subgraph_state(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=parent_state,
        )
        if is_forward_valid_subgraph_state(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=parent_state,
            analysis=parent_analysis,
        ):
            removable_edge_ids.append(int(edge_id))
    return tuple(int(edge_id) for edge_id in removable_edge_ids)


def analyze_subgraph_rollout_batch(
    *, prepared_batch: SubgraphPreparedBatch, rollout_batch: SubgraphRolloutBatch
) -> tuple[SubgraphAnalysis, ...]:
    cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis] = {}
    analyses: list[SubgraphAnalysis] = []
    for state_idx in range(rollout_batch.num_states):
        graph_idx = int(rollout_batch.graph_ids[state_idx].item())
        state = rollout_batch.states[state_idx]
        cache_key = (graph_idx, state.key())
        analysis = cache.get(cache_key)
        if analysis is None:
            analysis = analyze_subgraph_state(
                prepared_batch=prepared_batch,
                graph_idx=graph_idx,
                state=state,
            )
            cache[cache_key] = analysis
        analyses.append(analysis)
    return tuple(analyses)


def transition_subgraph_rollout_batch(
    *, rollout_batch: SubgraphRolloutBatch, chosen_actions: tuple[SubgraphAction, ...]
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
            next_done_mask[state_idx] = True
            next_states.append(state)
            continue
        if action.edge_id is None:
            raise RuntimeError("Expand actions must carry edge_id.")
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
    "forward_valid_removable_edge_ids",
    "initial_subgraph_state",
    "initialize_subgraph_rollout_batch",
    "is_forward_valid_subgraph_state",
    "transition_subgraph_rollout_batch",
]
