from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


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

    def with_edge(self, edge_id: int) -> "SubgraphState":
        edge_id = int(edge_id)
        if self.contains_edge(edge_id):
            raise ValueError(f"SubgraphState already contains edge_id={edge_id}.")
        return SubgraphState(edge_ids=tuple(self.edge_ids) + (edge_id,))

    def without_edge_id(self, edge_id: int) -> "SubgraphState":
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
    ) -> "SubgraphAction":
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
    def stop() -> "SubgraphAction":
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


from .state_kernel import (  # noqa: E402
    analyze_subgraph_rollout_batch,
    analyze_subgraph_state,
    forward_valid_removable_edge_ids,
    initial_subgraph_state,
    initialize_subgraph_rollout_batch,
    is_forward_valid_subgraph_state,
    transition_subgraph_rollout_batch,
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
