from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SubgraphState:
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


__all__ = [
    "SubgraphAction",
    "SubgraphAnalysis",
    "SubgraphRolloutBatch",
    "SubgraphState",
]
