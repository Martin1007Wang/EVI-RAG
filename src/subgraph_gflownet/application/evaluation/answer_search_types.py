from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _SubgraphPredictionCodec:
    kind = "subgraph_answer_search"

    @staticmethod
    def serialize_result(result: Any) -> dict[str, Any]:
        return dict(result)

    @staticmethod
    def serialize_label(label: Any) -> dict[str, Any]:
        return dict(label)

    @staticmethod
    def deserialize_result(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    @staticmethod
    def deserialize_label(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)


@dataclass
class _TerminalSampleAggregate:
    edge_ids: tuple[int, ...]
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    terminal_answer_set_entity_ids: tuple[int, ...]
    sample_count: int = 0
    score_sum: float = 0.0


@dataclass
class _GraphPredictionAccumulator:
    original_graph_idx: int
    sample_id: str
    question: str
    gold_answer_entity_ids: list[int]
    gold_answer_in_graph: bool
    candidate_answer_upper_bound: int
    answer_vote_counts: dict[int, float] = field(default_factory=dict)
    terminal_witnesses: dict[tuple[int, ...], _TerminalSampleAggregate] = field(
        default_factory=dict
    )
    nonempty_terminal_answer_set_rollout_count: int = 0
    gold_answer_in_state_rollout_count: int = 0
    total_stop_steps: float = 0.0
    total_terminal_component_count: float = 0.0
    rollout_count: int = 0
    early_stop_margin: float | None = None
    stopped_early: bool = False


@dataclass
class _PredictMetricsAccumulator:
    count: int = 0
    primary_sums: dict[str, float] = field(default_factory=dict)
    secondary_sums: dict[str, float] = field(default_factory=dict)


__all__ = [
    "_GraphPredictionAccumulator",
    "_PredictMetricsAccumulator",
    "_SubgraphPredictionCodec",
    "_TerminalSampleAggregate",
]
