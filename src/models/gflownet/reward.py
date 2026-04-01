from __future__ import annotations

from dataclasses import dataclass

from .prepared_batch import UNREACHABLE_DISTANCE, SubgraphPreparedBatch
from .state import SubgraphAnalysis


@dataclass(frozen=True)
class AdmissibleAnswerSet:
    entities: tuple[int, ...]
    gold_entities: tuple[int, ...]
    full_anchor_mask: int

    @property
    def count(self) -> int:
        return int(len(self.entities))

    @property
    def gold_count(self) -> int:
        return int(len(self.gold_entities))

    def contains(self, entity_id: int) -> bool:
        return int(entity_id) in self.entities

    def is_gold(self, entity_id: int) -> bool:
        return int(entity_id) in self.gold_entities


def resolve_admissible_answer_set(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> AdmissibleAnswerSet:
    # This is the structurally admissible answer set induced by multi-anchor
    # reachability, not a guarantee of semantic correctness.
    full_mask = int(prepared_batch.graph_anchor_full_mask[int(graph_idx)])
    if full_mask <= 0:
        return AdmissibleAnswerSet(
            entities=(),
            gold_entities=(),
            full_anchor_mask=int(full_mask),
        )
    admissible_entities = tuple(
        sorted(
            int(entity_id)
            for entity_id, bits in analysis.entity_reachability_bits.items()
            if int(bits) == int(full_mask)
        )
    )
    gold_answers = prepared_batch.graph_answer_entities[int(graph_idx)]
    gold_entities = tuple(
        int(entity_id)
        for entity_id in admissible_entities
        if int(entity_id) in gold_answers
    )
    return AdmissibleAnswerSet(
        entities=admissible_entities,
        gold_entities=gold_entities,
        full_anchor_mask=int(full_mask),
    )


def resolve_admissible_answer_commit_set(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> AdmissibleAnswerSet:
    return resolve_admissible_answer_set(
        prepared_batch=prepared_batch,
        graph_idx=graph_idx,
        analysis=analysis,
    )


def _redundancy_edge_count(analysis: SubgraphAnalysis) -> int:
    selected_nodes = int(max(analysis.num_state_nodes, len(analysis.selected_node_ids)))
    minimal_forest_edges = max(selected_nodes - int(analysis.anchor_component_count), 0)
    return max(int(analysis.num_selected_edges) - int(minimal_forest_edges), 0)


@dataclass(frozen=True)
class SubgraphTerminalReward:
    log_reward: float
    hit: bool
    answer_set: AdmissibleAnswerSet
    redundancy_edges: int

    @property
    def answer_entities(self) -> tuple[int, ...]:
        return tuple(int(entity_id) for entity_id in self.answer_set.entities)

    @property
    def chosen_answer_entity_id(self) -> int | None:
        if len(self.answer_set.entities) != 1:
            return None
        return int(self.answer_set.entities[0])


class SubgraphRewardModel:
    def __init__(
        self,
        *,
        gold_answer_bonus: float = 2.0,
        wrong_answer_penalty: float = 2.0,
        failure_penalty: float = 4.0,
        size_penalty: float = 0.1,
        redundancy_penalty: float = 0.25,
        component_penalty: float = 0.5,
    ) -> None:
        self.gold_answer_bonus = float(gold_answer_bonus)
        self.wrong_answer_penalty = float(wrong_answer_penalty)
        self.failure_penalty = float(failure_penalty)
        self.size_penalty = float(size_penalty)
        self.redundancy_penalty = float(redundancy_penalty)
        self.component_penalty = float(component_penalty)
        if self.gold_answer_bonus < 0.0:
            raise ValueError("training.answer_reward.gold_answer_bonus must be >= 0.")
        if self.wrong_answer_penalty < 0.0:
            raise ValueError(
                "training.answer_reward.wrong_answer_penalty must be >= 0."
            )
        if self.failure_penalty < 0.0:
            raise ValueError("training.answer_reward.failure_penalty must be >= 0.")
        if self.size_penalty < 0.0:
            raise ValueError("training.answer_reward.size_penalty must be >= 0.")
        if self.redundancy_penalty < 0.0:
            raise ValueError("training.answer_reward.redundancy_penalty must be >= 0.")
        if self.component_penalty < 0.0:
            raise ValueError("training.answer_reward.component_penalty must be >= 0.")

    def admissible_answer_set(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> AdmissibleAnswerSet:
        return resolve_admissible_answer_set(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def admissible_answer_commit_set(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> AdmissibleAnswerSet:
        return self.admissible_answer_set(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def count_gold_admissible_answers(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        answer_set = self.admissible_answer_set(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )
        gold_count = int(answer_set.gold_count)
        return gold_count, bool(gold_count > 0)

    def compute_terminal_log_reward(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
        answer_entity_id: int | None = None,
    ) -> SubgraphTerminalReward:
        del answer_entity_id
        answer_set = self.admissible_answer_set(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            analysis=analysis,
        )
        redundancy_edges = _redundancy_edge_count(analysis)
        structure_penalty = (
            float(self.size_penalty) * float(analysis.num_selected_edges)
            + float(self.redundancy_penalty) * float(redundancy_edges)
            + float(self.component_penalty)
            * float(max(int(analysis.anchor_component_count) - 1, 0))
        )
        if int(answer_set.gold_count) > 0:
            return SubgraphTerminalReward(
                log_reward=float(self.gold_answer_bonus - structure_penalty),
                hit=True,
                answer_set=answer_set,
                redundancy_edges=int(redundancy_edges),
            )
        if int(answer_set.count) > 0:
            return SubgraphTerminalReward(
                log_reward=float(-self.wrong_answer_penalty - structure_penalty),
                hit=False,
                answer_set=answer_set,
                redundancy_edges=int(redundancy_edges),
            )
        return SubgraphTerminalReward(
            log_reward=float(-self.failure_penalty - structure_penalty),
            hit=False,
            answer_set=answer_set,
            redundancy_edges=int(redundancy_edges),
        )

    def oracle_distance(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> int:
        oracle_distance_map = prepared_batch.graph_oracle_answer_distance[
            int(graph_idx)
        ]
        return min(
            (
                int(oracle_distance_map[node_id])
                for node_id in analysis.selected_node_ids
                if int(node_id) in oracle_distance_map
            ),
            default=UNREACHABLE_DISTANCE,
        )


__all__ = [
    "AdmissibleAnswerCommitSet",
    "AdmissibleAnswerSet",
    "SubgraphRewardModel",
    "SubgraphTerminalReward",
    "resolve_admissible_answer_set",
    "resolve_admissible_answer_commit_set",
]


AdmissibleAnswerCommitSet = AdmissibleAnswerSet
