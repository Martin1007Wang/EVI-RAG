from __future__ import annotations

from .semantic_oracles import (
    AdmissibleAnswerSet,
    RewardWeights,
    TerminalRewardSummary,
    compute_terminal_reward,
    gold_answer_entities_in_graph,
    oracle_distance,
    resolve_admissible_answer_set,
)
from .state import SubgraphAnalysis
from .subgraph_batch import SubgraphBatch


class SubgraphRewardModel:
    def __init__(
        self,
        *,
        hit_bonus: float = 5.0,
        frontier_bonus: float = 1.0,
        coverage_bonus: float = 0.2,
        size_penalty: float = 0.1,
        component_penalty: float = 0.5,
    ) -> None:
        self.weights = RewardWeights(
            hit_bonus=float(hit_bonus),
            frontier_bonus=float(frontier_bonus),
            coverage_bonus=float(coverage_bonus),
            size_penalty=float(size_penalty),
            component_penalty=float(component_penalty),
        )
        if self.weights.hit_bonus < 0.0:
            raise ValueError("training.answer_reward.hit_bonus must be >= 0.")
        if self.weights.frontier_bonus < 0.0:
            raise ValueError("training.answer_reward.frontier_bonus must be >= 0.")
        if self.weights.coverage_bonus < 0.0:
            raise ValueError("training.answer_reward.coverage_bonus must be >= 0.")
        if self.weights.size_penalty < 0.0:
            raise ValueError("training.answer_reward.size_penalty must be >= 0.")
        if self.weights.component_penalty < 0.0:
            raise ValueError("training.answer_reward.component_penalty must be >= 0.")

    def admissible_answer_set(
        self,
        *,
        batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> AdmissibleAnswerSet:
        return resolve_admissible_answer_set(
            batch=batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )

    def count_gold_answers_in_graph(
        self,
        *,
        batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        gold_count = int(
            len(
                gold_answer_entities_in_graph(
                    batch=batch,
                    graph_idx=graph_idx,
                    analysis=analysis,
                )
            )
        )
        return gold_count, bool(gold_count > 0)

    def compute_terminal_reward(
        self,
        *,
        batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> TerminalRewardSummary:
        return compute_terminal_reward(
            batch=batch,
            graph_idx=graph_idx,
            analysis=analysis,
            weights=self.weights,
        )

    def oracle_distance(
        self,
        *,
        batch: SubgraphBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> int:
        return oracle_distance(batch=batch, graph_idx=graph_idx, analysis=analysis)


SubgraphTerminalReward = TerminalRewardSummary


__all__ = [
    "AdmissibleAnswerSet",
    "SubgraphRewardModel",
    "SubgraphTerminalReward",
    "resolve_admissible_answer_set",
]
