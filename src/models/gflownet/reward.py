from __future__ import annotations

import math

from .prepared_batch import UNREACHABLE_DISTANCE, SubgraphPreparedBatch
from .state import SubgraphAnalysis


def _bit_count(value: int) -> int:
    return int(int(value).bit_count())


def _gold_answer_coverage_stats(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> tuple[int, int]:
    gold_answers = prepared_batch.graph_answer_entities[int(graph_idx)]
    if not gold_answers:
        return 0, 0
    full_mask = int(prepared_batch.graph_anchor_full_mask[int(graph_idx)])
    full_mask_bits = _bit_count(full_mask)
    if full_mask_bits <= 0:
        return 0, 0
    best_bits_by_entity: dict[int, int] = {}
    for node_id in analysis.selected_node_ids:
        entity_id = int(prepared_batch.node_entity_ids[int(node_id)].item())
        if entity_id not in gold_answers:
            continue
        covered_bits = _bit_count(int(analysis.reachability_bits.get(int(node_id), 0)))
        previous_best = int(best_bits_by_entity.get(entity_id, 0))
        if covered_bits > previous_best:
            best_bits_by_entity[entity_id] = covered_bits
    total_bit_coverage = int(sum(best_bits_by_entity.values()))
    full_answer_count = int(
        sum(
            1
            for covered_bits in best_bits_by_entity.values()
            if covered_bits >= full_mask_bits
        )
    )
    return total_bit_coverage, full_answer_count


def resolve_subgraph_answer_entities(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> tuple[int, ...]:
    full_mask = int(prepared_batch.graph_anchor_full_mask[int(graph_idx)])
    if full_mask <= 0:
        return ()
    answer_entities: set[int] = set()
    for node_id in analysis.selected_node_ids:
        node_bits = int(analysis.reachability_bits.get(int(node_id), 0))
        if node_bits != full_mask:
            continue
        entity_id = int(prepared_batch.node_entity_ids[int(node_id)].item())
        answer_entities.add(entity_id)
    return tuple(sorted(answer_entities))


class SubgraphRewardModel:
    def __init__(
        self,
        *,
        c_step: float = 0.1,
        lambda_conn: float = 0.5,
        beta_answer_bits: float = 0.0,
        beta_answer_full: float = 0.0,
        beta_hit: float = 2.0,
        beta_cnt: float = 0.25,
        beta_early: float = 1.0,
        min_stop_edges: int = 1,
        max_steps: int | None = None,
    ) -> None:
        self.c_step = float(c_step)
        self.lambda_conn = float(lambda_conn)
        self.beta_answer_bits = float(beta_answer_bits)
        self.beta_answer_full = float(beta_answer_full)
        self.beta_hit = float(beta_hit)
        self.beta_cnt = float(beta_cnt)
        self.beta_early = float(beta_early)
        self.min_stop_edges = int(min_stop_edges)
        self.max_steps = None if max_steps is None else int(max_steps)
        if self.c_step < 0.0:
            raise ValueError("training.subgraph_reward.c_step must be >= 0.")
        if self.lambda_conn < 0.0:
            raise ValueError("training.subgraph_reward.lambda_conn must be >= 0.")
        if self.beta_answer_bits < 0.0:
            raise ValueError("training.subgraph_reward.beta_answer_bits must be >= 0.")
        if self.beta_answer_full < 0.0:
            raise ValueError("training.subgraph_reward.beta_answer_full must be >= 0.")
        if self.beta_hit < 0.0:
            raise ValueError("training.subgraph_reward.beta_hit must be >= 0.")
        if self.beta_cnt < 0.0:
            raise ValueError("training.subgraph_reward.beta_cnt must be >= 0.")
        if self.beta_early < 0.0:
            raise ValueError("training.subgraph_reward.beta_early must be >= 0.")
        if self.min_stop_edges < 0:
            raise ValueError("training.subgraph_reward.min_stop_edges must be >= 0.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                "training.subgraph_reward.max_steps must be >= 1 when set."
            )

    def count_gold_answers(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        gold_answers = prepared_batch.graph_answer_entities[int(graph_idx)]
        answer_entities = {
            int(entity_id)
            for entity_id in resolve_subgraph_answer_entities(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                analysis=analysis,
            )
            if int(entity_id) in gold_answers
        }
        answer_count = int(len(answer_entities))
        return answer_count, bool(answer_count > 0)

    def compute_expand_log_reward(
        self,
        *,
        current_analysis: SubgraphAnalysis,
        next_analysis: SubgraphAnalysis,
        prepared_batch: SubgraphPreparedBatch | None = None,
        graph_idx: int | None = None,
    ) -> float:
        reward = -float(self.c_step) + float(self.lambda_conn) * float(
            max(
                int(current_analysis.anchor_component_count)
                - int(next_analysis.anchor_component_count),
                0,
            )
        )
        if prepared_batch is not None and graph_idx is not None:
            current_answer_bits, current_full_answers = _gold_answer_coverage_stats(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                analysis=current_analysis,
            )
            next_answer_bits, next_full_answers = _gold_answer_coverage_stats(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                analysis=next_analysis,
            )
            reward += float(self.beta_answer_bits) * float(
                max(int(next_answer_bits) - int(current_answer_bits), 0)
            )
            reward += float(self.beta_answer_full) * float(
                max(int(next_full_answers) - int(current_full_answers), 0)
            )
        return float(reward)

    def compute_stop_log_reward(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[float, int, bool]:
        answer_count, hit = self.count_gold_answers(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            analysis=analysis,
        )
        teacher_edge_count = prepared_batch.graph_teacher_edge_count[int(graph_idx)]
        feasible_within_horizon = True
        if self.max_steps is not None:
            feasible_within_horizon = teacher_edge_count is not None and int(
                teacher_edge_count
            ) <= int(self.max_steps)
        premature = feasible_within_horizon and (
            int(analysis.num_selected_edges) < int(self.min_stop_edges)
            or int(analysis.anchor_component_count)
            == len(prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)])
        )
        reward = 0.0
        if hit:
            reward += float(self.beta_hit)
            reward += float(self.beta_cnt) * math.log1p(float(answer_count))
        elif premature:
            reward -= float(self.beta_early)
        return float(reward), answer_count, bool(hit)

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


__all__ = ["SubgraphRewardModel", "resolve_subgraph_answer_entities"]
