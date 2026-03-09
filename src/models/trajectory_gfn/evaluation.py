from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from .analyzer import AnswerMassAnalysis
from .batch import TrajectoryBatch
from .posterior import (
    DiscoveredTrajectory,
    aggregate_rank_metrics,
    build_window_result,
    graph_gold_answers,
)
from .sampler import TrajectorySampleBatch


@dataclass(frozen=True)
class SampledEvaluationOutput:
    window_results: list
    rank_metrics: dict[str, float]


class SampledTrajectoryEvaluator:
    def __init__(
        self,
        *,
        mass_threshold: float,
        answer_top_ks: tuple[int, ...],
    ) -> None:
        self.mass_threshold = float(mass_threshold)
        self.answer_top_ks = tuple(int(k) for k in answer_top_ks)

    def evaluate(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectorySampleBatch,
    ) -> SampledEvaluationOutput:
        results = []
        for graph_idx in range(batch.num_graphs):
            sub_batch = batch.select_graph(graph_idx)
            discovered_paths = _extract_discovered_paths(
                batch=batch,
                sample_batch=sample_batch,
                graph_idx=graph_idx,
            )
            analysis = _analysis_from_discovered_paths(
                batch=sub_batch,
                discovered_paths=discovered_paths,
            )
            results.append(
                build_window_result(
                    batch=sub_batch,
                    discovered_paths=discovered_paths,
                    analysis=analysis,
                    inference_mode="sampled_snapshot",
                    answer_mass_threshold=self.mass_threshold,
                    support_mass_threshold=self.mass_threshold,
                    probe_count=int(sample_batch.start_nodes.size(1)),
                    remaining_mass_upper=max(
                        1.0 - sum(path.prob for path in discovered_paths),
                        0.0,
                    ),
                    stop_reason="snapshot",
                )
            )
        return SampledEvaluationOutput(
            window_results=results,
            rank_metrics=aggregate_rank_metrics(
                results=results,
                answer_top_ks=self.answer_top_ks,
            ),
        )


def _analysis_from_discovered_paths(
    *,
    batch: TrajectoryBatch,
    discovered_paths: list[DiscoveredTrajectory],
) -> AnswerMassAnalysis:
    answer_mass: dict[int, float] = defaultdict(float)
    terminal_mass = torch.zeros((batch.num_nodes_total,), dtype=torch.float32)
    for path in discovered_paths:
        answer_mass[path.answer_entity_id] += path.prob
        terminal_mass[path.terminal_node] += float(path.prob)
    gold_answers = graph_gold_answers(batch=batch)
    answer_ids = sorted(answer_mass)
    answer_probs = [answer_mass[answer_id] for answer_id in answer_ids]
    gold_total_mass = float(
        sum(answer_mass.get(answer_id, 0.0) for answer_id in gold_answers)
    )
    return AnswerMassAnalysis(
        terminal_mass=terminal_mass,
        answer_entity_ids=torch.tensor(answer_ids, dtype=torch.long),
        answer_probs=torch.tensor(answer_probs, dtype=torch.float32),
        gold_total_mass=gold_total_mass,
    )


def _extract_discovered_paths(
    *,
    batch: TrajectoryBatch,
    sample_batch: TrajectorySampleBatch,
    graph_idx: int,
) -> list[DiscoveredTrajectory]:
    gold_answers = graph_gold_answers(batch=batch.select_graph(graph_idx))
    log_probs = _rollout_log_probs(sample_batch=sample_batch, graph_idx=graph_idx)
    discovered: dict[tuple[int, tuple[int, ...], int], DiscoveredTrajectory] = {}
    for rollout_idx in range(int(sample_batch.start_nodes.size(1))):
        start_node = int(sample_batch.start_nodes[graph_idx, rollout_idx].item())
        terminal_node = int(sample_batch.stop_nodes[graph_idx, rollout_idx].item())
        edge_ids = _rollout_edge_ids(
            sample_batch=sample_batch,
            graph_idx=graph_idx,
            rollout_idx=rollout_idx,
        )
        answer_id = int(batch.node_global_ids[terminal_node].item())
        key = (start_node, edge_ids, terminal_node)
        discovered[key] = DiscoveredTrajectory(
            start_node=start_node,
            terminal_node=terminal_node,
            answer_entity_id=answer_id,
            edge_ids=edge_ids,
            log_prob=float(log_probs[rollout_idx].item()),
            is_gold=answer_id in gold_answers,
        )
    return list(discovered.values())


def _rollout_log_probs(
    *, sample_batch: TrajectorySampleBatch, graph_idx: int
) -> torch.Tensor:
    active_mask = sample_batch.active_steps[graph_idx].to(
        dtype=sample_batch.log_pf_steps.dtype
    )
    return sample_batch.start_log_probs[graph_idx] + (
        sample_batch.log_pf_steps[graph_idx] * active_mask
    ).sum(dim=-1)


def _rollout_edge_ids(
    *, sample_batch: TrajectorySampleBatch, graph_idx: int, rollout_idx: int
) -> tuple[int, ...]:
    move_mask = sample_batch.active_steps[graph_idx, rollout_idx] & (
        ~sample_batch.is_stop_steps[graph_idx, rollout_idx]
    )
    move_edge_ids = sample_batch.chosen_edge_ids_steps[graph_idx, rollout_idx][
        move_mask
    ]
    return tuple(
        int(edge_id) for edge_id in move_edge_ids.tolist() if int(edge_id) >= 0
    )


__all__ = ["SampledEvaluationOutput", "SampledTrajectoryEvaluator"]
