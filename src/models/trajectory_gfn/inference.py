from __future__ import annotations

from dataclasses import dataclass

from .analyzer import AnswerMassAnalysis
from .batch import TrajectoryBatch
from .policy import TrajectoryPolicy, TrajectoryPolicyContext
from .posterior import (
    DiscoveredTrajectory,
    aggregate_rank_metrics,
    build_answer_posterior,
    build_rank_only_result_from_discovered_paths,
    build_window_result,
    graph_gold_answers,
    support_targets,
)
from .sampler import ForwardRolloutSampler, TrajectorySampleBatch


@dataclass(frozen=True)
class PosteriorInferenceOutput:
    window_results: list
    rank_metrics: dict[str, float]


class AdaptivePosteriorInference:
    def __init__(
        self,
        *,
        answer_mass_threshold: float,
        support_mass_threshold: float,
        rollout_chunk_size: int,
        max_rollouts: int,
        answer_top_ks: tuple[int, ...],
        support_path_overlap_penalty: float = 0.25,
    ) -> None:
        self.answer_mass_threshold = float(answer_mass_threshold)
        self.support_mass_threshold = float(support_mass_threshold)
        self.support_path_overlap_penalty = float(support_path_overlap_penalty)
        self.rollout_chunk_size = int(rollout_chunk_size)
        self.max_rollouts = int(max_rollouts)
        self.answer_top_ks = tuple(int(k) for k in answer_top_ks)

    def infer_sampled_graph(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
        sampler: ForwardRolloutSampler,
        analysis: AnswerMassAnalysis,
    ):
        if batch.num_graphs != 1:
            raise ValueError("AdaptivePosteriorInference expects a single graph.")
        gold_answers = graph_gold_answers(batch=batch)
        answer_records, selected_answer_ids = build_answer_posterior(
            analysis=analysis,
            gold_answers=gold_answers,
            answer_mass_threshold=self.answer_mass_threshold,
        )
        targets = support_targets(
            answer_records=answer_records,
            selected_answer_ids=selected_answer_ids,
            support_mass_threshold=self.support_mass_threshold,
        )
        discovered: dict[tuple[int, tuple[int, ...], int], DiscoveredTrajectory] = {}
        probe_count = 0
        stop_reason = "max_rollouts"
        while probe_count < self.max_rollouts:
            chunk_size = min(self.rollout_chunk_size, self.max_rollouts - probe_count)
            sample_batch = sampler.sample(
                batch=batch,
                policy=policy,
                context=context,
                num_rollouts=chunk_size,
                is_training=False,
            )
            probe_count += chunk_size
            for path in self._extract_discovered_paths(
                batch=batch,
                sample_batch=sample_batch,
                gold_answers=gold_answers,
            ):
                discovered[(path.start_node, path.edge_ids, path.terminal_node)] = path
            if self._targets_met(
                discovered_paths=list(discovered.values()),
                targets=targets,
            ):
                stop_reason = "support_mass_reached"
                break
        discovered_paths = list(discovered.values())
        discovered_mass = sum(path.prob for path in discovered_paths)
        return build_window_result(
            batch=batch,
            discovered_paths=discovered_paths,
            analysis=analysis,
            inference_mode="sampled",
            answer_mass_threshold=self.answer_mass_threshold,
            support_mass_threshold=self.support_mass_threshold,
            support_path_overlap_penalty=self.support_path_overlap_penalty,
            probe_count=probe_count,
            remaining_mass_upper=max(1.0 - discovered_mass, 0.0),
            stop_reason=stop_reason,
        )

    def infer_sampled_rank_only_graph(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
        sampler: ForwardRolloutSampler,
    ):
        if batch.num_graphs != 1:
            raise ValueError("AdaptivePosteriorInference expects a single graph.")
        gold_answers = graph_gold_answers(batch=batch)
        discovered: dict[tuple[int, tuple[int, ...], int], DiscoveredTrajectory] = {}
        probe_count = 0
        while probe_count < self.max_rollouts:
            chunk_size = min(self.rollout_chunk_size, self.max_rollouts - probe_count)
            sample_batch = sampler.sample(
                batch=batch,
                policy=policy,
                context=context,
                num_rollouts=chunk_size,
                is_training=False,
            )
            probe_count += chunk_size
            for path in self._extract_discovered_paths(
                batch=batch,
                sample_batch=sample_batch,
                gold_answers=gold_answers,
            ):
                discovered[(path.start_node, path.edge_ids, path.terminal_node)] = path
        discovered_paths = list(discovered.values())
        discovered_mass = sum(path.prob for path in discovered_paths)
        return build_rank_only_result_from_discovered_paths(
            batch=batch,
            discovered_paths=discovered_paths,
            inference_mode="sampled_rank_only",
            answer_mass_threshold=self.answer_mass_threshold,
            probe_count=probe_count,
            remaining_mass_upper=max(1.0 - discovered_mass, 0.0),
            stop_reason="rank_only_sampled",
        )

    def aggregate_rank_metrics(self, *, results: list) -> dict[str, float]:
        return aggregate_rank_metrics(
            results=results,
            answer_top_ks=self.answer_top_ks,
        )

    @staticmethod
    def _targets_met(
        *,
        discovered_paths: list[DiscoveredTrajectory],
        targets: dict[int, float],
    ) -> bool:
        if not targets:
            return True
        covered: dict[int, float] = {answer_id: 0.0 for answer_id in targets}
        for path in discovered_paths:
            if path.answer_entity_id in covered:
                covered[path.answer_entity_id] += path.prob
        return all(
            covered[answer_id] + 1.0e-6 >= target
            for answer_id, target in targets.items()
        )

    @staticmethod
    def _extract_discovered_paths(
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectorySampleBatch,
        gold_answers: set[int],
    ) -> list[DiscoveredTrajectory]:
        rollout_log_probs = _rollout_log_probs(sample_batch=sample_batch)
        discovered_paths: list[DiscoveredTrajectory] = []
        num_rollouts = int(sample_batch.start_nodes.size(1))
        for rollout_idx in range(num_rollouts):
            start_node = int(sample_batch.start_nodes[0, rollout_idx].item())
            terminal_node = int(sample_batch.stop_nodes[0, rollout_idx].item())
            answer_id = int(batch.node_global_ids[terminal_node].item())
            discovered_paths.append(
                DiscoveredTrajectory(
                    start_node=start_node,
                    terminal_node=terminal_node,
                    answer_entity_id=answer_id,
                    edge_ids=_rollout_edge_ids(
                        sample_batch=sample_batch,
                        rollout_idx=rollout_idx,
                    ),
                    log_prob=float(rollout_log_probs[rollout_idx].item()),
                    is_gold=answer_id in gold_answers,
                )
            )
        return discovered_paths


def _rollout_log_probs(*, sample_batch: TrajectorySampleBatch):
    active_mask = sample_batch.active_steps[0].to(dtype=sample_batch.log_pf_steps.dtype)
    return sample_batch.start_log_probs[0] + (
        sample_batch.log_pf_steps[0] * active_mask
    ).sum(dim=-1)


def _rollout_edge_ids(
    *, sample_batch: TrajectorySampleBatch, rollout_idx: int
) -> tuple[int, ...]:
    move_mask = sample_batch.active_steps[0, rollout_idx] & (
        ~sample_batch.is_stop_steps[0, rollout_idx]
    )
    move_edge_ids = sample_batch.chosen_edge_ids_steps[0, rollout_idx][move_mask]
    return tuple(
        int(edge_id) for edge_id in move_edge_ids.tolist() if int(edge_id) >= 0
    )
