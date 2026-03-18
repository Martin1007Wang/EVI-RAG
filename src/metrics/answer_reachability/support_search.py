from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Protocol

import torch

from src.models.configs import SearchEvalConfig, HorizonConfig
from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import compute_constrained_policy_step
from src.models.gflownet import (
    PreparedSearchBatch,
    SearchPolicyProtocol,
    SearchState,
)

from .exact_analysis import ExactReachabilityAnalysis

from .posterior import (
    DiscoveredTrajectory,
    build_answer_posterior,
    build_window_result,
    graph_gold_answers,
    support_targets,
)
from .schema import SupportWindowResult

_MASS_TOLERANCE = 1.0e-6


class _ExactReachabilityAnalyzerProtocol(Protocol):
    def analyze(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ExactReachabilityAnalysis: ...


@dataclass(frozen=True)
class _SearchPrefix:
    log_prob: float
    upper_bound_log_mass: float
    current_node: int
    num_moves: int
    edge_ids: tuple[int, ...]
    start_node: int


class ExactSupportSearch:
    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        eval_cfg: SearchEvalConfig,
        analyzer: _ExactReachabilityAnalyzerProtocol,
    ) -> None:
        self.horizon_cfg = horizon_cfg
        self.eval_cfg = eval_cfg
        self.analyzer = analyzer

    def generate_window(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: ExactReachabilityAnalysis | None = None,
        include_answer_support: bool = True,
    ) -> SupportWindowResult:
        if batch.num_graphs != 1:
            raise ValueError("ExactSupportSearch expects a single-graph batch.")
        analysis = (
            self.analyzer.analyze(
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
            )
            if analysis is None
            else analysis
        )
        gold_answers = graph_gold_answers(batch=batch)
        gold_mask = self._gold_mask(batch=batch)
        frontier, next_tie = self._initialize_frontier(
            policy=policy,
            prepared_batch=prepared_batch,
            analysis=analysis,
        )
        discovered_paths: list[DiscoveredTrajectory] = []
        targets = self._support_targets(analysis=analysis, gold_answers=gold_answers)
        covered_mass = {answer_id: 0.0 for answer_id in targets}
        expansions = 0
        while not self._targets_met(covered_mass=covered_mass, targets=targets):
            if not frontier:
                return self._handle_incomplete_search(
                    batch=batch,
                    analysis=analysis,
                    discovered_paths=discovered_paths,
                    frontier=frontier,
                    expansions=expansions,
                    stop_reason="exact_frontier_exhausted",
                    error_message=(
                        "ExactSupportSearch exhausted candidates before covering the selected answer support mass."
                    ),
                    include_answer_support=include_answer_support,
                )
            _, _, candidate = heapq.heappop(frontier)
            if bool(gold_mask[candidate.current_node].item()):
                answer_id = int(batch.node_global_ids[candidate.current_node].item())
                if answer_id in targets:
                    path = self._to_discovered_path(
                        batch=batch,
                        candidate=candidate,
                        gold_answers=gold_answers,
                    )
                    discovered_paths.append(path)
                    covered_mass[answer_id] += path.prob
                continue
            if candidate.num_moves >= int(self.horizon_cfg.max_steps):
                continue
            if expansions >= int(self.eval_cfg.max_expansions):
                return self._handle_incomplete_search(
                    batch=batch,
                    analysis=analysis,
                    discovered_paths=discovered_paths,
                    frontier=frontier,
                    expansions=expansions,
                    stop_reason="exact_expansions_truncated",
                    error_message=(
                        "ExactSupportSearch exceeded max_expansions before covering the selected answer support mass."
                    ),
                    include_answer_support=include_answer_support,
                )
            expansions += 1
            next_tie = self._expand_prefix(
                frontier=frontier,
                candidate=candidate,
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
                analysis=analysis,
                gold_mask=gold_mask,
                tie=next_tie,
            )
            if len(frontier) > int(self.eval_cfg.max_frontier_size):
                return self._handle_incomplete_search(
                    batch=batch,
                    analysis=analysis,
                    discovered_paths=discovered_paths,
                    frontier=frontier,
                    expansions=expansions,
                    stop_reason="exact_frontier_truncated",
                    error_message=(
                        "ExactSupportSearch exceeded max_frontier_size before covering the selected answer support mass."
                    ),
                    include_answer_support=include_answer_support,
                )
        return build_window_result(
            batch=batch,
            discovered_paths=discovered_paths,
            analysis=analysis,
            inference_mode="exact",
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            probe_count=int(expansions),
            remaining_mass_upper=self._remaining_mass(frontier),
            stop_reason="support_mass_reached",
            coverage_certified=True,
            answer_mass_reference="exact",
            support_mass_reference="exact",
            answer_mass_reference_total=1.0,
            include_answer_support=include_answer_support,
        )

    def _handle_incomplete_search(
        self,
        *,
        batch: TrajectoryBatch,
        analysis: ExactReachabilityAnalysis,
        discovered_paths: list[DiscoveredTrajectory],
        frontier: list[tuple[float, int, _SearchPrefix]],
        expansions: int,
        stop_reason: str,
        error_message: str,
        include_answer_support: bool = True,
    ) -> SupportWindowResult:
        if bool(self.eval_cfg.strict_search):
            raise RuntimeError(error_message)
        return build_window_result(
            batch=batch,
            discovered_paths=discovered_paths,
            analysis=analysis,
            inference_mode="exact",
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            probe_count=int(expansions),
            remaining_mass_upper=self._remaining_mass(frontier),
            stop_reason=str(stop_reason),
            coverage_certified=False,
            answer_mass_reference="exact",
            support_mass_reference="partial_exact",
            answer_mass_reference_total=1.0,
            include_answer_support=include_answer_support,
        )

    def _initialize_frontier(
        self,
        *,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: ExactReachabilityAnalysis,
    ) -> tuple[list[tuple[float, int, _SearchPrefix]], int]:
        frontier: list[tuple[float, int, _SearchPrefix]] = []
        start_dist = policy.compute_start_distribution(prepared_batch)
        for idx in range(int(start_dist.candidate_nodes_abs.numel())):
            start_node = int(start_dist.candidate_nodes_abs[idx].item())
            upper_bound_log_mass = self._upper_bound_log_mass(
                analysis=analysis,
                log_prob=float(start_dist.log_probs[idx].item()),
                current_node=start_node,
                num_moves=0,
            )
            if not math.isfinite(upper_bound_log_mass):
                continue
            candidate = _SearchPrefix(
                log_prob=float(start_dist.log_probs[idx].item()),
                upper_bound_log_mass=upper_bound_log_mass,
                current_node=start_node,
                num_moves=0,
                edge_ids=(),
                start_node=start_node,
            )
            heapq.heappush(frontier, (-candidate.upper_bound_log_mass, idx, candidate))
        return frontier, int(start_dist.candidate_nodes_abs.numel())

    def _support_targets(
        self,
        *,
        analysis: ExactReachabilityAnalysis,
        gold_answers: set[int],
    ) -> dict[int, float]:
        answer_records, selected_answer_ids = build_answer_posterior(
            analysis=analysis,
            gold_answers=gold_answers,
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            total_mass_reference=1.0,
        )
        return support_targets(
            answer_records=answer_records,
            selected_answer_ids=selected_answer_ids,
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
        )

    @staticmethod
    def _targets_met(
        *, covered_mass: dict[int, float], targets: dict[int, float]
    ) -> bool:
        return all(
            covered_mass.get(answer_id, 0.0) + _MASS_TOLERANCE >= target_mass
            for answer_id, target_mass in targets.items()
        )

    def _expand_prefix(
        self,
        *,
        frontier: list[tuple[float, int, _SearchPrefix]],
        candidate: _SearchPrefix,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: ExactReachabilityAnalysis,
        gold_mask: torch.Tensor,
        tie: int,
    ) -> int:
        del gold_mask
        state = self._state_from_candidate(
            batch=batch,
            prepared_batch=prepared_batch,
            candidate=candidate,
            max_steps=int(self.horizon_cfg.max_steps),
        )
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
            state=state,
            max_steps=int(self.horizon_cfg.max_steps),
        )
        if not bool(step.has_values.view(-1)[0].item()):
            return tie
        distribution = step.distribution
        for edge_idx in range(int(step.move_log_probs.numel())):
            child_log_prob = float(step.move_log_probs[edge_idx].item())
            if not math.isfinite(child_log_prob):
                continue
            total_log_prob = candidate.log_prob + child_log_prob
            current_node = int(distribution.target_nodes[edge_idx].item())
            num_moves = candidate.num_moves + 1
            upper_bound_log_mass = self._upper_bound_log_mass(
                analysis=analysis,
                log_prob=total_log_prob,
                current_node=current_node,
                num_moves=num_moves,
            )
            if not math.isfinite(upper_bound_log_mass):
                continue
            prefix = _SearchPrefix(
                log_prob=total_log_prob,
                upper_bound_log_mass=upper_bound_log_mass,
                current_node=current_node,
                num_moves=num_moves,
                edge_ids=candidate.edge_ids
                + (int(distribution.edge_ids[edge_idx].item()),),
                start_node=candidate.start_node,
            )
            heapq.heappush(frontier, (-prefix.upper_bound_log_mass, tie, prefix))
            tie += 1
        return tie

    @staticmethod
    def _upper_bound_log_mass(
        *,
        analysis: ExactReachabilityAnalysis,
        log_prob: float,
        current_node: int,
        num_moves: int,
    ) -> float:
        if analysis.success_by_step is None:
            return float(log_prob)
        if num_moves >= int(analysis.success_by_step.size(0)):
            return float("-inf")
        suffix_success = float(analysis.success_by_step[num_moves, current_node].item())
        if suffix_success <= 0.0:
            return float("-inf")
        return float(log_prob + math.log(suffix_success))

    @staticmethod
    def _state_from_candidate(
        *,
        batch: TrajectoryBatch,
        prepared_batch: PreparedSearchBatch,
        candidate: _SearchPrefix,
        max_steps: int,
    ) -> SearchState:
        return SearchState.from_edge_path(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            start_node=int(candidate.start_node),
            edge_ids=candidate.edge_ids,
            max_steps=int(max_steps),
            device=batch.node_ptr.device,
        )

    @staticmethod
    def _gold_mask(*, batch: TrajectoryBatch) -> torch.Tensor:
        gold_mask = torch.zeros(
            (batch.num_nodes_total,), device=batch.node_ptr.device, dtype=torch.bool
        )
        if int(batch.a_local_indices.numel()) > 0:
            gold_mask.scatter_(0, batch.a_local_indices, True)
        return gold_mask

    @staticmethod
    def _to_discovered_path(
        *,
        batch: TrajectoryBatch,
        candidate: _SearchPrefix,
        gold_answers: set[int],
    ) -> DiscoveredTrajectory:
        answer_id = int(batch.node_global_ids[candidate.current_node].item())
        return DiscoveredTrajectory(
            start_node=int(candidate.start_node),
            terminal_node=int(candidate.current_node),
            answer_entity_id=answer_id,
            edge_ids=tuple(int(edge_id) for edge_id in candidate.edge_ids),
            log_prob=float(candidate.log_prob),
            is_gold=answer_id in gold_answers,
        )

    @staticmethod
    def _remaining_mass(frontier: list[tuple[float, int, _SearchPrefix]]) -> float:
        return float(
            sum(
                math.exp(candidate.upper_bound_log_mass) for _, _, candidate in frontier
            )
        )


__all__ = ["ExactSupportSearch"]
