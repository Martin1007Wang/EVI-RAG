from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import torch

from src.models.configs.trajectory_gfn import HorizonConfig, TrajectoryInferenceConfig

from .analyzer import AnswerMassAnalysis, AnswerMassAnalyzer
from .batch import TrajectoryBatch
from .policy import TrajectoryPolicy, TrajectoryPolicyContext
from .posterior import (
    DiscoveredTrajectory,
    build_answer_posterior,
    build_window_result,
    graph_gold_answers,
    support_targets,
)
from .schema import ElasticWindowResult
from .state import TrajectoryState
from .transition import apply_forward_constraints

_MASS_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class _PrefixCandidate:
    log_prob: float
    current_node: int
    num_moves: int
    edge_ids: tuple[int, ...]
    start_node: int


@dataclass(frozen=True)
class _TerminalCandidate:
    log_prob: float
    terminal_node: int
    edge_ids: tuple[int, ...]
    start_node: int


_SearchCandidate = _PrefixCandidate | _TerminalCandidate


class MassAdaptiveTrajectorySearch:
    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        inference_cfg: TrajectoryInferenceConfig,
        analyzer: AnswerMassAnalyzer,
    ) -> None:
        self.horizon_cfg = horizon_cfg
        self.inference_cfg = inference_cfg
        self.analyzer = analyzer

    def generate_window(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
    ) -> ElasticWindowResult:
        if batch.num_graphs != 1:
            raise ValueError(
                "MassAdaptiveTrajectorySearch expects a single-graph batch."
            )
        analysis = self.analyzer.analyze(batch=batch, policy=policy, context=context)
        gold_answers = graph_gold_answers(batch=batch)
        frontier, next_tie = self._initialize_frontier(policy=policy, context=context)
        discovered_paths: list[DiscoveredTrajectory] = []
        targets = self._support_targets(analysis=analysis, gold_answers=gold_answers)
        covered_mass = {answer_id: 0.0 for answer_id in targets}
        expansions = 0
        while not self._targets_met(covered_mass=covered_mass, targets=targets):
            if not frontier:
                raise RuntimeError(
                    "Exact trajectory search exhausted candidates before covering the selected answer support mass."
                )
            _, _, candidate = heapq.heappop(frontier)
            if isinstance(candidate, _TerminalCandidate):
                answer_id = int(batch.node_global_ids[candidate.terminal_node].item())
                if answer_id in targets:
                    path = self._to_discovered_path(
                        batch=batch,
                        candidate=candidate,
                        gold_answers=gold_answers,
                    )
                    discovered_paths.append(path)
                    covered_mass[answer_id] += path.prob
                continue
            if expansions >= int(self.inference_cfg.max_expansions):
                raise RuntimeError(
                    "Exact trajectory search exceeded max_expansions before covering the selected answer support mass."
                )
            expansions += 1
            next_tie = self._expand_prefix(
                frontier=frontier,
                candidate=candidate,
                batch=batch,
                policy=policy,
                context=context,
                tie=next_tie,
            )
            if len(frontier) > int(self.inference_cfg.max_frontier_size):
                raise RuntimeError(
                    "Exact trajectory search exceeded max_frontier_size before covering the selected answer support mass."
                )
        return build_window_result(
            batch=batch,
            discovered_paths=discovered_paths,
            analysis=analysis,
            inference_mode="exact",
            answer_mass_threshold=float(self.inference_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.inference_cfg.support_mass_threshold),
            probe_count=int(expansions),
            remaining_mass_upper=self._remaining_mass(frontier),
            stop_reason="support_mass_reached",
        )

    def _initialize_frontier(
        self,
        *,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
    ) -> tuple[list[tuple[float, int, _SearchCandidate]], int]:
        frontier: list[tuple[float, int, _SearchCandidate]] = []
        start_dist = policy.compute_start_distribution(context)
        for idx in range(int(start_dist.candidate_nodes_abs.numel())):
            start_node = int(start_dist.candidate_nodes_abs[idx].item())
            candidate = _PrefixCandidate(
                log_prob=float(start_dist.log_probs[idx].item()),
                current_node=start_node,
                num_moves=0,
                edge_ids=(),
                start_node=start_node,
            )
            heapq.heappush(frontier, (-candidate.log_prob, idx, candidate))
        return frontier, int(start_dist.candidate_nodes_abs.numel())

    def _support_targets(
        self,
        *,
        analysis: AnswerMassAnalysis,
        gold_answers: set[int],
    ) -> dict[int, float]:
        answer_records, selected_answer_ids = build_answer_posterior(
            analysis=analysis,
            gold_answers=gold_answers,
            answer_mass_threshold=float(self.inference_cfg.answer_mass_threshold),
        )
        return support_targets(
            answer_records=answer_records,
            selected_answer_ids=selected_answer_ids,
            support_mass_threshold=float(self.inference_cfg.support_mass_threshold),
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
        frontier: list[tuple[float, int, _SearchCandidate]],
        candidate: _PrefixCandidate,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
        tie: int,
    ) -> int:
        state = self._state_from_candidate(
            batch=batch,
            candidate=candidate,
            max_steps=int(self.horizon_cfg.max_steps),
        )
        distribution = policy.compute_forward_distribution(context, state)
        distribution = apply_forward_constraints(
            distribution,
            state=state,
            node_is_target=self._gold_mask(batch=batch),
            min_stop_steps=int(self.horizon_cfg.min_stop_steps),
            max_steps=int(self.horizon_cfg.max_steps),
        )
        if bool(distribution.invalid_rows.view(-1).any().item()):
            raise ValueError(
                "MassAdaptiveTrajectorySearch encountered states with empty forward support before min_stop_steps."
            )
        move_log_probs, stop_log_probs, _ = policy.compute_forward_log_probs(
            distribution
        )
        stop_log_prob = float(stop_log_probs.view(-1)[0].item())
        if math.isfinite(stop_log_prob):
            terminal = _TerminalCandidate(
                log_prob=candidate.log_prob + stop_log_prob,
                terminal_node=candidate.current_node,
                edge_ids=candidate.edge_ids,
                start_node=candidate.start_node,
            )
            heapq.heappush(frontier, (-terminal.log_prob, tie, terminal))
            tie += 1
        for edge_idx in range(int(move_log_probs.numel())):
            child_log_prob = float(move_log_probs[edge_idx].item())
            if not math.isfinite(child_log_prob):
                continue
            prefix = _PrefixCandidate(
                log_prob=candidate.log_prob + child_log_prob,
                current_node=int(distribution.target_nodes[edge_idx].item()),
                num_moves=candidate.num_moves + 1,
                edge_ids=candidate.edge_ids
                + (int(distribution.edge_ids[edge_idx].item()),),
                start_node=candidate.start_node,
            )
            heapq.heappush(frontier, (-prefix.log_prob, tie, prefix))
            tie += 1
        return tie

    @staticmethod
    def _state_from_candidate(
        *,
        batch: TrajectoryBatch,
        candidate: _PrefixCandidate,
        max_steps: int,
    ) -> TrajectoryState:
        return TrajectoryState.from_edge_path(
            start_node=int(candidate.start_node),
            edge_ids=candidate.edge_ids,
            edge_index=batch.edge_index,
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
        candidate: _TerminalCandidate,
        gold_answers: set[int],
    ) -> DiscoveredTrajectory:
        answer_id = int(batch.node_global_ids[candidate.terminal_node].item())
        return DiscoveredTrajectory(
            start_node=int(candidate.start_node),
            terminal_node=int(candidate.terminal_node),
            answer_entity_id=answer_id,
            edge_ids=candidate.edge_ids,
            log_prob=float(candidate.log_prob),
            is_gold=answer_id in gold_answers,
        )

    @staticmethod
    def _remaining_mass(frontier: list[tuple[float, int, _SearchCandidate]]) -> float:
        return float(sum(math.exp(candidate.log_prob) for _, _, candidate in frontier))
