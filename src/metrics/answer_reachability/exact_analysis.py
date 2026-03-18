from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import compute_constrained_policy_step
from src.models.gflownet import (
    PreparedSearchBatch,
    SearchPolicyProtocol,
    SearchState,
)
from src.utils.segment_ops import segment_logsumexp_1d


_LOG_ZERO = float("-inf")


@dataclass(frozen=True)
class ExactReachabilityAnalysis:
    terminal_mass: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    gold_total_mass: float
    retrieval_answer_entity_ids: torch.Tensor | None = None
    retrieval_answer_probs: torch.Tensor | None = None
    success_by_step: torch.Tensor | None = None
    log_terminal_mass: torch.Tensor | None = None
    log_answer_probs: torch.Tensor | None = None
    log_gold_total_mass: float | None = None
    log_retrieval_answer_probs: torch.Tensor | None = None
    log_success_by_step: torch.Tensor | None = None


@dataclass(frozen=True)
class ExactEdgeSupportAnalysis:
    edge_success_mass: torch.Tensor
    edge_conditional_success_prob: torch.Tensor
    gold_mass: float


@dataclass(frozen=True)
class _StepTransitionBatch:
    edge_agent_batch: torch.Tensor
    edge_ids: torch.Tensor
    target_nodes: torch.Tensor
    edge_probs: torch.Tensor
    has_values: torch.Tensor


@dataclass(frozen=True)
class ExactDynamicProgramResult:
    log_terminal_mass: torch.Tensor
    log_retrieval_terminal_mass: torch.Tensor
    log_success_by_step: torch.Tensor
    log_gold_mass: torch.Tensor
    log_gold_mass_by_graph: torch.Tensor
    log_edge_success_mass: torch.Tensor

    @property
    def terminal_mass(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_terminal_mass)

    @property
    def retrieval_terminal_mass(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_retrieval_terminal_mass)

    @property
    def success_by_step(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_success_by_step)

    @property
    def gold_mass(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_gold_mass)

    @property
    def gold_mass_by_graph(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_gold_mass_by_graph)

    @property
    def edge_success_mass(self) -> torch.Tensor:
        return _log_mass_to_mass(self.log_edge_success_mass)


def _log_mass_to_mass(log_mass: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.isfinite(log_mass), log_mass.exp(), torch.zeros_like(log_mass)
    )


def _log_scalar_to_float(log_value: torch.Tensor) -> float:
    scalar = float(log_value.detach().item())
    if not math.isfinite(scalar):
        return 0.0
    return float(math.exp(scalar))


def _probabilities_to_log_space(probabilities: torch.Tensor) -> torch.Tensor:
    if bool((probabilities < 0).any().item()):
        raise ValueError(
            "ExactReachabilityAnalyzer received negative transition probabilities."
        )
    return torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.full_like(probabilities, fill_value=_LOG_ZERO),
    )


def _segment_logsumexp(
    *, values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    aggregated, _ = segment_logsumexp_1d(
        values=values,
        segment_ids=segment_ids,
        num_segments=num_segments,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=_LOG_ZERO,
    )
    return aggregated


def _expand_group_ids(*, ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = (ptr[1:] - ptr[:-1]).to(device=device, dtype=torch.long)
    if int(counts.numel()) == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.arange(
        int(counts.numel()), device=device, dtype=torch.long
    ).repeat_interleave(counts)


def _resolve_absolute_local_indices(
    *,
    local_indices: torch.Tensor,
    local_ptr: torch.Tensor,
    node_ptr: torch.Tensor,
) -> torch.Tensor:
    if int(local_indices.numel()) == 0:
        return local_indices.new_empty((0,))
    graph_ids = _expand_group_ids(ptr=local_ptr, device=local_indices.device)
    node_offsets = node_ptr[:-1].to(device=local_indices.device, dtype=torch.long)
    return local_indices.to(dtype=torch.long) + node_offsets.index_select(0, graph_ids)


def aggregate_selected_log_masses(
    *,
    node_entity_ids: torch.Tensor,
    log_node_mass: torch.Tensor,
    entity_ids: torch.Tensor,
) -> torch.Tensor:
    if int(entity_ids.numel()) == 0:
        return log_node_mass.new_empty((0,))
    unique_entity_ids, inverse = torch.unique(
        entity_ids,
        sorted=True,
        return_inverse=True,
    )
    positions = torch.searchsorted(unique_entity_ids, node_entity_ids)
    within_range = positions < int(unique_entity_ids.numel())
    if not bool(within_range.any().item()):
        return log_node_mass.new_full((int(entity_ids.numel()),), fill_value=_LOG_ZERO)
    matched_positions = positions[within_range]
    matched_node_ids = node_entity_ids[within_range]
    is_match = unique_entity_ids.index_select(0, matched_positions) == matched_node_ids
    if not bool(is_match.any().item()):
        return log_node_mass.new_full((int(entity_ids.numel()),), fill_value=_LOG_ZERO)
    aggregated_unique = _segment_logsumexp(
        values=log_node_mass[within_range][is_match],
        segment_ids=matched_positions[is_match],
        num_segments=int(unique_entity_ids.numel()),
    )
    return aggregated_unique.index_select(0, inverse)


def _aggregate_answer_masses(
    *, batch: TrajectoryBatch, log_terminal_mass: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(batch.node_global_ids.numel()) == 0:
        empty_ids = batch.node_global_ids.new_empty((0,))
        empty_probs = log_terminal_mass.new_empty((0,))
        return empty_ids, empty_probs
    answer_entity_ids, inverse = torch.unique(
        batch.node_global_ids,
        sorted=True,
        return_inverse=True,
    )
    log_answer_probs = _segment_logsumexp(
        values=log_terminal_mass,
        segment_ids=inverse,
        num_segments=int(answer_entity_ids.numel()),
    )
    positive = torch.isfinite(log_answer_probs)
    return answer_entity_ids[positive], log_answer_probs[positive]


class ExactReachabilityAnalyzer:
    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

    def analyze(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ExactReachabilityAnalysis:
        if batch.num_graphs != 1:
            raise ValueError(
                "ExactReachabilityAnalyzer expects a single-graph TrajectoryBatch."
            )
        dp_result = self.compute_dynamic_program(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
        )
        return self._build_analysis(batch=batch, dp_result=dp_result)

    def analyze_edge_support(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ExactEdgeSupportAnalysis:
        if batch.num_graphs != 1:
            raise ValueError(
                "ExactReachabilityAnalyzer expects a single-graph TrajectoryBatch."
            )
        dp_result = self.compute_dynamic_program(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
        )
        if bool(torch.isfinite(dp_result.log_gold_mass).item()):
            edge_conditional_success_prob = _log_mass_to_mass(
                dp_result.log_edge_success_mass - dp_result.log_gold_mass
            )
        else:
            edge_conditional_success_prob = torch.zeros_like(
                dp_result.edge_success_mass
            )
        return ExactEdgeSupportAnalysis(
            edge_success_mass=dp_result.edge_success_mass,
            edge_conditional_success_prob=edge_conditional_success_prob,
            gold_mass=_log_scalar_to_float(dp_result.log_gold_mass),
        )

    def _build_analysis(
        self,
        *,
        batch: TrajectoryBatch,
        dp_result: ExactDynamicProgramResult,
    ) -> ExactReachabilityAnalysis:
        answer_entity_ids, log_answer_probs = _aggregate_answer_masses(
            batch=batch,
            log_terminal_mass=dp_result.log_terminal_mass,
        )
        retrieval_answer_entity_ids, log_retrieval_answer_probs = (
            _aggregate_answer_masses(
                batch=batch,
                log_terminal_mass=dp_result.log_retrieval_terminal_mass,
            )
        )
        return ExactReachabilityAnalysis(
            terminal_mass=dp_result.terminal_mass,
            answer_entity_ids=answer_entity_ids,
            answer_probs=_log_mass_to_mass(log_answer_probs),
            retrieval_answer_entity_ids=retrieval_answer_entity_ids,
            retrieval_answer_probs=_log_mass_to_mass(log_retrieval_answer_probs),
            gold_total_mass=_log_scalar_to_float(dp_result.log_gold_mass),
            success_by_step=dp_result.success_by_step,
            log_terminal_mass=dp_result.log_terminal_mass,
            log_answer_probs=log_answer_probs,
            log_gold_total_mass=float(dp_result.log_gold_mass.item()),
            log_retrieval_answer_probs=log_retrieval_answer_probs,
            log_success_by_step=dp_result.log_success_by_step,
        )

    def compute_dynamic_program(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ExactDynamicProgramResult:
        gold_mask = self._gold_mask(batch=batch)
        num_nodes = int(batch.num_nodes_total)
        num_graphs = int(batch.num_graphs)
        device = batch.node_ptr.device
        transitions = [
            self._compute_time_step_transitions(
                batch=batch,
                policy=policy,
                prepared_batch=prepared_batch,
                step_t=step_t,
            )
            for step_t in range(self.max_steps)
        ]
        log_success_by_step = torch.full(
            (self.max_steps + 1, num_nodes),
            device=device,
            dtype=torch.float32,
            fill_value=_LOG_ZERO,
        )
        log_success_by_step[:, gold_mask] = 0.0
        for step_t in range(self.max_steps - 1, -1, -1):
            transition = transitions[step_t]
            if int(transition.edge_probs.numel()) == 0:
                continue
            child_log_success = log_success_by_step[step_t + 1].index_select(
                0, transition.target_nodes
            )
            log_edge_probs = _probabilities_to_log_space(
                transition.edge_probs.to(dtype=torch.float32)
            )
            parent_log_success = _segment_logsumexp(
                values=log_edge_probs + child_log_success,
                segment_ids=transition.edge_agent_batch,
                num_segments=num_nodes,
            )
            log_success_by_step[step_t] = torch.where(
                gold_mask,
                torch.zeros_like(parent_log_success),
                parent_log_success,
            )
        start_dist = policy.compute_start_distribution(prepared_batch)
        log_start_mass = torch.full(
            (num_nodes,),
            device=device,
            dtype=torch.float32,
            fill_value=_LOG_ZERO,
        )
        if int(start_dist.candidate_nodes_abs.numel()) > 0:
            log_start_mass = _segment_logsumexp(
                values=start_dist.log_probs.to(dtype=torch.float32),
                segment_ids=start_dist.candidate_nodes_abs,
                num_segments=num_nodes,
            )
        log_gold_mass_by_graph = torch.full(
            (num_graphs,),
            device=device,
            dtype=torch.float32,
            fill_value=_LOG_ZERO,
        )
        if num_nodes > 0:
            log_gold_mass_by_graph = _segment_logsumexp(
                values=log_start_mass + log_success_by_step[0],
                segment_ids=batch.node_batch.to(device=device, dtype=torch.long),
                num_segments=num_graphs,
            )
        log_gold_mass = torch.tensor(_LOG_ZERO, device=device, dtype=torch.float32)
        if num_graphs > 0:
            log_gold_mass = torch.logsumexp(log_gold_mass_by_graph, dim=0)
        log_terminal_mass = torch.where(
            gold_mask,
            log_start_mass,
            torch.full_like(log_start_mass, fill_value=_LOG_ZERO),
        )
        log_edge_success_mass = torch.full(
            (int(prepared_batch.topology.edge_index.size(1)),),
            device=device,
            dtype=torch.float32,
            fill_value=_LOG_ZERO,
        )
        log_retrieval_terminal_mass = torch.full(
            (num_nodes,),
            device=device,
            dtype=torch.float32,
            fill_value=_LOG_ZERO,
        )
        log_alive_mass = log_start_mass.masked_fill(gold_mask, _LOG_ZERO)
        log_retrieval_alive_mass = log_start_mass.clone()
        for step_t in range(self.max_steps):
            transition = transitions[step_t]
            if int(transition.edge_probs.numel()) == 0:
                log_retrieval_terminal_mass = torch.logaddexp(
                    log_retrieval_terminal_mass,
                    log_retrieval_alive_mass,
                )
                break
            log_edge_probs = _probabilities_to_log_space(
                transition.edge_probs.to(dtype=torch.float32)
            )
            child_log_success = log_success_by_step[step_t + 1].index_select(
                0, transition.target_nodes
            )
            log_edge_mass = (
                log_alive_mass.index_select(0, transition.edge_agent_batch)
                + log_edge_probs
            )
            log_edge_success_mass = torch.logaddexp(
                log_edge_success_mass,
                _segment_logsumexp(
                    values=log_edge_mass + child_log_success,
                    segment_ids=transition.edge_ids,
                    num_segments=int(prepared_batch.topology.edge_index.size(1)),
                ),
            )
            next_log_alive_mass = torch.full_like(log_alive_mass, fill_value=_LOG_ZERO)
            target_is_gold = gold_mask.index_select(0, transition.target_nodes)
            if bool(target_is_gold.any().item()):
                log_terminal_mass = torch.logaddexp(
                    log_terminal_mass,
                    _segment_logsumexp(
                        values=log_edge_mass[target_is_gold],
                        segment_ids=transition.target_nodes[target_is_gold],
                        num_segments=num_nodes,
                    ),
                )
            non_gold_targets = ~target_is_gold
            if bool(non_gold_targets.any().item()) and step_t + 1 < self.max_steps:
                next_log_alive_mass = _segment_logsumexp(
                    values=log_edge_mass[non_gold_targets],
                    segment_ids=transition.target_nodes[non_gold_targets],
                    num_segments=num_nodes,
                )
            log_alive_mass = next_log_alive_mass

            retrieval_dead_end_mask = ~transition.has_values
            if bool(retrieval_dead_end_mask.any().item()):
                log_retrieval_terminal_mass = torch.logaddexp(
                    log_retrieval_terminal_mass,
                    log_retrieval_alive_mass.masked_fill(
                        ~retrieval_dead_end_mask, _LOG_ZERO
                    ),
                )
            log_retrieval_edge_mass = (
                log_retrieval_alive_mass.index_select(0, transition.edge_agent_batch)
                + log_edge_probs
            )
            retrieval_next_log_alive_mass = _segment_logsumexp(
                values=log_retrieval_edge_mass,
                segment_ids=transition.target_nodes,
                num_segments=num_nodes,
            )
            if step_t + 1 >= self.max_steps:
                log_retrieval_terminal_mass = torch.logaddexp(
                    log_retrieval_terminal_mass,
                    retrieval_next_log_alive_mass,
                )
            else:
                log_retrieval_alive_mass = retrieval_next_log_alive_mass
        return ExactDynamicProgramResult(
            log_terminal_mass=log_terminal_mass,
            log_retrieval_terminal_mass=log_retrieval_terminal_mass,
            log_success_by_step=log_success_by_step,
            log_gold_mass=log_gold_mass,
            log_gold_mass_by_graph=log_gold_mass_by_graph,
            log_edge_success_mass=log_edge_success_mass,
        )

    def _run_dynamic_program(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ExactDynamicProgramResult:
        return self.compute_dynamic_program(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
        )

    def _compute_time_step_transitions(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        step_t: int,
    ) -> _StepTransitionBatch:
        num_nodes = int(batch.num_nodes_total)
        current_nodes = torch.arange(
            num_nodes, device=batch.node_ptr.device, dtype=torch.long
        ).view(1, -1)
        state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=torch.zeros_like(current_nodes, dtype=torch.bool),
            num_steps=torch.full_like(
                current_nodes, fill_value=int(step_t), dtype=torch.long
            ),
        )
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
            state=state,
            max_steps=self.max_steps,
        )
        return _StepTransitionBatch(
            edge_agent_batch=step.distribution.edge_agent_batch,
            edge_ids=step.distribution.edge_ids,
            target_nodes=step.distribution.target_nodes,
            edge_probs=step.move_probs,
            has_values=step.has_values.view(-1),
        )

    @staticmethod
    def _gold_mask(*, batch: TrajectoryBatch) -> torch.Tensor:
        gold_mask = torch.zeros(
            (batch.num_nodes_total,), device=batch.node_ptr.device, dtype=torch.bool
        )
        if int(batch.a_local_indices.numel()) > 0:
            absolute_answer_nodes = _resolve_absolute_local_indices(
                local_indices=batch.a_local_indices,
                local_ptr=batch.a_ptr,
                node_ptr=batch.node_ptr,
            )
            gold_mask.scatter_(0, absolute_answer_nodes, True)
        return gold_mask


__all__ = [
    "ExactDynamicProgramResult",
    "ExactEdgeSupportAnalysis",
    "ExactReachabilityAnalysis",
    "ExactReachabilityAnalyzer",
    "aggregate_selected_log_masses",
]
