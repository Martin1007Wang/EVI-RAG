from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import compute_constrained_policy_step
from src.models.gflownet import (
    PreparedSearchBatch,
    SearchPolicyProtocol,
    SearchState,
)


@dataclass(frozen=True)
class ExactReachabilityAnalysis:
    terminal_mass: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    gold_total_mass: float
    retrieval_answer_entity_ids: torch.Tensor | None = None
    retrieval_answer_probs: torch.Tensor | None = None
    success_by_step: torch.Tensor | None = None


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
class _ExactDynamicProgramResult:
    terminal_mass: torch.Tensor
    retrieval_terminal_mass: torch.Tensor
    success_by_step: torch.Tensor
    gold_mass: torch.Tensor
    edge_success_mass: torch.Tensor


def _aggregate_answer_masses(
    *, batch: TrajectoryBatch, terminal_mass: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(batch.node_global_ids.numel()) == 0:
        empty_ids = batch.node_global_ids.new_empty((0,))
        empty_probs = terminal_mass.new_empty((0,))
        return empty_ids, empty_probs
    answer_entity_ids, inverse = torch.unique(
        batch.node_global_ids,
        sorted=True,
        return_inverse=True,
    )
    answer_probs = terminal_mass.new_zeros((int(answer_entity_ids.numel()),))
    answer_probs.scatter_add_(0, inverse, terminal_mass)
    positive = answer_probs > 0.0
    return answer_entity_ids[positive], answer_probs[positive]


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
        dp_result = self._run_dynamic_program(
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
        dp_result = self._run_dynamic_program(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
        )
        if float(dp_result.gold_mass.item()) > 0.0:
            edge_conditional_success_prob = (
                dp_result.edge_success_mass / dp_result.gold_mass
            )
        else:
            edge_conditional_success_prob = torch.zeros_like(
                dp_result.edge_success_mass
            )
        return ExactEdgeSupportAnalysis(
            edge_success_mass=dp_result.edge_success_mass,
            edge_conditional_success_prob=edge_conditional_success_prob,
            gold_mass=float(dp_result.gold_mass.item()),
        )

    def _build_analysis(
        self,
        *,
        batch: TrajectoryBatch,
        dp_result: _ExactDynamicProgramResult,
    ) -> ExactReachabilityAnalysis:
        answer_entity_ids, answer_probs = _aggregate_answer_masses(
            batch=batch,
            terminal_mass=dp_result.terminal_mass,
        )
        retrieval_answer_entity_ids, retrieval_answer_probs = _aggregate_answer_masses(
            batch=batch,
            terminal_mass=dp_result.retrieval_terminal_mass,
        )
        return ExactReachabilityAnalysis(
            terminal_mass=dp_result.terminal_mass,
            answer_entity_ids=answer_entity_ids,
            answer_probs=answer_probs,
            retrieval_answer_entity_ids=retrieval_answer_entity_ids,
            retrieval_answer_probs=retrieval_answer_probs,
            gold_total_mass=float(dp_result.gold_mass.item()),
            success_by_step=dp_result.success_by_step,
        )

    def _run_dynamic_program(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> _ExactDynamicProgramResult:
        gold_mask = self._gold_mask(batch=batch)
        num_nodes = int(batch.num_nodes_total)
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
        success_by_step = torch.zeros(
            (self.max_steps + 1, num_nodes),
            device=device,
            dtype=torch.float32,
        )
        success_by_step[:, gold_mask] = 1.0
        for step_t in range(self.max_steps - 1, -1, -1):
            transition = transitions[step_t]
            if int(transition.edge_probs.numel()) == 0:
                continue
            child_success = success_by_step[step_t + 1].index_select(
                0, transition.target_nodes
            )
            parent_success = success_by_step.new_zeros((num_nodes,))
            parent_success.scatter_add_(
                0,
                transition.edge_agent_batch,
                transition.edge_probs * child_success,
            )
            success_by_step[step_t] = torch.where(
                gold_mask,
                torch.ones_like(parent_success),
                parent_success,
            )
        start_dist = policy.compute_start_distribution(prepared_batch)
        start_mass = success_by_step.new_zeros((num_nodes,))
        if int(start_dist.candidate_nodes_abs.numel()) > 0:
            start_mass.scatter_add_(
                0, start_dist.candidate_nodes_abs, start_dist.log_probs.exp()
            )
        gold_mass = (start_mass * success_by_step[0]).sum()
        terminal_mass = success_by_step.new_zeros((num_nodes,))
        edge_success_mass = success_by_step.new_zeros(
            (int(prepared_batch.topology.edge_index.size(1)),)
        )
        retrieval_terminal_mass = success_by_step.new_zeros((num_nodes,))
        gold_start_mass = start_mass * gold_mask.to(dtype=start_mass.dtype)
        terminal_mass = terminal_mass + gold_start_mass
        alive_mass = start_mass.masked_fill(gold_mask, 0.0)
        retrieval_alive_mass = start_mass.clone()
        for step_t in range(self.max_steps):
            transition = transitions[step_t]
            if int(transition.edge_probs.numel()) == 0:
                retrieval_terminal_mass += retrieval_alive_mass
                break
            edge_parent_mass = alive_mass.index_select(0, transition.edge_agent_batch)
            edge_mass = edge_parent_mass * transition.edge_probs
            if int(edge_mass.numel()) == 0:
                retrieval_terminal_mass += retrieval_alive_mass
                break
            child_success = success_by_step[step_t + 1].index_select(
                0, transition.target_nodes
            )
            edge_success_mass.scatter_add_(
                0,
                transition.edge_ids,
                edge_mass * child_success,
            )
            next_alive_mass = alive_mass.new_zeros((num_nodes,))
            target_is_gold = gold_mask.index_select(0, transition.target_nodes)
            if bool(target_is_gold.any().item()):
                terminal_mass.scatter_add_(
                    0,
                    transition.target_nodes[target_is_gold],
                    edge_mass[target_is_gold],
                )
            non_gold_targets = ~target_is_gold
            if bool(non_gold_targets.any().item()) and step_t + 1 < self.max_steps:
                next_alive_mass.scatter_add_(
                    0,
                    transition.target_nodes[non_gold_targets],
                    edge_mass[non_gold_targets],
                )
            alive_mass = next_alive_mass

            retrieval_dead_end_mask = ~transition.has_values
            if bool(retrieval_dead_end_mask.any().item()):
                retrieval_terminal_mass += retrieval_alive_mass.masked_fill(
                    ~retrieval_dead_end_mask, 0.0
                )
            retrieval_edge_parent_mass = retrieval_alive_mass.index_select(
                0, transition.edge_agent_batch
            )
            retrieval_edge_mass = retrieval_edge_parent_mass * transition.edge_probs
            retrieval_next_alive_mass = retrieval_alive_mass.new_zeros((num_nodes,))
            if int(retrieval_edge_mass.numel()) > 0:
                retrieval_next_alive_mass.scatter_add_(
                    0,
                    transition.target_nodes,
                    retrieval_edge_mass,
                )
            if step_t + 1 >= self.max_steps:
                retrieval_terminal_mass += retrieval_next_alive_mass
            else:
                retrieval_alive_mass = retrieval_next_alive_mass
        return _ExactDynamicProgramResult(
            terminal_mass=terminal_mass,
            retrieval_terminal_mass=retrieval_terminal_mass,
            success_by_step=success_by_step,
            gold_mass=gold_mass,
            edge_success_mass=edge_success_mass,
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
            gold_mask.scatter_(0, batch.a_local_indices, True)
        return gold_mask


__all__ = [
    "ExactEdgeSupportAnalysis",
    "ExactReachabilityAnalysis",
    "ExactReachabilityAnalyzer",
]
