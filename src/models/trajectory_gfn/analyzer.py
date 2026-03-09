from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .batch import TrajectoryBatch
from .policy import TrajectoryPolicy, TrajectoryPolicyContext
from .state import TrajectoryState
from .transition import apply_forward_constraints


@dataclass(frozen=True)
class AnswerMassAnalysis:
    terminal_mass: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    gold_total_mass: float


@dataclass(frozen=True)
class _PrefixMassState:
    start_node: int
    current_node: int
    edge_ids: tuple[int, ...]

    @property
    def num_moves(self) -> int:
        return int(len(self.edge_ids))


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


def _state_from_prefix(
    *,
    batch: TrajectoryBatch,
    prefix: _PrefixMassState,
    max_steps: int,
) -> TrajectoryState:
    return TrajectoryState.from_edge_path(
        start_node=int(prefix.start_node),
        edge_ids=prefix.edge_ids,
        edge_index=batch.edge_index,
        max_steps=max_steps,
        device=batch.node_ptr.device,
    )


class AnswerMassAnalyzer:
    def __init__(self, *, max_steps: int, min_stop_steps: int) -> None:
        self.max_steps = int(max_steps)
        self.min_stop_steps = int(min_stop_steps)

    def analyze(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
    ) -> AnswerMassAnalysis:
        if batch.num_graphs != 1:
            raise ValueError(
                "AnswerMassAnalyzer expects a single-graph TrajectoryBatch."
            )
        terminal_mass = torch.zeros(
            (batch.num_nodes_total,),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        gold_mask = torch.zeros_like(terminal_mass, dtype=torch.bool)
        if int(batch.a_local_indices.numel()) > 0:
            gold_mask.scatter_(0, batch.a_local_indices, True)
        frontier = self._initialize_frontier(policy=policy, context=context)
        while frontier:
            frontier = self._propagate_frontier(
                batch=batch,
                policy=policy,
                context=context,
                frontier=frontier,
                terminal_mass=terminal_mass,
                gold_mask=gold_mask,
            )
        answer_entity_ids, answer_probs = _aggregate_answer_masses(
            batch=batch,
            terminal_mass=terminal_mass,
        )
        return AnswerMassAnalysis(
            terminal_mass=terminal_mass,
            answer_entity_ids=answer_entity_ids,
            answer_probs=answer_probs,
            gold_total_mass=float(terminal_mass[gold_mask].sum().item()),
        )

    def _initialize_frontier(
        self,
        *,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
    ) -> dict[_PrefixMassState, float]:
        frontier: dict[_PrefixMassState, float] = {}
        start_dist = policy.compute_start_distribution(context)
        for idx in range(int(start_dist.candidate_nodes_abs.numel())):
            start_node = int(start_dist.candidate_nodes_abs[idx].item())
            prefix = _PrefixMassState(
                start_node=start_node,
                current_node=start_node,
                edge_ids=(),
            )
            frontier[prefix] = frontier.get(prefix, 0.0) + float(
                start_dist.log_probs[idx].exp().item()
            )
        return frontier

    def _propagate_frontier(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
        frontier: dict[_PrefixMassState, float],
        terminal_mass: torch.Tensor,
        gold_mask: torch.Tensor,
    ) -> dict[_PrefixMassState, float]:
        next_frontier: dict[_PrefixMassState, float] = {}
        for prefix, prefix_mass in frontier.items():
            state = _state_from_prefix(
                batch=batch, prefix=prefix, max_steps=self.max_steps
            )
            distribution = policy.compute_forward_distribution(context, state)
            distribution = apply_forward_constraints(
                distribution,
                state=state,
                node_is_target=gold_mask,
                min_stop_steps=self.min_stop_steps,
                max_steps=self.max_steps,
            )
            if bool(distribution.invalid_rows.view(-1).any().item()):
                raise ValueError(
                    "AnswerMassAnalyzer encountered states with empty forward support before min_stop_steps."
                )
            move_log_probs, stop_log_probs, _ = policy.compute_forward_log_probs(
                distribution
            )
            terminal_mass[prefix.current_node] += float(prefix_mass) * float(
                stop_log_probs.view(-1)[0].exp().item()
            )
            for edge_idx in range(int(move_log_probs.numel())):
                child_log_prob = float(move_log_probs[edge_idx].item())
                if not math.isfinite(child_log_prob):
                    continue
                child = _PrefixMassState(
                    start_node=prefix.start_node,
                    current_node=int(distribution.target_nodes[edge_idx].item()),
                    edge_ids=prefix.edge_ids
                    + (int(distribution.edge_ids[edge_idx].item()),),
                )
                next_frontier[child] = next_frontier.get(child, 0.0) + float(
                    prefix_mass
                ) * math.exp(child_log_prob)
        return next_frontier
