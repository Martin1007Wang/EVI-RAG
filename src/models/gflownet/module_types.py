from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.configs import (
    ActionPriorConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    SearchEvalConfig,
)


@dataclass(frozen=True)
class GFlowNetConfig:
    horizon_cfg: HorizonConfig
    training_cfg: GFlowNetTrainingConfig
    action_prior_cfg: ActionPriorConfig
    policy_cfg: PolicyConfig
    eval_cfg: SearchEvalConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig


@dataclass(frozen=True)
class TrainingRolloutMetrics:
    unique_success_paths_per_100_rollouts: float
    new_success_paths: int
    start_node_entropy: torch.Tensor
    start_node_entropy_normalized: torch.Tensor
    proposal_start_target_kl: torch.Tensor
    active_forward_states: float
    unique_forward_states: float
    forward_state_dedup_keep_ratio: float
    raw_graph_candidates: float
    scored_graph_candidates: float
    raw_graph_candidates_per_unique_state: float
    scored_graph_candidates_per_unique_state: float


__all__ = ["GFlowNetConfig", "TrainingRolloutMetrics"]
