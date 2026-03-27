from __future__ import annotations

from typing import Any, Callable

import torch

from src.graph import TrajectoryBatch

from .losses import SubTrajectoryBalanceLossOutput
from .module_types import TrainingRolloutMetrics
from .sampler import TrajectoryGFNSampleBatch
from .success_paths import (
    collect_success_rollout_key_rows,
    deduplicate_success_rollout_key_rows,
)


def compute_training_rollout_metrics(
    *,
    batch: TrajectoryBatch,
    sample_batch: TrajectoryGFNSampleBatch,
) -> TrainingRolloutMetrics:
    total_rollouts = int(sample_batch.success_mask.numel())
    success_path_rows = collect_success_rollout_key_rows(
        batch=batch,
        sample_batch=sample_batch,
    )
    unique_success_path_rows = deduplicate_success_rollout_key_rows(success_path_rows)
    new_success_paths = (
        0 if unique_success_path_rows is None else int(unique_success_path_rows.size(0))
    )
    start_entropy = sample_batch.proposal_start_entropy
    start_entropy_normalized = sample_batch.proposal_start_entropy_normalized
    mean_start_entropy = (
        start_entropy.detach().to(dtype=torch.float32).mean()
        if start_entropy is not None and int(start_entropy.numel()) > 0
        else torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
    )
    mean_start_entropy_normalized = (
        start_entropy_normalized.detach().to(dtype=torch.float32).mean()
        if start_entropy_normalized is not None
        and int(start_entropy_normalized.numel()) > 0
        else torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
    )
    proposal_start_target_kl = sample_batch.proposal_start_target_kl
    mean_proposal_start_target_kl = (
        proposal_start_target_kl.detach().to(dtype=torch.float32).mean()
        if proposal_start_target_kl is not None
        and int(proposal_start_target_kl.numel()) > 0
        else torch.zeros((), device=batch.node_ptr.device, dtype=torch.float32)
    )
    unique_success_rate = (
        (100.0 * float(new_success_paths)) / float(total_rollouts)
        if total_rollouts > 0
        else 0.0
    )
    active_forward_states = float(sample_batch.total_active_agent_count)
    unique_forward_states = float(sample_batch.total_unique_active_state_count)
    raw_graph_candidates = float(sample_batch.total_raw_graph_candidate_count)
    scored_graph_candidates = float(sample_batch.total_scored_graph_candidate_count)
    forward_state_dedup_keep_ratio = (
        unique_forward_states / active_forward_states
        if active_forward_states > 0.0
        else 0.0
    )
    raw_graph_candidates_per_unique_state = (
        raw_graph_candidates / unique_forward_states
        if unique_forward_states > 0.0
        else 0.0
    )
    scored_graph_candidates_per_unique_state = (
        scored_graph_candidates / unique_forward_states
        if unique_forward_states > 0.0
        else 0.0
    )
    return TrainingRolloutMetrics(
        unique_success_paths_per_100_rollouts=unique_success_rate,
        new_success_paths=new_success_paths,
        start_node_entropy=mean_start_entropy,
        start_node_entropy_normalized=mean_start_entropy_normalized,
        proposal_start_target_kl=mean_proposal_start_target_kl,
        active_forward_states=active_forward_states,
        unique_forward_states=unique_forward_states,
        forward_state_dedup_keep_ratio=forward_state_dedup_keep_ratio,
        raw_graph_candidates=raw_graph_candidates,
        scored_graph_candidates=scored_graph_candidates,
        raw_graph_candidates_per_unique_state=raw_graph_candidates_per_unique_state,
        scored_graph_candidates_per_unique_state=scored_graph_candidates_per_unique_state,
    )


def safe_batch_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().to(dtype=torch.float32).view(-1)
    y = y.detach().to(dtype=torch.float32).view(-1)
    if int(x.numel()) < 2 or int(y.numel()) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y)
    if float(x_norm.item()) == 0.0 or float(y_norm.item()) == 0.0:
        return 0.0
    corr = torch.dot(x, y) / (x_norm * y_norm)
    return float(corr.clamp(min=-1.0, max=1.0).item())


def compute_root_diagnostics(
    *,
    prepared_batch: Any,
    sample_batch: TrajectoryGFNSampleBatch,
) -> dict[str, float]:
    topology = prepared_batch.topology
    device = sample_batch.graph_log_z.device
    node_counts = (
        topology.graph_node_offsets[1:] - topology.graph_node_offsets[:-1]
    ).to(
        device=device,
        dtype=torch.float32,
    )
    edge_counts = torch.zeros_like(node_counts)
    if int(topology.edge_index.size(1)) > 0:
        edge_graph_ids = topology.graph_index_from_nodes(
            topology.edge_index[0].to(device=device)
        )
        edge_counts.scatter_add_(
            0,
            edge_graph_ids,
            torch.ones_like(edge_graph_ids, dtype=torch.float32),
        )
    start_counts = prepared_batch.observation.q_local_indices.counts().to(
        device=device,
        dtype=torch.float32,
    )
    log_z = sample_batch.graph_log_z.detach().to(dtype=torch.float32)
    return {
        "log_z_num_nodes_corr": safe_batch_correlation(log_z, torch.log1p(node_counts)),
        "log_z_num_edges_corr": safe_batch_correlation(log_z, torch.log1p(edge_counts)),
        "log_z_start_candidates_corr": safe_batch_correlation(
            log_z,
            torch.log1p(start_counts),
        ),
    }


def build_training_metrics(
    *,
    cfg: Any,
    online_loss_output: SubTrajectoryBalanceLossOutput,
    total_loss: torch.Tensor,
    online_direct_entity_ranking_loss: torch.Tensor,
    online_direct_gold_entity_mass: torch.Tensor,
    online_direct_entity_count: torch.Tensor,
    rollouts_per_graph: int,
    sampling_temperature: float,
    action_prior_scale: float,
    rollout_metrics: TrainingRolloutMetrics,
    root_diagnostics: dict[str, float],
    success_replay_effective_mix_alpha: float,
    success_replay_buffer_size: int,
    success_replay_ready: bool,
    success_replay_added: int,
    success_replay_sampled: int,
    replay_subtb_loss: torch.Tensor,
    replay_direct_entity_ranking_loss: torch.Tensor,
    resolve_effective_pass: Callable[[bool], float | None],
) -> dict[str, Any]:
    proposal_root_beta = float(cfg.action_prior_cfg.root_beta or 0.0) * float(
        action_prior_scale
    )
    proposal_edge_beta = float(cfg.action_prior_cfg.edge_beta or 0.0) * float(
        action_prior_scale
    )
    proposal_stop_beta = float(cfg.action_prior_cfg.stop_beta) * float(
        action_prior_scale
    )
    proposal_intent_alignment_weight = float(
        cfg.action_prior_cfg.intent_alignment_weight
    )
    metrics: dict[str, Any] = {
        "loss": total_loss.detach(),
        "actor_loss": total_loss.detach(),
        "subtb_loss": total_loss.detach(),
        "online_subtb_loss": online_loss_output.subtb_loss,
        "replay_subtb_loss": replay_subtb_loss.detach(),
        "answer_quotient_direct_entity_ranking_loss": (
            online_direct_entity_ranking_loss.detach()
        ),
        "answer_quotient_direct_gold_entity_mass": (
            online_direct_gold_entity_mass.detach()
        ),
        "answer_quotient_direct_entity_count": online_direct_entity_count.detach(),
        "replay_answer_quotient_direct_entity_ranking_loss": (
            replay_direct_entity_ranking_loss.detach()
        ),
        "subtb_root_loss": online_loss_output.root_component_loss,
        "subtb_pairwise_loss": online_loss_output.pairwise_component_loss,
        "subtb_terminal_loss": online_loss_output.terminal_component_loss,
        "answer_quotient_loss": online_loss_output.answer_quotient_component_loss,
        "answer_quotient_residual": online_loss_output.answer_quotient_residual_abs,
        "answer_quotient_observed_sinks": online_loss_output.answer_quotient_observed_sink_count,
        "subtb_residual": online_loss_output.residual_abs,
        "subtb_residual_variance_per_batch": online_loss_output.residual_variance,
        "subtb_root": online_loss_output.root_abs,
        "rollout_success": online_loss_output.success_rate,
        "unique_success_paths_per_100_rollouts": (
            rollout_metrics.unique_success_paths_per_100_rollouts
        ),
        "new_success_paths": float(rollout_metrics.new_success_paths),
        "start_node_entropy": rollout_metrics.start_node_entropy,
        "start_node_entropy_normalized": rollout_metrics.start_node_entropy_normalized,
        "proposal_start_target_kl": rollout_metrics.proposal_start_target_kl,
        "active_forward_states": rollout_metrics.active_forward_states,
        "unique_forward_states": rollout_metrics.unique_forward_states,
        "forward_state_dedup_keep_ratio": (
            rollout_metrics.forward_state_dedup_keep_ratio
        ),
        "raw_graph_candidates": rollout_metrics.raw_graph_candidates,
        "scored_graph_candidates": rollout_metrics.scored_graph_candidates,
        "raw_graph_candidates_per_unique_state": (
            rollout_metrics.raw_graph_candidates_per_unique_state
        ),
        "scored_graph_candidates_per_unique_state": (
            rollout_metrics.scored_graph_candidates_per_unique_state
        ),
        "log_z_mean": online_loss_output.log_z_mean,
        "log_z_variance": online_loss_output.log_z_variance,
        "rollouts_per_graph": float(rollouts_per_graph),
        "sampling_temperature": sampling_temperature,
        "proposal_action_prior_scale": float(action_prior_scale),
        "proposal_root_beta": proposal_root_beta,
        "proposal_edge_beta": proposal_edge_beta,
        "proposal_stop_beta": proposal_stop_beta,
        "proposal_intent_alignment_weight": proposal_intent_alignment_weight,
        "proposal_intent_alignment_strength": (
            proposal_edge_beta * proposal_intent_alignment_weight
        ),
        "proposal_shortest_path_edge_weight": float(
            cfg.action_prior_cfg.shortest_path_edge_weight
        ),
        "proposal_answer_distance_weight": float(
            cfg.action_prior_cfg.answer_distance_weight
        ),
        "success_replay_mix_alpha": float(success_replay_effective_mix_alpha),
        "coverage_replay_mix_alpha": float(success_replay_effective_mix_alpha),
        "success_replay_buffer_size": float(success_replay_buffer_size),
        "success_replay_ready": float(success_replay_ready),
        "success_replay_added": float(success_replay_added),
        "success_replay_sampled": float(success_replay_sampled),
        "success_replay_shortest_path_guidance": float(
            cfg.training_cfg.success_replay.add_shortest_path_guidance
        ),
        "step_log_penalty": float(cfg.training_cfg.step_log_penalty or 0.0),
        "answer_stop_log_reward_bonus": float(
            cfg.training_cfg.answer_stop_log_reward_bonus or 0.0
        ),
        "answer_quotient_weight": float(cfg.training_cfg.answer_quotient.weight),
        "answer_quotient_enabled": float(cfg.training_cfg.answer_quotient.enabled),
        "answer_quotient_direct_entity_ranking_weight": float(
            cfg.training_cfg.answer_quotient.direct_entity_ranking_weight
        ),
        "answer_quotient_replace_terminal_loss": float(
            cfg.training_cfg.answer_quotient.replace_terminal_loss
        ),
        "answer_quotient_allocate_stop_mass": float(
            cfg.training_cfg.answer_quotient.allocate_stop_mass
        ),
        "potential_reward_answer_distance_weight": float(
            cfg.training_cfg.potential_reward.answer_distance_weight
        ),
        "terminal_failure_log_reward": float(
            cfg.training_cfg.terminal_failure_log_reward
        ),
    }
    metrics.update(root_diagnostics)
    effective_pass = resolve_effective_pass(True)
    if effective_pass is not None:
        metrics["effective_pass"] = effective_pass
    return metrics


def build_train_metrics_payload(metrics: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            scalar = float(value.detach().to(dtype=torch.float32).item())
        else:
            scalar = float(value)
        payload[f"train/{name}"] = scalar
    return payload


__all__ = [
    "build_train_metrics_payload",
    "build_training_metrics",
    "compute_root_diagnostics",
    "compute_training_rollout_metrics",
    "safe_batch_correlation",
]
