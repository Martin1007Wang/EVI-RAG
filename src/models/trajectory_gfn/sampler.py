from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.configs.trajectory_gfn import HorizonConfig, TrajectoryTrainingConfig
from src.models.environment.ops import build_node_membership_mask

from .action_sampler import ActionSampler
from .batch import TrajectoryBatch
from .policy import TrajectoryPolicy, TrajectoryPolicyContext
from .reward import TrajectoryReward
from .state import TrajectoryState
from .transition import advance_state, apply_forward_constraints


@dataclass(frozen=True)
class TrajectorySampleBatch:
    graph_log_z: torch.Tensor
    start_nodes: torch.Tensor
    start_log_probs: torch.Tensor
    start_state_log_f: torch.Tensor
    log_pf_steps: torch.Tensor
    state_log_f_steps: torch.Tensor
    next_state_log_f_steps: torch.Tensor
    chosen_edge_ids_steps: torch.Tensor
    active_steps: torch.Tensor
    is_stop_steps: torch.Tensor
    stop_nodes: torch.Tensor
    hit_mask: torch.Tensor
    terminal_rewards: torch.Tensor
    terminal_log_rewards: torch.Tensor

    @property
    def log_pb_steps(self) -> torch.Tensor:
        return torch.zeros_like(self.log_pf_steps)


class ForwardRolloutSampler:
    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: TrajectoryTrainingConfig,
        reward: TrajectoryReward,
    ) -> None:
        self.horizon_cfg = horizon_cfg
        self.training_cfg = training_cfg
        self.reward = reward
        self.action_sampler = ActionSampler()

    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: TrajectoryPolicy,
        context: TrajectoryPolicyContext,
        num_rollouts: int | None = None,
        is_training: bool = True,
        deterministic_start: bool = False,
        sampling_temperature: float | None = None,
    ) -> TrajectorySampleBatch:
        rollout_count = (
            int(num_rollouts)
            if num_rollouts is not None
            else int(self.training_cfg.rollout_batch_size)
        )
        temperature = (
            float(sampling_temperature)
            if sampling_temperature is not None
            else float(self.training_cfg.sampling_temperature)
        )
        start_dist = policy.compute_start_distribution(context)
        start_nodes, start_log_probs = policy.sample_start_nodes(
            start_dist,
            num_rollouts=rollout_count,
            deterministic=deterministic_start,
        )
        state = TrajectoryState.initialize(
            start_nodes=start_nodes,
            max_steps=int(self.horizon_cfg.max_steps),
        )
        start_state_log_f = policy.compute_log_flow(context, state)
        num_graphs, num_rollouts = start_nodes.shape
        max_action_steps = int(self.horizon_cfg.max_steps) + 1
        node_is_target = build_node_membership_mask(
            local_indices=batch.a_local_indices,
            ptr=batch.a_ptr,
            node_ptr=batch.node_ptr,
            num_nodes_total=batch.num_nodes_total,
            device=batch.node_ptr.device,
            field_name="a_local_indices",
        )
        log_pf_steps = torch.zeros(
            (num_graphs, num_rollouts, max_action_steps),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        state_log_f_steps = torch.zeros_like(log_pf_steps)
        next_state_log_f_steps = torch.zeros_like(log_pf_steps)
        chosen_edge_ids_steps = torch.full(
            (num_graphs, num_rollouts, max_action_steps),
            fill_value=-1,
            device=batch.node_ptr.device,
            dtype=torch.long,
        )
        active_steps = torch.zeros(
            (num_graphs, num_rollouts, max_action_steps),
            device=batch.node_ptr.device,
            dtype=torch.bool,
        )
        is_stop_steps = torch.zeros_like(active_steps)

        for step_idx in range(max_action_steps):
            active_mask = ~state.done_mask
            if not bool(active_mask.any().item()):
                break
            distribution = policy.compute_forward_distribution(context, state)
            distribution = apply_forward_constraints(
                distribution,
                state=state,
                node_is_target=node_is_target,
                min_stop_steps=int(self.horizon_cfg.min_stop_steps),
                max_steps=int(self.horizon_cfg.max_steps),
            )
            if bool(distribution.invalid_rows.view(-1).any().item()):
                raise ValueError(
                    "Encountered states with empty forward support before min_stop_steps."
                )
            action_info = self.action_sampler(
                distribution.to_policy_output(),
                is_training=is_training,
                sampling_temperature=temperature,
                invalid_logits_policy=str(self.training_cfg.invalid_logits_policy),
            )
            chosen_is_stop = action_info["is_stop"].view(num_graphs, num_rollouts)
            chosen_targets = action_info["chosen_target_nodes"].view(-1)
            chosen_edge_ids = action_info["chosen_edge_ids"].view(-1)
            chosen_edge_ids_view = chosen_edge_ids.view(num_graphs, num_rollouts)
            current_log_f = distribution.state_log_flows.to(dtype=torch.float32)
            next_state = advance_state(
                state,
                chosen_target_nodes=chosen_targets,
                chosen_edge_ids=chosen_edge_ids,
                is_stop=action_info["is_stop"].view(-1),
            )
            next_log_f = policy.compute_log_flow(context, next_state).to(
                dtype=torch.float32
            )
            log_pf = (
                action_info["log_prob"]
                .view(num_graphs, num_rollouts)
                .to(dtype=torch.float32)
            )
            log_pf_steps[:, :, step_idx] = torch.where(
                active_mask, log_pf, torch.zeros_like(log_pf)
            )
            state_log_f_steps[:, :, step_idx] = torch.where(
                active_mask,
                current_log_f,
                torch.zeros_like(current_log_f),
            )
            next_state_log_f_steps[:, :, step_idx] = torch.where(
                active_mask,
                next_log_f,
                torch.zeros_like(next_log_f),
            )
            chosen_edge_ids_steps[:, :, step_idx] = torch.where(
                active_mask & (~chosen_is_stop),
                chosen_edge_ids_view,
                torch.full_like(chosen_edge_ids_view, -1),
            )
            active_steps[:, :, step_idx] = active_mask
            is_stop_steps[:, :, step_idx] = active_mask & chosen_is_stop
            state = next_state

        hit_mask, rewards, log_rewards = self.reward.compute(
            batch=batch,
            stop_nodes=state.current_node,
        )
        return TrajectorySampleBatch(
            graph_log_z=context.graph_log_z,
            start_nodes=start_nodes,
            start_log_probs=start_log_probs.to(dtype=torch.float32),
            start_state_log_f=start_state_log_f.to(dtype=torch.float32),
            log_pf_steps=log_pf_steps,
            state_log_f_steps=state_log_f_steps,
            next_state_log_f_steps=next_state_log_f_steps,
            chosen_edge_ids_steps=chosen_edge_ids_steps,
            active_steps=active_steps,
            is_stop_steps=is_stop_steps,
            stop_nodes=state.current_node,
            hit_mask=hit_mask,
            terminal_rewards=rewards,
            terminal_log_rewards=log_rewards,
        )
