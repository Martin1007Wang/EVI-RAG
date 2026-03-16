from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.policy import (
    GFlowNetPolicy,
    PreparedGFlowNetBatch,
    SearchState,
    compute_constrained_forward_step,
)


class TrajectoryRolloutSupervisorProtocol(Protocol):
    def build_terminal_target_mask(self, *, batch: TrajectoryBatch) -> torch.Tensor: ...

    def compute_terminal_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
        success_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _sample_edges(
    *,
    edge_ids: torch.Tensor,
    target_nodes: torch.Tensor,
    out_degrees: torch.Tensor,
    edge_logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_agents = int(out_degrees.numel())
    chosen_edge_ids = torch.full(
        (total_agents,),
        fill_value=-1,
        device=edge_ids.device,
        dtype=torch.long,
    )
    chosen_target_nodes = torch.zeros(
        (total_agents,), device=target_nodes.device, dtype=torch.long
    )
    chosen_log_probs = torch.zeros(
        (total_agents,), device=edge_logits.device, dtype=torch.float32
    )
    if total_agents == 0:
        return chosen_edge_ids, chosen_target_nodes, chosen_log_probs
    edge_offsets = out_degrees.cumsum(0) - out_degrees
    for agent_idx in range(total_agents):
        degree = int(out_degrees[agent_idx].item())
        if degree <= 0:
            continue
        begin = int(edge_offsets[agent_idx].item())
        end = begin + degree
        logits = edge_logits[begin:end].to(dtype=torch.float32)
        if not bool(torch.isfinite(logits).any().item()):
            continue
        scaled_logits = logits / float(temperature)
        probs = torch.softmax(scaled_logits, dim=0)
        choice = torch.multinomial(probs, num_samples=1, replacement=True).item()
        chosen_edge_ids[agent_idx] = int(edge_ids[begin + choice].item())
        chosen_target_nodes[agent_idx] = int(target_nodes[begin + choice].item())
        chosen_log_probs[agent_idx] = torch.log(probs[choice].clamp_min(1.0e-12))
    return chosen_edge_ids, chosen_target_nodes, chosen_log_probs


@dataclass(frozen=True)
class TrajectoryGFNSampleBatch:
    graph_log_z: torch.Tensor
    start_nodes: torch.Tensor
    start_log_probs: torch.Tensor
    start_state_log_f: torch.Tensor
    log_pf_steps: torch.Tensor
    state_log_f_steps: torch.Tensor
    next_state_log_f_steps: torch.Tensor
    move_mask: torch.Tensor
    trace_nodes: torch.Tensor
    trace_num_steps: torch.Tensor
    trace_mask: torch.Tensor
    terminal_nodes: torch.Tensor
    terminal_num_steps: torch.Tensor
    terminal_state_log_f: torch.Tensor
    terminal_rewards: torch.Tensor
    terminal_log_rewards: torch.Tensor
    success_mask: torch.Tensor


class ForwardTrajectoryGFNSampler:
    def __init__(
        self,
        *,
        max_steps: int,
        trajectory_supervisor: TrajectoryRolloutSupervisorProtocol,
    ) -> None:
        self.max_steps = int(max_steps)
        self.trajectory_supervisor = trajectory_supervisor

    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicy,
        prepared_batch: PreparedGFlowNetBatch,
        rollout_batch_size: int,
        temperature: float,
    ) -> TrajectoryGFNSampleBatch:
        start_dist = policy.compute_start_distribution(prepared_batch)
        start_nodes, start_log_probs = policy.sample_start_nodes(
            start_dist,
            num_rollouts=int(rollout_batch_size),
            deterministic=False,
        )
        state = SearchState.initialize(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            start_nodes=start_nodes,
        )
        start_state_log_f = policy.compute_log_state_scores(prepared_batch, state)
        graph_log_z = policy.compute_graph_log_z(prepared_batch)
        num_graphs, num_rollouts = start_nodes.shape
        terminal_target_mask = self.trajectory_supervisor.build_terminal_target_mask(
            batch=batch
        )

        log_pf_steps = torch.zeros(
            (num_graphs, num_rollouts, self.max_steps),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        state_log_f_steps = torch.zeros_like(log_pf_steps)
        next_state_log_f_steps = torch.zeros_like(log_pf_steps)
        move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)
        trace_nodes = torch.zeros(
            (num_graphs, num_rollouts, self.max_steps),
            device=batch.node_ptr.device,
            dtype=torch.long,
        )
        trace_num_steps = torch.zeros_like(trace_nodes)
        trace_mask = torch.zeros_like(move_mask)

        current_nodes = state.current_nodes.clone()
        done_mask = state.done_mask.clone()
        num_steps = state.num_steps.clone()

        for step_idx in range(self.max_steps):
            active_mask = ~done_mask
            trace_nodes[:, :, step_idx] = current_nodes
            trace_num_steps[:, :, step_idx] = num_steps
            trace_mask[:, :, step_idx] = active_mask

            if not bool(active_mask.any().item()):
                break

            on_target = terminal_target_mask.index_select(
                0, current_nodes.view(-1)
            ).view_as(current_nodes)
            done_mask = done_mask | (active_mask & on_target)
            active_mask = ~done_mask
            if not bool(active_mask.any().item()):
                break

            search_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=current_nodes,
                done_mask=done_mask,
                num_steps=num_steps,
            )
            step = compute_constrained_forward_step(
                policy=policy,
                prepared_batch=prepared_batch,
                state=search_state,
                max_steps=self.max_steps,
            )
            distribution = step.distribution
            has_values = step.has_values.view_as(current_nodes)
            dead_end = active_mask & (~has_values)
            done_mask = done_mask | dead_end
            movable_mask = active_mask & has_values
            if not bool(movable_mask.any().item()):
                continue

            chosen_edge_ids, chosen_target_nodes, chosen_log_probs = _sample_edges(
                edge_ids=distribution.edge_ids,
                target_nodes=distribution.target_nodes,
                out_degrees=distribution.out_degrees.view(-1),
                edge_logits=distribution.edge_logits,
                temperature=float(temperature),
            )
            flat_movable = movable_mask.view(-1)
            flat_current = current_nodes.view(-1)
            flat_num_steps = num_steps.view(-1)
            flat_next_nodes = flat_current.clone()
            flat_next_nodes[flat_movable] = chosen_target_nodes[flat_movable]
            next_num_steps = flat_num_steps.clone()
            next_num_steps[flat_movable] = next_num_steps[flat_movable] + 1
            next_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=flat_next_nodes.view_as(current_nodes),
                done_mask=torch.zeros_like(done_mask),
                num_steps=next_num_steps.view_as(num_steps),
            )
            current_log_f = policy.compute_log_state_scores(
                prepared_batch, search_state
            )
            next_log_f = policy.compute_log_state_scores(prepared_batch, next_state)

            log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
            state_log_f_steps[:, :, step_idx] = current_log_f
            next_state_log_f_steps[:, :, step_idx] = next_log_f
            move_mask[:, :, step_idx] = movable_mask

            current_nodes = flat_next_nodes.view_as(current_nodes)
            num_steps = next_num_steps.view_as(num_steps)
            reached_horizon = num_steps >= int(self.max_steps)
            reached_target = terminal_target_mask.index_select(
                0, current_nodes.view(-1)
            ).view_as(current_nodes)
            done_mask = done_mask | reached_horizon | reached_target

        terminal_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=torch.zeros_like(done_mask),
            num_steps=num_steps,
        )
        terminal_state_log_f = policy.compute_log_state_scores(
            prepared_batch,
            terminal_state,
        )
        success_mask = terminal_target_mask.index_select(
            0, current_nodes.view(-1)
        ).view_as(current_nodes)
        terminal_rewards, terminal_log_rewards = (
            self.trajectory_supervisor.compute_terminal_rewards(
                batch=batch,
                terminal_nodes=current_nodes,
                success_mask=success_mask,
            )
        )
        return TrajectoryGFNSampleBatch(
            graph_log_z=graph_log_z,
            start_nodes=start_nodes,
            start_log_probs=start_log_probs.to(dtype=torch.float32),
            start_state_log_f=start_state_log_f.to(dtype=torch.float32),
            log_pf_steps=log_pf_steps,
            state_log_f_steps=state_log_f_steps,
            next_state_log_f_steps=next_state_log_f_steps,
            move_mask=move_mask,
            trace_nodes=trace_nodes,
            trace_num_steps=trace_num_steps,
            trace_mask=trace_mask,
            terminal_nodes=current_nodes,
            terminal_num_steps=num_steps,
            terminal_state_log_f=terminal_state_log_f.to(dtype=torch.float32),
            terminal_rewards=terminal_rewards,
            terminal_log_rewards=terminal_log_rewards,
            success_mask=success_mask,
        )


__all__ = [
    "ForwardTrajectoryGFNSampler",
    "TrajectoryRolloutSupervisorProtocol",
    "TrajectoryGFNSampleBatch",
]
