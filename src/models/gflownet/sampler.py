from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.graph_runtime import TrajectoryBatch
from .transitions import apply_forward_constraints
from .types import (
    ForwardActionDistribution,
    GFlowNetPolicyProtocol,
    PreparedGFlowNetBatch,
    SearchState,
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


def _build_answer_mask(batch: TrajectoryBatch) -> torch.Tensor:
    answer_mask = torch.zeros(
        (batch.num_nodes_total,), device=batch.node_ptr.device, dtype=torch.bool
    )
    if int(batch.a_local_indices.numel()) == 0:
        return answer_mask
    counts = batch.a_ptr[1:] - batch.a_ptr[:-1]
    offsets = batch.node_ptr[:-1].repeat_interleave(counts)
    absolute = batch.a_local_indices + offsets
    answer_mask.scatter_(0, absolute, True)
    return answer_mask


class AnswerReachabilityTrajectorySupervisor:
    def __init__(self, *, epsilon: float, failure_reward_mode: str) -> None:
        self.epsilon = float(epsilon)
        self.failure_reward_mode = str(failure_reward_mode)
        if self.failure_reward_mode not in {"constant", "graph_normalized"}:
            raise ValueError(
                "failure_reward_mode must be one of {'constant', 'graph_normalized'}."
            )

    def build_terminal_target_mask(self, *, batch: TrajectoryBatch) -> torch.Tensor:
        return _build_answer_mask(batch)

    def compute_terminal_rewards(
        self,
        *,
        batch: TrajectoryBatch,
        terminal_nodes: torch.Tensor,
        success_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.full(
            terminal_nodes.shape,
            fill_value=self.epsilon,
            device=terminal_nodes.device,
            dtype=torch.float32,
        )
        if self.failure_reward_mode == "graph_normalized":
            answer_counts = (batch.a_ptr[1:] - batch.a_ptr[:-1]).to(dtype=torch.float32)
            non_answer_counts = (
                (batch.node_ptr[1:] - batch.node_ptr[:-1]).to(dtype=torch.float32)
                - answer_counts
            ).clamp_min(1.0)
            graph_non_answer_counts = non_answer_counts.to(
                device=terminal_nodes.device
            ).unsqueeze(1)
            rewards = self.epsilon / graph_non_answer_counts.expand_as(terminal_nodes)
        rewards = torch.where(success_mask, torch.ones_like(rewards), rewards)
        return rewards, rewards.clamp_min(1.0e-12).log()


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


def _select_edge_log_probs(
    *,
    distribution: ForwardActionDistribution,
    selected_edge_ids: torch.Tensor,
    active_mask: torch.Tensor,
    policy: GFlowNetPolicyProtocol,
    error_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_nodes = torch.zeros_like(selected_edge_ids)
    selected_log_probs = torch.zeros(
        selected_edge_ids.shape,
        device=distribution.edge_logits.device,
        dtype=torch.float32,
    )
    move_log_probs, _, _ = policy.compute_move_log_probs(distribution)
    active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
    for agent_idx in active_indices:
        edge_id = int(selected_edge_ids[agent_idx].item())
        if edge_id < 0:
            raise ValueError(
                f"{error_prefix} is missing an edge id for an active step. "
                f"agent_idx={agent_idx}."
            )
        edge_mask = (distribution.edge_agent_batch == agent_idx) & (
            distribution.edge_ids == edge_id
        )
        if int(edge_mask.sum().item()) != 1:
            raise ValueError(
                f"{error_prefix} edge is invalid under the current policy state. "
                f"agent_idx={agent_idx} edge_id={edge_id}."
            )
        edge_position = int(torch.nonzero(edge_mask, as_tuple=False)[0].item())
        selected_nodes[agent_idx] = int(distribution.target_nodes[edge_position].item())
        selected_log_probs[agent_idx] = move_log_probs[edge_position]
    return selected_nodes, selected_log_probs


def _resolve_selected_start_values(
    *,
    prepared_batch: PreparedGFlowNetBatch,
    policy: GFlowNetPolicyProtocol,
    start_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_distribution = policy.compute_start_distribution(prepared_batch)
    log_prob_lookup: list[dict[int, float]] = []
    log_flow_lookup: list[dict[int, float]] = []
    num_graphs = int(prepared_batch.topology.num_graphs)
    for graph_idx in range(num_graphs):
        mask = start_distribution.candidate_graph_ids == graph_idx
        log_prob_lookup.append(
            {
                int(node.item()): float(log_prob.item())
                for node, log_prob in zip(
                    start_distribution.candidate_nodes_abs[mask],
                    start_distribution.log_probs[mask],
                )
            }
        )
        log_flow_lookup.append(
            {
                int(node.item()): float(log_flow.item())
                for node, log_flow in zip(
                    start_distribution.candidate_nodes_abs[mask],
                    start_distribution.log_flows[mask],
                )
            }
        )

    start_log_probs = torch.zeros_like(start_nodes, dtype=torch.float32)
    start_log_flows = torch.zeros_like(start_nodes, dtype=torch.float32)
    for graph_idx in range(int(start_nodes.size(0))):
        graph_log_probs = log_prob_lookup[graph_idx]
        graph_log_flows = log_flow_lookup[graph_idx]
        for rollout_idx in range(int(start_nodes.size(1))):
            node_id = int(start_nodes[graph_idx, rollout_idx].item())
            if node_id not in graph_log_probs or node_id not in graph_log_flows:
                raise ValueError(
                    "Sampled start node is not a valid target-policy start candidate. "
                    f"graph_idx={graph_idx} node_id={node_id}."
                )
            start_log_probs[graph_idx, rollout_idx] = graph_log_probs[node_id]
            start_log_flows[graph_idx, rollout_idx] = graph_log_flows[node_id]
    return (
        start_log_probs,
        start_log_flows,
        start_distribution.graph_log_z.to(dtype=torch.float32),
    )


def _rebuild_target_sample_batch(
    *,
    batch: TrajectoryBatch,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    trajectory_supervisor: TrajectoryRolloutSupervisorProtocol,
    start_nodes: torch.Tensor,
    planned_edge_ids: torch.Tensor,
    path_lengths: torch.Tensor,
    trace_nodes: torch.Tensor,
    trace_edge_ids: torch.Tensor,
    trace_num_steps: torch.Tensor,
    trace_mask: torch.Tensor,
    max_steps: int,
) -> TrajectoryGFNSampleBatch:
    start_log_probs, start_log_flows, graph_log_z = _resolve_selected_start_values(
        prepared_batch=prepared_batch,
        policy=policy,
        start_nodes=start_nodes,
    )
    terminal_target_mask = trajectory_supervisor.build_terminal_target_mask(batch=batch)

    log_pf_steps = torch.zeros(
        (batch.num_graphs, int(start_nodes.size(1)), max_steps),
        device=batch.node_ptr.device,
        dtype=torch.float32,
    )
    log_pb_steps = torch.zeros_like(log_pf_steps)
    state_log_f_steps = torch.zeros_like(log_pf_steps)
    next_state_log_f_steps = torch.zeros_like(log_pf_steps)
    move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)

    current_nodes = start_nodes.clone()
    num_steps = torch.zeros_like(start_nodes)
    total_agents = int(batch.num_graphs * int(start_nodes.size(1)))

    for step_idx in range(max_steps):
        active_mask = path_lengths > step_idx
        if not bool(active_mask.any().item()):
            break

        search_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=~active_mask,
            num_steps=num_steps,
        )
        distribution = apply_forward_constraints(
            policy.compute_forward_distribution(prepared_batch, search_state),
            state=search_state,
            max_steps=max_steps,
        )
        current_log_f = policy.compute_log_state_scores(prepared_batch, search_state)
        chosen_edge_ids = planned_edge_ids[:, :, step_idx].reshape(-1)
        flat_active = active_mask.reshape(-1)
        flat_current_nodes = current_nodes.reshape(-1)
        flat_num_steps = num_steps.reshape(-1)
        chosen_target_nodes = flat_current_nodes.clone()
        chosen_log_probs = torch.zeros(
            (total_agents,),
            device=batch.node_ptr.device,
            dtype=torch.float32,
        )
        chosen_log_pb = torch.zeros_like(chosen_log_probs)

        selected_nodes, selected_log_probs = _select_edge_log_probs(
            distribution=distribution,
            selected_edge_ids=chosen_edge_ids,
            active_mask=flat_active,
            policy=policy,
            error_prefix=(f"Sampled trajectory step={step_idx}"),
        )
        chosen_target_nodes[flat_active] = selected_nodes[flat_active]
        chosen_log_probs[flat_active] = selected_log_probs[flat_active]

        next_nodes = flat_current_nodes.clone()
        next_nodes[flat_active] = chosen_target_nodes[flat_active]
        next_num_steps = flat_num_steps.clone()
        next_num_steps[flat_active] = next_num_steps[flat_active] + 1
        next_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=next_nodes.view_as(current_nodes),
            done_mask=torch.zeros_like(active_mask),
            num_steps=next_num_steps.view_as(num_steps),
        )
        next_log_f = policy.compute_log_state_scores(prepared_batch, next_state)
        backward_distribution = policy.compute_backward_distribution(
            prepared_batch,
            next_state,
        )
        _, selected_log_pb = _select_edge_log_probs(
            distribution=backward_distribution,
            selected_edge_ids=chosen_edge_ids,
            active_mask=flat_active,
            policy=policy,
            error_prefix=(
                f"Sampled trajectory backward reconstruction step={step_idx}"
            ),
        )
        chosen_log_pb[flat_active] = selected_log_pb[flat_active]

        log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
        log_pb_steps[:, :, step_idx] = chosen_log_pb.view_as(current_nodes)
        state_log_f_steps[:, :, step_idx] = current_log_f
        next_state_log_f_steps[:, :, step_idx] = next_log_f
        move_mask[:, :, step_idx] = active_mask
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)

    terminal_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=current_nodes,
        done_mask=torch.zeros_like(num_steps, dtype=torch.bool),
        num_steps=path_lengths,
    )
    terminal_state_log_f = policy.compute_log_state_scores(
        prepared_batch, terminal_state
    )
    success_mask = terminal_target_mask.index_select(0, current_nodes.view(-1)).view_as(
        current_nodes
    )
    terminal_rewards, terminal_log_rewards = (
        trajectory_supervisor.compute_terminal_rewards(
            batch=batch,
            terminal_nodes=current_nodes,
            success_mask=success_mask,
        )
    )
    return TrajectoryGFNSampleBatch(
        graph_log_z=graph_log_z,
        start_nodes=start_nodes,
        start_log_probs=start_log_probs,
        start_state_log_f=start_log_flows.to(dtype=torch.float32),
        log_pf_steps=log_pf_steps,
        log_pb_steps=log_pb_steps,
        state_log_f_steps=state_log_f_steps,
        next_state_log_f_steps=next_state_log_f_steps,
        move_mask=move_mask,
        trace_nodes=trace_nodes,
        trace_edge_ids=trace_edge_ids,
        trace_num_steps=trace_num_steps,
        trace_mask=trace_mask,
        terminal_nodes=current_nodes,
        terminal_num_steps=path_lengths,
        terminal_state_log_f=terminal_state_log_f.to(dtype=torch.float32),
        terminal_rewards=terminal_rewards,
        terminal_log_rewards=terminal_log_rewards,
        success_mask=success_mask,
    )


@dataclass(frozen=True)
class TrajectoryGFNSampleBatch:
    graph_log_z: torch.Tensor
    start_nodes: torch.Tensor
    start_log_probs: torch.Tensor
    start_state_log_f: torch.Tensor
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    state_log_f_steps: torch.Tensor
    next_state_log_f_steps: torch.Tensor
    move_mask: torch.Tensor
    trace_nodes: torch.Tensor
    trace_edge_ids: torch.Tensor
    trace_num_steps: torch.Tensor
    trace_mask: torch.Tensor
    terminal_nodes: torch.Tensor
    terminal_num_steps: torch.Tensor
    terminal_state_log_f: torch.Tensor
    terminal_rewards: torch.Tensor
    terminal_log_rewards: torch.Tensor
    success_mask: torch.Tensor


class TrajectorySamplerProtocol(Protocol):
    def sample(
        self,
        *,
        batch: TrajectoryBatch,
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        rollout_batch_size: int,
        temperature: float,
    ) -> TrajectoryGFNSampleBatch: ...


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
        policy: GFlowNetPolicyProtocol,
        prepared_batch: PreparedGFlowNetBatch,
        rollout_batch_size: int,
        temperature: float,
    ) -> TrajectoryGFNSampleBatch:
        start_dist = policy.compute_behavior_start_distribution(prepared_batch)
        start_nodes, _, _ = policy.sample_start_nodes(
            start_dist,
            num_rollouts=int(rollout_batch_size),
            deterministic=False,
        )
        state = SearchState.initialize(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            start_nodes=start_nodes,
        )
        num_graphs, num_rollouts = start_nodes.shape
        terminal_target_mask = self.trajectory_supervisor.build_terminal_target_mask(
            batch=batch
        )
        trace_nodes = torch.zeros(
            (num_graphs, num_rollouts, self.max_steps),
            device=batch.node_ptr.device,
            dtype=torch.long,
        )
        trace_edge_ids = torch.full_like(trace_nodes, fill_value=-1)
        trace_num_steps = torch.zeros_like(trace_nodes)
        trace_mask = torch.zeros(
            (num_graphs, num_rollouts, self.max_steps),
            device=batch.node_ptr.device,
            dtype=torch.bool,
        )

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
            distribution = apply_forward_constraints(
                policy.compute_behavior_forward_distribution(
                    prepared_batch,
                    search_state,
                ),
                state=search_state,
                max_steps=self.max_steps,
            )
            move_log_probs, _, has_values = policy.compute_move_log_probs(distribution)
            has_values = has_values.view_as(current_nodes)
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
            flat_next_nodes = flat_current.clone()
            flat_next_nodes[flat_movable] = chosen_target_nodes[flat_movable]
            flat_num_steps = num_steps.view(-1)
            next_num_steps = flat_num_steps.clone()
            next_num_steps[flat_movable] = next_num_steps[flat_movable] + 1
            trace_edge_ids[:, :, step_idx] = chosen_edge_ids.view_as(current_nodes)

            current_nodes = flat_next_nodes.view_as(current_nodes)
            num_steps = next_num_steps.view_as(num_steps)
            reached_horizon = num_steps >= int(self.max_steps)
            reached_target = terminal_target_mask.index_select(
                0, current_nodes.view(-1)
            ).view_as(current_nodes)
            done_mask = done_mask | reached_horizon | reached_target

        return _rebuild_target_sample_batch(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
            trajectory_supervisor=self.trajectory_supervisor,
            start_nodes=start_nodes,
            planned_edge_ids=trace_edge_ids,
            path_lengths=num_steps,
            trace_nodes=trace_nodes,
            trace_edge_ids=trace_edge_ids,
            trace_num_steps=trace_num_steps,
            trace_mask=trace_mask,
            max_steps=self.max_steps,
        )


__all__ = [
    "AnswerReachabilityTrajectorySupervisor",
    "ForwardTrajectoryGFNSampler",
    "TrajectoryGFNSampleBatch",
    "TrajectoryRolloutSupervisorProtocol",
    "TrajectorySamplerProtocol",
]
