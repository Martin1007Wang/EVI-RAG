from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Sequence

import torch

from src.graph_runtime import TrajectoryBatch

from .path import append_relation_and_node_tokens, initialize_path_token_ids
from .sampler import (
    TrajectoryGFNSampleBatch,
    TrajectoryRolloutSupervisorProtocol,
    _select_edge_log_probs,
)
from .transitions import compute_constrained_policy_step
from .types import GFlowNetPolicyProtocol, PreparedGFlowNetBatch, SearchState


@dataclass(frozen=True)
class SuccessfulTrajectoryRecord:
    sample_id: str
    start_local_node: int
    local_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchReplayPlan:
    graph_indices: tuple[int, ...]
    records_by_graph: tuple[tuple[SuccessfulTrajectoryRecord, ...], ...]

    @property
    def num_trajectories(self) -> int:
        return sum(len(records) for records in self.records_by_graph)


def _edge_offsets(batch: TrajectoryBatch) -> torch.Tensor:
    edge_counts = torch.bincount(batch.edge_batch, minlength=batch.num_graphs)
    return edge_counts.cumsum(0) - edge_counts


class SuccessfulTrajectoryReplayBuffer:
    def __init__(
        self,
        *,
        max_buffer_size: int,
        max_trajectories_per_sample: int,
    ) -> None:
        self.max_buffer_size = int(max_buffer_size)
        self.max_trajectories_per_sample = int(max_trajectories_per_sample)
        self._lock = RLock()
        self._records_by_sample: dict[str, list[SuccessfulTrajectoryRecord]] = {}
        self._fifo: list[tuple[str, SuccessfulTrajectoryRecord]] = []
        self._size = 0

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def _append_record(self, record: SuccessfulTrajectoryRecord) -> bool:
        records = self._records_by_sample.setdefault(record.sample_id, [])
        if record in records:
            return False
        records.append(record)
        self._fifo.append((record.sample_id, record))
        self._size += 1
        while len(records) > self.max_trajectories_per_sample:
            dropped = records.pop(0)
            self._size -= 1
            if not records:
                self._records_by_sample.pop(record.sample_id, None)
        self._trim_to_capacity()
        return True

    def _trim_to_capacity(self) -> None:
        while self._size > self.max_buffer_size and self._fifo:
            sample_id, record = self._fifo.pop(0)
            records = self._records_by_sample.get(sample_id)
            if records is None:
                continue
            try:
                records.remove(record)
            except ValueError:
                continue
            self._size -= 1
            if not records:
                self._records_by_sample.pop(sample_id, None)

    def add_successes(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> int:
        node_offsets = batch.node_ptr[:-1]
        edge_offsets = _edge_offsets(batch)
        added = 0
        success_positions = torch.nonzero(sample_batch.success_mask, as_tuple=False)
        with self._lock:
            for graph_rollout in success_positions.tolist():
                graph_idx, rollout_idx = int(graph_rollout[0]), int(graph_rollout[1])
                sample_id = str(batch.sample_ids[graph_idx])
                start_local_node = int(
                    sample_batch.start_nodes[graph_idx, rollout_idx].item()
                    - node_offsets[graph_idx].item()
                )
                num_steps = int(
                    sample_batch.terminal_num_steps[graph_idx, rollout_idx].item()
                )
                edge_ids = sample_batch.trace_edge_ids[
                    graph_idx, rollout_idx, :num_steps
                ]
                local_edge_ids = tuple(
                    int(edge_id.item() - edge_offsets[graph_idx].item())
                    for edge_id in edge_ids
                )
                record = SuccessfulTrajectoryRecord(
                    sample_id=sample_id,
                    start_local_node=start_local_node,
                    local_edge_ids=local_edge_ids,
                )
                added += int(self._append_record(record))
        return added

    def plan_for_batch(
        self,
        *,
        batch: TrajectoryBatch,
        replay_rollouts_per_graph: int,
    ) -> BatchReplayPlan | None:
        with self._lock:
            if replay_rollouts_per_graph < 1 or self._size < 1:
                return None

            graph_indices: list[int] = []
            records_by_graph: list[tuple[SuccessfulTrajectoryRecord, ...]] = []
            for graph_idx, sample_id in enumerate(batch.sample_ids):
                records = self._records_by_sample.get(str(sample_id))
                if not records:
                    continue
                if len(records) >= replay_rollouts_per_graph:
                    perm = torch.randperm(len(records))[
                        :replay_rollouts_per_graph
                    ].tolist()
                    chosen = tuple(records[idx] for idx in perm)
                else:
                    chosen = tuple(
                        records[int(torch.randint(len(records), (1,)).item())]
                        for _ in range(replay_rollouts_per_graph)
                    )
                graph_indices.append(int(graph_idx))
                records_by_graph.append(chosen)

        if not graph_indices:
            return None
        return BatchReplayPlan(
            graph_indices=tuple(graph_indices),
            records_by_graph=tuple(records_by_graph),
        )


def _resolve_start_distribution_values(
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
        graph_log_prob_lookup = {
            int(node.item()): float(log_prob.item())
            for node, log_prob in zip(
                start_distribution.candidate_nodes_abs[mask],
                start_distribution.log_probs[mask],
            )
        }
        graph_log_flow_lookup = {
            int(node.item()): float(log_flow.item())
            for node, log_flow in zip(
                start_distribution.candidate_nodes_abs[mask],
                start_distribution.log_flows[mask],
            )
        }
        log_prob_lookup.append(graph_log_prob_lookup)
        log_flow_lookup.append(graph_log_flow_lookup)

    start_log_probs = torch.zeros_like(start_nodes, dtype=torch.float32)
    start_log_flows = torch.zeros_like(start_nodes, dtype=torch.float32)
    for graph_idx in range(int(start_nodes.size(0))):
        graph_log_probs = log_prob_lookup[graph_idx]
        graph_log_flows = log_flow_lookup[graph_idx]
        for rollout_idx in range(int(start_nodes.size(1))):
            node_id = int(start_nodes[graph_idx, rollout_idx].item())
            if node_id not in graph_log_probs or node_id not in graph_log_flows:
                raise ValueError(
                    "Replay trajectory start node is not a valid start candidate under the current batch. "
                    f"graph_idx={graph_idx} node_id={node_id}."
                )
            start_log_probs[graph_idx, rollout_idx] = graph_log_probs[node_id]
            start_log_flows[graph_idx, rollout_idx] = graph_log_flows[node_id]
    return (
        start_log_probs,
        start_log_flows,
        start_distribution.graph_log_z.to(dtype=torch.float32),
    )


def build_replay_sample_batch(
    *,
    batch: TrajectoryBatch,
    policy: GFlowNetPolicyProtocol,
    prepared_batch: PreparedGFlowNetBatch,
    trajectory_supervisor: TrajectoryRolloutSupervisorProtocol,
    replay_records: Sequence[Sequence[SuccessfulTrajectoryRecord]],
    max_steps: int,
) -> TrajectoryGFNSampleBatch:
    if int(batch.num_graphs) != len(replay_records):
        raise ValueError(
            "Replay records must align with replay batch graphs. "
            f"num_graphs={batch.num_graphs} len(records)={len(replay_records)}."
        )
    if max_steps < 1:
        raise ValueError("Replay max_steps must be >= 1.")
    num_rollouts = len(replay_records[0]) if replay_records else 0
    if num_rollouts < 1:
        raise ValueError("Replay requires at least one trajectory per graph.")
    if any(len(records) != num_rollouts for records in replay_records):
        raise ValueError(
            "Replay requires a fixed rollout count for every selected graph."
        )

    device = batch.node_ptr.device
    node_offsets = batch.node_ptr[:-1].view(-1, 1)
    edge_offsets = _edge_offsets(batch).view(-1, 1)
    planned_edge_ids = torch.full(
        (batch.num_graphs, num_rollouts, max_steps),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    start_nodes = torch.zeros(
        (batch.num_graphs, num_rollouts),
        device=device,
        dtype=torch.long,
    )
    path_lengths = torch.zeros_like(start_nodes)
    for graph_idx, records in enumerate(replay_records):
        for rollout_idx, record in enumerate(records):
            local_edge_ids = tuple(int(edge_id) for edge_id in record.local_edge_ids)
            if len(local_edge_ids) > max_steps:
                raise ValueError(
                    "Replay trajectory exceeds the configured horizon. "
                    f"sample_id={record.sample_id!r} len={len(local_edge_ids)} max_steps={max_steps}."
                )
            start_nodes[graph_idx, rollout_idx] = (
                int(record.start_local_node) + node_offsets[graph_idx, 0]
            )
            path_lengths[graph_idx, rollout_idx] = int(len(local_edge_ids))
            if local_edge_ids:
                planned_edge_ids[graph_idx, rollout_idx, : len(local_edge_ids)] = (
                    torch.tensor(local_edge_ids, device=device, dtype=torch.long)
                    + edge_offsets[graph_idx, 0]
                )

    start_log_probs, start_log_flows, graph_log_z = _resolve_start_distribution_values(
        prepared_batch=prepared_batch,
        policy=policy,
        start_nodes=start_nodes,
    )
    start_state_log_f = start_log_flows.to(dtype=torch.float32)
    terminal_target_mask = trajectory_supervisor.build_terminal_target_mask(batch=batch)

    log_pf_steps = torch.zeros(
        (batch.num_graphs, num_rollouts, max_steps),
        device=device,
        dtype=torch.float32,
    )
    log_pb_steps = torch.zeros_like(log_pf_steps)
    state_log_f_steps = torch.zeros_like(log_pf_steps)
    next_state_log_f_steps = torch.zeros_like(log_pf_steps)
    move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)
    trace_nodes = torch.zeros_like(planned_edge_ids)
    trace_edge_ids = torch.full_like(planned_edge_ids, fill_value=-1)
    trace_num_steps = torch.zeros_like(planned_edge_ids)
    trace_mask = torch.zeros_like(move_mask)

    current_nodes = start_nodes.clone()
    num_steps = torch.zeros_like(start_nodes)
    current_path_token_ids = initialize_path_token_ids(
        start_nodes=start_nodes,
        max_steps=max_steps,
    )
    total_agents = int(batch.num_graphs * num_rollouts)

    for step_idx in range(max_steps):
        active_mask = path_lengths > step_idx
        trace_nodes[:, :, step_idx] = current_nodes
        trace_num_steps[:, :, step_idx] = num_steps
        trace_mask[:, :, step_idx] = active_mask
        if not bool(active_mask.any().item()):
            break

        search_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=~active_mask,
            num_steps=num_steps,
            path_token_ids=current_path_token_ids,
        )
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
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
            (total_agents,), device=device, dtype=torch.float32
        )
        chosen_log_pb = torch.zeros_like(chosen_log_probs)

        selected_nodes, selected_log_probs = _select_edge_log_probs(
            distribution=step.distribution,
            selected_edge_ids=chosen_edge_ids,
            active_mask=flat_active,
            policy=policy,
            error_prefix=(f"Replay trajectory step={step_idx}"),
        )
        chosen_target_nodes[flat_active] = selected_nodes[flat_active]
        chosen_log_probs[flat_active] = selected_log_probs[flat_active]

        next_nodes = flat_current_nodes.clone()
        next_nodes[flat_active] = chosen_target_nodes[flat_active]
        next_num_steps = flat_num_steps.clone()
        next_num_steps[flat_active] = next_num_steps[flat_active] + 1
        safe_edge_ids = chosen_edge_ids.clamp(min=0)
        chosen_relation_ids = prepared_batch.topology.edge_type.index_select(
            0, safe_edge_ids
        )
        next_path_token_ids = append_relation_and_node_tokens(
            path_token_ids=current_path_token_ids,
            num_steps=num_steps,
            relation_ids=chosen_relation_ids.view_as(current_nodes),
            target_nodes=next_nodes.view_as(current_nodes),
            active_mask=active_mask,
        )
        next_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=next_nodes.view_as(current_nodes),
            done_mask=torch.zeros_like(active_mask),
            num_steps=next_num_steps.view_as(num_steps),
            path_token_ids=next_path_token_ids,
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
            error_prefix=(f"Replay trajectory backward reconstruction step={step_idx}"),
        )
        chosen_log_pb[flat_active] = selected_log_pb[flat_active]

        log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
        log_pb_steps[:, :, step_idx] = chosen_log_pb.view_as(current_nodes)
        state_log_f_steps[:, :, step_idx] = current_log_f
        next_state_log_f_steps[:, :, step_idx] = next_log_f
        move_mask[:, :, step_idx] = active_mask
        trace_edge_ids[:, :, step_idx] = chosen_edge_ids.view_as(current_nodes)
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)
        current_path_token_ids = next_path_token_ids

    terminal_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=current_nodes,
        done_mask=torch.zeros_like(num_steps, dtype=torch.bool),
        num_steps=path_lengths,
        path_token_ids=current_path_token_ids,
    )
    terminal_state_log_f = policy.compute_log_state_scores(
        prepared_batch, terminal_state
    )
    success_mask = terminal_target_mask.index_select(0, current_nodes.view(-1)).view_as(
        current_nodes
    )
    if not bool(success_mask.all().item()):
        raise ValueError(
            "Replay buffer produced a trajectory that no longer lands on an answer node."
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
        start_state_log_f=start_state_log_f.to(dtype=torch.float32),
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


__all__ = [
    "BatchReplayPlan",
    "SuccessfulTrajectoryRecord",
    "SuccessfulTrajectoryReplayBuffer",
    "build_replay_sample_batch",
]
