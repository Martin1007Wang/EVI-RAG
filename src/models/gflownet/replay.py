from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Sequence

import torch

from src.graph_runtime import TrajectoryBatch

from .path import (
    append_relation_and_node_tokens_inplace,
    count_path_node_revisits,
    initialize_path_token_ids,
)
from .success_paths import (
    collect_success_rollout_key_rows,
    compute_edge_offsets,
    deduplicate_success_rollout_key_rows,
)
from .sampler import (
    TerminalTransitionBatch,
    TrajectoryGFNSampleBatch,
    TrajectoryRolloutSupervisorProtocol,
    _apply_terminal_submit_backward_log_probs,
    _mask_terminal_submit_backward_log_probs,
    _resolve_chosen_relation_ids,
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


def _extract_success_records(
    *,
    batch: TrajectoryBatch,
    sample_batch: TrajectoryGFNSampleBatch,
) -> list[SuccessfulTrajectoryRecord]:
    success_path_rows = collect_success_rollout_key_rows(
        batch=batch,
        sample_batch=sample_batch,
    )
    unique_rows = deduplicate_success_rollout_key_rows(success_path_rows)
    if unique_rows is None:
        return []
    unique_rows_cpu = unique_rows.detach().to(device="cpu", dtype=torch.long)
    records: list[SuccessfulTrajectoryRecord] = []
    for row in unique_rows_cpu.tolist():
        graph_idx = int(row[0])
        step_count = int(row[3])
        records.append(
            SuccessfulTrajectoryRecord(
                sample_id=str(batch.sample_ids[graph_idx]),
                start_local_node=int(row[1]),
                local_edge_ids=tuple(
                    int(edge_id)
                    for edge_id in row[4 : 4 + step_count]
                    if int(edge_id) >= 0
                ),
            )
        )
    return records


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
        self._record_set_by_sample: dict[str, set[SuccessfulTrajectoryRecord]] = {}
        self._fifo: list[tuple[str, SuccessfulTrajectoryRecord]] = []
        self._size = 0

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def _append_record(self, record: SuccessfulTrajectoryRecord) -> bool:
        records = self._records_by_sample.setdefault(record.sample_id, [])
        record_set = self._record_set_by_sample.setdefault(record.sample_id, set())
        if record in record_set:
            return False
        records.append(record)
        record_set.add(record)
        self._fifo.append((record.sample_id, record))
        self._size += 1
        while len(records) > self.max_trajectories_per_sample:
            dropped = records.pop(0)
            record_set.discard(dropped)
            self._size -= 1
            if not records:
                self._records_by_sample.pop(record.sample_id, None)
                self._record_set_by_sample.pop(record.sample_id, None)
        self._trim_to_capacity()
        return True

    def _trim_to_capacity(self) -> None:
        while self._size > self.max_buffer_size and self._fifo:
            sample_id, record = self._fifo.pop(0)
            records = self._records_by_sample.get(sample_id)
            record_set = self._record_set_by_sample.get(sample_id)
            if records is None:
                continue
            try:
                records.remove(record)
            except ValueError:
                continue
            if record_set is not None:
                record_set.discard(record)
            self._size -= 1
            if not records:
                self._records_by_sample.pop(sample_id, None)
                self._record_set_by_sample.pop(sample_id, None)

    def add_successes(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> int:
        added = 0
        success_records = _extract_success_records(
            batch=batch, sample_batch=sample_batch
        )
        with self._lock:
            for record in success_records:
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
    start_log_probs = torch.zeros_like(start_nodes, dtype=torch.float32)
    start_log_flows = torch.zeros_like(start_nodes, dtype=torch.float32)
    num_graphs, num_rollouts = start_nodes.shape
    if num_graphs == 0 or num_rollouts == 0:
        return (
            start_log_probs,
            start_log_flows,
            start_distribution.graph_log_z.to(dtype=torch.float32),
        )
    if int(start_distribution.candidate_nodes_abs.numel()) == 0:
        raise ValueError(
            "Replay trajectory start node is not a valid start candidate under the current batch. "
            "The batch has no start candidates."
        )
    flat_graph_ids = (
        torch.arange(num_graphs, device=start_nodes.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(start_nodes)
        .reshape(-1)
    )
    flat_start_nodes = start_nodes.reshape(-1)
    lookup_base = (
        torch.maximum(
            start_distribution.candidate_nodes_abs.to(dtype=torch.long).max(),
            flat_start_nodes.to(dtype=torch.long).max(),
        )
        + 1
    )
    candidate_keys = (
        start_distribution.candidate_graph_ids.to(dtype=torch.long) * lookup_base
    ) + start_distribution.candidate_nodes_abs.to(dtype=torch.long)
    sorted_keys, order = torch.sort(candidate_keys)
    selected_keys = (flat_graph_ids * lookup_base) + flat_start_nodes.to(
        dtype=torch.long
    )
    match_idx = torch.searchsorted(sorted_keys, selected_keys)
    in_range = match_idx < int(sorted_keys.numel())
    exact_match = torch.zeros_like(in_range)
    exact_match[in_range] = (
        sorted_keys.index_select(0, match_idx[in_range]) == selected_keys[in_range]
    )
    if not bool(exact_match.all().item()):
        invalid_graphs = flat_graph_ids[~exact_match].tolist()
        invalid_nodes = flat_start_nodes[~exact_match].tolist()
        raise ValueError(
            "Replay trajectory start node is not a valid start candidate under the current batch. "
            f"graph_idx={invalid_graphs} node_ids={invalid_nodes}."
        )
    selected_positions = order.index_select(0, match_idx)
    start_log_probs = (
        start_distribution.log_probs.to(dtype=torch.float32)
        .index_select(0, selected_positions)
        .view_as(start_nodes)
    )
    start_log_flows = (
        start_distribution.log_flows.to(dtype=torch.float32)
        .index_select(0, selected_positions)
        .view_as(start_nodes)
    )
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
    edge_offsets = compute_edge_offsets(batch).view(-1, 1)
    max_actions = max_steps + 1
    planned_edge_ids = torch.full(
        (batch.num_graphs, num_rollouts, max_actions),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    planned_submit_mask = torch.zeros(
        (batch.num_graphs, num_rollouts, max_actions),
        device=device,
        dtype=torch.bool,
    )
    start_nodes = torch.zeros(
        (batch.num_graphs, num_rollouts),
        device=device,
        dtype=torch.long,
    )
    path_lengths = torch.zeros_like(start_nodes)
    terminal_action_counts = torch.zeros_like(start_nodes)
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
            terminal_action_counts[graph_idx, rollout_idx] = (
                int(len(local_edge_ids)) + 1
            )
            if local_edge_ids:
                planned_edge_ids[graph_idx, rollout_idx, : len(local_edge_ids)] = (
                    torch.tensor(local_edge_ids, device=device, dtype=torch.long)
                    + edge_offsets[graph_idx, 0]
                )
            planned_submit_mask[graph_idx, rollout_idx, len(local_edge_ids)] = True

    start_log_probs, start_log_flows, graph_log_z = _resolve_start_distribution_values(
        prepared_batch=prepared_batch,
        policy=policy,
        start_nodes=start_nodes,
    )
    start_state_log_f = start_log_flows.to(dtype=torch.float32)
    terminal_target_mask = trajectory_supervisor.build_terminal_target_mask(batch=batch)

    log_pf_steps = torch.zeros(
        (batch.num_graphs, num_rollouts, max_actions),
        device=device,
        dtype=torch.float32,
    )
    log_pb_steps = torch.zeros_like(log_pf_steps)
    next_state_log_f_steps = torch.zeros_like(log_pf_steps)
    move_mask = torch.zeros_like(log_pf_steps, dtype=torch.bool)
    trace_nodes = torch.zeros_like(planned_edge_ids)
    trace_edge_ids = torch.full_like(planned_edge_ids, fill_value=-1)
    trace_num_steps = torch.zeros_like(planned_edge_ids)
    trace_mask = torch.zeros_like(move_mask)
    trace_submit_mask = planned_submit_mask

    current_nodes = start_nodes.clone()
    num_steps = torch.zeros_like(start_nodes)
    current_path_token_ids = initialize_path_token_ids(
        start_nodes=start_nodes,
        max_steps=max_steps,
    )
    current_control_states = policy.build_start_control_states(
        prepared_batch,
        start_nodes,
    )
    total_agents = int(batch.num_graphs * num_rollouts)

    for step_idx in range(max_actions):
        active_mask = terminal_action_counts > step_idx
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
            control_state=current_control_states,
        )
        chosen_edge_ids = planned_edge_ids[:, :, step_idx].reshape(-1)
        chosen_is_submit = planned_submit_mask[:, :, step_idx].reshape(-1)
        # Replay keeps the recorded edge in the shortlist support so SubTB can
        # reconstruct a finite log-prob for the teacher-forced transition.
        step = compute_constrained_policy_step(
            policy=policy,
            prepared_batch=prepared_batch,
            state=search_state,
            max_steps=max_steps,
            required_edge_ids=chosen_edge_ids,
        )
        current_log_f = step.distribution.current_log_f
        if current_log_f is None:
            current_log_f = policy.compute_log_state_scores(
                prepared_batch, search_state
            )
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
            selected_is_submit=chosen_is_submit,
            active_mask=flat_active,
            policy=policy,
            error_prefix=(f"Replay trajectory step={step_idx}"),
        )
        chosen_target_nodes[flat_active] = selected_nodes[flat_active]
        chosen_log_probs[flat_active] = selected_log_probs[flat_active]

        flat_graph_move = flat_active & (~chosen_is_submit)
        flat_submit = flat_active & chosen_is_submit
        next_nodes = flat_current_nodes.clone()
        next_nodes[flat_graph_move] = chosen_target_nodes[flat_graph_move]
        next_num_steps = flat_num_steps.clone()
        next_num_steps[flat_graph_move] = next_num_steps[flat_graph_move] + 1
        chosen_relation_ids = _resolve_chosen_relation_ids(
            edge_type=prepared_batch.topology.edge_type,
            chosen_edge_ids=chosen_edge_ids,
            view_shape=current_nodes.shape,
        )
        next_path_token_ids = append_relation_and_node_tokens_inplace(
            path_token_ids=current_path_token_ids,
            num_steps=num_steps,
            relation_ids=chosen_relation_ids,
            target_nodes=next_nodes.view_as(current_nodes),
            active_mask=flat_graph_move.view_as(active_mask),
        )
        next_control_states = current_control_states.clone()
        if bool(flat_graph_move.any().item()):
            flat_next_control_states = next_control_states.view(
                -1, int(next_control_states.size(-1))
            )
            flat_current_control_states = current_control_states.view(
                -1, int(current_control_states.size(-1))
            )
            flat_relation_ids = chosen_relation_ids.view(-1)
            flat_next_control_states[flat_graph_move] = (
                policy.compute_next_control_states(
                    prepared_batch,
                    control_states=flat_current_control_states[flat_graph_move],
                    next_nodes=next_nodes[flat_graph_move],
                    relation_ids=flat_relation_ids[flat_graph_move],
                )
            )
        next_log_f = torch.zeros_like(current_log_f)
        if bool(flat_graph_move.any().item()):
            next_state = SearchState(
                topology=prepared_batch.topology,
                observation=prepared_batch.observation,
                current_nodes=next_nodes.view_as(current_nodes),
                done_mask=torch.zeros_like(active_mask),
                num_steps=next_num_steps.view_as(num_steps),
                path_token_ids=next_path_token_ids,
                control_state=next_control_states,
            )
            next_log_f = policy.compute_log_state_scores(prepared_batch, next_state)
            backward_distribution = policy.compute_backward_distribution(
                prepared_batch,
                next_state,
            )
            _, selected_log_pb = _select_edge_log_probs(
                distribution=backward_distribution,
                selected_edge_ids=chosen_edge_ids,
                selected_is_submit=torch.zeros_like(chosen_is_submit),
                active_mask=flat_graph_move,
                policy=policy,
                error_prefix=(
                    f"Replay trajectory backward reconstruction step={step_idx}"
                ),
            )
            chosen_log_pb[flat_graph_move] = selected_log_pb[flat_graph_move]
        chosen_log_pb[flat_submit] = 0.0

        log_pf_steps[:, :, step_idx] = chosen_log_probs.view_as(current_nodes)
        log_pb_steps[:, :, step_idx] = chosen_log_pb.view_as(current_nodes)
        next_state_log_f_steps[:, :, step_idx] = next_log_f
        move_mask[:, :, step_idx] = active_mask
        trace_edge_ids[:, :, step_idx] = chosen_edge_ids.view_as(current_nodes)
        current_nodes = next_nodes.view_as(current_nodes)
        num_steps = next_num_steps.view_as(num_steps)
        current_path_token_ids = next_path_token_ids
        current_control_states = next_control_states

    success_mask = terminal_target_mask.index_select(0, current_nodes.view(-1)).view_as(
        current_nodes
    )
    if not bool(success_mask.all().item()):
        raise ValueError(
            "Replay buffer produced a trajectory that no longer lands on an answer node."
        )
    terminal_cycle_counts = count_path_node_revisits(
        path_token_ids=current_path_token_ids,
        num_steps=path_lengths,
    )
    terminal_transition: TerminalTransitionBatch = (
        trajectory_supervisor.resolve_terminal_transitions(
            batch=batch,
            terminal_nodes=current_nodes,
            success_mask=success_mask,
            terminal_num_steps=path_lengths,
            terminal_cycle_counts=terminal_cycle_counts,
        )
    )
    masked_terminal_backward_log_probs = _mask_terminal_submit_backward_log_probs(
        terminal_action_counts=terminal_action_counts,
        terminal_num_steps=path_lengths,
        terminal_backward_log_probs=terminal_transition.terminal_backward_log_probs,
    )
    log_pb_steps = _apply_terminal_submit_backward_log_probs(
        log_pb_steps=log_pb_steps,
        terminal_action_counts=terminal_action_counts,
        terminal_num_steps=path_lengths,
        terminal_backward_log_probs=masked_terminal_backward_log_probs,
    )
    return TrajectoryGFNSampleBatch(
        graph_log_z=graph_log_z,
        start_nodes=start_nodes,
        start_log_probs=start_log_probs,
        start_state_log_f=start_state_log_f.to(dtype=torch.float32),
        log_pf_steps=log_pf_steps,
        log_pb_steps=log_pb_steps,
        state_log_f_steps=None,
        next_state_log_f_steps=next_state_log_f_steps,
        move_mask=move_mask,
        trace_nodes=trace_nodes,
        trace_edge_ids=trace_edge_ids,
        trace_num_steps=trace_num_steps,
        trace_mask=trace_mask,
        trace_submit_mask=trace_submit_mask,
        terminal_nodes=current_nodes,
        terminal_entity_ids=terminal_transition.terminal_entity_ids,
        terminal_num_steps=path_lengths,
        terminal_action_counts=terminal_action_counts,
        terminal_state_log_f=None,
        terminal_rewards=terminal_transition.terminal_rewards,
        terminal_log_rewards=terminal_transition.terminal_log_rewards,
        terminal_backward_log_probs=masked_terminal_backward_log_probs,
        success_mask=success_mask,
    )


__all__ = [
    "BatchReplayPlan",
    "SuccessfulTrajectoryRecord",
    "SuccessfulTrajectoryReplayBuffer",
    "build_replay_sample_batch",
]
