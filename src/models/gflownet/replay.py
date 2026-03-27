from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch

from src.data.preprocess.labels.edge_retrieval import (
    resolve_forward_shortest_path_trajectory,
)
from src.graph import TrajectoryBatch
from src.models.configs import SuccessReplayConfig

from .sampler import TrajectoryGFNSampleBatch


@dataclass(frozen=True)
class ReplayTrajectoryRecord:
    graph_key: str
    start_node_local: int
    planned_edge_ids_local: torch.Tensor
    planned_stop_mask: torch.Tensor
    path_length: int
    termination_action_steps: int
    trace_nodes_local: torch.Tensor
    trace_edge_ids_local: torch.Tensor
    trace_num_steps: torch.Tensor
    trace_mask: torch.Tensor
    trace_stop_mask: torch.Tensor
    signature: str


@dataclass(frozen=True)
class ReplayTrajectoryBatch:
    batch: TrajectoryBatch
    start_nodes: torch.Tensor
    planned_edge_ids: torch.Tensor
    planned_stop_mask: torch.Tensor
    path_lengths: torch.Tensor
    termination_action_steps: torch.Tensor
    trace_nodes: torch.Tensor
    trace_edge_ids: torch.Tensor
    trace_num_steps: torch.Tensor
    trace_mask: torch.Tensor
    trace_stop_mask: torch.Tensor


@dataclass(frozen=True)
class ReplayGraphPayload:
    key: str
    batch: TrajectoryBatch


def _localize_trace_nodes(
    *, trace_nodes: torch.Tensor, trace_mask: torch.Tensor, node_offset: int
) -> torch.Tensor:
    local_trace_nodes = torch.zeros_like(trace_nodes)
    active_mask = trace_mask.to(dtype=torch.bool)
    if bool(active_mask.any().item()):
        local_trace_nodes[active_mask] = trace_nodes[active_mask] - int(node_offset)
    return local_trace_nodes


def _localize_trace_edge_ids(
    *, trace_edge_ids: torch.Tensor, edge_offset: int
) -> torch.Tensor:
    return torch.where(
        trace_edge_ids >= 0,
        trace_edge_ids - int(edge_offset),
        torch.full_like(trace_edge_ids, fill_value=-1),
    )


def _signature_from_local_trajectory(
    *,
    sample_id: str,
    start_node_local: int,
    planned_edge_ids_local: torch.Tensor,
    planned_stop_mask: torch.Tensor,
    termination_action_steps: int,
) -> str:
    edge_tokens = planned_edge_ids_local.to(dtype=torch.long).cpu().tolist()
    stop_tokens = planned_stop_mask.to(dtype=torch.long).cpu().tolist()
    return (
        f"{sample_id}|start={int(start_node_local)}|term={int(termination_action_steps)}|"
        f"edges={edge_tokens}|stop={stop_tokens}"
    )


def _graph_key_from_batch(batch: TrajectoryBatch) -> str:
    return (
        f"{batch.dataset_scope}|sample={batch.sample_ids[0]}|"
        f"nodes={batch.num_nodes_total}|edges={int(batch.edge_index.size(1))}"
    )


class SuccessReplayBuffer:
    def __init__(self, *, config: SuccessReplayConfig) -> None:
        self.config = config
        self._items: deque[ReplayTrajectoryRecord] = deque()
        self._signatures: set[str] = set()
        self._graph_payloads: dict[str, ReplayGraphPayload] = {}
        self._graph_refcounts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._items)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def ready(self) -> bool:
        return len(self) >= int(self.config.min_buffer_size)

    def add_successes(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: TrajectoryGFNSampleBatch,
    ) -> int:
        if not self.enabled:
            return 0
        if sample_batch.trace_stop_mask is None:
            raise ValueError(
                "Replay buffer requires trace_stop_mask for success storage."
            )
        if sample_batch.termination_action_steps is None:
            raise ValueError(
                "Replay buffer requires termination_action_steps for success storage."
            )
        trace_stop_mask_steps = sample_batch.trace_stop_mask
        termination_action_steps_tensor = sample_batch.termination_action_steps
        success_positions = torch.nonzero(sample_batch.success_mask, as_tuple=False)
        added = 0
        edge_offsets = batch.edge_ptr[:-1].to(dtype=torch.long)
        node_offsets = batch.node_ptr[:-1].to(dtype=torch.long)
        graph_payload_cache: dict[int, ReplayGraphPayload] = {}
        if int(success_positions.numel()) > 0:
            for graph_idx_tensor, rollout_idx_tensor in success_positions:
                graph_idx = int(graph_idx_tensor.item())
                rollout_idx = int(rollout_idx_tensor.item())
                graph_payload = graph_payload_cache.get(graph_idx)
                if graph_payload is None:
                    single_graph_batch = batch.select_graph(graph_idx).to(
                        torch.device("cpu")
                    )
                    graph_payload = self._get_or_create_graph_payload(
                        single_graph_batch
                    )
                    graph_payload_cache[graph_idx] = graph_payload
                node_offset = int(node_offsets[graph_idx].item())
                edge_offset = int(edge_offsets[graph_idx].item())

                trace_mask = (
                    sample_batch.trace_mask[graph_idx, rollout_idx]
                    .detach()
                    .to(device=torch.device("cpu"), dtype=torch.bool)
                )
                trace_stop_mask = (
                    trace_stop_mask_steps[graph_idx, rollout_idx]
                    .detach()
                    .to(device=torch.device("cpu"), dtype=torch.bool)
                )
                trace_nodes_local = _localize_trace_nodes(
                    trace_nodes=sample_batch.trace_nodes[graph_idx, rollout_idx]
                    .detach()
                    .to(device=torch.device("cpu"), dtype=torch.long),
                    trace_mask=trace_mask,
                    node_offset=node_offset,
                )
                trace_edge_ids_local = _localize_trace_edge_ids(
                    trace_edge_ids=sample_batch.trace_edge_ids[graph_idx, rollout_idx]
                    .detach()
                    .to(device=torch.device("cpu"), dtype=torch.long),
                    edge_offset=edge_offset,
                )
                start_node_local = int(
                    sample_batch.start_nodes[graph_idx, rollout_idx].detach().item()
                    - node_offset
                )
                termination_action_steps = int(
                    termination_action_steps_tensor[graph_idx, rollout_idx]
                    .detach()
                    .item()
                )
                signature = _signature_from_local_trajectory(
                    sample_id=graph_payload.batch.sample_ids[0],
                    start_node_local=start_node_local,
                    planned_edge_ids_local=trace_edge_ids_local,
                    planned_stop_mask=trace_stop_mask,
                    termination_action_steps=termination_action_steps,
                )
                if bool(self.config.deduplicate) and signature in self._signatures:
                    continue

                record = ReplayTrajectoryRecord(
                    graph_key=graph_payload.key,
                    start_node_local=start_node_local,
                    planned_edge_ids_local=trace_edge_ids_local,
                    planned_stop_mask=trace_stop_mask,
                    path_length=int(
                        sample_batch.terminal_num_steps[graph_idx, rollout_idx]
                        .detach()
                        .item()
                    ),
                    termination_action_steps=termination_action_steps,
                    trace_nodes_local=trace_nodes_local,
                    trace_edge_ids_local=trace_edge_ids_local,
                    trace_num_steps=sample_batch.trace_num_steps[graph_idx, rollout_idx]
                    .detach()
                    .to(device=torch.device("cpu"), dtype=torch.long),
                    trace_mask=trace_mask,
                    trace_stop_mask=trace_stop_mask,
                    signature=signature,
                )
                self._items.append(record)
                self._signatures.add(signature)
                self._retain_graph_payload(graph_payload.key)
                added += 1
                while len(self._items) > int(self.config.capacity):
                    evicted = self._items.popleft()
                    self._signatures.discard(evicted.signature)
                    self._release_graph_payload(evicted.graph_key)
        if bool(self.config.add_shortest_path_guidance):
            added += self._add_shortest_path_guidance(
                batch=batch,
                max_actions=int(sample_batch.trace_edge_ids.size(-1)),
            )
        return added

    def _add_shortest_path_guidance(
        self,
        *,
        batch: TrajectoryBatch,
        max_actions: int,
    ) -> int:
        if max_actions < 1:
            return 0
        added = 0
        graph_payload_cache: dict[int, ReplayGraphPayload] = {}
        for graph_idx in range(batch.num_graphs):
            graph_payload = graph_payload_cache.get(graph_idx)
            if graph_payload is None:
                single_graph_batch = batch.select_graph(graph_idx).to(
                    torch.device("cpu")
                )
                graph_payload = self._get_or_create_graph_payload(single_graph_batch)
                graph_payload_cache[graph_idx] = graph_payload
            guidance = resolve_forward_shortest_path_trajectory(
                edge_index=graph_payload.batch.edge_index,
                q_local_indices=graph_payload.batch.q_local_indices,
                a_local_indices=graph_payload.batch.a_local_indices,
                num_nodes=int(graph_payload.batch.num_nodes_total),
            )
            if guidance is None:
                continue
            hop_length = int(guidance.hop_length)
            if hop_length + 1 > int(max_actions):
                continue
            planned_edge_ids_local = torch.full(
                (max_actions,), fill_value=-1, dtype=torch.long
            )
            planned_stop_mask = torch.zeros((max_actions,), dtype=torch.bool)
            trace_nodes_local = torch.zeros((max_actions,), dtype=torch.long)
            trace_edge_ids_local = torch.full(
                (max_actions,), fill_value=-1, dtype=torch.long
            )
            trace_num_steps = torch.zeros((max_actions,), dtype=torch.long)
            trace_mask = torch.zeros((max_actions,), dtype=torch.bool)
            trace_stop_mask = torch.zeros((max_actions,), dtype=torch.bool)
            if hop_length > 0:
                edge_ids_tensor = torch.as_tensor(
                    guidance.path_edge_ids, dtype=torch.long
                )
                planned_edge_ids_local[:hop_length] = edge_ids_tensor
                trace_edge_ids_local[:hop_length] = edge_ids_tensor
            path_nodes_tensor = torch.as_tensor(guidance.path_nodes, dtype=torch.long)
            trace_nodes_local[: hop_length + 1] = path_nodes_tensor
            trace_num_steps[: hop_length + 1] = torch.arange(
                hop_length + 1, dtype=torch.long
            )
            trace_mask[: hop_length + 1] = True
            planned_stop_mask[hop_length] = True
            trace_stop_mask[hop_length] = True
            termination_action_steps = hop_length + 1
            signature = _signature_from_local_trajectory(
                sample_id=graph_payload.batch.sample_ids[0],
                start_node_local=int(guidance.start_node),
                planned_edge_ids_local=planned_edge_ids_local,
                planned_stop_mask=planned_stop_mask,
                termination_action_steps=termination_action_steps,
            )
            if bool(self.config.deduplicate) and signature in self._signatures:
                continue
            record = ReplayTrajectoryRecord(
                graph_key=graph_payload.key,
                start_node_local=int(guidance.start_node),
                planned_edge_ids_local=planned_edge_ids_local,
                planned_stop_mask=planned_stop_mask,
                path_length=hop_length,
                termination_action_steps=termination_action_steps,
                trace_nodes_local=trace_nodes_local,
                trace_edge_ids_local=trace_edge_ids_local,
                trace_num_steps=trace_num_steps,
                trace_mask=trace_mask,
                trace_stop_mask=trace_stop_mask,
                signature=signature,
            )
            self._items.append(record)
            self._signatures.add(signature)
            self._retain_graph_payload(graph_payload.key)
            added += 1
            while len(self._items) > int(self.config.capacity):
                evicted = self._items.popleft()
                self._signatures.discard(evicted.signature)
                self._release_graph_payload(evicted.graph_key)
        return added

    def sample_replay_batch(
        self,
        *,
        device: torch.device,
        replay_trajectories_per_step: int,
    ) -> ReplayTrajectoryBatch | None:
        if not self.ready or replay_trajectories_per_step < 1:
            return None
        sample_count = min(int(replay_trajectories_per_step), len(self._items))
        if sample_count < 1:
            return None
        sampled_records = random.sample(tuple(self._items), k=sample_count)
        concatenated_batch = TrajectoryBatch.concatenate(
            [
                self._graph_payloads[record.graph_key].batch
                for record in sampled_records
            ],
            validate=True,
        ).to(device)
        max_actions = int(sampled_records[0].planned_edge_ids_local.numel())
        start_nodes = torch.zeros((sample_count, 1), device=device, dtype=torch.long)
        planned_edge_ids = torch.full(
            (sample_count, 1, max_actions),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        planned_stop_mask = torch.zeros(
            (sample_count, 1, max_actions),
            device=device,
            dtype=torch.bool,
        )
        path_lengths = torch.zeros((sample_count, 1), device=device, dtype=torch.long)
        termination_action_steps = torch.zeros_like(path_lengths)
        trace_nodes = torch.zeros(
            (sample_count, 1, max_actions),
            device=device,
            dtype=torch.long,
        )
        trace_edge_ids = torch.full_like(trace_nodes, fill_value=-1)
        trace_num_steps = torch.zeros_like(trace_nodes)
        trace_mask = torch.zeros(
            (sample_count, 1, max_actions),
            device=device,
            dtype=torch.bool,
        )
        trace_stop_mask = torch.zeros_like(trace_mask)
        node_offsets = concatenated_batch.node_ptr[:-1]
        edge_offsets = concatenated_batch.edge_ptr[:-1]
        for record_idx, record in enumerate(sampled_records):
            node_offset = int(node_offsets[record_idx].item())
            edge_offset = int(edge_offsets[record_idx].item())
            start_nodes[record_idx, 0] = int(record.start_node_local) + node_offset
            local_edge_ids = record.planned_edge_ids_local.to(device=device)
            planned_edge_ids[record_idx, 0] = torch.where(
                local_edge_ids >= 0,
                local_edge_ids + edge_offset,
                torch.full_like(local_edge_ids, fill_value=-1),
            )
            planned_stop_mask[record_idx, 0] = record.planned_stop_mask.to(
                device=device
            )
            path_lengths[record_idx, 0] = int(record.path_length)
            termination_action_steps[record_idx, 0] = int(
                record.termination_action_steps
            )
            local_trace_nodes = record.trace_nodes_local.to(device=device)
            local_trace_mask = record.trace_mask.to(device=device)
            trace_nodes[record_idx, 0] = torch.where(
                local_trace_mask,
                local_trace_nodes + node_offset,
                torch.zeros_like(local_trace_nodes),
            )
            local_trace_edge_ids = record.trace_edge_ids_local.to(device=device)
            trace_edge_ids[record_idx, 0] = torch.where(
                local_trace_edge_ids >= 0,
                local_trace_edge_ids + edge_offset,
                torch.full_like(local_trace_edge_ids, fill_value=-1),
            )
            trace_num_steps[record_idx, 0] = record.trace_num_steps.to(device=device)
            trace_mask[record_idx, 0] = local_trace_mask
            trace_stop_mask[record_idx, 0] = record.trace_stop_mask.to(device=device)
        return ReplayTrajectoryBatch(
            batch=concatenated_batch,
            start_nodes=start_nodes,
            planned_edge_ids=planned_edge_ids,
            planned_stop_mask=planned_stop_mask,
            path_lengths=path_lengths,
            termination_action_steps=termination_action_steps,
            trace_nodes=trace_nodes,
            trace_edge_ids=trace_edge_ids,
            trace_num_steps=trace_num_steps,
            trace_mask=trace_mask,
            trace_stop_mask=trace_stop_mask,
        )

    def _get_or_create_graph_payload(
        self,
        single_graph_batch: TrajectoryBatch,
    ) -> ReplayGraphPayload:
        graph_key = _graph_key_from_batch(single_graph_batch)
        payload = self._graph_payloads.get(graph_key)
        if payload is not None:
            return payload
        payload = ReplayGraphPayload(key=graph_key, batch=single_graph_batch)
        self._graph_payloads[graph_key] = payload
        self._graph_refcounts[graph_key] = 0
        return payload

    def _retain_graph_payload(self, graph_key: str) -> None:
        self._graph_refcounts[graph_key] = self._graph_refcounts.get(graph_key, 0) + 1

    def _release_graph_payload(self, graph_key: str) -> None:
        remaining = self._graph_refcounts.get(graph_key, 0) - 1
        if remaining > 0:
            self._graph_refcounts[graph_key] = remaining
            return
        self._graph_refcounts.pop(graph_key, None)
        self._graph_payloads.pop(graph_key, None)


__all__ = [
    "ReplayGraphPayload",
    "ReplayTrajectoryBatch",
    "ReplayTrajectoryRecord",
    "SuccessReplayBuffer",
]
