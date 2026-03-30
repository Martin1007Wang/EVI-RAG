from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from src.data.preprocess.labels.edge_retrieval import (
    resolve_forward_multi_anchor_union_trajectory,
    resolve_forward_shortest_path_trajectory,
)
from src.graph import TrajectoryBatch


UNREACHABLE_DISTANCE = -1


def _dedup_preserve_order(values: list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _oracle_distance_map(
    *,
    num_nodes: int,
    edge_src: list[int],
    edge_dst: list[int],
    answer_nodes: list[int],
) -> dict[int, int]:
    if num_nodes <= 0 or not answer_nodes:
        return {}
    reverse_adj: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for src, dst in zip(edge_src, edge_dst):
        reverse_adj[int(dst)].append(int(src))
    distance = [UNREACHABLE_DISTANCE for _ in range(int(num_nodes))]
    queue: deque[int] = deque()
    for answer in sorted(
        {int(node) for node in answer_nodes if 0 <= int(node) < num_nodes}
    ):
        distance[int(answer)] = 0
        queue.append(int(answer))
    while queue:
        current = queue.popleft()
        next_distance = int(distance[current]) + 1
        for predecessor in reverse_adj[current]:
            if distance[predecessor] != UNREACHABLE_DISTANCE:
                continue
            distance[predecessor] = next_distance
            queue.append(predecessor)
    return {
        int(node_id): int(node_distance)
        for node_id, node_distance in enumerate(distance)
        if int(node_distance) != UNREACHABLE_DISTANCE
    }


@dataclass(frozen=True)
class SubgraphPreparedBatch:
    # This static context stores the graph/question semantics needed to
    # reconstruct the full Markov state from the lightweight rollout trace.
    topology: Any
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    node_entity_ids: torch.Tensor
    node_ptr: torch.Tensor
    edge_ptr: torch.Tensor
    edge_batch: torch.Tensor
    sample_ids: tuple[str, ...]
    questions: tuple[str, ...]
    anchor_local_indices: torch.Tensor
    anchor_ptr: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    graph_anchor_abs_node_ids: tuple[tuple[int, ...], ...]
    graph_anchor_full_mask: tuple[int, ...]
    graph_outgoing_edge_ids: tuple[dict[int, tuple[int, ...]], ...]
    graph_answer_entities: tuple[frozenset[int], ...]
    graph_oracle_answer_distance: tuple[dict[int, int], ...]
    graph_teacher_action_edge_ids: tuple[tuple[int, ...] | None, ...]
    graph_teacher_edge_count: tuple[int | None, ...]

    @property
    def device(self) -> torch.device:
        return self.node_tokens.device

    @property
    def num_graphs(self) -> int:
        return len(self.sample_ids)


def build_subgraph_prepared_batch(
    *,
    batch: TrajectoryBatch,
    topology: Any,
    observation: Any,
    encoded: Any,
) -> SubgraphPreparedBatch:
    graph_anchor_abs_node_ids: list[tuple[int, ...]] = []
    graph_anchor_full_mask: list[int] = []
    graph_outgoing_edge_ids: list[dict[int, tuple[int, ...]]] = []
    graph_answer_entities: list[frozenset[int]] = []
    graph_oracle_answer_distance: list[dict[int, int]] = []
    graph_teacher_action_edge_ids: list[tuple[int, ...] | None] = []
    graph_teacher_edge_count: list[int | None] = []
    for graph_idx in range(int(batch.num_graphs)):
        node_start = int(batch.node_ptr[graph_idx].item())
        node_end = int(batch.node_ptr[graph_idx + 1].item())
        edge_start = int(batch.edge_ptr[graph_idx].item())
        edge_end = int(batch.edge_ptr[graph_idx + 1].item())
        anchor_start = int(batch.anchor_ptr[graph_idx].item())
        anchor_end = int(batch.anchor_ptr[graph_idx + 1].item())
        answer_start = int(batch.answer_ptr[graph_idx].item())
        answer_end = int(batch.answer_ptr[graph_idx + 1].item())
        anchor_abs_node_ids = _dedup_preserve_order(
            [
                int(node_start + int(local_idx))
                for local_idx in batch.anchor_local_indices[
                    anchor_start:anchor_end
                ].tolist()
            ]
        )
        graph_anchor_abs_node_ids.append(anchor_abs_node_ids)
        graph_anchor_full_mask.append(
            (1 << len(anchor_abs_node_ids)) - 1 if anchor_abs_node_ids else 0
        )
        outgoing: dict[int, list[int]] = {}
        local_edge_src = (
            topology.edge_index[0, edge_start:edge_end].detach().cpu().tolist()
        )
        for edge_id, src in enumerate(local_edge_src, start=edge_start):
            outgoing.setdefault(int(src), []).append(int(edge_id))
        graph_outgoing_edge_ids.append(
            {
                int(node_id): tuple(int(edge_id) for edge_id in edge_ids)
                for node_id, edge_ids in outgoing.items()
            }
        )
        answer_entities = frozenset(
            int(value)
            for value in batch.answer_entity_ids[answer_start:answer_end].tolist()
        )
        graph_answer_entities.append(answer_entities)
        answer_nodes: list[int] = []
        if answer_entities:
            local_node_entities = batch.node_entity_ids[node_start:node_end].tolist()
            for local_idx, entity_id in enumerate(local_node_entities):
                if int(entity_id) in answer_entities:
                    answer_nodes.append(int(node_start + local_idx))
        graph_oracle_answer_distance.append(
            _oracle_distance_map(
                num_nodes=int(topology.num_nodes),
                edge_src=[int(value) for value in topology.edge_index[0].tolist()],
                edge_dst=[int(value) for value in topology.edge_index[1].tolist()],
                answer_nodes=answer_nodes,
            )
        )
        local_edge_index = batch.edge_index[:, edge_start:edge_end] - int(node_start)
        num_local_nodes = int(node_end - node_start)
        local_anchor_indices = batch.anchor_local_indices[anchor_start:anchor_end]
        local_a = batch.a_local_indices[answer_start:answer_end]
        teacher_edge_ids: tuple[int, ...] | None = None
        if int(local_anchor_indices.numel()) > 1:
            teacher = resolve_forward_multi_anchor_union_trajectory(
                edge_index=local_edge_index,
                anchor_local_indices=local_anchor_indices,
                a_local_indices=local_a,
                num_nodes=num_local_nodes,
            )
            if teacher is not None:
                teacher_edge_ids = tuple(
                    int(edge_start + edge_id) for edge_id in teacher.ordered_edge_ids
                )
        else:
            teacher = resolve_forward_shortest_path_trajectory(
                edge_index=local_edge_index,
                anchor_local_indices=local_anchor_indices,
                a_local_indices=local_a,
                num_nodes=num_local_nodes,
            )
            if teacher is not None:
                teacher_edge_ids = tuple(
                    int(edge_start + edge_id) for edge_id in teacher.path_edge_ids
                )
        graph_teacher_action_edge_ids.append(teacher_edge_ids)
        teacher_edge_count = (
            None if teacher_edge_ids is None else int(len(teacher_edge_ids))
        )
        graph_teacher_edge_count.append(teacher_edge_count)
    return SubgraphPreparedBatch(
        topology=topology,
        node_tokens=encoded.node_tokens,
        relation_tokens=encoded.relation_tokens,
        question_tokens=encoded.question_tokens,
        node_entity_ids=observation.node_entity_ids,
        node_ptr=batch.node_ptr,
        edge_ptr=batch.edge_ptr,
        edge_batch=batch.edge_batch,
        sample_ids=tuple(str(sample_id) for sample_id in batch.sample_ids),
        questions=tuple(str(question) for question in batch.questions),
        anchor_local_indices=batch.anchor_local_indices,
        anchor_ptr=batch.anchor_ptr,
        answer_entity_ids=batch.answer_entity_ids,
        answer_ptr=batch.answer_ptr,
        graph_anchor_abs_node_ids=tuple(graph_anchor_abs_node_ids),
        graph_anchor_full_mask=tuple(graph_anchor_full_mask),
        graph_outgoing_edge_ids=tuple(graph_outgoing_edge_ids),
        graph_answer_entities=tuple(graph_answer_entities),
        graph_oracle_answer_distance=tuple(graph_oracle_answer_distance),
        graph_teacher_action_edge_ids=tuple(graph_teacher_action_edge_ids),
        graph_teacher_edge_count=tuple(graph_teacher_edge_count),
    )


__all__ = [
    "UNREACHABLE_DISTANCE",
    "SubgraphPreparedBatch",
    "build_subgraph_prepared_batch",
]
