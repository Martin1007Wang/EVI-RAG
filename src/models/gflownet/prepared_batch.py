from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

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
    edge_question_similarity: torch.Tensor
    graph_answer_entities: tuple[frozenset[int], ...]
    graph_node_entities: tuple[frozenset[int], ...]
    graph_node_counts: tuple[int, ...]
    graph_edge_counts: tuple[int, ...]
    graph_oracle_answer_distance: tuple[dict[int, int], ...]
    graph_teacher_action_edge_ids: tuple[tuple[int, ...] | None, ...]
    graph_teacher_edge_count: tuple[int | None, ...]

    @property
    def device(self) -> torch.device:
        return self.node_tokens.device

    @property
    def num_graphs(self) -> int:
        return len(self.sample_ids)

    def select_graphs(
        self, graph_indices: list[int] | tuple[int, ...]
    ) -> "SubgraphPreparedBatch":
        if not graph_indices:
            raise ValueError("graph_indices must be non-empty.")
        normalized_indices = [int(graph_idx) for graph_idx in graph_indices]
        for graph_idx in normalized_indices:
            if graph_idx < 0 or graph_idx >= int(self.num_graphs):
                raise IndexError(
                    f"graph_idx out of range for SubgraphPreparedBatch.select_graphs: {graph_idx}."
                )
        graph_index_tensor = torch.tensor(
            normalized_indices,
            device=self.question_tokens.device,
            dtype=torch.long,
        )
        graph_node_counts = tuple(
            int(self.graph_node_counts[graph_idx]) for graph_idx in normalized_indices
        )
        graph_edge_counts = tuple(
            int(self.graph_edge_counts[graph_idx]) for graph_idx in normalized_indices
        )
        anchor_counts = [
            int(
                self.anchor_ptr[graph_idx + 1].item()
                - self.anchor_ptr[graph_idx].item()
            )
            for graph_idx in normalized_indices
        ]
        answer_counts = [
            int(
                self.answer_ptr[graph_idx + 1].item()
                - self.answer_ptr[graph_idx].item()
            )
            for graph_idx in normalized_indices
        ]

        def _compact_ptr(
            counts: list[int] | tuple[int, ...], *, device: torch.device
        ) -> torch.Tensor:
            ptr = torch.zeros((len(counts) + 1,), device=device, dtype=torch.long)
            if counts:
                ptr[1:] = torch.tensor(counts, device=device, dtype=torch.long).cumsum(
                    0
                )
            return ptr

        def _gather_segments(
            values: torch.Tensor,
            ptr: torch.Tensor,
        ) -> torch.Tensor:
            segments = [
                values[int(ptr[graph_idx].item()) : int(ptr[graph_idx + 1].item())]
                for graph_idx in normalized_indices
                if int(ptr[graph_idx + 1].item()) > int(ptr[graph_idx].item())
            ]
            if segments:
                return torch.cat(segments, dim=0)
            return values.new_empty((0,), dtype=values.dtype)

        edge_batch = self.edge_batch.new_empty((0,), dtype=torch.long)
        if graph_edge_counts:
            repeated_graph_ids = [
                torch.full(
                    (int(edge_count),),
                    fill_value=local_graph_idx,
                    device=self.edge_batch.device,
                    dtype=torch.long,
                )
                for local_graph_idx, edge_count in enumerate(graph_edge_counts)
                if int(edge_count) > 0
            ]
            if repeated_graph_ids:
                edge_batch = torch.cat(repeated_graph_ids, dim=0)
        return SubgraphPreparedBatch(
            topology=self.topology,
            node_tokens=self.node_tokens,
            relation_tokens=self.relation_tokens,
            question_tokens=self.question_tokens.index_select(0, graph_index_tensor),
            node_entity_ids=self.node_entity_ids,
            node_ptr=_compact_ptr(list(graph_node_counts), device=self.node_ptr.device),
            edge_ptr=_compact_ptr(list(graph_edge_counts), device=self.edge_ptr.device),
            edge_batch=edge_batch,
            sample_ids=tuple(
                self.sample_ids[graph_idx] for graph_idx in normalized_indices
            ),
            questions=tuple(
                self.questions[graph_idx] for graph_idx in normalized_indices
            ),
            anchor_local_indices=_gather_segments(
                self.anchor_local_indices,
                self.anchor_ptr,
            ),
            anchor_ptr=_compact_ptr(anchor_counts, device=self.anchor_ptr.device),
            answer_entity_ids=_gather_segments(
                self.answer_entity_ids,
                self.answer_ptr,
            ),
            answer_ptr=_compact_ptr(answer_counts, device=self.answer_ptr.device),
            graph_anchor_abs_node_ids=tuple(
                self.graph_anchor_abs_node_ids[graph_idx]
                for graph_idx in normalized_indices
            ),
            graph_anchor_full_mask=tuple(
                self.graph_anchor_full_mask[graph_idx]
                for graph_idx in normalized_indices
            ),
            graph_outgoing_edge_ids=tuple(
                self.graph_outgoing_edge_ids[graph_idx]
                for graph_idx in normalized_indices
            ),
            edge_question_similarity=self.edge_question_similarity,
            graph_answer_entities=tuple(
                self.graph_answer_entities[graph_idx]
                for graph_idx in normalized_indices
            ),
            graph_node_entities=tuple(
                self.graph_node_entities[graph_idx] for graph_idx in normalized_indices
            ),
            graph_node_counts=graph_node_counts,
            graph_edge_counts=graph_edge_counts,
            graph_oracle_answer_distance=tuple(
                self.graph_oracle_answer_distance[graph_idx]
                for graph_idx in normalized_indices
            ),
            graph_teacher_action_edge_ids=tuple(
                self.graph_teacher_action_edge_ids[graph_idx]
                for graph_idx in normalized_indices
            ),
            graph_teacher_edge_count=tuple(
                self.graph_teacher_edge_count[graph_idx]
                for graph_idx in normalized_indices
            ),
        )


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
    graph_node_entities: list[frozenset[int]] = []
    graph_node_counts: list[int] = []
    graph_edge_counts: list[int] = []
    graph_oracle_answer_distance: list[dict[int, int]] = []
    graph_teacher_action_edge_ids: list[tuple[int, ...] | None] = []
    graph_teacher_edge_count: list[int | None] = []
    edge_question_similarity = torch.zeros(
        (int(topology.edge_index.size(1)),),
        device=encoded.question_tokens.device,
        dtype=torch.float32,
    )
    full_edge_src = topology.edge_index[0].detach().cpu().tolist()
    full_edge_dst = topology.edge_index[1].detach().cpu().tolist()
    for graph_idx in range(int(batch.num_graphs)):
        node_start = int(batch.node_ptr[graph_idx].item())
        node_end = int(batch.node_ptr[graph_idx + 1].item())
        edge_start = int(batch.edge_ptr[graph_idx].item())
        edge_end = int(batch.edge_ptr[graph_idx + 1].item())
        anchor_start = int(batch.anchor_ptr[graph_idx].item())
        anchor_end = int(batch.anchor_ptr[graph_idx + 1].item())
        answer_start = int(batch.answer_ptr[graph_idx].item())
        answer_end = int(batch.answer_ptr[graph_idx + 1].item())
        graph_node_counts.append(int(node_end - node_start))
        graph_edge_counts.append(int(edge_end - edge_start))
        local_anchor_indices = batch.anchor_local_indices[
            anchor_start:anchor_end
        ].tolist()
        anchor_abs_node_ids = _dedup_preserve_order(
            [int(node_start + int(local_idx)) for local_idx in local_anchor_indices]
        )
        graph_anchor_abs_node_ids.append(anchor_abs_node_ids)
        graph_anchor_full_mask.append(
            (1 << len(anchor_abs_node_ids)) - 1 if anchor_abs_node_ids else 0
        )
        local_node_entity_values = batch.node_entity_ids[node_start:node_end].tolist()
        local_node_entities = frozenset(
            int(value) for value in local_node_entity_values
        )
        graph_node_entities.append(local_node_entities)
        outgoing: dict[int, list[int]] = {}
        local_edge_src = full_edge_src[edge_start:edge_end]
        local_relation_ids = topology.edge_type[edge_start:edge_end]
        local_edge_ids = torch.arange(
            edge_start,
            edge_end,
            device=encoded.question_tokens.device,
            dtype=torch.long,
        )
        if int(local_edge_ids.numel()) > 0:
            with torch.no_grad():
                local_similarity = F.cosine_similarity(
                    (
                        encoded.relation_tokens.index_select(0, local_relation_ids)
                        + encoded.node_tokens.index_select(
                            0,
                            topology.edge_index[1, edge_start:edge_end],
                        )
                    ).to(dtype=torch.float32),
                    encoded.question_tokens[graph_idx]
                    .unsqueeze(0)
                    .expand(int(local_edge_ids.numel()), -1)
                    .to(dtype=torch.float32),
                    dim=-1,
                )
            edge_question_similarity.index_copy_(0, local_edge_ids, local_similarity)
        for edge_id, src in enumerate(local_edge_src, start=edge_start):
            outgoing.setdefault(int(src), []).append(int(edge_id))
        graph_outgoing_edge_ids.append(
            {
                int(node_id): tuple(
                    int(edge_id)
                    for edge_id in sorted(
                        edge_ids,
                        key=lambda value: (
                            -float(edge_question_similarity[int(value)].item()),
                            int(value),
                        ),
                    )
                )
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
            for local_idx, entity_id in enumerate(local_node_entity_values):
                if int(entity_id) in answer_entities:
                    answer_nodes.append(int(node_start + local_idx))
        graph_oracle_answer_distance.append(
            _oracle_distance_map(
                num_nodes=int(topology.num_nodes),
                edge_src=full_edge_src,
                edge_dst=full_edge_dst,
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
        edge_question_similarity=edge_question_similarity,
        graph_answer_entities=tuple(graph_answer_entities),
        graph_node_entities=tuple(graph_node_entities),
        graph_node_counts=tuple(graph_node_counts),
        graph_edge_counts=tuple(graph_edge_counts),
        graph_oracle_answer_distance=tuple(graph_oracle_answer_distance),
        graph_teacher_action_edge_ids=tuple(graph_teacher_action_edge_ids),
        graph_teacher_edge_count=tuple(graph_teacher_edge_count),
    )


__all__ = [
    "UNREACHABLE_DISTANCE",
    "SubgraphPreparedBatch",
    "build_subgraph_prepared_batch",
]
