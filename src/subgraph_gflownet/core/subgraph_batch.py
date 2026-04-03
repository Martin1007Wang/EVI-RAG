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
class TeacherBank:
    sequences: tuple[tuple[int, ...], ...] = ()
    subgraphs: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_sequences(
        cls,
        sequences: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
    ) -> "TeacherBank":
        if not sequences:
            return cls()
        canonical_sequences = tuple(
            tuple(int(edge_id) for edge_id in edge_ids) for edge_ids in sequences
        )
        canonical_subgraphs = tuple(
            tuple(sorted({int(edge_id) for edge_id in edge_ids}))
            for edge_ids in canonical_sequences
        )
        return cls(sequences=canonical_sequences, subgraphs=canonical_subgraphs)

    @property
    def representative_sequence(self) -> tuple[int, ...] | None:
        if not self.sequences:
            return None
        return self.sequences[0]

    @property
    def representative_edge_count(self) -> int | None:
        representative = self.representative_sequence
        if representative is None:
            return None
        return int(len(representative))


@dataclass(frozen=True)
class GraphSubgraphContext:
    sample_id: str
    question: str
    anchor_abs_node_ids: tuple[int, ...]
    anchor_full_mask: int
    outgoing_edge_ids: dict[int, tuple[int, ...]]
    answer_entities: frozenset[int]
    node_entities: frozenset[int]
    node_count: int
    edge_count: int
    oracle_answer_distance: dict[int, int]
    teacher_bank: TeacherBank


@dataclass(frozen=True)
class SubgraphBatchBuildOptions:
    include_edge_question_similarity: bool = True
    include_oracle_distance: bool = True
    include_teacher_banks: bool = True


@dataclass(frozen=True)
class SubgraphBatch:
    topology: Any
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    node_entity_ids: torch.Tensor
    node_ptr: torch.Tensor
    edge_ptr: torch.Tensor
    edge_batch: torch.Tensor
    anchor_local_indices: torch.Tensor
    anchor_ptr: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    edge_question_similarity: torch.Tensor | None
    graphs: tuple[GraphSubgraphContext, ...]

    @property
    def device(self) -> torch.device:
        return self.node_tokens.device

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(graph.sample_id for graph in self.graphs)

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(graph.question for graph in self.graphs)

    @property
    def graph_anchor_abs_node_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(graph.anchor_abs_node_ids for graph in self.graphs)

    @property
    def graph_anchor_full_mask(self) -> tuple[int, ...]:
        return tuple(int(graph.anchor_full_mask) for graph in self.graphs)

    @property
    def graph_outgoing_edge_ids(self) -> tuple[dict[int, tuple[int, ...]], ...]:
        return tuple(graph.outgoing_edge_ids for graph in self.graphs)

    @property
    def graph_answer_entities(self) -> tuple[frozenset[int], ...]:
        return tuple(graph.answer_entities for graph in self.graphs)

    @property
    def graph_node_entities(self) -> tuple[frozenset[int], ...]:
        return tuple(graph.node_entities for graph in self.graphs)

    @property
    def graph_node_counts(self) -> tuple[int, ...]:
        return tuple(int(graph.node_count) for graph in self.graphs)

    @property
    def graph_edge_counts(self) -> tuple[int, ...]:
        return tuple(int(graph.edge_count) for graph in self.graphs)

    @property
    def graph_oracle_answer_distance(self) -> tuple[dict[int, int], ...]:
        return tuple(graph.oracle_answer_distance for graph in self.graphs)

    @property
    def graph_teacher_sequence_bank(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(graph.teacher_bank.sequences for graph in self.graphs)

    @property
    def graph_teacher_subgraph_bank(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(graph.teacher_bank.subgraphs for graph in self.graphs)

    @property
    def graph_teacher_action_edge_ids(self) -> tuple[tuple[int, ...] | None, ...]:
        return tuple(
            graph.teacher_bank.representative_sequence for graph in self.graphs
        )

    @property
    def graph_teacher_edge_count(self) -> tuple[int | None, ...]:
        return tuple(
            graph.teacher_bank.representative_edge_count for graph in self.graphs
        )

    def graph(self, graph_idx: int) -> GraphSubgraphContext:
        normalized_idx = int(graph_idx)
        if normalized_idx < 0 or normalized_idx >= int(self.num_graphs):
            raise IndexError(f"graph_idx out of range for SubgraphBatch: {graph_idx}.")
        return self.graphs[normalized_idx]

    def select_graphs(
        self, graph_indices: list[int] | tuple[int, ...]
    ) -> "SubgraphBatch":
        if not graph_indices:
            raise ValueError("graph_indices must be non-empty.")
        normalized_indices = [int(graph_idx) for graph_idx in graph_indices]
        for graph_idx in normalized_indices:
            self.graph(graph_idx)
        graph_index_tensor = torch.tensor(
            normalized_indices,
            device=self.question_tokens.device,
            dtype=torch.long,
        )
        graph_node_counts = tuple(
            int(self.graphs[graph_idx].node_count) for graph_idx in normalized_indices
        )
        graph_edge_counts = tuple(
            int(self.graphs[graph_idx].edge_count) for graph_idx in normalized_indices
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

        def _gather_segments(values: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
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
        return SubgraphBatch(
            topology=self.topology,
            node_tokens=self.node_tokens,
            relation_tokens=self.relation_tokens,
            question_tokens=self.question_tokens.index_select(0, graph_index_tensor),
            node_entity_ids=self.node_entity_ids,
            node_ptr=_compact_ptr(list(graph_node_counts), device=self.node_ptr.device),
            edge_ptr=_compact_ptr(list(graph_edge_counts), device=self.edge_ptr.device),
            edge_batch=edge_batch,
            anchor_local_indices=_gather_segments(
                self.anchor_local_indices, self.anchor_ptr
            ),
            anchor_ptr=_compact_ptr(anchor_counts, device=self.anchor_ptr.device),
            answer_entity_ids=_gather_segments(self.answer_entity_ids, self.answer_ptr),
            answer_ptr=_compact_ptr(answer_counts, device=self.answer_ptr.device),
            edge_question_similarity=self.edge_question_similarity,
            graphs=tuple(self.graphs[graph_idx] for graph_idx in normalized_indices),
        )


def build_subgraph_batch(
    *,
    batch: TrajectoryBatch,
    topology: Any,
    observation: Any,
    encoded: Any,
    options: SubgraphBatchBuildOptions | None = None,
) -> SubgraphBatch:
    build_options = options or SubgraphBatchBuildOptions()
    graphs: list[GraphSubgraphContext] = []
    edge_question_similarity: torch.Tensor | None = None
    if build_options.include_edge_question_similarity:
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
        local_anchor_indices = batch.anchor_local_indices[
            anchor_start:anchor_end
        ].tolist()
        anchor_abs_node_ids = _dedup_preserve_order(
            [int(node_start + int(local_idx)) for local_idx in local_anchor_indices]
        )
        anchor_full_mask = (
            (1 << len(anchor_abs_node_ids)) - 1 if anchor_abs_node_ids else 0
        )
        local_node_entity_values = batch.node_entity_ids[node_start:node_end].tolist()
        node_entities = frozenset(int(value) for value in local_node_entity_values)
        local_edge_src = full_edge_src[edge_start:edge_end]
        local_relation_ids = topology.edge_type[edge_start:edge_end]
        local_edge_ids = torch.arange(
            edge_start,
            edge_end,
            device=encoded.question_tokens.device,
            dtype=torch.long,
        )
        if edge_question_similarity is not None and int(local_edge_ids.numel()) > 0:
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
        outgoing: dict[int, list[int]] = {}
        for edge_id, src in enumerate(local_edge_src, start=edge_start):
            outgoing.setdefault(int(src), []).append(int(edge_id))
        outgoing_edge_ids = {
            int(node_id): tuple(
                int(edge_id)
                for edge_id in sorted(
                    edge_ids,
                    key=(
                        (
                            lambda value: (
                                -float(edge_question_similarity[int(value)].item()),
                                int(value),
                            )
                        )
                        if edge_question_similarity is not None
                        else (lambda value: int(value))
                    ),
                )
            )
            for node_id, edge_ids in outgoing.items()
        }
        answer_entities = frozenset(
            int(value)
            for value in batch.answer_entity_ids[answer_start:answer_end].tolist()
        )
        answer_nodes: list[int] = []
        if answer_entities:
            for local_idx, entity_id in enumerate(local_node_entity_values):
                if int(entity_id) in answer_entities:
                    answer_nodes.append(int(node_start + local_idx))
        oracle_answer_distance = (
            _oracle_distance_map(
                num_nodes=int(topology.num_nodes),
                edge_src=full_edge_src,
                edge_dst=full_edge_dst,
                answer_nodes=answer_nodes,
            )
            if build_options.include_oracle_distance
            else {}
        )
        teacher_bank = TeacherBank()
        if build_options.include_teacher_banks:
            local_edge_index = batch.edge_index[:, edge_start:edge_end] - int(
                node_start
            )
            num_local_nodes = int(node_end - node_start)
            local_anchor_indices_tensor = batch.anchor_local_indices[
                anchor_start:anchor_end
            ]
            local_a = batch.a_local_indices[answer_start:answer_end]
            teacher_edge_ids: tuple[int, ...] | None = None
            if int(local_anchor_indices_tensor.numel()) > 1:
                teacher = resolve_forward_multi_anchor_union_trajectory(
                    edge_index=local_edge_index,
                    anchor_local_indices=local_anchor_indices_tensor,
                    a_local_indices=local_a,
                    num_nodes=num_local_nodes,
                )
                if teacher is not None:
                    teacher_edge_ids = tuple(
                        int(edge_start + edge_id)
                        for edge_id in teacher.ordered_edge_ids
                    )
            else:
                teacher = resolve_forward_shortest_path_trajectory(
                    edge_index=local_edge_index,
                    anchor_local_indices=local_anchor_indices_tensor,
                    a_local_indices=local_a,
                    num_nodes=num_local_nodes,
                )
                if teacher is not None:
                    teacher_edge_ids = tuple(
                        int(edge_start + edge_id) for edge_id in teacher.path_edge_ids
                    )
            teacher_bank = TeacherBank.from_sequences(
                () if teacher_edge_ids is None else (teacher_edge_ids,)
            )
        graphs.append(
            GraphSubgraphContext(
                sample_id=str(batch.sample_ids[graph_idx]),
                question=str(batch.questions[graph_idx]),
                anchor_abs_node_ids=anchor_abs_node_ids,
                anchor_full_mask=int(anchor_full_mask),
                outgoing_edge_ids=outgoing_edge_ids,
                answer_entities=answer_entities,
                node_entities=node_entities,
                node_count=int(node_end - node_start),
                edge_count=int(edge_end - edge_start),
                oracle_answer_distance=oracle_answer_distance,
                teacher_bank=teacher_bank,
            )
        )
    return SubgraphBatch(
        topology=topology,
        node_tokens=encoded.node_tokens,
        relation_tokens=encoded.relation_tokens,
        question_tokens=encoded.question_tokens,
        node_entity_ids=observation.node_entity_ids,
        node_ptr=batch.node_ptr,
        edge_ptr=batch.edge_ptr,
        edge_batch=batch.edge_batch,
        anchor_local_indices=batch.anchor_local_indices,
        anchor_ptr=batch.anchor_ptr,
        answer_entity_ids=batch.answer_entity_ids,
        answer_ptr=batch.answer_ptr,
        edge_question_similarity=edge_question_similarity,
        graphs=tuple(graphs),
    )


__all__ = [
    "GraphSubgraphContext",
    "SubgraphBatch",
    "SubgraphBatchBuildOptions",
    "TeacherBank",
    "UNREACHABLE_DISTANCE",
    "build_subgraph_batch",
]
