from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .schema.batch import ReplayProgramBatch, ReplayProgramSample, RetrievalBatch
from .schema.fields import SampleFields

_DEFAULT_FOLLOW_BATCH = (
    SampleFields.REACHABLE_TARGET_NODE_IDS,
)

_REPLAY_EXCLUDE_KEYS = (
    SampleFields.REPLAY_CANDIDATE_EDGE_IDS,
    SampleFields.REPLAY_CANDIDATE_PTR,
    SampleFields.REPLAY_CANDIDATE_TARGET_POSITIONS,
    SampleFields.REPLAY_CANDIDATE_TARGET_PTR,
    SampleFields.REPLAY_EDGE_TO_CANDIDATE_IDS,
    SampleFields.REPLAY_EDGE_TO_CANDIDATE_PTR,
    SampleFields.REPLAY_PATH_TRUNCATED,
    "replay_program",
)


class RetrievalCollator:
    """
    Collator for RetrievalData samples.

    Responsibilities:
    - batch graph fields with PyG semantics
    - stack graph-level question embeddings into [B, H]
    - attach edge_batch as [E], where edge_batch[e] is the graph id of edge e

    Non-responsibilities:
    - no schema fallback
    - no legacy field handling
    - no mask construction
    - no path recomputation
    """

    def __init__(
        self,
        *,
        follow_batch: Sequence[str] = (),
        exclude_keys: Sequence[str] = (),
    ) -> None:
        self.follow_batch = _merge_follow_batch(follow_batch)
        self.exclude_keys = _exclude_question_embedding(exclude_keys)

    def __call__(self, samples: list[Any]) -> RetrievalBatch:
        if not samples:
            raise ValueError("samples must be non-empty.")

        batch = RetrievalBatch.from_data_list(
            samples,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )

        batch.sample_id = [str(sample.sample_id) for sample in samples]
        batch.question_emb = stack_question_embeddings(samples)
        batch.edge_batch = edge_batch_from_node_batch(batch)
        batch.replay_program = build_replay_program_batch(
            samples=samples,
            batch=batch,
        )

        return batch


def _exclude_question_embedding(exclude_keys: Sequence[str]) -> list[str]:
    keys = list(exclude_keys)
    if SampleFields.QUESTION_EMB not in keys:
        keys.append(SampleFields.QUESTION_EMB)
    for key in _REPLAY_EXCLUDE_KEYS:
        if key not in keys:
            keys.append(key)
    return keys


def _merge_follow_batch(follow_batch: Sequence[str]) -> list[str]:
    keys = list(follow_batch)
    for key in _DEFAULT_FOLLOW_BATCH:
        if key not in keys:
            keys.append(key)
    return keys


def stack_question_embeddings(samples: Sequence[Any]) -> torch.Tensor:
    question_embs = [
        torch.as_tensor(sample.question_emb, dtype=torch.float32).flatten()
        for sample in samples
    ]

    hidden_dim = int(question_embs[0].numel())

    for index, question_emb in enumerate(question_embs):
        actual_dim = int(question_emb.numel())
        if actual_dim != hidden_dim:
            raise ValueError(
                "question_emb hidden_dim mismatch "
                f"at sample {index}: got {actual_dim}, expected {hidden_dim}."
            )

    return torch.stack(question_embs, dim=0).contiguous()


def edge_batch_from_node_batch(batch: RetrievalBatch) -> torch.Tensor:
    edge_index = batch.edge_index
    node_batch = batch.batch

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )

    if node_batch.ndim != 1:
        raise ValueError(
            f"batch.batch must have shape [num_nodes], got {tuple(node_batch.shape)}."
        )

    num_edges = int(edge_index.size(1))
    if num_edges == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    src_graph = node_batch.index_select(0, edge_index[0])
    dst_graph = node_batch.index_select(0, edge_index[1])

    if not torch.equal(src_graph, dst_graph):
        raise ValueError("edge_index contains cross-graph edges after batching.")

    return src_graph.contiguous()


def build_replay_program_batch(
    *,
    samples: Sequence[Any],
    batch: RetrievalBatch,
) -> ReplayProgramBatch:
    device = batch.edge_index.device
    sample_programs = [_sample_replay_program(sample) for sample in samples]

    candidate_edge_chunks: list[torch.Tensor] = []
    candidate_ptr_parts = [0]
    candidate_target_position_chunks: list[torch.Tensor] = []
    candidate_target_ptr_parts = [0]
    edge_to_candidate_id_chunks: list[torch.Tensor] = []
    edge_to_candidate_ptr_parts = [0]
    candidate_graph_ptr_parts = [0]
    path_truncated_values: list[int] = []

    edge_offset = 0
    candidate_offset = 0
    total_candidate_edges = 0
    total_candidate_targets = 0
    total_edge_candidate_refs = 0
    total_candidates = 0

    for graph_id, (sample, program) in enumerate(zip(samples, sample_programs, strict=True)):
        num_edges = int(sample.num_edges)
        local_candidate_count = int(program.candidate_ptr.numel()) - 1

        if int(program.candidate_edge_ids_local.numel()) > 0:
            candidate_edge_chunks.append(
                program.candidate_edge_ids_local.to(device=device, dtype=torch.long) + int(edge_offset)
            )
        if int(program.candidate_target_positions.numel()) > 0:
            candidate_target_position_chunks.append(
                program.candidate_target_positions.to(device=device, dtype=torch.long)
            )
        if int(program.edge_to_candidate_ids_local.numel()) > 0:
            edge_to_candidate_id_chunks.append(
                program.edge_to_candidate_ids_local.to(device=device, dtype=torch.long) + int(candidate_offset)
            )

        candidate_ptr_parts.extend(
            int(total_candidate_edges) + int(value)
            for value in program.candidate_ptr.to(dtype=torch.long).tolist()[1:]
        )
        candidate_target_ptr_parts.extend(
            int(total_candidate_targets) + int(value)
            for value in program.candidate_target_ptr.to(dtype=torch.long).tolist()[1:]
        )
        edge_to_candidate_ptr_parts.extend(
            int(total_edge_candidate_refs) + int(value)
            for value in program.edge_to_candidate_ptr.to(dtype=torch.long).tolist()[1:]
        )
        candidate_graph_ptr_parts.append(int(candidate_offset) + int(local_candidate_count))
        path_truncated_values.append(int(program.path_truncated.item()))

        total_candidate_edges += int(program.candidate_edge_ids_local.numel())
        total_candidate_targets += int(program.candidate_target_positions.numel())
        total_edge_candidate_refs += int(program.edge_to_candidate_ids_local.numel())
        total_candidates += int(local_candidate_count)
        edge_offset += int(num_edges)
        candidate_offset += int(local_candidate_count)

    return ReplayProgramBatch(
        candidate_edge_ids=_cat_or_empty(candidate_edge_chunks, device=device),
        candidate_ptr=torch.tensor(candidate_ptr_parts, dtype=torch.long, device=device).contiguous(),
        candidate_target_positions=_cat_or_empty(candidate_target_position_chunks, device=device),
        candidate_target_ptr=torch.tensor(candidate_target_ptr_parts, dtype=torch.long, device=device).contiguous(),
        edge_to_candidate_ids=_cat_or_empty(edge_to_candidate_id_chunks, device=device),
        edge_to_candidate_ptr=torch.tensor(edge_to_candidate_ptr_parts, dtype=torch.long, device=device).contiguous(),
        candidate_graph_ptr=torch.tensor(candidate_graph_ptr_parts, dtype=torch.long, device=device).contiguous(),
        path_truncated_by_graph=torch.tensor(path_truncated_values, dtype=torch.long, device=device).contiguous(),
    )


def _sample_replay_program(sample: Any) -> ReplayProgramSample:
    program = getattr(sample, "replay_program", None)
    if not isinstance(program, ReplayProgramSample):
        raise TypeError("sample.replay_program must be a ReplayProgramSample.")
    return program


def _cat_or_empty(chunks: Sequence[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(tuple(chunk.contiguous() for chunk in chunks), dim=0).contiguous()
