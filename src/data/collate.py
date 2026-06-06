from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .schema.batch import ReplayBankBatch, ReplayBankSample, RetrievalBatch
from .schema.fields import SampleFields

_DEFAULT_FOLLOW_BATCH = (
    SampleFields.REACHABLE_TARGET_NODE_IDS,
)

_REPLAY_EXCLUDE_KEYS = (
    SampleFields.REPLAY_BANK_EDGE_IDS,
    SampleFields.REPLAY_BANK_EDGE_COUNT,
    "replay_bank",
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
    - no deprecated field handling
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
        batch.replay_bank = build_replay_bank_batch(
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


def build_replay_bank_batch(
    *,
    samples: Sequence[Any],
    batch: RetrievalBatch,
) -> ReplayBankBatch:
    device = batch.edge_index.device
    sample_banks = [_sample_replay_bank(sample) for sample in samples]
    edge_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    priority_chunks: list[torch.Tensor] = []
    edge_offset = 0
    shape = sample_banks[0].edge_ids_local.shape
    count_shape = sample_banks[0].edge_count.shape
    priority_shape = sample_banks[0].priority.shape
    for sample, bank in zip(samples, sample_banks, strict=True):
        num_edges = int(sample.num_edges)
        if (
            bank.edge_ids_local.shape != shape
            or bank.edge_count.shape != count_shape
            or bank.priority.shape != priority_shape
        ):
            raise ValueError("All replay banks in a batch must have the same shape.")
        local_edge_ids = bank.edge_ids_local.to(device=device, dtype=torch.long)
        valid_local_edge_ids = local_edge_ids[local_edge_ids.ge(0)]
        if bool(valid_local_edge_ids.ge(num_edges).any()):
            raise ValueError("replay bank contains an edge id outside the sample-local edge range.")
        edge_chunks.append(torch.where(local_edge_ids.ge(0), local_edge_ids + edge_offset, local_edge_ids))
        count_chunks.append(bank.edge_count.to(device=device, dtype=torch.long))
        priority_chunks.append(bank.priority.to(device=device, dtype=torch.float32))
        edge_offset += int(num_edges)

    return ReplayBankBatch(
        edge_ids=torch.stack(edge_chunks, dim=0).contiguous(),
        edge_count=torch.stack(count_chunks, dim=0).contiguous(),
        priority=torch.stack(priority_chunks, dim=0).contiguous(),
    )


def _sample_replay_bank(sample: Any) -> ReplayBankSample:
    bank = getattr(sample, "replay_bank", None)
    if not isinstance(bank, ReplayBankSample):
        raise TypeError("sample.replay_bank must be a ReplayBankSample.")
    return bank
