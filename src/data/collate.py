from __future__ import annotations
from typing import Any, Sequence
import torch
from .schema.batch import RetrievalBatch


class RetrievalCollator:
    """
    Collator for RetrievalData samples.
    Responsibilities:
    - use PyG batching semantics for graph fields
    - stack graph-level question embeddings as [batch_size, hidden_dim]
    - expose node_ptr as an alias of ptr
    - construct edge_batch and edge_ptr
    Non-responsibilities:
    - no schema fallback
    - no legacy field handling
    - no mask construction
    - no path recomputation
    """

    def __init__(
        self,
        *,
        follow_batch: Sequence[str] | None = None,
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        self.follow_batch = list(follow_batch or [])
        self.exclude_keys = list(exclude_keys or [])

    def __call__(self, batch_list: list[Any]) -> RetrievalBatch:
        if not batch_list:
            raise ValueError("batch_list must be non-empty")
        batch = RetrievalBatch.from_data_list(
            batch_list,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )
        batch.question_emb = _stack_question_embeddings(batch_list)
        batch.node_ptr = batch.ptr
        _attach_edge_batch(batch)
        return batch


def _stack_question_embeddings(batch_list: Sequence[Any]) -> torch.Tensor:
    question_embs = [torch.as_tensor(sample.question_emb, dtype=torch.float32).reshape(-1) for sample in batch_list]
    hidden_dim = int(question_embs[0].numel())
    for idx, question_emb in enumerate(question_embs):
        if int(question_emb.numel()) != hidden_dim:
            raise ValueError(f"question_emb hidden_dim mismatch at batch index {idx}: " f"got {int(question_emb.numel())}, expected {hidden_dim}")
    return torch.stack(question_embs, dim=0).contiguous()


def _attach_edge_batch(batch: RetrievalBatch) -> None:
    edge_index = batch.edge_index
    node_ptr = batch.ptr
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}")
    if node_ptr.ndim != 1:
        raise ValueError(f"batch.ptr must be 1D, got {tuple(node_ptr.shape)}")
    num_graphs = int(node_ptr.numel()) - 1
    if num_graphs < 0:
        raise ValueError("batch.ptr is invalid")
    device = edge_index.device
    num_edges = int(edge_index.size(1))
    if num_edges == 0:
        batch.edge_batch = torch.empty(0, dtype=torch.long, device=device)
        batch.edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        return
    edge_batch = (
        torch.searchsorted(
            node_ptr.to(device=device),
            edge_index[0],
            side="right",
        )
        - 1
    )
    if edge_batch.numel() != num_edges:
        raise ValueError(f"edge_batch length mismatch: got {edge_batch.numel()}, expected {num_edges}")
    if edge_batch.min().item() < 0 or edge_batch.max().item() >= num_graphs:
        raise ValueError("edge_batch contains graph ids outside valid range")
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = torch.cumsum(edge_counts, dim=0)
    batch.edge_batch = edge_batch.contiguous()
    batch.edge_ptr = edge_ptr.contiguous()
