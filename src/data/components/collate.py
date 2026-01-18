from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.loader.dataloader import Collater

from src.utils.graph import (
    build_edge_batch_debug_context,
    compute_edge_batch,
)
_ZERO = 0
_ONE = 1


def _expand_multi_start_answer_samples(
    batch_list: list[Any],
    *,
    expand_multi_start: bool,
    expand_multi_answer: bool,
) -> list[Any]:
    if not expand_multi_start and not expand_multi_answer:
        return batch_list
    expanded: list[Any] = []
    for data in batch_list:
        q_local = getattr(data, "q_local_indices", None)
        a_local = getattr(data, "a_local_indices", None)
        answer_ids = getattr(data, "answer_entity_ids", None)
        if q_local is None:
            raise AttributeError("Batch missing q_local_indices required for expansion.")
        if a_local is None or answer_ids is None:
            raise AttributeError("Batch missing a_local_indices/answer_entity_ids required for expansion.")
        if q_local is not None and not torch.is_tensor(q_local):
            q_local = torch.as_tensor(q_local, dtype=torch.long)
        if a_local is not None and not torch.is_tensor(a_local):
            a_local = torch.as_tensor(a_local, dtype=torch.long)
        if answer_ids is not None and not torch.is_tensor(answer_ids):
            answer_ids = torch.as_tensor(answer_ids, dtype=torch.long)
        q_vals = q_local.view(-1)
        a_vals = a_local.view(-1)
        answer_vals = answer_ids.view(-1)
        if a_vals.numel() != answer_vals.numel():
            raise ValueError("a_local_indices length mismatch with answer_entity_ids.")
        q_candidates: list[tuple[torch.Tensor, int | None]] = []
        if not expand_multi_start or q_vals.numel() <= _ONE:
            q_candidates = [(q_vals, None)]
        else:
            q_candidates = [(q_vals[idx].view(_ONE), idx) for idx in range(q_vals.numel())]
        a_candidates: list[tuple[torch.Tensor, torch.Tensor, int | None]] = []
        if not expand_multi_answer or a_vals.numel() <= _ONE:
            a_candidates = [(a_vals, answer_vals, None)]
        else:
            a_candidates = [
                (a_vals[idx].view(_ONE), answer_vals[idx].view(_ONE), idx) for idx in range(a_vals.numel())
            ]
        base_id = str(getattr(data, "sample_id", ""))
        for q_val, q_idx in q_candidates:
            for a_val, ans_val, a_idx in a_candidates:
                clone = data.clone()
                clone.q_local_indices = q_val
                clone.a_local_indices = a_val
                clone.answer_entity_ids = ans_val
                if base_id:
                    suffixes: list[str] = []
                    if q_idx is not None:
                        suffixes.append(f"q{q_idx}")
                    if a_idx is not None:
                        suffixes.append(f"a{a_idx}")
                    if suffixes:
                        clone.sample_id = f"{base_id}::" + "::".join(suffixes)
                expanded.append(clone)
    return expanded


def _attach_answer_ids(batch: Any) -> None:
    if not hasattr(batch, "answer_entity_ids"):
        raise AttributeError("Batch missing answer_entity_ids required for metrics.")
    batch.answer_entity_ids = torch.as_tensor(batch.answer_entity_ids, dtype=torch.long, device="cpu")
    answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
    if answer_ptr is None and hasattr(batch, "_slice_dict"):
        answer_ptr = batch._slice_dict.get("answer_entity_ids")
    if answer_ptr is None:
        raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
    batch.answer_entity_ids_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device="cpu")


def _attach_edge_batch(batch: Any, *, validate: bool) -> None:
    edge_index = getattr(batch, "edge_index", None)
    node_ptr = getattr(batch, "ptr", None)
    if edge_index is None or node_ptr is None:
        raise AttributeError("Batch missing edge_index/ptr; cannot precompute edge_batch.")
    edge_index = edge_index.to(device="cpu")
    node_ptr = node_ptr.to(device="cpu")
    num_graphs = int(node_ptr.numel() - _ONE)
    if num_graphs <= _ZERO:
        raise ValueError("ptr must encode at least one graph when precomputing edge_batch.")
    debug_context = build_edge_batch_debug_context(batch) if validate else None
    edge_batch, edge_ptr = compute_edge_batch(
        edge_index,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
        device=edge_index.device,
        debug_context=debug_context,
        validate=validate,
    )
    batch.edge_batch = edge_batch
    batch.edge_ptr = edge_ptr


class BatchAugmenter:
    """Attach derived fields to a PyG batch."""

    def __init__(
        self,
        *,
        precompute_edge_batch: bool,
        validate_edge_batch: bool,
    ) -> None:
        self._precompute_edge_batch = bool(precompute_edge_batch)
        self._validate_edge_batch = bool(validate_edge_batch)

    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, list):
            raise TypeError("RetrievalCollater received a list batch; dataset must return GraphData.")
        _attach_answer_ids(batch)
        if self._precompute_edge_batch:
            _attach_edge_batch(batch, validate=self._validate_edge_batch)
        return batch


class RetrievalCollater:
    """Collate PyG graphs and apply optional batch augmentation."""

    def __init__(
        self,
        dataset: Any,
        *,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        augmenter: Optional[BatchAugmenter] = None,
        expand_multi_start: bool = False,
        expand_multi_answer: bool = False,
    ) -> None:
        self._augmenter = augmenter
        self._expand_multi_start = bool(expand_multi_start)
        self._expand_multi_answer = bool(expand_multi_answer)
        self._collater = Collater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> Any:
        if self._expand_multi_start or self._expand_multi_answer:
            batch_list = _expand_multi_start_answer_samples(
                batch_list,
                expand_multi_start=self._expand_multi_start,
                expand_multi_answer=self._expand_multi_answer,
            )
        batch = self._collater(batch_list)
        if self._augmenter is None:
            return batch
        return self._augmenter(batch)
