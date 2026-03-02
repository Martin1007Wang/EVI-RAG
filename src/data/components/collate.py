from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.loader.dataloader import Collater

from src.utils.graph import compute_edge_batch


def _as_1d_long(value: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        return tensor.view(-1)
    tensor = value
    target_device = tensor.device if device is None else device
    if tensor.device != target_device:
        tensor = tensor.to(device=target_device)
    if tensor.dtype != torch.long:
        tensor = tensor.to(dtype=torch.long)
    return tensor.view(-1)


def _expand_answer_samples(
    batch_list: list[Any],
    *,
    expand_multi_answer: bool,
    filter_zero_hop: bool,
) -> list[Any]:
    if not expand_multi_answer and not filter_zero_hop:
        return batch_list
    expanded: list[Any] = []
    for data in batch_list:
        q_local = getattr(data, "q_local_indices", None)
        a_local = getattr(data, "a_local_indices", None)
        answer_ids = getattr(data, "answer_entity_ids", None)
        node_global_ids = getattr(data, "node_global_ids", None)
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
        if node_global_ids is not None and not torch.is_tensor(node_global_ids):
            node_global_ids = torch.as_tensor(node_global_ids, dtype=torch.long)
        q_vals = q_local.view(-1)
        a_vals = a_local.view(-1)
        answer_vals = answer_ids.view(-1)
        if a_vals.numel() != answer_vals.numel():
            if node_global_ids is None:
                raise AttributeError("Batch missing node_global_ids required to align answers.")
            if a_vals.numel() == 0:
                answer_vals = a_vals.new_empty((0,))
            else:
                answer_vals = node_global_ids.view(-1).index_select(0, a_vals)
        a_candidates: list[tuple[torch.Tensor, torch.Tensor, int | None]] = []
        if not expand_multi_answer or a_vals.numel() <= 1:
            a_candidates = [(a_vals, answer_vals, None)]
        else:
            a_candidates = [
                (a_vals[idx].view(1), answer_vals[idx].view(1), idx) for idx in range(a_vals.numel())
            ]
        base_id = str(getattr(data, "sample_id", ""))
        for a_val, ans_val, a_idx in a_candidates:
            if filter_zero_hop and q_vals.numel() > 0 and a_val.numel() > 0:
                if bool((q_vals.view(-1, 1) == a_val.view(1, -1)).any().item()):
                    continue
            clone = data.clone()
            clone.a_local_indices = a_val
            clone.answer_entity_ids = ans_val
            if base_id and a_idx is not None:
                clone.sample_id = f"{base_id}::a{a_idx}"
            expanded.append(clone)
    if filter_zero_hop and not expanded:
        raise ValueError("All samples filtered by zero-hop guard; disable filter_zero_hop to proceed.")
    return expanded


def _attach_answer_ids(batch: Any) -> None:
    if not hasattr(batch, "answer_entity_ids"):
        raise AttributeError("Batch missing answer_entity_ids required for metrics.")
    batch.answer_entity_ids = _as_1d_long(batch.answer_entity_ids, device=torch.device("cpu"))
    answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
    if answer_ptr is None and hasattr(batch, "_slice_dict"):
        answer_ptr = batch._slice_dict.get("answer_entity_ids")
    if answer_ptr is None:
        raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
    batch.answer_entity_ids_ptr = _as_1d_long(answer_ptr, device=torch.device("cpu"))
    batch.answer_ptr = batch.answer_entity_ids_ptr
    answer_counts = batch.answer_entity_ids_ptr[1:] - batch.answer_entity_ids_ptr[:-1]
    batch.num_valid_graphs = int((answer_counts > 0).sum().item())
    batch.dummy_mask = answer_counts <= 0


def _attach_qa_ptrs(batch: Any) -> None:
    slice_dict = getattr(batch, "_slice_dict", None)
    if not isinstance(slice_dict, dict):
        raise AttributeError("Batch missing _slice_dict required for q_ptr/a_ptr.")
    q_ptr = slice_dict.get("q_local_indices")
    a_ptr = slice_dict.get("a_local_indices")
    if q_ptr is None or a_ptr is None:
        raise AttributeError("Batch _slice_dict missing q_local_indices/a_local_indices pointers.")
    batch.q_ptr = _as_1d_long(q_ptr, device=torch.device("cpu"))
    batch.a_ptr = _as_1d_long(a_ptr, device=torch.device("cpu"))


def _attach_local_indices(batch: Any) -> None:
    if not hasattr(batch, "q_local_indices"):
        raise AttributeError("Batch missing q_local_indices required for graph alignment.")
    if not hasattr(batch, "a_local_indices"):
        raise AttributeError("Batch missing a_local_indices required for graph alignment.")
    batch.q_local_indices = _as_1d_long(batch.q_local_indices, device=torch.device("cpu"))
    batch.a_local_indices = _as_1d_long(batch.a_local_indices, device=torch.device("cpu"))


def _attach_graph_stats(batch: Any) -> None:
    node_ptr = getattr(batch, "ptr", None)
    if node_ptr is None:
        raise AttributeError("Batch missing ptr; cannot infer graph counts.")
    node_ptr = _as_1d_long(node_ptr, device=torch.device("cpu"))
    num_graphs = int(node_ptr.numel() - 1)
    num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
    batch.num_graphs = num_graphs
    batch.num_nodes_total = num_nodes_total
    batch.node_ptr = node_ptr


def _attach_edge_batch(batch: Any) -> None:
    edge_index = getattr(batch, "edge_index", None)
    node_ptr = getattr(batch, "ptr", None)
    if edge_index is None or node_ptr is None:
        raise AttributeError("Batch missing edge_index/ptr; cannot precompute edge_batch.")
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    elif edge_index.dtype != torch.long:
        edge_index = edge_index.to(dtype=torch.long)
    node_ptr = _as_1d_long(node_ptr, device=edge_index.device)
    num_graphs = int(node_ptr.numel() - 1)
    if num_graphs <= 0:
        raise ValueError("ptr must encode at least one graph when precomputing edge_batch.")
    edge_batch, edge_ptr = compute_edge_batch(
        edge_index,
        node_ptr=node_ptr,
        num_graphs=num_graphs,
        device=edge_index.device,
        validate=False,
    )
    batch.edge_batch = edge_batch
    batch.edge_ptr = edge_ptr


def _validate_ptrs(batch: Any) -> None:
    num_graphs = getattr(batch, "num_graphs", None)
    if not isinstance(num_graphs, int) or num_graphs <= 0:
        raise ValueError("Batch missing valid num_graphs; cannot validate ptrs.")
    q_ptr = getattr(batch, "q_ptr", None)
    a_ptr = getattr(batch, "a_ptr", None)
    answer_ptr = getattr(batch, "answer_ptr", None)
    if q_ptr is None or a_ptr is None or answer_ptr is None:
        raise AttributeError("Batch missing q_ptr/a_ptr/answer_ptr required for validation.")
    if q_ptr.numel() != num_graphs + 1:
        raise ValueError("q_ptr length mismatch with num_graphs.")
    if a_ptr.numel() != num_graphs + 1:
        raise ValueError("a_ptr length mismatch with num_graphs.")
    if answer_ptr.numel() != num_graphs + 1:
        raise ValueError("answer_ptr length mismatch with num_graphs.")
    if int(q_ptr[-1].item()) != int(batch.q_local_indices.numel()):
        raise ValueError("q_ptr[-1] mismatch q_local_indices length.")
    if int(a_ptr[-1].item()) != int(batch.a_local_indices.numel()):
        raise ValueError("a_ptr[-1] mismatch a_local_indices length.")
    if int(answer_ptr[-1].item()) != int(batch.answer_entity_ids.numel()):
        raise ValueError("answer_ptr[-1] mismatch answer_entity_ids length.")
    if not hasattr(batch, "dummy_mask"):
        raise AttributeError("Batch missing dummy_mask derived from answer_ptr.")
    if batch.dummy_mask.numel() != num_graphs:
        raise ValueError("dummy_mask length mismatch with num_graphs.")


class BatchAugmenter:
    """Attach derived fields to a PyG batch."""

    def __init__(
        self,
        *,
        precompute_edge_batch: bool,
    ) -> None:
        self._precompute_edge_batch = bool(precompute_edge_batch)

    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, list):
            raise TypeError("RetrievalCollater received a list batch; dataset must return GraphData.")
        _attach_graph_stats(batch)
        _attach_local_indices(batch)
        _attach_qa_ptrs(batch)
        _attach_answer_ids(batch)
        if self._precompute_edge_batch:
            _attach_edge_batch(batch)
        _validate_ptrs(batch)
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
        expand_multi_answer: bool = False,
        filter_zero_hop: bool = True,
    ) -> None:
        self._augmenter = augmenter
        self._expand_multi_answer = bool(expand_multi_answer)
        self._filter_zero_hop = bool(filter_zero_hop)
        self._collater = Collater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> Any:
        if self._expand_multi_answer or self._filter_zero_hop:
            batch_list = _expand_answer_samples(
                batch_list,
                expand_multi_answer=self._expand_multi_answer,
                filter_zero_hop=self._filter_zero_hop,
            )
        batch = self._collater(batch_list)
        if self._augmenter is None:
            return batch
        return self._augmenter(batch)
