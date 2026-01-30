from __future__ import annotations

from typing import Any, Optional

import torch
from torch_geometric.loader.dataloader import Collater

from src.utils.graph import (
    build_edge_batch_debug_context,
    build_edge_inverse_map,
    compute_edge_batch,
)
_ZERO = 0
_ONE = 1


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
            if a_vals.numel() == _ZERO:
                answer_vals = a_vals.new_empty((_ZERO,))
            else:
                answer_vals = node_global_ids.view(-1).index_select(0, a_vals)
        a_candidates: list[tuple[torch.Tensor, torch.Tensor, int | None]] = []
        if not expand_multi_answer or a_vals.numel() <= _ONE:
            a_candidates = [(a_vals, answer_vals, None)]
        else:
            a_candidates = [
                (a_vals[idx].view(_ONE), answer_vals[idx].view(_ONE), idx) for idx in range(a_vals.numel())
            ]
        base_id = str(getattr(data, "sample_id", ""))
        for a_val, ans_val, a_idx in a_candidates:
            if filter_zero_hop and q_vals.numel() > _ZERO and a_val.numel() > _ZERO:
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
    batch.answer_entity_ids = torch.as_tensor(batch.answer_entity_ids, dtype=torch.long, device="cpu")
    answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
    if answer_ptr is None and hasattr(batch, "_slice_dict"):
        answer_ptr = batch._slice_dict.get("answer_entity_ids")
    if answer_ptr is None:
        raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
    batch.answer_entity_ids_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device="cpu")
    answer_counts = batch.answer_entity_ids_ptr[1:] - batch.answer_entity_ids_ptr[:-1]
    batch.num_valid_graphs = int((answer_counts > _ZERO).sum().item())


def _attach_graph_stats(batch: Any) -> None:
    node_ptr = getattr(batch, "ptr", None)
    if node_ptr is None:
        raise AttributeError("Batch missing ptr; cannot infer graph counts.")
    node_ptr = torch.as_tensor(node_ptr, dtype=torch.long, device="cpu").view(-1)
    num_graphs = int(node_ptr.numel() - _ONE)
    num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > _ZERO else _ZERO
    batch.num_graphs = num_graphs
    batch.num_nodes_total = num_nodes_total


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


def _attach_edge_inverse_map(batch: Any, *, inverse_map: torch.Tensor) -> None:
    edge_index = getattr(batch, "edge_index", None)
    edge_attr = getattr(batch, "edge_attr", None)
    num_nodes_total = getattr(batch, "num_nodes_total", None)
    if edge_index is None or edge_attr is None:
        raise AttributeError("Batch missing edge_index/edge_attr; cannot precompute edge_inverse_map.")
    if num_nodes_total is None:
        raise AttributeError("Batch missing num_nodes_total; call _attach_graph_stats first.")
    if edge_index.device.type != "cpu":
        edge_index = edge_index.to(device="cpu")
    edge_relations = edge_attr
    if edge_relations.device.type != "cpu":
        edge_relations = edge_relations.to(device="cpu")
    if inverse_map.device.type != "cpu":
        inverse_map = inverse_map.to(device="cpu")
    edge_inverse_map = build_edge_inverse_map(
        edge_index=edge_index,
        edge_relations=edge_relations,
        num_nodes_total=int(num_nodes_total),
        inverse_map=inverse_map,
        num_relations=int(inverse_map.numel()),
    )
    batch.edge_inverse_map = edge_inverse_map


class BatchAugmenter:
    """Attach derived fields to a PyG batch."""

    def __init__(
        self,
        *,
        precompute_edge_batch: bool,
        validate_edge_batch: bool,
        precompute_edge_inverse_map: bool,
        relation_inverse_map: Optional[torch.Tensor],
    ) -> None:
        self._precompute_edge_batch = bool(precompute_edge_batch)
        self._validate_edge_batch = bool(validate_edge_batch)
        self._precompute_edge_inverse_map = bool(precompute_edge_inverse_map)
        self._relation_inverse_map = None
        if relation_inverse_map is not None:
            self._relation_inverse_map = torch.as_tensor(relation_inverse_map, dtype=torch.long, device="cpu")
        if self._precompute_edge_inverse_map and self._relation_inverse_map is None:
            raise ValueError("precompute_edge_inverse_map=True requires relation_inverse_map.")

    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, list):
            raise TypeError("RetrievalCollater received a list batch; dataset must return GraphData.")
        _attach_graph_stats(batch)
        _attach_answer_ids(batch)
        if self._precompute_edge_batch:
            _attach_edge_batch(batch, validate=self._validate_edge_batch)
        if self._precompute_edge_inverse_map:
            _attach_edge_inverse_map(batch, inverse_map=self._relation_inverse_map)
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
