from __future__ import annotations

import functools
import random
from typing import Any, Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but optional
    np = None

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from ...utils.graph import (
    build_edge_batch_debug_context,
    compute_edge_batch,
    compute_undirected_degree,
)
from ...utils.logging_utils import get_logger, log_event
from ..schema.constants import _NON_TEXT_EMBEDDING_ID, _ONE
from ..g_retrieval_dataset import GRetrievalDataset

logger = get_logger(__name__)

_NON_TEXT_RELATION_MEAN_AGG = True
_NUMPY_SEED_MOD = 2**32 - 1


def _init_worker_seed(
    worker_id: int,
    *,
    base_seed: Optional[int],
    user_init_fn: Optional[Callable[[int], None]],
) -> None:
    if base_seed is not None:
        worker_seed = int(base_seed) + int(worker_id)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(worker_seed)
        random.seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed % _NUMPY_SEED_MOD)
    if user_init_fn is not None:
        user_init_fn(worker_id)


def _apply_non_text_relation_embeddings(batch: Any) -> None:
    node_embeddings = getattr(batch, "node_embeddings", None)
    if not torch.is_tensor(node_embeddings):
        raise AttributeError("Batch missing node_embeddings for non-text embedding update.")
    node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device=node_embeddings.device)
    cvt_mask = node_embedding_ids == _NON_TEXT_EMBEDDING_ID
    if not torch.any(cvt_mask):
        return
    edge_index = getattr(batch, "edge_index", None)
    edge_embeddings = getattr(batch, "edge_embeddings", None)
    if edge_index is None or not torch.is_tensor(edge_embeddings):
        raise AttributeError("Batch missing edge_index/edge_embeddings for non-text embedding update.")
    edge_index = edge_index.to(device=node_embeddings.device, dtype=torch.long, non_blocking=True)
    edge_embeddings = edge_embeddings.to(device=node_embeddings.device, dtype=node_embeddings.dtype, non_blocking=True)
    num_nodes = int(node_embeddings.size(0))
    if num_nodes <= 0 or edge_embeddings.numel() == 0:
        return
    num_edges = int(edge_embeddings.size(0))
    rel_dim = int(edge_embeddings.size(-1))
    agg = torch.zeros((num_nodes, rel_dim), device=edge_embeddings.device, dtype=edge_embeddings.dtype)
    counts = torch.zeros((num_nodes,), device=edge_embeddings.device, dtype=edge_embeddings.dtype)
    ones = torch.ones((num_edges,), device=edge_embeddings.device, dtype=edge_embeddings.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    agg.index_add_(0, src, edge_embeddings)
    agg.index_add_(0, dst, edge_embeddings)
    counts.index_add_(0, src, ones)
    counts.index_add_(0, dst, ones)
    if _NON_TEXT_RELATION_MEAN_AGG:
        denom = counts.clamp(min=float(_ONE)).unsqueeze(-1)
        agg = agg / denom
    updated = node_embeddings.clone()
    updated[cvt_mask] = agg[cvt_mask]
    batch.node_embeddings = updated


class RetrievalCollater:
    """Worker-side collate that attaches embeddings and optional edge_batch."""

    def __init__(
        self,
        dataset: GRetrievalDataset,
        *,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        attach_embeddings: bool = True,
        precompute_edge_batch: bool = True,
        precompute_node_in_degree: bool = True,
    ) -> None:
        self._dataset = dataset
        self._attach_embeddings_enabled = bool(attach_embeddings)
        self._precompute_edge_batch = bool(precompute_edge_batch)
        self._precompute_node_in_degree = bool(precompute_node_in_degree)
        self._collater = Collater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> Any:
        batch = self._collater(batch_list)
        if not hasattr(batch, "node_embedding_ids") or not hasattr(batch, "edge_attr"):
            return batch
        if self._attach_embeddings_enabled:
            self._attach_embeddings(batch)
        self._attach_answer_ids(batch)
        if self._precompute_edge_batch:
            self._attach_edge_batch(batch)
        if self._precompute_node_in_degree:
            self._attach_node_in_degree(batch)
        return batch

    def _resolve_global_embeddings(self):
        global_embeddings = getattr(self._dataset, "global_embeddings", None)
        if global_embeddings is None:
            raise RuntimeError("Dataset missing global_embeddings required for embedding lookup.")
        return global_embeddings

    def _attach_embeddings(self, batch: Any) -> None:
        global_embeddings = self._resolve_global_embeddings()
        node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device="cpu")
        relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")
        batch.node_embeddings = global_embeddings.get_entity_embeddings(node_embedding_ids)
        batch.edge_embeddings = global_embeddings.get_relation_embeddings(relation_ids)
        _apply_non_text_relation_embeddings(batch)

    def _attach_answer_ids(self, batch: Any) -> None:
        if not hasattr(batch, "answer_entity_ids"):
            raise AttributeError("Batch missing answer_entity_ids required for metrics.")
        batch.answer_entity_ids = torch.as_tensor(batch.answer_entity_ids, dtype=torch.long, device="cpu")

        answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr is None and hasattr(batch, "_slice_dict"):
            answer_ptr = batch._slice_dict.get("answer_entity_ids")
        if answer_ptr is None:
            raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
        batch.answer_entity_ids_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device="cpu")
        if hasattr(batch, "answer_entity_ids_len"):
            del batch.answer_entity_ids_len

    def _attach_edge_batch(self, batch: Any) -> None:
        edge_index = getattr(batch, "edge_index", None)
        node_ptr = getattr(batch, "ptr", None)
        if edge_index is None or node_ptr is None:
            raise AttributeError("Batch missing edge_index/ptr; cannot precompute edge_batch.")
        edge_index = edge_index.to(device="cpu")
        node_ptr = node_ptr.to(device="cpu")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("ptr must encode at least one graph when precomputing edge_batch.")
        debug_context = build_edge_batch_debug_context(batch)
        edge_batch, edge_ptr = compute_edge_batch(
            edge_index,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=edge_index.device,
            debug_context=debug_context,
            validate=True,
        )
        batch.edge_batch = edge_batch
        batch.edge_ptr = edge_ptr

    def _attach_node_in_degree(self, batch: Any) -> None:
        edge_index = getattr(batch, "edge_index", None)
        node_ptr = getattr(batch, "ptr", None)
        if edge_index is None or node_ptr is None:
            raise AttributeError("Batch missing edge_index/ptr; cannot precompute node_in_degree.")
        edge_index = edge_index.to(device="cpu")
        num_nodes = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        if num_nodes <= 0:
            raise ValueError("ptr must encode positive node counts when precomputing node_in_degree.")
        batch.node_in_degree = compute_undirected_degree(edge_index, num_nodes=num_nodes).to(dtype=torch.long)


class UnifiedDataLoader(DataLoader):
    """PyG DataLoader with worker-side embedding lookups."""

    def __init__(
        self,
        dataset: GRetrievalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        random_seed: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        worker_embed_lookup: bool = True,
        precompute_edge_batch: bool = True,
        precompute_node_in_degree: bool = True,
        embeddings_device: Optional[str] = None,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.random_seed = random_seed
        self._worker_embed_lookup = bool(worker_embed_lookup)
        self._embeddings_device = None if embeddings_device is None else str(embeddings_device)
        if self._embeddings_device not in (None, "cpu", "cuda"):
            raise ValueError(f"embeddings_device must be cpu or cuda, got {self._embeddings_device!r}.")
        if self._embeddings_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embeddings_device=cuda requested but CUDA is not available.")

        user_init_fn: Optional[Callable[[int], None]] = kwargs.pop("worker_init_fn", None)
        base_seed = int(random_seed) if random_seed is not None else None
        kwargs["worker_init_fn"] = functools.partial(
            _init_worker_seed,
            base_seed=base_seed,
            user_init_fn=user_init_fn,
        )

        if base_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(base_seed)
            kwargs.setdefault("generator", generator)

        if num_workers > 0 and "multiprocessing_context" not in kwargs:
            if torch.cuda.is_available():
                kwargs["multiprocessing_context"] = "spawn"

        if prefetch_factor is not None and num_workers > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)

        collate_fn = RetrievalCollater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            attach_embeddings=self._worker_embed_lookup,
            precompute_edge_batch=precompute_edge_batch,
            precompute_node_in_degree=precompute_node_in_degree,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
        self.shuffle = shuffle
        self._global_embeddings = None
        log_event(
            logger,
            "unified_dataloader_init",
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def __iter__(self):
        iterator = super().__iter__()
        if self._worker_embed_lookup:
            if self._embeddings_device not in (None, "cpu"):
                raise RuntimeError("embeddings_device must be cpu when worker_embed_lookup is enabled.")
            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                if isinstance(batch, list):
                    raise TypeError("UnifiedDataLoader received a list batch; dataset must return GraphData via __getitem__.")
                yield batch
        else:
            global_embeddings = self._global_embeddings
            if global_embeddings is None:
                global_embeddings = self.dataset.global_embeddings
                self._global_embeddings = global_embeddings
            if global_embeddings.entity_embeddings is None or global_embeddings.relation_embeddings is None:
                raise RuntimeError("Global embeddings not loaded before iteration")

            pin_outputs = bool(getattr(self, "pin_memory", False)) and torch.cuda.is_available()
            embeddings_device = self._embeddings_device
            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                if isinstance(batch, list):
                    raise TypeError("UnifiedDataLoader received a list batch; dataset must return GraphData via __getitem__.")
                if not hasattr(batch, "node_embeddings") or not hasattr(batch, "edge_embeddings"):
                    node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device="cpu")
                    relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")
                    batch.node_embeddings = global_embeddings.get_entity_embeddings(
                        node_embedding_ids,
                        device=embeddings_device,
                    )
                    batch.edge_embeddings = global_embeddings.get_relation_embeddings(
                        relation_ids,
                        device=embeddings_device,
                    )
                    _apply_non_text_relation_embeddings(batch)
                    if pin_outputs:
                        if batch.node_embeddings.device.type == "cpu":
                            batch.node_embeddings = batch.node_embeddings.pin_memory()
                        if batch.edge_embeddings.device.type == "cpu":
                            batch.edge_embeddings = batch.edge_embeddings.pin_memory()
                yield batch
