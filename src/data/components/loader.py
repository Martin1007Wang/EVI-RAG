from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but optional
    np = None

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from ...utils.graph_utils import compute_edge_batch
from ..g_retrieval_dataset import GRetrievalDataset

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._dataset = dataset
        self._attach_embeddings = bool(attach_embeddings)
        self._precompute_edge_batch = bool(precompute_edge_batch)
        self._collater = Collater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> Any:
        batch = self._collater(batch_list)
        if not hasattr(batch, "node_embedding_ids") or not hasattr(batch, "edge_attr"):
            return batch
        if self._attach_embeddings:
            self._attach_embeddings(batch)
        self._attach_answer_ids(batch)
        if self._precompute_edge_batch:
            self._attach_edge_batch(batch)
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
            return
        edge_index = edge_index.to(device="cpu")
        node_ptr = node_ptr.to(device="cpu")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            return
        edge_batch, edge_ptr = compute_edge_batch(
            edge_index,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=edge_index.device,
            debug_batch=batch,
        )
        batch.edge_batch = edge_batch
        batch.edge_ptr = edge_ptr


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
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.random_seed = random_seed
        self._worker_embed_lookup = bool(worker_embed_lookup)

        worker_init_fn: Optional[Callable[[int], None]] = kwargs.pop("worker_init_fn", None)
        if random_seed is not None:
            base_seed = int(random_seed)

            def _seed_worker(worker_id: int) -> None:
                worker_seed = base_seed + worker_id
                torch.manual_seed(worker_seed)
                torch.cuda.manual_seed_all(worker_seed)
                random.seed(worker_seed)
                if np is not None:
                    np.random.seed(worker_seed % (2**32 - 1))
                if worker_init_fn is not None:
                    worker_init_fn(worker_id)

            kwargs["worker_init_fn"] = _seed_worker

            generator = torch.Generator()
            generator.manual_seed(base_seed)
            kwargs.setdefault("generator", generator)
        elif worker_init_fn is not None:
            kwargs["worker_init_fn"] = worker_init_fn

        if prefetch_factor is not None and num_workers > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)

        collate_fn = RetrievalCollater(
            dataset,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            attach_embeddings=self._worker_embed_lookup,
            precompute_edge_batch=True,
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
        self._global_embeddings = dataset.global_embeddings
        logger.info("UnifiedDataLoader initialized: batch_size=%s shuffle=%s", batch_size, shuffle)

    def __iter__(self):
        if self._worker_embed_lookup:
            yield from super().__iter__()
            return

        global_embeddings = self._global_embeddings
        if global_embeddings.entity_embeddings is None or global_embeddings.relation_embeddings is None:
            raise RuntimeError("Global embeddings not loaded before iteration")

        pin_outputs = bool(getattr(self, "pin_memory", False)) and torch.cuda.is_available()
        for batch in super().__iter__():
            if not hasattr(batch, "node_embeddings") or not hasattr(batch, "edge_embeddings"):
                node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device="cpu")
                relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")
                batch.node_embeddings = global_embeddings.get_entity_embeddings(node_embedding_ids)
                batch.edge_embeddings = global_embeddings.get_relation_embeddings(relation_ids)
                if pin_outputs:
                    batch.node_embeddings = batch.node_embeddings.pin_memory()
                    batch.edge_embeddings = batch.edge_embeddings.pin_memory()
            yield batch
