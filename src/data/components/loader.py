from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but optional
    np = None

import torch
from torch_geometric.loader import DataLoader

from ..g_retrieval_dataset import GRetrievalDataset

logger = logging.getLogger(__name__)


class UnifiedDataLoader(DataLoader):
    """PyG DataLoader with deferred, batched embedding lookups."""

    def __init__(
        self,
        dataset: GRetrievalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.random_seed = random_seed

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

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
        self.shuffle = shuffle
        self.global_embeddings = dataset.global_embeddings
        logger.info("UnifiedDataLoader initialized: batch_size=%s shuffle=%s", batch_size, shuffle)

    def __iter__(self):
        entity_table = self.global_embeddings.entity_embeddings
        relation_table = self.global_embeddings.relation_embeddings
        if entity_table is None or relation_table is None:
            raise RuntimeError("Global embeddings not loaded before iteration")

        pin_outputs = bool(getattr(self, "pin_memory", False)) and torch.cuda.is_available()

        for batch in super().__iter__():
            node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device="cpu")
            relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")

            if hasattr(batch, "edge_index") and batch.edge_index is not None:
                batch.reverse_edge_index = batch.edge_index.flip(0)

            batch.node_embeddings = self.global_embeddings.get_entity_embeddings(node_embedding_ids)
            batch.edge_embeddings = self.global_embeddings.get_relation_embeddings(relation_ids)
            if pin_outputs:
                batch.node_embeddings = batch.node_embeddings.pin_memory()
                batch.edge_embeddings = batch.edge_embeddings.pin_memory()

            if not hasattr(batch, "answer_entity_ids"):
                raise AttributeError("Batch missing answer_entity_ids required for metrics.")
            batch.answer_entity_ids = torch.as_tensor(batch.answer_entity_ids, dtype=torch.long, device="cpu")

            answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
            if answer_ptr is None and hasattr(batch, "_slice_dict"):
                answer_ptr = batch._slice_dict.get("answer_entity_ids")
            if answer_ptr is None and hasattr(batch, "answer_entity_ids_len"):
                lens = torch.as_tensor(batch.answer_entity_ids_len, dtype=torch.long, device="cpu").view(-1)
                answer_ptr = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)], dim=0)
            if answer_ptr is None:
                raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
            batch.answer_entity_ids_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device="cpu")
            if pin_outputs:
                batch.answer_entity_ids = batch.answer_entity_ids.pin_memory()
                batch.answer_entity_ids_ptr = batch.answer_entity_ids_ptr.pin_memory()
            if hasattr(batch, "answer_entity_ids_len"):
                del batch.answer_entity_ids_len

            yield batch
