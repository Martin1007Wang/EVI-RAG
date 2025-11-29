import logging
import math
import random
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected but optional
    np = None

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
from torch_geometric.loader import DataLoader

from ..g_retrieval_dataset import GRetrievalDataset

logger = logging.getLogger(__name__)


class RollingBatchSampler(BatchSampler):
    """Yield a fixed number of full batches per epoch without traversing the dataset each time."""

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        batches_per_epoch: Optional[int] = None,
        *,
        samples_per_epoch: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if dataset_len <= 0:
            raise ValueError("dataset_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batches_per_epoch is None and samples_per_epoch is None:
            raise ValueError("Specify batches_per_epoch or samples_per_epoch")
        if batches_per_epoch is None:
            batches_per_epoch = max(1, math.ceil(samples_per_epoch / batch_size))  # type: ignore[arg-type]

        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self._batches_per_epoch = int(batches_per_epoch)
        self.shuffle = bool(shuffle)
        self._cursor = 0
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed) if seed is not None else int(torch.seed() % (2**31 - 1)))
        self._perm = self._make_perm()

    def _make_perm(self) -> List[int]:
        perm = list(range(self.dataset_len))
        if self.shuffle:
            idx = torch.randperm(self.dataset_len, generator=self._rng).tolist()
            perm = [perm[i] for i in idx]
        return perm

    def __len__(self) -> int:
        return self._batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        bs = self.batch_size
        n = self.dataset_len
        for _ in range(self._batches_per_epoch):
            batch: List[int] = []
            while len(batch) < bs:
                if self._cursor >= n:
                    self._perm = self._make_perm()
                    self._cursor = 0
                take = min(bs - len(batch), n - self._cursor)
                if take <= 0:
                    raise RuntimeError("RollingBatchSampler dataset unexpectedly empty")
                batch.extend(self._perm[self._cursor : self._cursor + take])
                self._cursor += take
            yield batch


class UnifiedDataLoader(DataLoader):
    """PyG DataLoader with deferred, batched embedding lookups."""

    def __init__(
        self,
        dataset: GRetrievalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        samples_per_epoch: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        random_seed: Optional[int] = None,
        hard_negative_k: int = 0,
        hard_negative_similarity: str = "cosine",
        **kwargs,
    ):
        """Initialize the data loader with deferred embedding lookup."""
        self.random_seed = random_seed
        self.hard_negative_k = int(max(0, hard_negative_k))
        self.hard_negative_similarity = str(hard_negative_similarity).strip().lower()
        if self.hard_negative_similarity not in {"cosine", "dot"}:
            logger.warning(
                "Unknown hard_negative_similarity=%s; falling back to 'cosine'",
                self.hard_negative_similarity,
            )
            self.hard_negative_similarity = "cosine"
        if self.hard_negative_k > 0:
            logger.info(
                "Hard negative attachment enabled: k=%d similarity=%s",
                self.hard_negative_k,
                self.hard_negative_similarity,
            )

        # Wrap existing worker_init_fn if reproducibility is requested
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
        elif worker_init_fn is not None:
            kwargs["worker_init_fn"] = worker_init_fn

        # If a partial-epoch policy is requested, use a rolling batch sampler.
        use_rolling = (samples_per_epoch is not None) or (batches_per_epoch is not None)
        if use_rolling:
            logger.info(
                "RollingBatchSampler enabled (batches_per_epoch=%s, samples_per_epoch=%s)",
                batches_per_epoch,
                samples_per_epoch,
            )

        generator: Optional[torch.Generator] = None
        if random_seed is not None and not use_rolling:
            generator = torch.Generator()
            generator.manual_seed(int(random_seed))
            kwargs.setdefault("generator", generator)

        if use_rolling:
            # Build a batch sampler that yields fixed number of full batches each epoch
            batch_sampler = RollingBatchSampler(
                dataset_len=len(dataset),
                batch_size=batch_size,
                batches_per_epoch=batches_per_epoch,
                samples_per_epoch=samples_per_epoch,
                shuffle=shuffle,
                seed=random_seed,
            )

            # In batch_sampler mode, DataLoader must not receive batch_size/shuffle/drop_last
            for k in ["batch_size", "shuffle", "drop_last"]:
                if k in kwargs:
                    kwargs.pop(k)

            super().__init__(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                **kwargs,
            )
        else:
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                **kwargs,
            )
        self.shuffle = shuffle
        self.global_embeddings = dataset.global_embeddings
        logger.info("UnifiedDataLoader initialized: batch_size=%s", batch_size)

    def __iter__(self):
        entity_table = self.global_embeddings.entity_embeddings
        relation_table = self.global_embeddings.relation_embeddings
        if entity_table is None or relation_table is None:
            raise RuntimeError("Global embeddings not loaded before iteration")

        for batch in super().__iter__():
            node_embedding_ids = torch.as_tensor(batch.node_embedding_ids, dtype=torch.long, device="cpu")
            relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")

            if hasattr(batch, "edge_index") and batch.edge_index is not None:
                batch.reverse_edge_index = batch.edge_index.flip(0)

            batch.node_embeddings = self.global_embeddings.get_entity_embeddings(node_embedding_ids)
            batch.edge_embeddings = self.global_embeddings.get_relation_embeddings(relation_ids)

            if not hasattr(batch, "answer_entity_ids"):
                raise AttributeError("Batch missing answer_entity_ids required for metrics.")
            ans = batch.answer_entity_ids
            ans = torch.as_tensor(ans, dtype=torch.long, device="cpu")
            batch.answer_entity_ids = ans

            answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
            if answer_ptr is None and hasattr(batch, "_slice_dict"):
                answer_ptr = batch._slice_dict.get("answer_entity_ids")
            if answer_ptr is None and hasattr(batch, "answer_entity_ids_len"):
                lens = torch.as_tensor(batch.answer_entity_ids_len, dtype=torch.long, device="cpu").view(-1)
                answer_ptr = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)], dim=0)
            if answer_ptr is None:
                raise AttributeError("Batch missing answer_entity_ids_ptr; PyG collate may have failed.")
            batch.answer_entity_ids_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device="cpu")
            # Clean up intermediate length to keep batch light
            if hasattr(batch, "answer_entity_ids_len"):
                del batch.answer_entity_ids_len

            self._attach_hard_negatives(batch)

            yield batch

    def _attach_hard_negatives(self, batch: Any) -> None:
        if self.hard_negative_k <= 0:
            batch.hard_negatives = None
            return

        labels = getattr(batch, "labels", None)
        if labels is None:
            batch.hard_negatives = None
            return

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.cpu()

        if labels.numel() == 0:
            batch.hard_negatives = None
            return

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            batch.hard_negatives = None
            return

        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            batch.hard_negatives = None
            return
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index)
        edge_index = edge_index.cpu()

        head_idx = edge_index[0].long()
        tail_idx = edge_index[1].long()
        node_emb = batch.node_embeddings
        rel_emb = batch.edge_embeddings

        head_feat = node_emb[head_idx]
        tail_feat = node_emb[tail_idx]
        edge_feat = torch.cat([head_feat, rel_emb, tail_feat], dim=-1)
        if self.hard_negative_similarity == "cosine":
            edge_feat = F.normalize(edge_feat, dim=-1)

        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)

        if hasattr(batch, "batch") and getattr(batch, "batch") is not None:
            node_to_graph = batch.batch.to(head_idx.device)
            edge_graph_ids = node_to_graph[head_idx]
        else:
            edge_graph_ids = head_idx.new_zeros(edge_index.size(1))

        selected_pos: List[torch.Tensor] = []
        selected_neg: List[torch.Tensor] = []
        selected_sim: List[torch.Tensor] = []

        unique_graphs = torch.unique(edge_graph_ids[pos_indices])
        for gid in unique_graphs:
            pos_mask_gid = edge_graph_ids[pos_indices] == gid
            neg_mask_gid = edge_graph_ids[neg_indices] == gid
            pos_idx = pos_indices[pos_mask_gid]
            neg_idx = neg_indices[neg_mask_gid]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_feat = edge_feat[pos_idx]
            neg_feat = edge_feat[neg_idx]
            sim_matrix = pos_feat @ neg_feat.t()

            k = min(self.hard_negative_k, neg_feat.size(0))
            if k <= 0:
                continue

            values, indices = torch.topk(sim_matrix, k=k, dim=1, largest=True)
            selected_pos.append(pos_idx)
            selected_neg.append(neg_idx[indices])
            selected_sim.append(values)

        if not selected_pos:
            batch.hard_negatives = None
            return

        batch.hard_negatives = {
            "k": self.hard_negative_k,
            "pos_indices": torch.cat(selected_pos, dim=0).detach(),
            "neg_indices": torch.cat(selected_neg, dim=0).detach(),
            "similarity": torch.cat(selected_sim, dim=0).detach(),
        }
