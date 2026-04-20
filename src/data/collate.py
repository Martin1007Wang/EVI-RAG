from __future__ import annotations

from typing import Any, List, Optional, cast

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader.dataloader import Collater

from .retrieval import DataResource
from .schema import SampleFields
from .schema import RetrievalBatch


class RetrievalCollator:
    def __init__(
        self,
        data_resource: DataResource,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> None:
        self.data_resource = data_resource
        resolved_exclude_keys = list(exclude_keys or [])
        if SampleFields.BOUNDED_SUFFIX_COUNT not in resolved_exclude_keys:
            resolved_exclude_keys.append(SampleFields.BOUNDED_SUFFIX_COUNT)
        self._pyg_collater = Collater(
            dataset=cast(Dataset, None),
            follow_batch=follow_batch,
            exclude_keys=resolved_exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> RetrievalBatch:
        batch: RetrievalBatch = self._pyg_collater(batch_list)

        # 显式修正图级字段，避免被 PyG 按节点/边属性错误拼接
        batch.question_emb = torch.stack(
            [
                torch.as_tensor(sample.question_emb, dtype=torch.float32).reshape(-1)
                for sample in batch_list
            ],
            dim=0,
        )

        if batch.question_emb.ndim != 2:
            raise ValueError(
                f"question_emb must be 2D after collation, got shape {tuple(batch.question_emb.shape)}."
            )

        bounded_suffix_values = [
            getattr(sample, SampleFields.BOUNDED_SUFFIX_COUNT, None) for sample in batch_list
        ]
        if any(value is not None for value in bounded_suffix_values):
            if not all(value is not None for value in bounded_suffix_values):
                raise ValueError(
                    "bounded_suffix_count must be present for all samples in a batch or none of them."
                )
            batch.bounded_suffix_count = torch.cat(
                [
                    torch.as_tensor(value, dtype=torch.float32)
                    for value in bounded_suffix_values
                ],
                dim=1,
            )

        self._attach_embeddings(batch)
        batch.node_ptr = batch.ptr
        _attach_edge_batch(batch)
        return batch

    def _attach_embeddings(self, batch: RetrievalBatch) -> None:
        node_ids = batch.node_entity_ids_global.long()
        emb_ids = self.data_resource.entity_embedding_map[node_ids]

        batch.node_tokens = self.data_resource.embedding_store.get_entity_embeddings(
            emb_ids
        )
        batch.non_text_node_mask = emb_ids.eq(0)
        batch.is_cvt = self.data_resource.cvt_mask[node_ids]

        text_mask = ~batch.non_text_node_mask
        if text_mask.any() and not torch.isfinite(batch.node_tokens[text_mask]).all():
            raise ValueError(
                "Non-finite entity embeddings detected. Rebuild preprocess outputs."
            )

        batch.relation_tokens = self.data_resource.embedding_store.get_relation_embeddings(
            batch.edge_relation_ids_global.long()
        )
        if not torch.isfinite(batch.relation_tokens).all():
            raise ValueError(
                "Non-finite relation embeddings detected. Rebuild preprocess outputs."
            )


def _attach_edge_batch(batch: RetrievalBatch) -> None:
    edge_index = batch.edge_index
    node_ptr = batch.ptr
    num_graphs = node_ptr.numel() - 1
    device = edge_index.device

    if edge_index.numel() == 0:
        batch.edge_batch = torch.empty(0, dtype=torch.long, device=device)
        batch.edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        return

    edge_batch = torch.searchsorted(node_ptr, edge_index[0], side="right") - 1
    batch.edge_batch = edge_batch

    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    torch.cumsum(
        torch.bincount(edge_batch, minlength=num_graphs),
        dim=0,
        out=edge_ptr[1:],
    )
    batch.edge_ptr = edge_ptr
