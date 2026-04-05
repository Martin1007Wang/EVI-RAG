# src/data/collate.py
from __future__ import annotations

from typing import Any, List, Optional, cast
import torch
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Dataset

from .schema import RetrievalBatch
from .retrieval import DataResource


class RetrievalCollator:
    def __init__(
        self,
        data_resource: DataResource,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> None:
        self.data_resource = data_resource
        self._pyg_collater = Collater(
            dataset=cast(Dataset, None),  # PyG 运行时允许 None
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

    def __call__(self, batch_list: list[Any]) -> RetrievalBatch:
        batch: RetrievalBatch = self._pyg_collater(batch_list)
        self._attach_prepared_features(batch)
        _attach_graph_stats(batch)
        _attach_edge_batch(batch)
        return batch

    def _attach_prepared_features(self, batch: RetrievalBatch) -> None:
        # 将全局语义 ID 在 collate 阶段水合为 backbone 可直接消费的对齐张量。
        node_ids = batch.node_entity_ids_global.long()
        emb_ids = self.data_resource.entity_embedding_map[node_ids]
        batch.node_tokens = self.data_resource.embedding_store.get_entity_embeddings(
            emb_ids
        )
        batch.is_cvt = self.data_resource.cvt_mask[node_ids]
        batch.edge_relation_tokens = (
            self.data_resource.embedding_store.get_relation_embeddings(
                batch.edge_relation_ids_global.long()
            )
        )


def _attach_graph_stats(batch: RetrievalBatch) -> None:
    batch.node_ptr = batch.ptr


def _attach_edge_batch(batch: RetrievalBatch) -> None:
    """在不引入外部文件的情况下，高效计算边所属的图批次和指针"""
    edge_index = batch.edge_index
    node_ptr = batch.ptr
    num_graphs = node_ptr.numel() - 1

    if edge_index.numel() == 0:
        batch.edge_batch = torch.empty(0, dtype=torch.long, device=edge_index.device)
        batch.edge_ptr = torch.zeros(
            num_graphs + 1, dtype=torch.long, device=edge_index.device
        )
        return

    # 利用 PyTorch 的 searchsorted 快速定位每条边的 source 节点属于哪张图
    row = edge_index[0]
    edge_batch = torch.searchsorted(node_ptr, row, side="right") - 1
    batch.edge_batch = edge_batch

    # 计算 edge_ptr (类似 node_ptr，表示每张图的边在 edge_index 中的起始偏移)
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=edge_index.device)
    torch.cumsum(edge_counts, dim=0, out=edge_ptr[1:])
    batch.edge_ptr = edge_ptr
