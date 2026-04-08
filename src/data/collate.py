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
        batch.node_tokens = _resolve_non_text_node_tokens(
            node_tokens=batch.node_tokens,
            edge_relation_tokens=batch.edge_relation_tokens,
            edge_index=batch.edge_index,
            embedding_ids=emb_ids,
        )


def _resolve_non_text_node_tokens(
    *,
    node_tokens: torch.Tensor,
    edge_relation_tokens: torch.Tensor,
    edge_index: torch.Tensor,
    embedding_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Give emb_id==0 nodes a local DeepSets-style initialisation.

    Priority:
    1. Mean of 1-hop textual neighbour embeddings.
    2. If none exists, mean of incident relation embeddings.
    3. If the node is isolated, keep the original zero-row embedding.
    """
    if node_tokens.numel() == 0:
        return node_tokens

    non_text_mask = embedding_ids.eq(0)
    if not bool(non_text_mask.any().item()):
        return node_tokens

    if edge_index.numel() == 0:
        return node_tokens

    resolved = node_tokens.clone()
    base_node_tokens = node_tokens
    num_nodes = int(node_tokens.size(0))
    feature_dim = int(node_tokens.size(1))
    device = node_tokens.device
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    text_mask = ~non_text_mask

    text_sum = base_node_tokens.new_zeros((num_nodes, feature_dim))
    text_count = base_node_tokens.new_zeros(num_nodes)

    dst_is_text = text_mask.index_select(0, dst)
    if bool(dst_is_text.any().item()):
        text_sum.index_add_(0, src[dst_is_text], base_node_tokens[dst[dst_is_text]])
        text_count.index_add_(
            0,
            src[dst_is_text],
            torch.ones_like(src[dst_is_text], dtype=text_count.dtype),
        )

    src_is_text = text_mask.index_select(0, src)
    if bool(src_is_text.any().item()):
        text_sum.index_add_(0, dst[src_is_text], base_node_tokens[src[src_is_text]])
        text_count.index_add_(
            0,
            dst[src_is_text],
            torch.ones_like(dst[src_is_text], dtype=text_count.dtype),
        )

    has_text_neighbour = text_count.gt(0)
    if bool((non_text_mask & has_text_neighbour).any().item()):
        text_mean = text_sum / text_count.clamp_min(1.0).unsqueeze(-1)
        target_mask = non_text_mask & has_text_neighbour
        resolved[target_mask] = text_mean[target_mask]

    remaining_mask = non_text_mask & ~has_text_neighbour
    if not bool(remaining_mask.any().item()):
        return resolved

    rel_sum = edge_relation_tokens.new_zeros((num_nodes, feature_dim))
    rel_count = edge_relation_tokens.new_zeros(num_nodes)
    rel_sum.index_add_(0, src, edge_relation_tokens)
    rel_sum.index_add_(0, dst, edge_relation_tokens)
    ones = torch.ones_like(src, dtype=rel_count.dtype)
    rel_count.index_add_(0, src, ones)
    rel_count.index_add_(0, dst, ones)

    has_relation_context = rel_count.gt(0)
    if bool((remaining_mask & has_relation_context).any().item()):
        rel_mean = rel_sum / rel_count.clamp_min(1.0).unsqueeze(-1)
        target_mask = remaining_mask & has_relation_context
        resolved[target_mask] = rel_mean[target_mask]

    return resolved


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
