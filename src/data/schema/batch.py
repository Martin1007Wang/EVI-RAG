# src/schema/batch.py
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import torch
from torch_geometric.data import Batch
from .fields import SampleFields


class RetrievalBatch(Batch):
    """
    知识图谱检索的全局批次数据结构。
    核心哲学：环境拓扑与状态掩码在维度上天然对齐，由 PyG 原生 cat 机制接管。
    """

    if TYPE_CHECKING:
        # --- PyG 原生内置属性 ---
        ptr: torch.Tensor
        batch: torch.Tensor
        edge_index: torch.Tensor
        num_nodes: int

        # --- 从 LMDB 加载的原始全局语义张量 ---
        edge_relation_ids_global: torch.Tensor
        node_entity_ids_global: torch.Tensor
        question_emb: torch.Tensor

        # 核心修改：直接承接拼接后的全局布尔掩码 (shape: [Total_Nodes])
        is_anchor_mask: torch.Tensor
        is_target_mask: torch.Tensor
        answer_entity_ids_global: torch.Tensor

        # --- Collator 附加的、与计算轴对齐的张量 ---
        node_tokens: torch.Tensor
        edge_relation_tokens: torch.Tensor
        is_cvt: torch.Tensor
        heuristic_log_v: torch.Tensor

        # --- Collator 附加的辅助统计 ---
        node_ptr: torch.Tensor
        edge_batch: torch.Tensor
        edge_ptr: torch.Tensor

    @property
    def num_nodes_total(self) -> int:
        """安全地获取全局总节点数"""
        return self.num_nodes
