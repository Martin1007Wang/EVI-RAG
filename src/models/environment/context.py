from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CsrAdjacency:
    """Lightweight CSR adjacency wrapper that preserves multi-edges."""

    crow: torch.Tensor
    col: torch.Tensor
    edge_ids: torch.Tensor
    size: tuple[int, int]

    def crow_indices(self) -> torch.Tensor:
        return self.crow

    def col_indices(self) -> torch.Tensor:
        return self.col

    def values(self) -> torch.Tensor:
        return self.edge_ids


@dataclass(frozen=True)
class GraphEnvContext:
    """
    [环境基座] 静态图上下文，整个 Episode 生命周期内绝对不可变。
    由 Builder 在 DataLoader 输出后单次构建。
    """

    # 宏观物理统计
    num_graphs: int
    num_nodes_total: int
    # 基础图结构 (PyG Batch 语义)
    node_ptr: torch.Tensor  # [B+1]
    edge_index: torch.Tensor  # [2, E]
    edge_relations: torch.Tensor  # [E]
    edge_rel_global: torch.Tensor  # [E]
    edge_batch: torch.Tensor  # [E]
    node_batch: torch.Tensor  # [N]
    # 现代稀疏张量拓扑 (核心优化点：替代手写的 8 个张量)
    adj_t_fwd: CsrAdjacency
    adj_t_bwd: CsrAdjacency
    # 神经网络连续表征
    node_embeddings: torch.Tensor  # [N, d]
    node_tokens: torch.Tensor  # [N, d]
    relation_tokens: torch.Tensor  # [num_relations, d]
    question_emb: torch.Tensor  # [B, d]
    # 任务信标 (绝对坐标)
    q_local_indices: torch.Tensor  # [num_q]
    a_local_indices: torch.Tensor  # [num_answers]
    q_ptr: torch.Tensor  # [B+1]
    a_ptr: torch.Tensor  # [B+1]
    answer_entity_ids: torch.Tensor  # [num_answers]
    answer_ptr: torch.Tensor  # [B+1]
    # 辅助与元数据
    node_global_ids: torch.Tensor  # [N]
    dummy_mask: torch.Tensor  # [B]
    sample_ids: list[str]  # [B]
    # Runtime contract: current policy requires token-level question context.
    question_ctx: torch.Tensor  # [B, L, d] or [B, L, emb_dim]
    question_ctx_mask: torch.Tensor  # [B, L] bool mask where True denotes valid token
    heuristic_log_v: torch.Tensor | None = None  # [N], optional frozen guidance


__all__ = [
    "CsrAdjacency",
    "GraphEnvContext",
]
