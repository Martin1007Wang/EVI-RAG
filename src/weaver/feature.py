from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class FeaturePack:
    question_h: torch.Tensor  # [G, H]   — projected question
    entity_h: torch.Tensor  # [N, H]   — projected entity, including CVT fallback
    edge_h: torch.Tensor  # [E, H]   — encoded edge (src+relation+dst)
    relation_h: torch.Tensor  # [E, H]   — pure relation semantics for Path1
    device: torch.device


class EdgeEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, *, src_h, relation_h, dst_h) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([src_h, relation_h, dst_h], dim=-1)))


class StateEncoder(nn.Module):
    """
    Compress (question, selected edges) into a single state vector.
    """

    def __init__(self, *, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        self.hidden_dim = hidden_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.empty_state = nn.Parameter(torch.zeros(hidden_dim))
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        *,
        question_h: torch.Tensor,
        selected_edge_h: torch.Tensor | None,
    ) -> torch.Tensor:
        question_h = question_h.float().reshape(1, self.hidden_dim)

        if selected_edge_h is None or selected_edge_h.numel() == 0:
            aggregated = self.empty_state.view(1, self.hidden_dim)
        else:
            q = question_h.unsqueeze(0)
            kv = selected_edge_h.float().reshape(-1, self.hidden_dim).unsqueeze(0)
            aggregated, _ = self.cross_attn(q, kv, kv)
            aggregated = aggregated.squeeze(1)

        return self.fusion(torch.cat([aggregated, question_h], dim=-1))


class FeatureEncoder(nn.Module):
    """
    把 L2-normed BGE 语义特征投影到 GFlowNet 可训练空间。

    sem_dim  : BGE 输出维度（默认 1024），来自 semantic table 宽度
    hidden_dim: GFlowNet 工作维度（默认 512），允许 != sem_dim
    """

    entity_text_semantic_table: torch.Tensor
    text_row_by_entity_id: torch.Tensor
    entity_relation_neighborhood_semantic_table: torch.Tensor
    relation_neighborhood_row_by_entity_id: torch.Tensor
    relation_semantic_table: torch.Tensor

    def __init__(
        self,
        *,
        entity_text_semantic_table: torch.Tensor,
        text_row_by_entity_id: torch.Tensor,
        entity_relation_neighborhood_semantic_table: torch.Tensor,
        relation_neighborhood_row_by_entity_id: torch.Tensor,
        relation_semantic_table: torch.Tensor,
        sem_dim: int = 1024,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        sem_dim = int(sem_dim)
        hidden_dim = int(hidden_dim)
        if sem_dim <= 0:
            raise ValueError("sem_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        _validate_semantic_width(entity_text_semantic_table, sem_dim, "entity_text_semantic_table")
        _validate_semantic_width(entity_relation_neighborhood_semantic_table, sem_dim, "entity_relation_neighborhood_semantic_table")
        _validate_semantic_width(relation_semantic_table, sem_dim, "relation_semantic_table")

        # ── 注册只读 buffer（不参与梯度） ──────────────────────────────
        self.register_buffer("entity_text_semantic_table", entity_text_semantic_table.float(), persistent=False)
        self.register_buffer("text_row_by_entity_id", text_row_by_entity_id.long(), persistent=False)
        self.register_buffer("entity_relation_neighborhood_semantic_table", entity_relation_neighborhood_semantic_table.float(), persistent=False)
        self.register_buffer("relation_neighborhood_row_by_entity_id", relation_neighborhood_row_by_entity_id.long(), persistent=False)
        self.register_buffer("relation_semantic_table", relation_semantic_table.float(), persistent=False)

        # ── 三个独立 Linear Projector（bias=False，L2-normed 输入无需偏置）──
        # 投影后接 LayerNorm：把各向同性约束交给 LN，不强制 L2 球面
        self._sem_dim = sem_dim
        self._hidden_dim = hidden_dim

        self.question_proj = nn.Linear(sem_dim, hidden_dim, bias=False)
        self.entity_proj = nn.Linear(sem_dim, hidden_dim, bias=False)
        self.relation_proj = nn.Linear(sem_dim, hidden_dim, bias=False)

        self.question_norm = nn.LayerNorm(hidden_dim)
        self.entity_norm = nn.LayerNorm(hidden_dim)
        self.relation_norm = nn.LayerNorm(hidden_dim)

        self.edge_encoder = EdgeEncoder(hidden_dim=hidden_dim)

    @property
    def sem_dim(self) -> int:
        return self._sem_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _lookup_entity_features(
        self,
        entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        查表逻辑：有文本的实体取文本 embedding，无文本的 CVT 实体取邻域聚合 embedding。
        返回 raw sem 向量 [N, sem_dim]，**未投影**。
        """
        text_rows = self.text_row_by_entity_id.index_select(0, entity_ids)
        entity_has_text = text_rows.ge(0)

        neigh_rows = self.relation_neighborhood_row_by_entity_id.index_select(0, entity_ids)
        # CVT fallback：只在没有文本时使用邻域特征
        entity_has_neigh = neigh_rows.ge(0) & ~entity_has_text

        missing = ~entity_has_text & ~entity_has_neigh
        if missing.any():
            raise ValueError(
                "Every non-text entity must have a relation-neighborhood feature; "
                f"missing catalog entity ids: {torch.unique(entity_ids[missing]).tolist()}"
            )

        raw = entity_ids.new_empty(entity_ids.size(0), self._sem_dim, dtype=torch.float32)

        if entity_has_neigh.any():
            raw[entity_has_neigh] = self.entity_relation_neighborhood_semantic_table.index_select(0, neigh_rows[entity_has_neigh])

        if entity_has_text.any():
            # 文本特征优先级高于邻域特征，最后写入覆盖
            raw[entity_has_text] = self.entity_text_semantic_table.index_select(0, text_rows[entity_has_text])

        return raw

    def forward(self, batch: RetrievalBatch) -> FeaturePack:
        edge_src = batch.edge_index[0].long()
        edge_dst = batch.edge_index[1].long()

        # ── Question 投影 ───────────────────────────────────────────
        question_sem = batch.question_emb.float()
        if question_sem.size(-1) != self._sem_dim:
            raise ValueError(f"batch.question_emb dim {question_sem.size(-1)} != sem_dim {self._sem_dim}")
        question_h = self.question_norm(self.question_proj(question_sem))  # [G, H]

        # ── Entity 查表 + 投影 ──────────────────────────────────────
        entity_sem = self._lookup_entity_features(batch.node_entity_catalog_ids.long())  # [N, sem_dim]
        entity_h = self.entity_norm(self.entity_proj(entity_sem))  # [N, H]

        # ── Relation 去重查表 + 投影 ────────────────────────────────
        relation_ids = batch.edge_relation_catalog_ids.long()
        unique_relation_ids, inverse_relation_ids = torch.unique(relation_ids, return_inverse=True)
        unique_relation_sem = self.relation_semantic_table.index_select(0, unique_relation_ids)
        unique_relation_h = self.relation_norm(self.relation_proj(unique_relation_sem))
        relation_h = unique_relation_h.index_select(0, inverse_relation_ids)  # [E, H]

        # ── Edge encoding ───────────────────────────────────────────
        src_h = entity_h.index_select(0, edge_src)
        dst_h = entity_h.index_select(0, edge_dst)
        edge_h = self.edge_encoder(src_h=src_h, relation_h=relation_h, dst_h=dst_h)

        return FeaturePack(
            question_h=question_h,
            entity_h=entity_h,
            edge_h=edge_h,
            relation_h=relation_h,
            device=edge_src.device,
        )


__all__ = [
    "EdgeEncoder",
    "FeatureEncoder",
    "FeaturePack",
    "StateEncoder",
]


def _validate_semantic_width(table: torch.Tensor, sem_dim: int, name: str) -> None:
    if table.dim() != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(table.shape)}.")
    if int(table.size(-1)) != int(sem_dim):
        raise ValueError(f"{name} width {int(table.size(-1))} != sem_dim {int(sem_dim)}.")
