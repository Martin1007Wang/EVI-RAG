from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from src.data.schema import RetrievalBatch


@dataclass(frozen=True, slots=True)
class FeaturePack:
    """
    batch 内所有特征的统一容器，所有张量均已投影到 hidden_dim 空间。

    字段语义：
    question_h  [G, H]  — 投影后问题嵌入（G = batch 内图数量）
    entity_h    [N, H]  — 投影后实体嵌入（N = batch 内节点数，含 CVT fallback）
    edge_h      [E, H]  — EdgeEncoder 输出（含 src/rel/dst 三路融合）
      relation_h  [E, H]  — 纯关系语义，按边展开，供 EdgeEncoder 使用
    frontier_prune_score [E] — 原始 question/relation 语义空间上的静态剪枝分数

    注：device 字段已删除。任何 tensor 字段本身携带 .device 属性；
        存储冗余字段会引入一致性隐患（tensor 迁移后 device 字段不自动更新）。
        需要时请使用 feature_pack.question_h.device。
    """

    question_h: torch.Tensor  # [G, H]
    entity_h: torch.Tensor  # [N, H]
    edge_h: torch.Tensor  # [E, H]
    relation_h: torch.Tensor  # [E, H]
    frontier_prune_score: torch.Tensor  # [E]


class EdgeEncoder(nn.Module):
    """
    将 (src_h, relation_h, dst_h) 三路特征线性融合为单条边的表示。

    设计原则：
      - 三类输入来自同一 BGE 球面，经同规格投影后已在统一空间，
        W_e 的作用是学习三路融合权重，不引入额外非线性。
      - 非线性推理能力交由下游 FlowEstimator 的 MLP 完成，
        保持 EdgeEncoder 输出的可解释性。
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # W_e ∈ R^{H × 3H}，bias=False（输入已 LN，原点对称）
        self.proj = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        src_h: torch.Tensor,  # [E, H]
        relation_h: torch.Tensor,  # [E, H]
        dst_h: torch.Tensor,  # [E, H]
    ) -> torch.Tensor:  # [E, H]
        return self.norm(self.proj(torch.cat([src_h, relation_h, dst_h], dim=-1)))


class StateEncoder(nn.Module):
    """
    将 (question, 已选边集合) 压缩为单个状态向量 state_h。

    接口：
      forward(question_h, selected_edge_h)
        — 单 state 接口，兼容旧代码路径，内部委托给 forward_batched。
        — selected_edge_h=None 或空 tensor 时走空状态路径。

      forward_batched(question_h, selected_edge_h, key_padding_mask, is_empty)
        — 批量接口，由 ForwardPolicy._build_state_h_batched 调用。
        — selected_edge_h 为 padded tensor [S, L_max, H]，
          key_padding_mask=True 表示 padding 位（被 attention 忽略）。
        — is_empty=True 的 state 用 empty_state_emb 替代 attention 聚合结果。

    架构：
      CrossAttn(query=question_h, key/value=selected_edge_h)
      → fusion MLP([question_h ‖ attn_out]) → state_h [*, H]

    empty_state_emb 是可学习参数，为空状态提供有意义的初始流估计起点，
    梯度可以回传到初始状态 z_0 的 log F(z_0) ≈ log Z。
    """

    def __init__(self, *, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        self.hidden_dim = hidden_dim

        # query: question_h；key/value: selected_edge_h
        # batch_first=True：输入形状 [B, L, H]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
            dropout=0.1,  # 作用在 attention weights，而不是 attn 输出
        )
        # 空状态（edge_count=0）的可学习表示
        self.empty_state_emb = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.empty_state_emb.view(1, -1))

        # [question_h ‖ attn_out] → state_h
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        *,
        question_h: torch.Tensor,  # [S, H] or [H]
        selected_edge_h: torch.Tensor | None,  # [S, L_max, H], [L, H], or None
        key_padding_mask: torch.Tensor | None = None,  # [S, L_max]，True = padding 位
        is_empty: torch.Tensor | None = None,  # [S] bool，True = 该 state 无已选边
    ) -> torch.Tensor:  # [S, H]
        if key_padding_mask is None or is_empty is None:
            return self._forward_single(
                question_h=question_h,
                selected_edge_h=selected_edge_h,
            )
        if selected_edge_h is None:
            raise ValueError(
                "selected_edge_h must be provided for batched state encoding."
            )
        return self.forward_batched(
            question_h=question_h,
            selected_edge_h=selected_edge_h,
            key_padding_mask=key_padding_mask,
            is_empty=is_empty,
        )

    def forward_batched(
        self,
        *,
        question_h: torch.Tensor,  # [S, H]
        selected_edge_h: torch.Tensor,  # [S, L_max, H]，padding 用 0 填充
        key_padding_mask: torch.Tensor,  # [S, L_max]，True = padding 位
        is_empty: torch.Tensor,  # [S] bool，True = 该 state 无已选边
    ) -> torch.Tensor:  # [S, H]
        """
        批量编码 S 个 state。

        CrossAttn 的 query 是 question_h（shape [S, 1, H]），
        key/value 是 padded 已选边集合（shape [S, L_max, H]）。
        key_padding_mask 屏蔽 padding 位，保证 attention 仅在有效边上聚合。

        空 state 的处理：
          仅对非空 state 计算 attention；空 state 直接用 empty_state_emb
          替代 attn_out，再统一送入 fusion。
        """
        S, H = question_h.shape

        attn_agg = self.empty_state_emb.unsqueeze(0).expand(S, H).clone()
        if bool(is_empty.all()):
            return self.fusion(torch.cat([question_h, attn_agg], dim=-1))

        non_empty = ~is_empty
        q = question_h[non_empty].unsqueeze(1)  # [S_nonempty, 1, H]
        attn_out, _ = self.cross_attn(
            query=q,
            key=selected_edge_h[non_empty],
            value=selected_edge_h[non_empty],
            key_padding_mask=key_padding_mask[non_empty],
        )  # [S_nonempty, 1, H]
        attn_agg[non_empty] = attn_out.squeeze(1).to(dtype=attn_agg.dtype)

        return self.fusion(torch.cat([question_h, attn_agg], dim=-1))  # [S, H]

    def _forward_single(
        self,
        *,
        question_h: torch.Tensor,
        selected_edge_h: torch.Tensor | None,
    ) -> torch.Tensor:
        question_h = question_h.float().reshape(1, self.hidden_dim)
        if selected_edge_h is None or int(selected_edge_h.numel()) == 0:
            selected = question_h.new_zeros((1, 1, self.hidden_dim))
            key_padding_mask = torch.ones(
                (1, 1), dtype=torch.bool, device=question_h.device
            )
            is_empty = torch.ones((1,), dtype=torch.bool, device=question_h.device)
        else:
            selected = selected_edge_h.float().reshape(1, -1, self.hidden_dim)
            key_padding_mask = torch.zeros(
                (1, int(selected.size(1))), dtype=torch.bool, device=question_h.device
            )
            is_empty = torch.zeros((1,), dtype=torch.bool, device=question_h.device)
        return self.forward_batched(
            question_h=question_h,
            selected_edge_h=selected,
            key_padding_mask=key_padding_mask,
            is_empty=is_empty,
        )


class FeatureEncoder(nn.Module):
    """
    把 L2-normed BGE 语义特征投影到 GFlowNet 可训练空间。

    sem_dim   : BGE 输出维度（默认 1024），来自 semantic table 宽度
    hidden_dim: GFlowNet 工作维度（默认 512），允许 != sem_dim

    投影设计：
      - bias=False：BGE 输出已 L2 归一化，原点对称，平移偏置无意义
      - LayerNorm(hidden_dim)：替代 L2 球面约束，允许各向异性决策空间
      - 三个投影独立：question / entity / relation 语义角色不同，保留任务特化能力
    """

    # 类型注解：register_buffer 注册的只读张量
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
        self.register_buffer(
            "entity_text_semantic_table",
            entity_text_semantic_table.float(),
            persistent=False,
        )
        self.register_buffer(
            "text_row_by_entity_id",
            text_row_by_entity_id.long(),
            persistent=False,
        )
        self.register_buffer(
            "entity_relation_neighborhood_semantic_table",
            entity_relation_neighborhood_semantic_table.float(),
            persistent=False,
        )
        self.register_buffer(
            "relation_neighborhood_row_by_entity_id",
            relation_neighborhood_row_by_entity_id.long(),
            persistent=False,
        )
        self.register_buffer(
            "relation_semantic_table",
            relation_semantic_table.float(),
            persistent=False,
        )

        self._sem_dim = sem_dim
        self._hidden_dim = hidden_dim

        # ── 三个独立 Linear Projector + LayerNorm ────────────────────
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
        entity_ids: torch.Tensor,  # [N]
    ) -> torch.Tensor:  # [N, sem_dim]
        """
        查表逻辑：有文本的实体取文本 embedding，
        无文本的 CVT 实体取邻域聚合 embedding（fallback）。

        查表优先级：
          1. entity_has_text  = text_row_by_entity_id[id] >= 0
          2. entity_has_neigh = relation_neighborhood_row_by_entity_id[id] >= 0
                                AND NOT entity_has_text
          3. 两者都没有 → ValueError（catalog 数据不完整）

        返回原始 sem 向量 [N, sem_dim]，未投影。
        """
        text_rows = self.text_row_by_entity_id.index_select(0, entity_ids)
        entity_has_text = text_rows.ge(0)

        neigh_rows = self.relation_neighborhood_row_by_entity_id.index_select(
            0, entity_ids
        )
        entity_has_neigh = neigh_rows.ge(0) & ~entity_has_text

        missing = ~entity_has_text & ~entity_has_neigh
        if missing.any():
            raise ValueError(
                "Every non-text entity must have a relation-neighborhood feature; "
                f"missing catalog entity ids: "
                f"{torch.unique(entity_ids[missing]).tolist()}"
            )

        raw = entity_ids.new_empty(
            entity_ids.size(0), self._sem_dim, dtype=torch.float32
        )

        # CVT fallback 先写，文本特征后写（文本优先级更高，覆盖 CVT）
        if entity_has_neigh.any():
            raw[entity_has_neigh] = (
                self.entity_relation_neighborhood_semantic_table.index_select(
                    0, neigh_rows[entity_has_neigh]
                )
            )
        if entity_has_text.any():
            raw[entity_has_text] = self.entity_text_semantic_table.index_select(
                0, text_rows[entity_has_text]
            )

        return raw

    def forward(self, batch: RetrievalBatch) -> FeaturePack:
        edge_src = batch.edge_index[0].long()
        edge_dst = batch.edge_index[1].long()

        question_sem = batch.question_emb.float()
        if question_sem.size(-1) != self._sem_dim:
            raise ValueError(
                f"batch.question_emb dim {question_sem.size(-1)} != sem_dim {self._sem_dim}"
            )
        question_h = self.question_norm(self.question_proj(question_sem))  # [G, H]

        entity_sem = self._lookup_entity_features(
            batch.node_entity_catalog_ids.long()
        )  # [N, sem_dim]
        entity_h = self.entity_norm(self.entity_proj(entity_sem))  # [N, H]

        relation_ids = batch.edge_relation_catalog_ids.long()
        unique_relation_ids, inverse_relation_ids = torch.unique(
            relation_ids, return_inverse=True
        )
        unique_relation_sem = self.relation_semantic_table.index_select(
            0, unique_relation_ids
        )
        unique_relation_h = self.relation_norm(self.relation_proj(unique_relation_sem))
        relation_h = unique_relation_h.index_select(0, inverse_relation_ids)
        frontier_prune_score = (
            question_sem.index_select(
                0,
                batch.edge_graph_ids.long(),
            )
            .mul(
                self.relation_semantic_table.index_select(0, relation_ids),
            )
            .sum(dim=-1)
        )

        src_h = entity_h.index_select(0, edge_src)  # [E, H]
        dst_h = entity_h.index_select(0, edge_dst)  # [E, H]
        edge_h = self.edge_encoder(src_h=src_h, relation_h=relation_h, dst_h=dst_h)

        return FeaturePack(
            question_h=question_h,
            entity_h=entity_h,
            edge_h=edge_h,
            relation_h=relation_h,
            frontier_prune_score=frontier_prune_score,
        )


__all__ = [
    "EdgeEncoder",
    "FeatureEncoder",
    "FeaturePack",
    "StateEncoder",
]
