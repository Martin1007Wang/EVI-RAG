"""
policy.py — 增量式子图扩张 MDP 策略网络 (纯粹 Subgraph 模式)

核心设计原则（第一性原理）：
  - Policy 是无状态函数 f_θ(s, t)，不持有任何环境规则（如 expand_budget）。
  - forward 一次性返回所有动作分布 (Actor) 与流量估计 (Critic)。
  - flow_state_h 仅由 active_nodes 在 node_h_state（Markov-faithful 通道）上
    注意力池化得到，绝不向外暴露，也不混入 policy 通道的表示。

backbone 双通道约定：
  backbone_output.node_h_state  → Markov-faithful，仅 active_edges 传播
                                   用于 FlowHead / _encode_flow_state / anchor_pool
  backbone_output.node_h_policy → frontier-aware，active_edges ∪ frontier 传播
                                   用于 EdgeScorer / active_edge_encoder / candidate_edge_encoder
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from .modules.backbone import PreparedGNNInput, BackboneOutput, NBFBackbone
from .modules.heads import (
    ActionHead,
    EdgeScoreBreakdown,
    EdgeScorerInputs,
    ExpandEdgeScorer,
    FlowHead,
    ZHead,
)
from .state import State


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateEdges:
    edge_ids: torch.Tensor
    expand_logits: torch.Tensor
    batch_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.edge_ids.shape[0])


@dataclass(frozen=True)
class PolicyStepOutput:
    """Actor-Critic 统一输出容器，不向外泄漏任何隐藏层表示。"""
    type_logits: torch.Tensor
    candidates: CandidateEdges
    state_log_flow: torch.Tensor
    root_log_z: torch.Tensor | None = None
    edge_score_breakdown: EdgeScoreBreakdown | None = None


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        hidden_dim: int = 1024,
        action_stats_dim: int = 6,  # ← CHANGED: 7→6，删除冗余的 (1 - expand_ratio)
        action_head_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        backbone_hidden_dim = int(dict(backbone_cfg).get("hidden_dim", hidden_dim))
        if backbone_hidden_dim != hidden_dim:
            raise ValueError(f"Dim mismatch: policy={hidden_dim}, backbone={backbone_hidden_dim}")

        self.backbone = NBFBackbone(**backbone_cfg)
        self.z_head = ZHead(hidden_dim=hidden_dim)
        self.flow_head = FlowHead(hidden_dim=hidden_dim)

        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            type_feature_dim=hidden_dim,
            **(action_head_cfg or {}),
        )
        self.expand_edge_scorer = ExpandEdgeScorer(
            hidden_dim=hidden_dim,
            **(edge_scorer_cfg or {}),
        )

        # 独立编码器：已激活边 vs 候选边
        self.active_edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 动作类型编码器：action_stats_dim 现在是 6
        self.action_type_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + action_stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 节点级注意力和图级状态聚合
        self.node_state_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def prepare_rollout_context(self, batch: RetrievalBatch) -> PreparedGNNInput:
        return self.backbone.precompute_static(batch)

    # -----------------------------------------------------------------------
    # 池化工具（纯函数式）
    # -----------------------------------------------------------------------

    @staticmethod
    def _pool_masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        masked_values = values[mask]
        masked_batch = batch_index[mask]
        pooled_sum = scatter_sum(masked_values, masked_batch, dim=0, dim_size=num_graphs)
        pooled_count = (
            torch.bincount(masked_batch, minlength=num_graphs)
            .to(values.dtype)
            .clamp_min(1.0)
        )
        return pooled_sum / pooled_count.unsqueeze(-1)

    @staticmethod
    def _pool_selected_mean(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        pooled_sum = scatter_sum(values, batch_index, dim=0, dim_size=num_graphs)
        pooled_count = (
            torch.bincount(batch_index, minlength=num_graphs)
            .to(values.dtype)
            .clamp_min(1.0)
        )
        return pooled_sum / pooled_count.unsqueeze(-1)

    # -----------------------------------------------------------------------
    # 特征构建：流状态（Critic 用）
    # -----------------------------------------------------------------------

    def _encode_flow_state(
        self,
        values: torch.Tensor,
        query_h: torch.Tensor,
        mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Markov-faithful flow readout.

        values 必须来自 backbone_output.node_h_state（仅 active_edges 传播通道），
        而非 node_h_policy。两者调用签名相同，由 forward() 负责传入正确的张量。
        """
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        masked_values = values[mask]
        if masked_values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))

        masked_batch = batch_index[mask]
        masked_query = query_h.index_select(0, masked_batch)

        masked_logits = self.node_state_attn(
            torch.cat([masked_values, masked_query], dim=-1)
        ).squeeze(-1)
        masked_weights = pyg_softmax(masked_logits, masked_batch, num_nodes=num_graphs)
        return scatter_sum(
            masked_weights.unsqueeze(-1) * masked_values,
            masked_batch,
            dim=0,
            dim_size=num_graphs,
        )

    # -----------------------------------------------------------------------
    # 特征构建：边编码（Actor 用）
    # -----------------------------------------------------------------------

    def _encode_edges(
        self,
        batch: RetrievalBatch,
        backbone_output: BackboneOutput,
        edge_mask: torch.Tensor,
        encoder: nn.Module,
        use_static_dst_in_ctx: bool,
        return_scorer_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, EdgeScorerInputs | None]:
        edge_src = batch.edge_index[0][edge_mask]
        edge_dst = batch.edge_index[1][edge_mask]
        edge_batch_index = batch.edge_batch[edge_mask]

        # ← CHANGED: src 改用 node_h_policy（frontier-aware 通道）
        # 原代码用 backbone_output.node_h（现已是 node_h_state），
        # 导致边打分器的 src 表示看不到候选边邻域信息。
        # node_h_policy 在 active_edges ∪ frontier 上传播，
        # 使 src 节点能感知到其 frontier 邻居，提升边打分质量。
        src_dyn_node_h = backbone_output.node_h_policy[edge_src]  # ← CHANGED

        # dst 保持静态特征不变：EdgeScorer 的先验分设计为无结构依赖，
        # 用静态 dst 是语义正确的（dst 未被激活时没有动态表示可用）。
        dst_stat_node_h = backbone_output.feature_bank.node_h[edge_dst]

        # ctx_input 中的 dst：
        #   active_edge_encoder (use_static_dst_in_ctx=False): dst 已在图中，可用动态表示
        #   candidate_edge_encoder (use_static_dst_in_ctx=True): dst 尚未激活，只有静态表示
        # ← CHANGED: 动态 dst 也改用 node_h_policy，理由同 src
        dst_ctx_node_h = (
            dst_stat_node_h
            if use_static_dst_in_ctx
            else backbone_output.node_h_policy[edge_dst]  # ← CHANGED
        )

        ctx_input = torch.cat([
            src_dyn_node_h,
            backbone_output.rel_h[edge_mask],
            dst_ctx_node_h,
            backbone_output.query_h.index_select(0, edge_batch_index),
        ], dim=-1)

        scorer_inputs = (
            EdgeScorerInputs(
                src_dyn_node_h=src_dyn_node_h,
                edge_batch_index=edge_batch_index,
                dst_stat_node_h=dst_stat_node_h,
                rel_h=backbone_output.feature_bank.rel_h[edge_mask],
                query_h=backbone_output.query_h,
            )
            if return_scorer_inputs
            else None
        )

        return encoder(ctx_input), edge_batch_index, scorer_inputs

    # -----------------------------------------------------------------------
    # 特征构建：动作类型统计特征
    # -----------------------------------------------------------------------

    def _build_action_type_features(
        self,
        batch: RetrievalBatch,
        state: State,
        query_h: torch.Tensor,
        candidate_pool: torch.Tensor,
        candidate_batch_index: torch.Tensor,
        has_valid_edges: torch.Tensor,
    ) -> torch.Tensor:
        """构造 6 维图环境动态与时序统计量。

        维度说明（共 6 维，已删除冗余的 remaining_ratio）：
          0  active_node_ratio    = |V_t| / |V|        图节点覆盖率
          1  active_edge_ratio    = |E_t| / |E|        已选边密度
          2  candidate_ratio      = |frontier| / |E|   待扩张边密度
          3  has_valid_edges                            是否有可扩张边（0/1）
          4  expand_ratio         = num_expands / expand_budget  时间进度
          5  is_last_step                               是否到达 horizon（0/1）

        删除 (1 - expand_ratio) 的原因：
          它与 expand_ratio 线性相关（相关系数 = -1），对 MLP 不提供任何额外信息，
          只在训练初期引入梯度共线性噪声。如需"剩余时间感"，
          expand_ratio 本身已经编码了这一信息。
        """
        num_graphs = batch.num_graphs
        dtype, device = query_h.dtype, query_h.device

        active_node_count = scatter_sum(
            state.active_nodes.to(dtype), batch.batch, dim=0, dim_size=num_graphs
        ).unsqueeze(-1)
        active_edge_count = scatter_sum(
            state.active_edges.to(dtype), batch.edge_batch, dim=0, dim_size=num_graphs
        ).unsqueeze(-1)
        candidate_count = (
            torch.bincount(candidate_batch_index, minlength=num_graphs)
            .to(dtype=dtype, device=device)
            .unsqueeze(-1)
        )

        total_node_count = (
            torch.bincount(batch.batch, minlength=num_graphs)
            .to(dtype=dtype, device=device)
            .unsqueeze(-1)
            .clamp_min(1.0)
        )
        total_edge_count = (
            torch.bincount(batch.edge_batch, minlength=num_graphs)
            .to(dtype=dtype, device=device)
            .unsqueeze(-1)
            .clamp_min(1.0)
        )

        expand_ratio = torch.full(
            (num_graphs, 1), float(state.expand_ratio), device=device, dtype=dtype
        )
        is_last_step = torch.full(
            (num_graphs, 1), float(state.remaining_budget <= 0), device=device, dtype=dtype
        )

        stats = torch.cat([
            active_node_count / total_node_count,   # dim 0
            active_edge_count / total_edge_count,   # dim 1
            candidate_count / total_edge_count,     # dim 2
            has_valid_edges.to(dtype).unsqueeze(-1),# dim 3
            expand_ratio,                           # dim 4
            # ← CHANGED: 删除 1.0 - expand_ratio（原 dim 5，冗余）
            is_last_step,                           # dim 5（原 dim 6）
        ], dim=-1)  # [B, 6]

        return self.action_type_encoder(
            torch.cat([query_h, candidate_pool, stats], dim=-1)
        )

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        batch: RetrievalBatch,
        state: State,
        rollout_context: PreparedGNNInput | None = None,
    ) -> PolicyStepOutput:

        # ── 1. 骨干网双通道前向 ───────────────────────────────────────────
        backbone_output = self.backbone(
            batch,
            active_edges=state.active_edges,
            active_nodes=state.active_nodes,
            static_context=rollout_context,
        )
        query_h = backbone_output.query_h

        # ── 2. Markov-faithful 流状态（Critic 通道）────────────────────────
        # ← CHANGED: 显式传入 node_h_state，不再依赖 .node_h 别名。
        # node_h_state 仅在 active_edges 上传播，满足严格马尔可夫性：
        #   active_edges(s₁) == active_edges(s₂) ⟹ flow_state_h(s₁) == flow_state_h(s₂)
        flow_state_h = self._encode_flow_state(
            values=backbone_output.node_h_state,  # ← CHANGED
            query_h=query_h,
            mask=state.active_nodes,
            batch_index=batch.batch,
            num_graphs=batch.num_graphs,
        )

        # ── 3. Critic 价值估计 ────────────────────────────────────────────
        state_log_flow = self.flow_head(query_h=query_h, state_h=flow_state_h)
        root_log_z = (
            self.z_head(query_h=query_h, root_state_h=flow_state_h)
            if state.num_expands == 0
            else None
        )

        # ── 4. Actor 状态表示（state_h = flow ⊕ edge_pool ⊕ anchor_pool）──
        # active_edge_encoder 使用 node_h_policy（frontier-aware），
        # 让 edge_pool 能感知到活跃边附近的候选扩张方向。
        active_edge_ctx_h, active_edge_batch_index, _ = self._encode_edges(
            batch,
            backbone_output,
            state.active_edges,
            self.active_edge_encoder,
            use_static_dst_in_ctx=False,
        )
        edge_pool = self._pool_selected_mean(
            active_edge_ctx_h, active_edge_batch_index, batch.num_graphs
        )

        # ← CHANGED: anchor_pool 改用 node_h_state。
        # anchor 节点是 E_0 的端点，属于已确定的图结构，其语义应与 flow_state_h 一致。
        # 用 node_h_policy 会引入 frontier 信息，污染 state_h 与 flow_state_h 的对齐。
        anchor_pool = self._pool_masked_mean(
            backbone_output.node_h_state,  # ← CHANGED
            batch.is_anchor_mask,
            batch.batch,
            batch.num_graphs,
        )

        state_h = self.state_encoder(
            torch.cat([flow_state_h, edge_pool, anchor_pool], dim=-1)
        )

        # ── 5. 生成扩张候选集（frontier）────────────────────────────────────
        edge_src, edge_dst = batch.edge_index[0], batch.edge_index[1]
        candidate_mask = (
            (state.active_nodes[edge_src] | state.active_nodes[edge_dst])
            & ~state.active_edges
        )
        candidate_edge_ids = torch.nonzero(candidate_mask, as_tuple=False).view(-1)

        # candidate_edge_encoder 使用 node_h_policy，
        # 使 src 的表示已经聚合了候选边侧的邻域信息，提升打分质量。
        candidate_edge_ctx_h, candidate_edge_batch_index, scorer_inputs = self._encode_edges(
            batch,
            backbone_output,
            candidate_mask,
            self.candidate_edge_encoder,
            use_static_dst_in_ctx=True,
            return_scorer_inputs=True,
        )
        has_valid_edges = scatter_sum(
            candidate_mask.int(), batch.edge_batch, dim=0, dim_size=batch.num_graphs
        ).bool()

        # ── 6. Actor 动作打分 ─────────────────────────────────────────────
        candidate_pool = self._pool_selected_mean(
            candidate_edge_ctx_h, candidate_edge_batch_index, batch.num_graphs
        )
        type_features = self._build_action_type_features(
            batch, state, query_h, candidate_pool, candidate_edge_batch_index, has_valid_edges
        )

        type_logits = self.action_head(
            state_h=state_h, type_features=type_features
        )["type_logits"]
        edge_score_breakdown = self.expand_edge_scorer(scorer_inputs, return_breakdown=True)

        return PolicyStepOutput(
            type_logits=type_logits,
            candidates=CandidateEdges(
                edge_ids=candidate_edge_ids,
                expand_logits=edge_score_breakdown.final_logits,
                batch_index=candidate_edge_batch_index,
            ),
            state_log_flow=state_log_flow,
            root_log_z=root_log_z,
            edge_score_breakdown=edge_score_breakdown,
        )


__all__ = ["CandidateEdges", "Policy", "PolicyStepOutput"]
