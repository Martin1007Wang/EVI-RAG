"""
policy.py — 增量式子图扩张 MDP 策略网络

核心设计原则（第一性原理）：
  - query_h 是全局条件变量，不充当任何"特征"本身
  - active_edge（状态描述）和 candidate_edge（动作打分）使用独立编码器
  - 注意力权重要么显式使用，要么不计算
  - anchor_mask 缺失时强制 warning，不静默降级
  - 无向图对称边在候选集中去重（前提：edge_index 包含双向边）

flow_state_h vs state_h
───────────────────────
  state_h       供 action scoring（ExpandEdgeScorer / type_logits）使用，
                融合了 node_pool + edge_pool + anchor_pool 三路信息。
  flow_state_h  供 FlowHead / ZHead 估计 log F~(s) 使用，
                仅由 active_nodes 的注意力池化得到（Markov-faithful），
                梯度不会反流到 active_edge_encoder 或 anchor_pool 分支。

max_steps
─────────
  由 Policy.__init__ 的 max_steps 参数持有，不依赖 State.max_steps
  （该字段已从 State 移除，改由 RolloutEngine 管理）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
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


_ACTION_TYPE_STATS_DIM = 7


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEdges:
    """Aligned candidate-edge container.

    Row i across all fields describes the same candidate edge.
    """

    edge_ids: torch.Tensor
    expand_logits: torch.Tensor
    batch_index: torch.Tensor
    relation_only_logits: torch.Tensor | None = None
    residual_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.edge_ids.ndim != 1:
            raise ValueError(f"edge_ids must be 1-D, got shape {tuple(self.edge_ids.shape)}.")
        if self.edge_ids.dtype != torch.long:
            raise ValueError(f"edge_ids must have dtype torch.long, got {self.edge_ids.dtype}.")
        n = int(self.edge_ids.shape[0])
        if self.expand_logits.shape != (n,):
            raise ValueError(f"expand_logits shape {tuple(self.expand_logits.shape)} != ({n},).")
        if self.batch_index.shape != (n,):
            raise ValueError(f"batch_index shape {tuple(self.batch_index.shape)} != ({n},).")
        if self.batch_index.dtype != torch.long:
            raise ValueError(f"batch_index must have dtype torch.long, got {self.batch_index.dtype}.")
        for name, value in (
            ("relation_only_logits", self.relation_only_logits),
            ("residual_logits", self.residual_logits),
        ):
            if value is None:
                continue
            if value.shape != (n,):
                raise ValueError(f"{name} shape {tuple(value.shape)} != ({n},).")

    def __len__(self) -> int:
        return int(self.edge_ids.shape[0])

    def filter(self, mask: torch.Tensor) -> "CandidateEdges":
        if mask.shape != (len(self),):
            raise ValueError(f"mask shape {tuple(mask.shape)} != candidate count ({len(self)},).")
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must have dtype torch.bool, got {mask.dtype}.")
        return CandidateEdges(
            edge_ids=self.edge_ids[mask],
            expand_logits=self.expand_logits[mask],
            batch_index=self.batch_index[mask],
            relation_only_logits=(
                None if self.relation_only_logits is None else self.relation_only_logits[mask]
            ),
            residual_logits=(
                None if self.residual_logits is None else self.residual_logits[mask]
            ),
        )


@dataclass(frozen=True)
class PolicyStepOutput:
    """Full-batch policy output for one MDP step.

    ``state_h``      action-state representation used by action heads.
    ``flow_state_h`` cleaner state-flow readout used by ``FlowHead`` / ``ZHead``.
                     Computed solely from active_nodes (Markov-faithful); its
                     gradients are isolated from the action-scoring pathway.
    """

    query_h: torch.Tensor
    state_h: torch.Tensor
    type_logits: torch.Tensor
    candidates: CandidateEdges
    type_features: torch.Tensor | None = None
    flow_state_h: torch.Tensor | None = None
    edge_score_breakdown: EdgeScoreBreakdown | None = None
    # Computed in __post_init__; default=0 ensures pickle / dataclasses.replace safety.
    _batch_size: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.query_h.ndim != 2:
            raise ValueError(f"query_h must be 2-D (B, H), got shape {tuple(self.query_h.shape)}.")
        batch_size = int(self.query_h.shape[0])
        object.__setattr__(self, "_batch_size", batch_size)

        self._validate_graph_rows("state_h", self.state_h, batch_size)
        self._validate_graph_rows("type_logits", self.type_logits, batch_size)
        if self.type_logits.ndim != 2 or self.type_logits.shape[1] != 2:
            raise ValueError(f"type_logits must have shape (B, 2), got {tuple(self.type_logits.shape)}.")
        if self.type_features is not None:
            self._validate_graph_rows("type_features", self.type_features, batch_size)
        if self.flow_state_h is not None:
            self._validate_graph_rows("flow_state_h", self.flow_state_h, batch_size)

        if len(self.candidates) > 0:
            lo = int(self.candidates.batch_index.min().item())
            hi = int(self.candidates.batch_index.max().item())
            if lo < 0 or hi >= batch_size:
                raise ValueError(f"candidates.batch_index range [{lo}, {hi}] out of [0, {batch_size}).")

    @staticmethod
    def _validate_graph_rows(name: str, tensor: torch.Tensor, batch_size: int) -> None:
        if tensor.ndim < 1:
            raise ValueError(f"{name} must have a leading batch dimension.")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch dim {tensor.shape[0]} != query_h batch dim {batch_size}.")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def action_logits(self) -> dict[str, torch.Tensor]:
        return {"type_logits": self.type_logits}

    def validate_full_batch(self, num_graphs: int) -> None:
        if self._batch_size != int(num_graphs):
            raise ValueError(
                f"PolicyStepOutput must be full-batch: "
                f"query_h has {self._batch_size} rows but batch has {num_graphs} graphs."
            )


@dataclass(frozen=True)
class EdgeFeatureSlice:
    """Per-edge feature bundle after slicing by edge_ids."""

    edge_ids: torch.Tensor
    edge_index: torch.Tensor  # [2, E_slice]
    edge_batch_index: torch.Tensor  # [E_slice]
    edge_ctx_h: torch.Tensor  # [E_slice, H]  编码器输出（给 state/type 摘要使用）
    src_dyn_node_h: torch.Tensor  # [E_slice, H]  当前状态下的动态源节点表示
    dst_stat_node_h: torch.Tensor  # [E_slice, H]  静态目标节点语义
    rel_h: torch.Tensor  # [E_slice, H]  关系语义
    edge_state_ids: torch.Tensor  # [E_slice]


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------


class Policy(nn.Module):
    """增量式子图扩张 MDP 策略网络。

    特征消费分工
    ─────────────
    query_h          全局查询条件，注入所有打分/编码模块，不作为"特征"本身
    node_h           NBF 迭代后的动态节点表示
    rel_h            关系语义，用于边上下文编码和最终边打分
    feature_bank     静态 node/rel 特征（不被 NBF 平滑），直接送入 ExpandEdgeScorer
    flow_state_h     仅由 active_nodes readout 得到的流状态表示，供 FlowHead / ZHead 使用

    编码器分工
    ─────────────
    active_edge_encoder    编码已激活边（描述当前子图结构）→ state_h 的 edge_pool
    candidate_edge_encoder 编码候选边（描述扩张价值）→ expand/stop 的候选摘要

    注意
    ────
    anchor 节点现在使用门控 query 融合而不是硬替换，因此 anchor_pool
    重新成为有信息量的状态摘要：它保留 anchor 实体语义差异，并带有
    轻量 query 条件偏移。
    """

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        hidden_dim: int = 1024,
        max_steps: int = 0,
        action_head_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
        undirected: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.undirected = undirected
        backbone_hidden_dim = int(dict(backbone_cfg).get("hidden_dim", hidden_dim))
        if backbone_hidden_dim != hidden_dim:
            raise ValueError(
                "Policy hidden_dim must match backbone hidden_dim. "
                f"Got policy hidden_dim={hidden_dim} and backbone hidden_dim={backbone_hidden_dim}."
            )
        # max_steps 由 Policy 持有，不依赖 State.max_steps（已从 State 移除）。
        # 用于 _build_action_type_features 中的剩余预算特征计算。
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
        self.max_steps = max_steps

        resolved_action_head_cfg = dict(action_head_cfg or {})
        configured_type_feature_dim = resolved_action_head_cfg.pop("type_feature_dim", None)
        if configured_type_feature_dim not in (None, hidden_dim):
            raise ValueError(
                "action_head.type_feature_dim is derived from policy hidden_dim and "
                f"must equal {hidden_dim}, got {configured_type_feature_dim}."
            )

        self.backbone = NBFBackbone(**backbone_cfg)
        self.z_head = ZHead(hidden_dim=hidden_dim)
        self.flow_head = FlowHead(hidden_dim=hidden_dim)
        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            type_feature_dim=hidden_dim,
            **resolved_action_head_cfg,
        )
        self.expand_edge_scorer = ExpandEdgeScorer(
            hidden_dim=hidden_dim,
            **dict(edge_scorer_cfg or {}),
        )

        # active / candidate 边使用独立编码器
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
        self.action_type_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + _ACTION_TYPE_STATS_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 节点注意力池化（query 条件）
        self.node_state_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 状态摘要：node_pool ⊕ edge_pool ⊕ anchor_pool → state_h
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # -----------------------------------------------------------------------
    # 静态预计算
    # -----------------------------------------------------------------------

    def precompute_backbone_static_context(self, batch: RetrievalBatch) -> PreparedGNNInput:
        return self.backbone.precompute_static(batch)

    def update_training_schedule(self, *, global_step: int) -> None:
        del global_step

    # -----------------------------------------------------------------------
    # 池化工具
    # -----------------------------------------------------------------------

    @staticmethod
    def _pool_masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """mask=True 的行做 per-graph 均值池化。"""
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        masked_values = values[mask]
        masked_batch = batch_index[mask]
        pooled_sum = scatter_sum(masked_values, masked_batch, dim=0, dim_size=num_graphs)
        pooled_count = (
            torch.bincount(masked_batch, minlength=num_graphs)
            .to(dtype=values.dtype, device=values.device)
            .clamp_min(1.0)
        )
        return pooled_sum / pooled_count.unsqueeze(-1)

    @staticmethod
    def _pool_selected_mean(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """已选行做 per-graph 均值池化。"""
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        pooled_sum = scatter_sum(values, batch_index, dim=0, dim_size=num_graphs)
        pooled_count = (
            torch.bincount(batch_index, minlength=num_graphs)
            .to(dtype=values.dtype, device=values.device)
            .clamp_min(1.0)
        )
        return pooled_sum / pooled_count.unsqueeze(-1)

    def _pool_masked_attention(
        self,
        *,
        values: torch.Tensor,
        query_h: torch.Tensor,
        mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """per-graph query 条件注意力池化（只对 mask=True 的行）。"""
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        masked_values = values[mask]
        if masked_values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))
        masked_batch = batch_index[mask]
        masked_query = query_h.index_select(0, masked_batch)
        masked_logits = self.node_state_attn(torch.cat([masked_values, masked_query], dim=-1)).squeeze(-1)
        masked_weights = pyg_softmax(masked_logits, masked_batch, num_nodes=num_graphs)
        return scatter_sum(
            masked_weights.unsqueeze(-1) * masked_values,
            masked_batch,
            dim=0,
            dim_size=num_graphs,
        )

    # -----------------------------------------------------------------------
    # 边特征构建
    # -----------------------------------------------------------------------

    def _encode_edge_ctx(
        self,
        *,
        src_node_h: torch.Tensor,
        dst_node_h: torch.Tensor,
        rel_h: torch.Tensor,
        query_h: torch.Tensor,
        edge_batch_index: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """拼接显式边因子后过编码器。"""
        query_per_edge = query_h.index_select(0, edge_batch_index)
        ctx_input = torch.cat(
            [
                src_node_h,
                rel_h,
                dst_node_h,
                query_per_edge,
            ],
            dim=-1,
        )
        return encoder(ctx_input)

    def _slice_edge_features(
        self,
        *,
        batch: RetrievalBatch,
        backbone_output: BackboneOutput,
        edge_ids: torch.Tensor,
        encoder: nn.Module,
        use_static_dst_in_ctx: bool,
    ) -> EdgeFeatureSlice:
        """按 edge_ids 切片，构建 EdgeFeatureSlice。

        ``edge_ctx_h`` 供 state/type 摘要使用；最终 ExpandEdgeScorer 则显式
        消费 ``[query, src_dyn_node_h, rel_h, dst_stat_node_h]``。
        """
        d = self.hidden_dim
        dtype = backbone_output.node_h.dtype

        if edge_ids.numel() == 0:
            return EdgeFeatureSlice(
                edge_ids=edge_ids,
                edge_index=edge_ids.new_zeros(2, 0),
                edge_batch_index=edge_ids.new_zeros(0),
                edge_ctx_h=edge_ids.new_zeros(0, d, dtype=dtype),
                src_dyn_node_h=edge_ids.new_zeros(0, d, dtype=dtype),
                dst_stat_node_h=edge_ids.new_zeros(0, d, dtype=dtype),
                rel_h=edge_ids.new_zeros(0, d, dtype=dtype),
                edge_state_ids=edge_ids.new_zeros(0),
            )

        fb = backbone_output.feature_bank
        edge_index = batch.edge_index.index_select(1, edge_ids)
        edge_batch_index = batch.edge_batch.index_select(0, edge_ids)
        src = edge_index[0]
        dst = edge_index[1]
        src_dyn_node_h = backbone_output.node_h.index_select(0, src)
        dst_stat_node_h = fb.node_h.index_select(0, dst)
        if use_static_dst_in_ctx:
            dst_ctx_node_h = dst_stat_node_h
        else:
            dst_ctx_node_h = backbone_output.node_h.index_select(0, dst)

        edge_ctx_h = self._encode_edge_ctx(
            src_node_h=src_dyn_node_h,
            dst_node_h=dst_ctx_node_h,
            rel_h=backbone_output.rel_h.index_select(0, edge_ids),
            query_h=backbone_output.query_h,
            edge_batch_index=edge_batch_index,
            encoder=encoder,
        )

        return EdgeFeatureSlice(
            edge_ids=edge_ids,
            edge_index=edge_index,
            edge_batch_index=edge_batch_index,
            edge_ctx_h=edge_ctx_h,
            src_dyn_node_h=src_dyn_node_h,
            dst_stat_node_h=dst_stat_node_h,
            rel_h=fb.rel_h.index_select(0, edge_ids),
            edge_state_ids=backbone_output.edge_state_ids.index_select(0, edge_ids),
        )

    # -----------------------------------------------------------------------
    # 状态编码
    # -----------------------------------------------------------------------

    def _encode_flow_state(
        self,
        *,
        batch: RetrievalBatch,
        node_h: torch.Tensor,
        query_h: torch.Tensor,
        active_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Markov-faithful flow readout built only from active nodes.

        仅使用 active_nodes 做注意力池化，保证 flow_state_h 是严格 Markov 的：
        F~(s) 只依赖当前激活节点集合，不依赖 active_edges 或 anchor 信息。
        """
        return self._pool_masked_attention(
            values=node_h,
            query_h=query_h,
            mask=active_nodes,
            batch_index=batch.batch,
            num_graphs=batch.num_graphs,
        )

    def _encode_state(
        self,
        *,
        batch: RetrievalBatch,
        node_pool: torch.Tensor,
        active_edge_ctx_h: torch.Tensor,
        active_edge_batch_index: torch.Tensor,
        node_h: torch.Tensor,
    ) -> torch.Tensor:
        """三路融合：node_pool ⊕ edge_pool ⊕ anchor_pool → state_h。

        node_pool 由调用方传入（复用 _encode_flow_state 的结果），
        避免对 node_state_attn 的重复前向计算。
        """
        num_graphs = batch.num_graphs

        edge_pool = self._pool_selected_mean(active_edge_ctx_h, active_edge_batch_index, num_graphs)
        anchor_mask = getattr(batch, "is_anchor_mask", None)
        if anchor_mask is None:
            raise ValueError("RetrievalBatch.is_anchor_mask is required to build anchor_pool.")
        anchor_pool = self._pool_masked_mean(node_h, anchor_mask, batch.batch, num_graphs)
        return self.state_encoder(torch.cat([node_pool, edge_pool, anchor_pool], dim=-1))

    # -----------------------------------------------------------------------
    # 候选边
    # -----------------------------------------------------------------------

    def _build_candidate_mask(
        self,
        *,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """构建候选边掩码：至少一端已激活且自身未激活。

        undirected=True 时通过 src < dst 去除对称冗余边，
        前提是 edge_index 中每条无向边以双向形式存储（PyG 默认约定）。
        """
        valid = (active_nodes[edge_src] | active_nodes[edge_dst]) & ~active_edges
        if self.undirected:
            valid = valid & (edge_src < edge_dst)
        return valid

    def _build_action_type_features(
        self,
        *,
        batch: RetrievalBatch,
        state: State,
        query_h: torch.Tensor,
        candidate_edge_ctx_h: torch.Tensor,
        candidate_edge_batch_index: torch.Tensor,
        has_valid_edges: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = batch.num_graphs
        dtype = query_h.dtype
        device = query_h.device

        candidate_pool = self._pool_selected_mean(candidate_edge_ctx_h, candidate_edge_batch_index, num_graphs)
        active_node_count = scatter_sum(
            state.active_nodes.to(dtype=dtype),
            batch.batch,
            dim=0,
            dim_size=num_graphs,
        ).unsqueeze(-1)
        active_edge_count = scatter_sum(
            state.active_edges.to(dtype=dtype),
            batch.edge_batch,
            dim=0,
            dim_size=num_graphs,
        ).unsqueeze(-1)
        candidate_count = (
            torch.bincount(candidate_edge_batch_index, minlength=num_graphs)
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
            torch.bincount(batch.edge_batch, minlength=num_graphs).to(dtype=dtype, device=device).unsqueeze(-1)
        )
        safe_total_edge_count = total_edge_count.clamp_min(1.0)

        # 剩余预算特征：使用 self.max_steps（由 Policy 持有），
        # 不再依赖已废弃的 State.max_steps。
        rollout_step = max(int(getattr(state, "rollout_step", 0)), 0)
        max_steps = self.max_steps
        step_value = torch.full((num_graphs, 1), float(rollout_step), device=device, dtype=dtype)
        max_step_value = torch.full((num_graphs, 1), float(max_steps), device=device, dtype=dtype)
        safe_max_step_value = max_step_value.clamp_min(1.0)
        remaining_budget = (max_step_value - step_value).clamp_min(0.0)

        stats = torch.cat(
            [
                active_node_count / total_node_count,
                active_edge_count / safe_total_edge_count,
                candidate_count / safe_total_edge_count,
                has_valid_edges.to(dtype=dtype).unsqueeze(-1),
                step_value / safe_max_step_value,
                remaining_budget / safe_max_step_value,
                remaining_budget.eq(0.0).to(dtype=dtype),
            ],
            dim=-1,
        )
        # 保护性断言：stats 维度必须与 action_type_encoder 输入层一致
        assert stats.shape[-1] == _ACTION_TYPE_STATS_DIM, (
            f"stats dim {stats.shape[-1]} != _ACTION_TYPE_STATS_DIM "
            f"{_ACTION_TYPE_STATS_DIM}. "
            "Update _ACTION_TYPE_STATS_DIM if you add/remove stat features."
        )
        return self.action_type_encoder(torch.cat([query_h, candidate_pool, stats], dim=-1))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        batch: RetrievalBatch,
        state: State,
        backbone_static_context: PreparedGNNInput | None = None,
    ) -> PolicyStepOutput:
        active_nodes = state.active_nodes
        active_edges = state.active_edges

        # 1. Backbone
        backbone_output = self.backbone(
            batch,
            active_edges=active_edges,
            active_nodes=active_nodes,
            static_context=backbone_static_context,
        )
        node_h = backbone_output.node_h
        query_h = backbone_output.query_h

        # 2. flow_state_h：仅由 active_nodes 池化得到（Markov-faithful）
        #    同时作为 state_h 中 node_pool 分量，避免重复计算 node_state_attn。
        flow_state_h = self._encode_flow_state(
            batch=batch,
            node_h=node_h,
            query_h=query_h,
            active_nodes=active_nodes,
        )

        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]

        # 3. 已激活边 → active_edge_encoder
        active_edge_ids = torch.nonzero(active_edges, as_tuple=False).view(-1)
        active_edge_slice = self._slice_edge_features(
            batch=batch,
            backbone_output=backbone_output,
            edge_ids=active_edge_ids,
            encoder=self.active_edge_encoder,
            use_static_dst_in_ctx=False,
        )

        # 4. 状态编码（复用 flow_state_h 作为 node_pool，避免二次注意力计算）
        state_h = self._encode_state(
            batch=batch,
            node_pool=flow_state_h,
            active_edge_ctx_h=active_edge_slice.edge_ctx_h,
            active_edge_batch_index=active_edge_slice.edge_batch_index,
            node_h=node_h,
        )

        # 5. 候选边 → candidate_edge_encoder
        candidate_mask = self._build_candidate_mask(
            edge_src=edge_src,
            edge_dst=edge_dst,
            active_nodes=active_nodes,
            active_edges=active_edges,
        )
        candidate_edge_ids = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
        candidate_edge_slice = self._slice_edge_features(
            batch=batch,
            backbone_output=backbone_output,
            edge_ids=candidate_edge_ids,
            encoder=self.candidate_edge_encoder,
            use_static_dst_in_ctx=True,
        )
        has_valid_edges = scatter_sum(
            candidate_mask.int(),
            batch.edge_batch,
            dim=0,
            dim_size=batch.num_graphs,
        ).bool()

        # 6. 动作类型打分
        type_features = self._build_action_type_features(
            batch=batch,
            state=state,
            query_h=query_h,
            candidate_edge_ctx_h=candidate_edge_slice.edge_ctx_h,
            candidate_edge_batch_index=candidate_edge_slice.edge_batch_index,
            has_valid_edges=has_valid_edges,
        )
        type_logits = self.action_head(
            state_h=state_h,
            type_features=type_features,
        )["type_logits"]

        # 7. 候选边打分
        edge_score_breakdown = self.expand_edge_scorer(
            EdgeScorerInputs(
                src_dyn_node_h=candidate_edge_slice.src_dyn_node_h,
                edge_batch_index=candidate_edge_slice.edge_batch_index,
                dst_stat_node_h=candidate_edge_slice.dst_stat_node_h,
                rel_h=candidate_edge_slice.rel_h,
                query_h=query_h,
            ),
            return_breakdown=True,
        )

        output = PolicyStepOutput(
            query_h=query_h,
            state_h=state_h,
            type_logits=type_logits,
            type_features=type_features,
            flow_state_h=flow_state_h,
            edge_score_breakdown=edge_score_breakdown,
            candidates=CandidateEdges(
                edge_ids=candidate_edge_ids,
                expand_logits=edge_score_breakdown.final_logits,
                batch_index=candidate_edge_slice.edge_batch_index,
                relation_only_logits=edge_score_breakdown.relation_only_logits,
                residual_logits=edge_score_breakdown.residual_logits,
            ),
        )
        output.validate_full_batch(batch.num_graphs)
        return output

    # -----------------------------------------------------------------------
    # GFlowNet 辅助接口
    # -----------------------------------------------------------------------

    def root_log_z(self, query_h: torch.Tensor, root_state_h: torch.Tensor) -> torch.Tensor:
        """Full-batch root log Z, shape (B,)."""
        if query_h.ndim != 2:
            raise ValueError(f"query_h must be 2-D (B, H), got {tuple(query_h.shape)}.")
        if root_state_h.ndim != 2:
            raise ValueError(f"root_state_h must be 2-D (B, H), got {tuple(root_state_h.shape)}.")
        if root_state_h.shape[0] != query_h.shape[0]:
            raise ValueError(
                f"root_state_h batch dim {root_state_h.shape[0]} " f"!= query_h batch dim {query_h.shape[0]}."
            )
        result = self.z_head(query_h=query_h, root_state_h=root_state_h)
        if result.shape != query_h.shape[:1]:
            raise ValueError(f"z_head returned shape {tuple(result.shape)}, " f"expected {tuple(query_h.shape[:1])}.")
        return result

    def state_log_flow(self, query_h: torch.Tensor, flow_state_h: torch.Tensor) -> torch.Tensor:
        """Full-batch state log flows, shape (B,).

        参数名改为 flow_state_h（原为 state_h），与 PolicyStepOutput.flow_state_h
        以及 rollout.py 的调用约定保持一致，避免误传 action-scoring 的 state_h。
        """
        if query_h.ndim != 2:
            raise ValueError(f"query_h must be 2-D (B, H), got {tuple(query_h.shape)}.")
        if flow_state_h.ndim != 2:
            raise ValueError(f"flow_state_h must be 2-D (B, H), got {tuple(flow_state_h.shape)}.")
        if flow_state_h.shape[0] != query_h.shape[0]:
            raise ValueError(
                f"flow_state_h batch dim {flow_state_h.shape[0]} " f"!= query_h batch dim {query_h.shape[0]}."
            )
        result = self.flow_head(query_h=query_h, state_h=flow_state_h)
        if result.shape != query_h.shape[:1]:
            raise ValueError(
                f"flow_head returned shape {tuple(result.shape)}, " f"expected {tuple(query_h.shape[:1])}."
            )
        return result


__all__ = ["CandidateEdges", "Policy", "PolicyStepOutput"]
