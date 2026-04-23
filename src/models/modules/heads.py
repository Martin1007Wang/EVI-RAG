from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_sum

from src.utils.nn_utils import (
    build_mlp,
    init_xavier,
    require_finite,
    zero_last_linear,
)
from .backbone import BackboneOutput


class _ProjectedDotScalar(nn.Module):
    """Projected dot-product scorer with a small residual MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_residual_layers: int = 1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.score_scale = hidden_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residual = build_mlp(
            hidden_dim * 3,
            1,
            max(hidden_dim // 2, 1),
            num_residual_layers,
            dropout,
        )
        init_xavier(self.q_proj)
        init_xavier(self.k_proj)
        if zero_init:
            zero_last_linear(self.residual)

    def _score(self, ctx_q: torch.Tensor, ctx_s: torch.Tensor) -> torch.Tensor:
        projected_q = self.q_proj(ctx_q)
        projected_s = self.k_proj(ctx_s)
        bilinear = (projected_q * projected_s).sum(dim=-1) * self.score_scale
        residual = self.residual(
            torch.cat([ctx_q, ctx_s, ctx_q * ctx_s], dim=-1)
        ).squeeze(-1)
        return bilinear + residual


class ZHead(_ProjectedDotScalar):
    """log Z(q, s0) 估计器：图级别的条件配分函数常量，由查询和根状态学习而来。"""

    def forward(self, query_h: torch.Tensor, root_state_h: torch.Tensor) -> torch.Tensor:
        return self._score(query_h, root_state_h)


class FlowHead(_ProjectedDotScalar):
    """log F(s | q) 估计器：严格马尔可夫的边缘流量评估。"""

    def forward(self, query_h: torch.Tensor, state_h: torch.Tensor) -> torch.Tensor:
        return self._score(query_h, state_h)


@dataclass(frozen=True)
class EdgeScorerInputs:
    """ExpandEdgeScorer 的严格输入契约（仅包含候选边切片数据）。

    src_dyn_node_h : 候选边起点的动态表示，来自 backbone node_h_policy 通道
                     （frontier-aware，已聚合候选边邻域信息）。
    dst_stat_node_h: 候选边终点的静态表示，来自 feature_bank.node_h
                     （dst 尚未激活，无动态表示可用；先验分不依赖拓扑状态）。
    """
    src_dyn_node_h: torch.Tensor    # [C, H] 候选边起点（policy 通道动态特征）
    edge_batch_index: torch.Tensor  # [C]    候选边的图归属索引
    dst_stat_node_h: torch.Tensor   # [C, H] 候选边终点（静态/环境特征）
    rel_h: torch.Tensor             # [C, H] 候选边的关系语义（静态）
    query_h: torch.Tensor           # [B, H] 全局查询语义


@dataclass(frozen=True)
class EdgeScoreBreakdown:
    """边打分的物理量分解，用于后续的分析、正则化或消融实验。"""
    prior_logits: torch.Tensor
    residual_logits: torch.Tensor
    final_logits: torch.Tensor


class ExpandEdgeScorer(nn.Module):
    """候选边价值打分器 (Prior-Regularized Topology Reranker)."""

    def __init__(
        self,
        hidden_dim: int,
        prior_scale_init: float = 5.0,
        prior_scale_trainable: bool = True,
        residual_scale: float = 1.0,
        num_residual_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        if self.residual_scale < 0.0:
            raise ValueError(f"residual_scale must be >= 0, got {self.residual_scale}.")

        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale_init)))
        self.prior_scale.requires_grad_(bool(prior_scale_trainable))

        self.residual_scorer = build_mlp(
            self.hidden_dim * 4 + 1,
            1,
            self.hidden_dim,
            num_residual_layers,
            dropout,
        )
        zero_last_linear(self.residual_scorer)

    @staticmethod
    def _center_scores_per_graph(
        scores: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        per_graph_sum = scatter_sum(scores, batch_index, dim=0, dim_size=num_graphs)
        per_graph_count = (
            torch.bincount(batch_index, minlength=num_graphs)
            .to(dtype=scores.dtype, device=scores.device)
            .clamp_min(1.0)
        )
        per_graph_mean = per_graph_sum / per_graph_count
        return scores - per_graph_mean.index_select(0, batch_index)

    def forward(
        self,
        inp: EdgeScorerInputs,
        *,
        return_breakdown: bool = False,
    ) -> torch.Tensor | EdgeScoreBreakdown:
        if inp.src_dyn_node_h.numel() == 0:
            empty = inp.src_dyn_node_h.new_zeros((0,))
            if not return_breakdown:
                return empty
            return EdgeScoreBreakdown(
                prior_logits=empty, residual_logits=empty, final_logits=empty
            )

        query_per_edge = inp.query_h.index_select(0, inp.edge_batch_index)  # [C, H]

        prior_ctx = F.normalize(query_per_edge, p=2, dim=-1)
        prior_logits = self.prior_scale * F.cosine_similarity(prior_ctx, inp.rel_h, dim=-1)

        residual_delta = self.residual_scorer(
            torch.cat([
                inp.src_dyn_node_h,
                inp.dst_stat_node_h,
                query_per_edge * inp.src_dyn_node_h,
                query_per_edge * inp.dst_stat_node_h,
                prior_logits.unsqueeze(-1),
            ], dim=-1)
        ).squeeze(-1)

        residual_logits = self.residual_scale * torch.tanh(
            self._center_scores_per_graph(
                residual_delta, inp.edge_batch_index, inp.query_h.size(0)
            )
        )

        final_logits = prior_logits + residual_logits

        if not return_breakdown:
            return final_logits

        return EdgeScoreBreakdown(
            prior_logits=prior_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
        )


class ActionHead(nn.Module):
    """图级别的二元决策打分器 (Expand vs Stop)

    返回 (B, 2) logits，其中 col 0 = Expand, col 1 = Stop。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        type_feature_dim: int = 0,
        zero_init_type_output: bool = True,
    ) -> None:
        super().__init__()
        if type_feature_dim < 0:
            raise ValueError(f"type_feature_dim must be >= 0, got {type_feature_dim}.")
        self.type_feature_dim = int(type_feature_dim)

        self.type_scorer = build_mlp(
            hidden_dim + self.type_feature_dim,
            2,
            hidden_dim,
            num_layers,
            dropout,
        )
        if zero_init_type_output:
            zero_last_linear(self.type_scorer)

    def forward(
        self,
        state_h: torch.Tensor,
        type_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        finite_state_h = require_finite(state_h, name="action_state_h")

        if self.type_feature_dim > 0:
            if type_features is None:
                raise ValueError(
                    f"type_feature_dim={self.type_feature_dim} but type_features is None."
                )
            if type_features.shape != (state_h.size(0), self.type_feature_dim):
                raise ValueError(
                    f"type_features shape mismatch: expected "
                    f"({state_h.size(0)}, {self.type_feature_dim}), "
                    f"got {tuple(type_features.shape)}."
                )
            type_ctx = torch.cat(
                [
                    finite_state_h,
                    require_finite(
                        type_features.to(finite_state_h.dtype), name="type_features"
                    ),
                ],
                dim=-1,
            )
        else:
            type_ctx = finite_state_h

        return {"type_logits": self.type_scorer(type_ctx)}


def build_edge_scorer_inputs(
    backbone_out: BackboneOutput,
    edge_index: torch.Tensor,
    edge_batch_index: torch.Tensor,
) -> EdgeScorerInputs:
    """根据 BackboneOutput 全量构建 EdgeScorerInputs。

    此函数用于全量转换场景（非 rollout 热路径）。
    热路径中 policy.py 的 _encode_edges 直接传入 candidate_mask 切片以节省显存。

    src_dyn_node_h 使用 node_h_policy（frontier-aware 通道），
    与热路径行为保持一致。
    """
    src = edge_index[0]
    dst = edge_index[1]
    return EdgeScorerInputs(
        src_dyn_node_h=backbone_out.node_h_policy.index_select(0, src),  # ← CHANGED
        edge_batch_index=edge_batch_index,
        dst_stat_node_h=backbone_out.feature_bank.node_h.index_select(0, dst),
        rel_h=backbone_out.rel_h,
        query_h=backbone_out.query_h,
    )


__all__ = [
    "ActionHead",
    "EdgeScoreBreakdown",
    "EdgeScorerInputs",
    "ExpandEdgeScorer",
    "FlowHead",
    "ZHead",
    "build_edge_scorer_inputs",
]