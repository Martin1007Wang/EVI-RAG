from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.state import (
    FrontierEncoding,
    NodeSelection,
    StateBatch,
    frontier_from_graph,
)

from .output import PolicyOutput, STOP_EDGE_ID


@dataclass(frozen=True, slots=True)
class PolicyCache:
    question_h_by_graph: torch.Tensor  # [G, H]
    edge_h: torch.Tensor  # [E, H]
    relation_h: torch.Tensor  # [E, H]  relation 特征，供 Path1 使用


@dataclass(frozen=True, slots=True)
class PolicyActionSpace:
    active: NodeSelection
    frontier: FrontierEncoding


class StateFlowHead(nn.Module):
    """state_h → log F(s)，用于 trajectory balance。"""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        head_dim = min(int(state_dim), 256)
        self.net = nn.Sequential(
            nn.Linear(state_dim, head_dim, bias=False),
            nn.LayerNorm(head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1, bias=True),
        )

    def forward(self, *, state_h: torch.Tensor) -> torch.Tensor:  # [S]
        return self.net(state_h).squeeze(-1)


class FlowEstimator(nn.Module):
    """
    Unified edge and STOP scorer in the shared hidden space.
    """

    def __init__(self, *, hidden_dim: int, relation_lambda: float = 0.5) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.relation_lambda = float(relation_lambda)

        self.marginal_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def score_edges(
        self,
        *,
        question_h: torch.Tensor,
        state_h: torch.Tensor,
        frontier_edge_h: torch.Tensor,
        frontier_relation_h: torch.Tensor,
    ) -> torch.Tensor:
        relation_score = (question_h * frontier_relation_h).sum(dim=-1) * self.scale
        edge_score = (question_h * frontier_edge_h).sum(dim=-1) * self.scale
        phi_relation = relation_score + self.relation_lambda * edge_score
        phi_mgn = self.marginal_mlp(
            torch.cat([state_h, frontier_edge_h, state_h * frontier_edge_h], dim=-1)
        ).squeeze(-1)
        return phi_relation + phi_mgn

    def score_stop(
        self,
        *,
        question_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.stop_head(torch.cat([state_h, question_h], dim=-1)).squeeze(-1)

    def forward(
        self,
        *,
        question_h: torch.Tensor,
        state_h: torch.Tensor,
        frontier_edge_h: torch.Tensor,
        frontier_relation_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.score_edges(
                question_h=question_h,
                state_h=state_h,
                frontier_edge_h=frontier_edge_h,
                frontier_relation_h=frontier_relation_h,
            ),
            self.score_stop(question_h=question_h, state_h=state_h),
        )


class ForwardPolicy(nn.Module):
    """GFlowNet forward policy for KG subgraph retrieval.

        数据流：
        FeaturePack
          ├─ question_h  [G, H]   问题嵌入
          ├─ edge_h   [E, H]   边特征（EdgeEncoder 输出，已含 src/relation/dst）
          └─ relation_h [E, H]   relation 特征（FeatureEncoder 输出，Path1 专用）

        每步 forward：
          1. build_cache：缓存 question_h / edge_h / relation_h（batch 内不变）
          2. prepare_action_space：构建 frontier edges
          3. StateEncoder（逐 state 调用）：
               CrossAttn(question_h, selected_edge_h) + fusion → state_h
          4. FlowEstimator：
               Path1: question_h · relation_h + λ · question_h · edge_h  → φ_relation [E]
               Path2: MLP([s, e, s⊙e])                                    → φ_mgn [E]
               STOP:  MLP([s, question])                                   → stop  [S]
          5. 组装 PolicyOutput
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        flow_estimator: FlowEstimator,
        state_flow_head: StateFlowHead,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.flow_estimator = flow_estimator
        self.state_flow_head = state_flow_head

    # ------------------------------------------------------------------
    def build_cache(self, features: FeaturePack) -> PolicyCache:
        return PolicyCache(
            question_h_by_graph=features.question_h.float(),
            edge_h=features.edge_h.float(),
            relation_h=features.relation_h.float(),
        )

    def prepare_action_space(self, *, state: StateBatch, graph_context: GraphContext) -> PolicyActionSpace:
        active = state.active_node_index(graph_context)
        frontier = frontier_from_graph(state=state, graph=graph_context, active=active)
        return PolicyActionSpace(
            active=active,
            frontier=frontier,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        state: StateBatch,
        features: FeaturePack,
        graph_context: GraphContext,
        cache: PolicyCache | None = None,
        action_space: PolicyActionSpace | None = None,
        compute_log_flow: bool = False,
    ) -> PolicyOutput:
        if cache is None:
            cache = self.build_cache(features)
        if action_space is None:
            action_space = self.prepare_action_space(state=state, graph_context=graph_context)

        frontier = action_space.frontier
        S = state.num_states
        dev = state.device

        # ── per-state question_h ───────────────────────────────────
        # state.graph_ids: [S]，每个 state 归属的 graph
        question_h_per_state = cache.question_h_by_graph.index_select(0, state.graph_ids)  # [S, H]

        # ── StateEncoder：逐 state 编码（简化版 StateEncoder 是单 state 接口）
        # 批量处理：对每个 state 单独调用，收集结果
        state_h_list: list[torch.Tensor] = []
        selected = state.selected_edge_index()  # (row_ids [E_sel], edge_ids [E_sel])

        for i in range(S):
            # 取当前 state i 的已选边
            mask = selected.row_ids == i
            sel_edge_h = cache.edge_h.index_select(0, selected.edge_ids[mask]) if mask.any() else None
            state_h_i = self.state_encoder(
                question_h=question_h_per_state[i],  # [H] 或 [1, H]
                selected_edge_h=sel_edge_h,  # [E_i, H] 或 None
            )  # [1, H]
            state_h_list.append(state_h_i)

        state_h = torch.cat(state_h_list, dim=0)  # [S, H]

        # ── frontier edge 特征 ─────────────────────────────────────
        f_edge_ids = frontier.edge_ids  # [F]
        frontier_edge_h = cache.edge_h.index_select(0, f_edge_ids)  # [F, H]
        frontier_relation_h = cache.relation_h.index_select(0, f_edge_ids)  # [F, H]

        # ── FlowEstimator：批量打分 ────────────────────────────────
        # state_h[frontier.row_ids]：把每条 frontier 边映射到对应 state 的 state_h
        edge_logits = self.flow_estimator.score_edges(
            question_h=question_h_per_state.index_select(0, frontier.row_ids),  # [F, H]
            state_h=state_h.index_select(0, frontier.row_ids),  # [F, H]
            frontier_edge_h=frontier_edge_h,  # [F, H]
            frontier_relation_h=frontier_relation_h,  # [F, H]
        )  # edge_logits: [F]

        stop_logits = self.flow_estimator.score_stop(
            question_h=question_h_per_state,
            state_h=state_h,
        )

        # ── state flow（trajectory balance）───────────────────────
        log_flow = self.state_flow_head(state_h=state_h).float() if compute_log_flow else None

        # ── 组装 PolicyOutput ──────────────────────────────────────
        rows = torch.arange(S, dtype=torch.long, device=dev)

        return PolicyOutput(
            action_logits=torch.cat([stop_logits.float(), edge_logits.float()]),
            action_row_ids=torch.cat([rows, frontier.row_ids]),
            action_edge_ids=torch.cat([torch.full_like(rows, STOP_EDGE_ID), f_edge_ids]),
            frontier=frontier,
            log_flow=log_flow,
        )


__all__ = [
    "FlowEstimator",
    "ForwardPolicy",
    "PolicyActionSpace",
    "PolicyCache",
    "StateFlowHead",
]
