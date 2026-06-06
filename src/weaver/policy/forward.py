from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_logsumexp
from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.state import (
    FrontierEncoding,
    NodeSelection,
    StateBatch,
    frontier_from_graph,
)

from .backward import BackwardPolicy, BackwardPolicyOutput, removable_edges
from .edge_scorer import QuestionConditionedEdgeScorer
from .output import PolicyOutput, STOP_EDGE_ID


@dataclass(frozen=True, slots=True)
class PolicyInput:
    question_h_by_graph: torch.Tensor  # [G, H]
    edge_h: torch.Tensor  # [E, H]
    relation_h: torch.Tensor  # [E, H]  relation 特征，供 Path1 使用
    align_score: torch.Tensor | None = None  # [E] rollout/eval 可缓存的预计算 φ_align


@dataclass(frozen=True, slots=True)
class PolicyActionSpace:
    active: NodeSelection
    frontier: FrontierEncoding


class StateFlowHead(nn.Module):
    """state_h -> base state flow f_theta(s)."""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        head_dim = min(int(state_dim), 256)
        self.net = nn.Sequential(
            nn.Linear(state_dim, head_dim, bias=False),
            nn.LayerNorm(head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1, bias=True),
        )

    def forward(
        self,
        *,
        state_h: torch.Tensor,
    ) -> torch.Tensor:  # [S]
        return self.net(state_h).squeeze(-1)


class FlowEstimator(nn.Module):
    """
    Unified edge and STOP scorer in the shared hidden space.

    边打分双路融合：
      Path1 (question-driven alignment):
        φ_relation = (q · r) * scale + λ * (q · e) * scale
        捕获问题与谓词/三元组的方向对齐，无状态依赖。

      Path2 (state-edge compatibility):
        φ_mgn = MLP([s_h ‖ e ‖ s_h⊙e])
        s_h⊙e：当前子图状态与候选边的结构兼容度（state-aware）
        问题只通过 state_h 的构造间接影响该项，不直接提供 q⊙e 交叉特征。

    STOP/CONTINUE factorization：
        rank_e = φ_align + φ_state 只决定 frontier 内部边排序。
        u_stop = MLP_stop(s_h)
        u_cont = MLP_cont([s_h | detached frontier soft summary | log1p(|C(s)|)])
        L_e = u_cont + rank_e - logsumexp(rank_e)
        因此所有边动作的总 logit 质量严格等于 CONTINUE logit，
        且 rank logits 的整体平移不会改变 STOP-vs-CONTINUE 概率。
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        relation_lambda: float = 0.5,
        stop_initial_bias: float = 1.5,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.hidden_dim = hidden_dim
        self.relation_lambda = float(relation_lambda)

        self.edge_scorer = QuestionConditionedEdgeScorer(
            hidden_dim=hidden_dim,
            relation_lambda=relation_lambda,
        )
        # STOP: [s_h] → 1
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        nn.init.constant_(self.stop_head[-1].bias, float(stop_initial_bias))
        # CONTINUE: [s_h | detached frontier_soft_summary | log1p(frontier_count)] → 1
        self.cont_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def score_edges(
        self,
        *,
        state_h: torch.Tensor,  # [F, H]
        frontier_edge_h: torch.Tensor,  # [F, H]
        frontier_align_score: torch.Tensor,  # [F]
    ) -> torch.Tensor:  # [F]
        phi_state = self.edge_scorer.score_state(
            state_h=state_h,
            edge_h=frontier_edge_h,
        )
        return frontier_align_score.float() + phi_state.float()

    def score_stop(
        self,
        *,
        state_h: torch.Tensor,  # [S, H]
    ) -> torch.Tensor:  # [S]
        return self.stop_head(state_h).squeeze(-1)

    def forward(
        self,
        *,
        state_h: torch.Tensor,  # [S, H]
        frontier_row_ids: torch.Tensor,  # [F]
        frontier_edge_h: torch.Tensor,  # [F, H]
        frontier_align_score: torch.Tensor,  # [F]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        S = int(state_h.size(0))
        if int(frontier_row_ids.numel()) == 0:
            return state_h.new_empty((0,), dtype=torch.float32), state_h.new_zeros((S,), dtype=torch.float32)

        rank_logits = self.score_edges(
            state_h=state_h.index_select(0, frontier_row_ids),
            frontier_edge_h=frontier_edge_h,
            frontier_align_score=frontier_align_score,
        )
        rank_log_z = segment_logsumexp(
            values=rank_logits,
            segment_ids=frontier_row_ids,
            num_segments=S,
        )
        rank_prob = (rank_logits - rank_log_z.index_select(0, frontier_row_ids)).exp().detach()
        opportunity_h = state_h.new_zeros((S, self.hidden_dim))
        opportunity_h.scatter_add_(
            0,
            frontier_row_ids.view(-1, 1).expand(-1, self.hidden_dim),
            frontier_edge_h * rank_prob.unsqueeze(-1),
        )
        has_frontier = torch.isfinite(rank_log_z)
        frontier_count = torch.bincount(frontier_row_ids, minlength=S).to(dtype=state_h.dtype)
        stop_logits = state_h.new_zeros((S,), dtype=torch.float32)
        stop_logits[has_frontier] = self.score_stop(state_h=state_h[has_frontier]).float()
        cont_logits = state_h.new_zeros((S,), dtype=torch.float32)
        cont_input = torch.cat(
            [
                state_h[has_frontier],
                opportunity_h[has_frontier],
                torch.log1p(frontier_count[has_frontier]).unsqueeze(-1),
            ],
            dim=-1,
        )
        cont_logits[has_frontier] = self.cont_head(cont_input).squeeze(-1).float()
        edge_logits = cont_logits.index_select(0, frontier_row_ids) + rank_logits.float() - rank_log_z.index_select(0, frontier_row_ids)
        return edge_logits, stop_logits


class ForwardPolicy(nn.Module):
    """GFlowNet forward policy for KG subgraph retrieval.

    数据流：
    FeaturePack
      ├─ question_h  [G, H]   问题嵌入
      ├─ edge_h      [E, H]   边特征（EdgeEncoder 输出，已含 src/relation/dst）
      └─ relation_h  [E, H]   relation 特征（FeatureEncoder 输出，Path1 专用）

    每步 forward：
      1. build_policy_input：缓存 question_h / edge_h / relation_h（batch 内不变）
      2. prepare_action_space：构建 frontier edges
      3. StateEncoder（批量化）：
           fusion([question_h, CrossAttn(question_h, selected_edge_h)]) → state_h [S, H]
           空 state（edge_count=0）使用可学习 empty_state_emb 替代 CrossAttn 输出
      4. FlowEstimator：
           Path1: q·r + λ·q·e                      → φ_relation [F]
           Path2: MLP([s ‖ e ‖ s⊙e])                → φ_mgn      [F]
           STOP:  MLP([s])                           → stop        [S]
      5. 组装 PolicyOutput
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        flow_estimator: FlowEstimator,
        state_flow_head: StateFlowHead,
        backward_policy: BackwardPolicy | None = None,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.flow_estimator = flow_estimator
        self.state_flow_head = state_flow_head
        self.backward_policy = backward_policy or BackwardPolicy(hidden_dim=flow_estimator.hidden_dim)

    # ------------------------------------------------------------------
    def build_policy_input(
        self,
        features: FeaturePack,
        graph_context: GraphContext | None = None,
        *,
        compute_align_score: bool = True,
    ) -> PolicyInput:
        return PolicyInput(
            question_h_by_graph=features.question_h.float(),
            edge_h=features.edge_h.float(),
            relation_h=features.relation_h.float(),
            align_score=(
                self._compute_align_score(features=features, graph_context=graph_context)
                if graph_context is not None and compute_align_score
                else None
            ),
        )

    def _compute_align_score(
        self,
        *,
        features: FeaturePack,
        graph_context: GraphContext,
    ) -> torch.Tensor:
        edge_graph_ids = graph_context.edge_to_graph.to(
            device=features.edge_h.device,
            dtype=torch.long,
        )
        question_h = features.question_h.float().index_select(0, edge_graph_ids)
        return self.flow_estimator.edge_scorer.score_alignment(
            question_h=question_h,
            edge_h=features.edge_h.float(),
            relation_h=features.relation_h.float(),
        )

    def _resolve_align_score(
        self,
        *,
        policy_input: PolicyInput,
        features: FeaturePack,
        graph_context: GraphContext,
    ) -> torch.Tensor:
        if policy_input.align_score is None:
            return self._compute_align_score(
                features=features,
                graph_context=graph_context,
            )
        return policy_input.align_score

    def prepare_action_space(
        self,
        *,
        state: StateBatch,
        graph_context: GraphContext,
    ) -> PolicyActionSpace:
        active = state.active_node_index(graph_context)
        frontier = frontier_from_graph(state=state, graph=graph_context, active=active)
        return PolicyActionSpace(active=active, frontier=frontier)

    # ------------------------------------------------------------------
    def _build_state_h_batched(
        self,
        *,
        S: int,
        question_h_per_state: torch.Tensor,  # [S, H]
        selected_row_ids: torch.Tensor,  # [E_sel]，每条已选边归属的 state 行号
        selected_edge_ids: torch.Tensor,  # [E_sel]，已选边在全局 edge_h 中的索引
        edge_h: torch.Tensor,  # [E_total, H]
        device: torch.device,
    ) -> torch.Tensor:  # [S, H]
        """
        将变长的已选边集合 pad 到统一长度 L_max，
        一次性批量送入 StateEncoder.forward，
        替代原有的 Python-level for loop。

        构造过程：
          1. bincount → edge_counts [S]，确定 L_max
          2. cumsum → state_offsets，计算每条边在其 state 内的局部位置 local_pos
          3. scatter 填充 padded [S, L_max, H] 和 key_padding_mask [S, L_max]
          4. 调用 StateEncoder.forward，空 state 由 is_empty mask 特殊处理
        """
        H = question_h_per_state.shape[-1]

        if selected_row_ids.numel() == 0:
            # 全部为空 state（rollout 第 0 步或所有 state 刚初始化）
            is_empty = torch.ones(S, dtype=torch.bool, device=device)
            dummy_kv = torch.zeros(S, 1, H, device=device)
            full_mask = torch.ones(S, 1, dtype=torch.bool, device=device)
            return self.state_encoder(
                question_h=question_h_per_state,
                selected_edge_h=dummy_kv,
                key_padding_mask=full_mask,
                is_empty=is_empty,
            )

        # ── Step 1: 统计每个 state 的已选边数 ──────────────────────────
        _validate_row_ids(selected_row_ids, upper=S, name="selected_row_ids")
        edge_counts = torch.bincount(selected_row_ids, minlength=S)  # [S]
        L_max = int(edge_counts.max().item())
        is_empty = edge_counts.eq(0)  # [S]

        # ── Step 2: 计算每条已选边在其 state 内的局部位置 ───────────────
        # state_offsets[i] = 前 i 个 state 的已选边总数（即第 i 个 state 的起始偏移）
        state_offsets = torch.zeros(S + 1, dtype=torch.long, device=device)
        state_offsets[1:] = edge_counts.cumsum(0)
        # 全局序号 - 所属 state 的起始偏移 = 局部位置
        local_pos = torch.arange(selected_row_ids.numel(), device=device) - state_offsets[selected_row_ids]  # [E_sel]
        if bool((local_pos.lt(0) | local_pos.ge(edge_counts.index_select(0, selected_row_ids))).any()):
            raise ValueError("selected_row_ids must be non-decreasing for local_pos computation.")

        # ── Step 3: 构造 padded tensor 和 attention mask ─────────────────
        # padding 用 0 填充；key_padding_mask=True 表示该位置是 padding（被 attn 忽略）
        padded = torch.zeros(S, L_max, H, device=device)
        key_padding_mask = torch.ones(S, L_max, dtype=torch.bool, device=device)

        sel_edge_h = edge_h.index_select(0, selected_edge_ids)  # [E_sel, H]
        padded[selected_row_ids, local_pos] = sel_edge_h
        key_padding_mask[selected_row_ids, local_pos] = False  # 有效位取消 mask

        # ── Step 4: 批量编码 ─────────────────────────────────────────────
        state_h = self.state_encoder(
            question_h=question_h_per_state,
            selected_edge_h=padded,
            key_padding_mask=key_padding_mask,
            is_empty=is_empty,
        )
        _require_finite(state_h, "state_h")
        return state_h

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        state: StateBatch,
        features: FeaturePack,
        graph_context: GraphContext,
        policy_input: PolicyInput | None = None,
        action_space: PolicyActionSpace | None = None,
        compute_log_flow: bool = False,
    ) -> PolicyOutput:
        if policy_input is None:
            policy_input = self.build_policy_input(features, graph_context=graph_context)
        if action_space is None:
            action_space = self.prepare_action_space(state=state, graph_context=graph_context)

        frontier = action_space.frontier
        S = state.num_states
        dev = state.device
        _validate_row_ids(frontier.row_ids, upper=S, name="frontier.row_ids")

        # ── per-state question_h ───────────────────────────────────────
        # state.graph_ids: [S]，每个 state 归属的 graph（batch-local 索引）
        question_h_per_state = policy_input.question_h_by_graph.index_select(0, state.graph_ids)  # [S, H]

        # ── StateEncoder：批量化，替代原 Python for loop ────────────────
        selected = state.selected_edge_index()  # (row_ids [E_sel], edge_ids [E_sel])
        state_h = self._build_state_h_batched(
            S=S,
            question_h_per_state=question_h_per_state,
            selected_row_ids=selected.row_ids,
            selected_edge_ids=selected.edge_ids,
            edge_h=policy_input.edge_h,
            device=dev,
        )  # [S, H]

        # ── frontier edge 特征 ─────────────────────────────────────────
        f_edge_ids = frontier.edge_ids  # [F]
        frontier_edge_h = policy_input.edge_h.index_select(0, f_edge_ids)  # [F, H]
        align_score = self._resolve_align_score(
            policy_input=policy_input,
            features=features,
            graph_context=graph_context,
        )
        frontier_align_score = align_score.index_select(0, f_edge_ids)  # [F]

        # ── FlowEstimator：批量打分 ────────────────────────────────────
        # state_h[frontier.row_ids]：把每条 frontier 边映射到对应 state 的 state_h
        edge_logits, stop_logits = self.flow_estimator(
            state_h=state_h,  # [S, H]
            frontier_row_ids=frontier.row_ids,  # [F]
            frontier_edge_h=frontier_edge_h,  # [F, H]
            frontier_align_score=frontier_align_score,  # [F]
        )
        frontier_count = torch.bincount(frontier.row_ids, minlength=S)
        empty_with_frontier = state.edge_count.eq(0) & frontier_count.gt(0)
        if bool(empty_with_frontier.any()):
            stop_logits = stop_logits.clone()
            stop_logits[empty_with_frontier] = -1.0e9
        _require_finite(edge_logits, "edge_logits")
        _require_finite(stop_logits, "stop_logits")

        # ── state flow（trajectory balance，training-only）─────────────
        log_flow_base = None
        if compute_log_flow:
            log_flow_base = self.state_flow_head(state_h=state_h).float()
        if log_flow_base is not None:
            _require_finite(log_flow_base, "log_flow_base")

        # ── 组装 PolicyOutput ──────────────────────────────────────────
        rows = torch.arange(S, dtype=torch.long, device=dev)

        return PolicyOutput(
            action_logits=torch.cat([stop_logits.float(), edge_logits.float()]),
            action_row_ids=torch.cat([rows, frontier.row_ids]),
            action_edge_ids=torch.cat([torch.full_like(rows, STOP_EDGE_ID), f_edge_ids]),
            frontier=frontier,
            log_flow_base=log_flow_base,
            state_h=state_h if compute_log_flow else None,
        )

    def score_backward(
        self,
        *,
        child_state: StateBatch,
        graph_context: GraphContext,
        policy_input: PolicyInput,
        forward_output: PolicyOutput,
    ) -> BackwardPolicyOutput:
        child_state_h = forward_output.require_state_h()
        if int(child_state_h.size(0)) != int(child_state.num_states):
            raise ValueError("forward_output.state_h must have one row per child state.")
        removable = removable_edges(
            child_state=child_state,
            graph_context=graph_context,
        )
        counts = torch.bincount(removable.row_ids, minlength=child_state.num_states)
        non_root = child_state.edge_count.gt(0)
        if bool(counts[non_root].le(0).any()):
            raise ValueError("Every non-root child state must have a removable predecessor.")
        return self.backward_policy(
            child_state_h=child_state_h,
            question_h_by_graph=policy_input.question_h_by_graph,
            edge_h=policy_input.edge_h,
            relation_h=policy_input.relation_h,
            removable=removable,
        )


__all__ = [
    "FlowEstimator",
    "ForwardPolicy",
    "PolicyActionSpace",
    "PolicyInput",
    "StateFlowHead",
    "BackwardPolicy",
    "BackwardPolicyOutput",
]


def _validate_row_ids(row_ids: torch.Tensor, *, upper: int, name: str) -> None:
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(upper)).any()):
        bad = ((row_ids.lt(0)) | (row_ids.ge(int(upper)))).nonzero(as_tuple=False)[:8]
        raise ValueError(f"{name} contains out-of-range rows; sample positions={bad.tolist()}.")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if bool(torch.isfinite(tensor).all()):
        return
    bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
    preview = bad[:8].tolist()
    raise ValueError(f"{name} contains non-finite values at indices {preview}.")
