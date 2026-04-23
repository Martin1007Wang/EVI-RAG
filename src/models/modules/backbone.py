from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_softmax, scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import init_xavier


@dataclass(frozen=True)
class PreparedGNNInput:
    """Static features computed once per batch / rollout.

    这些特征在整个 rollout 期间保持不变，由 precompute_static() 生成一次后
    缓存在 rollout_context 里复用，避免重复计算。
    """
    node_h: torch.Tensor      # [N, H] 原始节点语义嵌入（未经 anchor gate）
    rel_h: torch.Tensor       # [E, H] 关系语义嵌入
    query_h: torch.Tensor     # [B, H] 问题嵌入
    node_input_h: torch.Tensor  # [N, H] anchor-gated 初始节点特征，GNN 的起点


@dataclass(frozen=True)
class BackboneOutput:
    """backbone.forward() 的输出契约。

    Fields
    ------
    node_h_state : [N, H]
        **Markov-faithful 状态表示**。
        消息传递仅沿 active_edges（已选入图的边）传播。
        用途：FlowHead 估计 log F(s)，以及 _encode_flow_state 的 attention pooling。
        不变式：active_edges(s₁) == active_edges(s₂)  ⟹  node_h_state(s₁) == node_h_state(s₂)

    node_h_policy : [N, H]
        **策略感知表示**。
        消息传递沿 active_edges ∪ frontier_edges（候选边）传播，
        使得待扩张的目标节点（frontier dst）也能从邻域收到消息。
        用途：ExpandEdgeScorer 的 src_dyn_node_h，以及 active_edge_encoder。
        注意：同一节点在 node_h_state 和 node_h_policy 中的表示可以不同。

    rel_h : [E, H]
        关系嵌入，直接透传自 static context（rel 不参与 GNN 更新）。

    query_h : [B, H]
        问题嵌入，直接透传。

    feature_bank : PreparedGNNInput
        静态特征缓存，供下游模块按需取用（如 dst_stat_node_h）。
    """
    node_h_state: torch.Tensor   # Markov-faithful，仅 active_edges 传播
    node_h_policy: torch.Tensor  # frontier-aware，active_edges ∪ frontier 传播
    rel_h: torch.Tensor
    query_h: torch.Tensor
    feature_bank: PreparedGNNInput

    # ── 向后兼容 ───────────────────────────────────────────────────────────
    # 旧代码通过 backbone_output.node_h 或 iter(backbone_output) 访问节点表示。
    # 这里保留 .node_h 作为 node_h_state 的别名，并在迁移完成后移除。
    @property
    def node_h(self) -> torch.Tensor:
        """Deprecated: 请明确使用 .node_h_state 或 .node_h_policy。"""
        return self.node_h_state

    def as_triple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 注意：triple 里的 node_h 是 state 版本，调用方若需要 policy 版本须显式访问
        return self.node_h_state, self.rel_h, self.query_h

    def __iter__(self):
        yield from self.as_triple()


class _NBFLayer(nn.Module):
    """One untied NBF-style Bellman-Ford message-passing layer.

    该层本身无状态偏好：edge_mask 由调用方决定传播拓扑，
    层内逻辑对"active"还是"frontier"边一视同仁。
    两次不同 mask 的传播复用同一套权重（untied 指层间权重不共享，层内无此限制）。
    """

    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fwd_msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        for m in self.fwd_msg_mlp:
            if isinstance(m, nn.Linear):
                init_xavier(m)
        init_xavier(self.attn_score)
        init_xavier(self.update)

    def forward(
        self,
        *,
        node_h: torch.Tensor,
        rel_h: torch.Tensor,
        query_h: torch.Tensor,
        edge_index: torch.Tensor,
        node_graph_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = int(node_h.size(0))
        if num_nodes == 0 or edge_index.size(1) == 0:
            return node_h

        src = edge_index[0]
        dst = edge_index[1]

        if edge_mask is not None:
            if not bool(edge_mask.any()):
                return node_h
            src = src[edge_mask]
            dst = dst[edge_mask]
            rel_h = rel_h[edge_mask]

        query_per_edge = query_h.index_select(0, node_graph_index.index_select(0, src))
        msg_fwd = self.fwd_msg_mlp(
            torch.cat([node_h.index_select(0, src), rel_h, query_per_edge], dim=-1)
        )

        attn_logit = self.attn_score(msg_fwd)                              # [E, 1]
        attn_weight = scatter_softmax(attn_logit, dst, dim=0)              # [E, 1]
        agg = scatter_sum(msg_fwd * attn_weight, dst, dim=0, dim_size=num_nodes)  # [N, H]

        return self.norm(node_h + self.dropout(self.update(agg)))


class NBFBackbone(nn.Module):

    def __init__(
        self,
        *,
        embedding_dim: int = 1024,
        hidden_dim: int = 1024,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.gnn_num_layers = int(gnn_num_layers)

        if self.gnn_num_layers < 0:
            raise ValueError(f"gnn_num_layers must be >= 0, got {self.gnn_num_layers}.")
        if self.hidden_dim != self.emb_dim:
            raise ValueError(
                f"NBFBackbone hidden_dim must equal embedding_dim. "
                f"Got hidden_dim={self.hidden_dim} and embedding_dim={self.emb_dim}."
            )

        self.non_text_embedding = nn.Parameter(torch.randn(self.emb_dim) * 0.02)
        self.anchor_gate = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.nbf_layers = nn.ModuleList(
            _NBFLayer(hidden_dim=self.hidden_dim, dropout=gnn_dropout)
            for _ in range(self.gnn_num_layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        init_xavier(self.anchor_gate)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_node_tokens(self, batch: RetrievalBatch) -> torch.Tensor:
        tokens = batch.node_tokens
        mask = getattr(batch, "non_text_node_mask", None)
        if mask is None or not bool(mask.any()):
            return tokens
        placeholder = self.non_text_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return torch.where(mask.unsqueeze(-1), placeholder.view(1, -1), tokens)

    def _build_node_input_h(
        self,
        *,
        node_h: torch.Tensor,
        query_h: torch.Tensor,
        batch_index: torch.Tensor,
        anchor_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        node_input_h = node_h.clone()
        if anchor_mask is None or not bool(anchor_mask.any()):
            return node_input_h

        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).view(-1)
        anchor_batch_idx = batch_index.index_select(0, anchor_idx)
        anchor_node_h = node_h.index_select(0, anchor_idx)
        anchor_query_h = query_h.index_select(0, anchor_batch_idx)

        anchor_gate = torch.sigmoid(
            self.anchor_gate(
                torch.cat([anchor_node_h, anchor_query_h, anchor_node_h * anchor_query_h], dim=-1)
            )
        )
        fused_anchor_h = (1.0 - anchor_gate) * anchor_node_h + anchor_gate * anchor_query_h
        node_input_h[anchor_mask] = F.normalize(fused_anchor_h, p=2, dim=-1)
        return node_input_h

    def _run_gnn(
        self,
        node_input_h: torch.Tensor,
        rel_h: torch.Tensor,
        query_h: torch.Tensor,
        edge_index: torch.Tensor,
        node_graph_index: torch.Tensor,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """在给定 edge_mask 下跑完整的 GNN 栈，返回归一化后的节点表示。

        抽出这个辅助方法的原因：forward() 需要用不同的 mask 调用两次，
        把共同的循环逻辑收拢在此处避免重复。
        """
        node_h = node_input_h
        for layer in self.nbf_layers:
            node_h = layer(
                node_h=node_h,
                rel_h=rel_h,
                query_h=query_h,
                edge_index=edge_index,
                node_graph_index=node_graph_index,
                edge_mask=edge_mask,
            )
        return self.output_norm(node_h)

    # ── Public API ────────────────────────────────────────────────────────

    def precompute_static(self, batch: RetrievalBatch) -> PreparedGNNInput:
        """Prepare node / relation / query features once per batch."""
        node_h = self._resolve_node_tokens(batch)
        rel_h = batch.relation_tokens
        query_h = batch.question_emb

        node_input_h = self._build_node_input_h(
            node_h=node_h,
            query_h=query_h,
            batch_index=batch.batch,
            anchor_mask=getattr(batch, "is_anchor_mask", None),
        )
        return PreparedGNNInput(
            node_h=node_h,
            rel_h=rel_h,
            query_h=query_h,
            node_input_h=node_input_h,
        )

    def forward(
        self,
        batch: RetrievalBatch,
        active_edges: torch.Tensor | None = None,
        active_nodes: torch.Tensor | None = None,
        static_context: PreparedGNNInput | None = None,
    ) -> BackboneOutput:
        """Run the dual-pass GNN to produce two semantically distinct node representations.

        Pass 1 — state pass（状态通道）
        --------------------------------
        edge_mask = active_edges（若有），否则 active_nodes[src] & active_nodes[dst]（交集近似）

        消息只沿「已选入图的边」传播。
        设 V_t = active nodes，E_t = active edges，则：
          node_h_state 是 f(V_t, E_t, q) 的严格函数。
        因此 active_edges(s₁) == active_edges(s₂) ⟹ node_h_state(s₁) == node_h_state(s₂)，
        满足 GFlowNet 对 log F(s) 的 Markov 性要求。

        Pass 2 — policy pass（策略通道）
        ----------------------------------
        edge_mask = active_edges | frontier_edges
          其中 frontier_edges = {e ∉ E_t : endpoints(e) ∩ V_t ≠ ∅}

        消息沿「已选边 ∪ 候选边」传播，使 frontier 目标节点（dst）
        也能聚合来自活跃邻域的信息，为边打分器提供更丰富的表示。
        代价：node_h_policy 不是 E_t 的严格函数，不能用于 FlowHead。

        Parameters
        ----------
        active_edges : BoolTensor [E_total] | None
            当前已选入图的边的布尔掩码。
            为 None 时（如 rollout 第 0 步之前的静态预计算），两个 pass 均用全图边。
        active_nodes : BoolTensor [N_total] | None
            当前活跃节点的布尔掩码，用于推导 frontier_edges。
            若 active_edges 为 None 则用于近似 state mask。
        """
        fb = static_context if static_context is not None else self.precompute_static(batch)

        if not (fb.node_input_h.shape[-1] == fb.rel_h.shape[-1] == fb.query_h.shape[-1] == self.hidden_dim):
            raise ValueError(f"Feature dim mismatch. Expected {self.hidden_dim}.")

        # ── 计算两套 edge_mask ────────────────────────────────────────────
        #
        # state_edge_mask：Pass 1 使用，仅含已选边，保证 Markov 性。
        # policy_edge_mask：Pass 2 使用，含已选边 + frontier，提升策略视野。
        #
        # 为什么要分开，不能共用一套？
        #   共用 frontier mask（原代码做法）时，node_h 隐式编码了"哪些边在 frontier"，
        #   而 frontier 在不同 rollout 步骤是不同的；两个 active_edges 相同但
        #   frontier 不同的状态（因为候选图不同）会得到不同 node_h，污染 flow 估计。
        #   共用 active_edges mask 则 frontier 节点表示退化为静态嵌入，削弱策略能力。

        src_all, dst_all = batch.edge_index[0], batch.edge_index[1]

        if active_edges is not None:
            # 标准 rollout 路径：active_edges 是精确的 E_t 指示器
            state_edge_mask = active_edges  # [E_total] bool

            if active_nodes is not None:
                # frontier = 有一个端点在 V_t 中、且自身不在 E_t 中的边
                # 注意：这里用 | 而非 &，因为 frontier 允许只有 src 或只有 dst 在 V_t
                frontier_mask = (active_nodes[src_all] | active_nodes[dst_all]) & ~active_edges
                policy_edge_mask = active_edges | frontier_mask
            else:
                # 没有 active_nodes 信息时，policy pass 退化为和 state pass 相同
                # 这种情况在正常 rollout 中不应出现，此处为防御性处理
                policy_edge_mask = active_edges

        elif active_nodes is not None:
            # 降级路径：只有 active_nodes，没有精确的 active_edges
            # state mask 用交集近似（两端均激活 ≈ 边已选入）
            # policy mask 用并集（端点任一激活 = 原代码行为，用于策略）
            state_edge_mask = active_nodes[src_all] & active_nodes[dst_all]
            policy_edge_mask = active_nodes[src_all] | active_nodes[dst_all]

        else:
            # 没有任何动态信息（全图传播，如 precompute 阶段）
            state_edge_mask = None
            policy_edge_mask = None

        # ── Pass 1：状态通道，Markov-faithful ─────────────────────────────
        node_h_state = self._run_gnn(
            node_input_h=fb.node_input_h,
            rel_h=fb.rel_h,
            query_h=fb.query_h,
            edge_index=batch.edge_index,
            node_graph_index=batch.batch,
            edge_mask=state_edge_mask,
        )

        # ── Pass 2：策略通道，frontier-aware ─────────────────────────────
        # 两次 pass 共享 nbf_layers 权重，不增加参数量，只增加一倍前向计算。
        # 梯度从两个 pass 分别回传，训练信号互不干扰（因为 node_input_h 是同一个起点）。
        node_h_policy = self._run_gnn(
            node_input_h=fb.node_input_h,
            rel_h=fb.rel_h,
            query_h=fb.query_h,
            edge_index=batch.edge_index,
            node_graph_index=batch.batch,
            edge_mask=policy_edge_mask,
        )

        return BackboneOutput(
            node_h_state=node_h_state,
            node_h_policy=node_h_policy,
            rel_h=fb.rel_h,
            query_h=fb.query_h,
            feature_bank=fb,
        )


__all__ = [
    "PreparedGNNInput",
    "BackboneOutput",
    "NBFBackbone",
]