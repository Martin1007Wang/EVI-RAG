from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Any
from torch import nn
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.graph_utils import compute_component_labels, count_components
from .modules.backbone import GNNBackbone
from .modules.heads import ActionHead, ZHead
from .subgraph_state import SubgraphState


@dataclass(frozen=True)
class PolicyStepOutput:
    action_logits: dict[str, torch.Tensor]
    question_h: torch.Tensor
    subgraph_h: torch.Tensor


class Policy(nn.Module):
    """增量式子图扩张 MDP 策略网络。"""

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        hidden_dim: int = 512,
        relation_prior_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = GNNBackbone(**backbone_cfg)
        self.z_head = ZHead(hidden_dim=hidden_dim)
        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            **(relation_prior_cfg or {}),
        )
        self.edge_state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _pool_masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        对 mask=True 的行做 per-graph 均值池化。

        mask 必须来自 rollout 状态的不可变快照，而不是可继续被 inplace
        改写的原始状态张量。这样 values 的梯度路径可以安全保留。
        """
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))

        masked_values = values[mask]
        masked_batch = batch_index[mask]  # batch_index 是静态拓扑，无需 detach

        pooled_sum = scatter_sum(
            masked_values, masked_batch, dim=0, dim_size=num_graphs
        )
        # [优化] 用 bincount 代替 scatter_sum(ones)，避免分配临时全 1 张量
        pooled_count = (
            torch.bincount(masked_batch, minlength=num_graphs)
            .to(dtype=values.dtype, device=values.device)
            .clamp_min(1.0)
        )

        return pooled_sum / pooled_count.unsqueeze(-1)

    def _summarize_subgraph(
        self,
        *,
        batch: RetrievalBatch,
        node_h: torch.Tensor,
        edge_state_h: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = batch.num_graphs

        component_labels = compute_component_labels(
            num_nodes=batch.num_nodes,
            active_nodes=active_nodes,
            active_edges=active_edges,
            edge_index=batch.edge_index,
        )

        # node_h / edge_state_h 保留梯度；state mask 已在 rollout 边界快照化
        node_pool = self._pool_masked_mean(
            node_h, active_nodes, batch.batch, num_graphs
        )
        edge_pool = self._pool_masked_mean(
            edge_state_h, active_edges, batch.edge_batch, num_graphs
        )

        # ── 拓扑统计量（全部在 no_grad 下计算，不需要梯度）──
        with torch.no_grad():
            node_counts = torch.bincount(
                batch.batch[active_nodes], minlength=num_graphs
            ).to(dtype=node_h.dtype, device=node_h.device)

            edge_counts = torch.bincount(
                batch.edge_batch[active_edges], minlength=num_graphs
            ).to(dtype=node_h.dtype, device=node_h.device)

            component_counts = count_components(
                component_labels=component_labels,
                active_nodes=active_nodes,
                batch_index=batch.batch,
                num_graphs=num_graphs,
            ).to(dtype=node_h.dtype)

            cycle_ranks = (edge_counts - node_counts + component_counts).clamp_min(0.0)

            # ── 度数计算：用 scatter_sum 替代 index_add_，全程 out-of-place ──
            # [修复] 原实现在 active_degree 上调用 index_add_（inplace），
            #        且 bool(x.any().item()) 会强制 CUDA 同步并阻止 torch.compile。
            #        改用 scatter_sum：纯 out-of-place，无同步，编译友好。
            edge_src = batch.edge_index[0]
            edge_dst = batch.edge_index[1]
            active_src = edge_src[active_edges]
            active_dst = edge_dst[active_edges]
            ones = torch.ones(
                active_src.size(0), dtype=node_h.dtype, device=node_h.device
            )

            # 每条活跃边给两个端点各贡献 1 度
            degree_contrib = torch.cat([ones, ones], dim=0)
            degree_nodes = torch.cat([active_src, active_dst], dim=0)
            active_degree = scatter_sum(
                degree_contrib, degree_nodes, dim=0, dim_size=batch.num_nodes
            )

            leaf_mask = active_nodes & (active_degree <= 1.0)
            leaf_counts = torch.bincount(
                batch.batch[leaf_mask], minlength=num_graphs
            ).to(dtype=node_h.dtype, device=node_h.device)

        stats = torch.stack(
            [
                torch.log1p(node_counts),
                torch.log1p(edge_counts),
                torch.log1p(component_counts),
                torch.log1p(cycle_ranks),
                torch.log1p(leaf_counts),
            ],
            dim=-1,
        )  # (B, 5) — 无梯度，但与 node_pool / edge_pool cat 后经 state_encoder 得到梯度

        subgraph_h = self.state_encoder(
            torch.cat([node_pool, edge_pool, stats], dim=-1)
        )
        return subgraph_h, component_labels

    def _build_edge_state(
        self,
        *,
        batch: RetrievalBatch,
        node_h: torch.Tensor,
        edge_relation_h: torch.Tensor,
        active_nodes: torch.Tensor,
        relation_prior_logits: torch.Tensor,
        node_query_scores: torch.Tensor,
    ) -> torch.Tensor:
        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        active_h, inactive_h, src_active, dst_active, single_frontier = (
            self._resolve_edge_roles(
                edge_index=batch.edge_index,
                node_h=node_h,
                active_nodes=active_nodes,
            )
        )

        src_query = node_query_scores[edge_src]
        dst_query = node_query_scores[edge_dst]
        active_query = torch.where(
            src_active & ~dst_active,
            src_query,
            torch.where(
                dst_active & ~src_active, dst_query, torch.maximum(src_query, dst_query)
            ),
        )
        inactive_query = torch.where(
            src_active & ~dst_active,
            dst_query,
            torch.where(
                dst_active & ~src_active, src_query, torch.maximum(src_query, dst_query)
            ),
        )
        inactive_query = torch.where(
            single_frontier, inactive_query, torch.zeros_like(inactive_query)
        )

        edge_semantic_features = torch.stack(
            [active_query, inactive_query, relation_prior_logits],
            dim=-1,
        )
        edge_state_input = torch.cat(
            [active_h, inactive_h, edge_relation_h, edge_semantic_features],
            dim=-1,
        )
        return self.edge_state_encoder(edge_state_input)

    @staticmethod
    def _resolve_edge_roles(
        *,
        edge_index: torch.Tensor,
        node_h: torch.Tensor,
        active_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        src_h = node_h[edge_src]
        dst_h = node_h[edge_dst]

        src_active = active_nodes[edge_src]
        dst_active = active_nodes[edge_dst]
        src_active_col = src_active.unsqueeze(-1)
        dst_active_col = dst_active.unsqueeze(-1)
        single_frontier = src_active ^ dst_active

        active_h = torch.where(
            src_active_col & ~dst_active_col,
            src_h,
            torch.where(dst_active_col & ~src_active_col, dst_h, 0.5 * (src_h + dst_h)),
        )
        inactive_h = torch.where(
            src_active_col & ~dst_active_col,
            dst_h,
            torch.where(dst_active_col & ~src_active_col, src_h, 0.5 * (src_h + dst_h)),
        )
        return active_h, inactive_h, src_active, dst_active, single_frontier

    @staticmethod
    def _build_edge_discrimination_features(
        *,
        batch: RetrievalBatch,
        node_h: torch.Tensor,
        edge_relation_h: torch.Tensor,
        q_h: torch.Tensor,
        active_nodes: torch.Tensor,
        relation_prior_logits: torch.Tensor,
        node_query_scores: torch.Tensor,
    ) -> torch.Tensor:
        active_h, inactive_h, src_active, dst_active, single_frontier = (
            Policy._resolve_edge_roles(
                edge_index=batch.edge_index,
                node_h=node_h,
                active_nodes=active_nodes,
            )
        )
        q_per_edge = q_h.index_select(0, batch.edge_batch)
        dynamic_relation = F.cosine_similarity(
            q_per_edge,
            edge_relation_h,
            dim=-1,
            eps=1e-8,
        )
        dynamic_inactive = F.cosine_similarity(
            q_per_edge,
            inactive_h,
            dim=-1,
            eps=1e-8,
        )
        dynamic_active = F.cosine_similarity(
            q_per_edge,
            active_h,
            dim=-1,
            eps=1e-8,
        )

        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        src_query = node_query_scores[edge_src]
        dst_query = node_query_scores[edge_dst]
        static_tail = torch.where(
            src_active & ~dst_active,
            dst_query,
            torch.where(
                dst_active & ~src_active,
                src_query,
                torch.maximum(src_query, dst_query),
            ),
        )
        static_tail = torch.where(
            single_frontier, static_tail, torch.zeros_like(static_tail)
        )

        return torch.nan_to_num(
            torch.stack(
                [
                    relation_prior_logits,
                    dynamic_relation,
                    static_tail,
                    dynamic_inactive,
                    dynamic_inactive - dynamic_active,
                    single_frontier.float(),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def _build_node_query_scores(batch: RetrievalBatch) -> torch.Tensor:
        question_emb = torch.nan_to_num(batch.question_emb.float())
        node_emb = torch.nan_to_num(batch.node_tokens.float())
        if question_emb.dim() != 2:
            raise ValueError(
                f"question_emb must be 2D, got shape {tuple(question_emb.shape)}."
            )
        if node_emb.dim() != 2:
            raise ValueError(
                f"node_tokens must be 2D, got shape {tuple(node_emb.shape)}."
            )
        if question_emb.size(-1) != node_emb.size(-1):
            raise ValueError(
                "question_emb and node_tokens must share the same width for node-query "
                f"similarity, got {question_emb.size(-1)} and {node_emb.size(-1)}."
            )

        node_question_emb = question_emb.index_select(0, batch.batch)
        node_query_scores = F.cosine_similarity(
            node_question_emb,
            node_emb,
            dim=-1,
            eps=1e-8,
        )
        return torch.nan_to_num(node_query_scores)

    @staticmethod
    def _build_edge_struct_features(
        *,
        edge_index: torch.Tensor,
        active_nodes: torch.Tensor,
        component_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        构建每条边的结构特征（5维 float），不需要梯度。

        active_nodes 来自不可变状态快照；component_labels 是离散拓扑标签。
        这里直接读取即可，不需要额外 detach。
        """
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        src_active = active_nodes[edge_src]
        dst_active = active_nodes[edge_dst]
        both_active = src_active & dst_active

        src_labels = component_labels[edge_src]
        dst_labels = component_labels[edge_dst]
        cross_component = both_active & (src_labels != dst_labels)
        cycle_closure = both_active & ~cross_component

        # stack 转 float，全程无梯度
        return torch.stack(
            [
                src_active.float(),
                dst_active.float(),
                both_active.float(),
                cross_component.float(),
                cycle_closure.float(),
            ],
            dim=-1,
        )

    def _build_relation_prior_logits(self, batch: RetrievalBatch) -> torch.Tensor:
        """Build zero-shot prior logits from question-relation cosine similarity."""
        question_emb = torch.nan_to_num(batch.question_emb.float())
        relation_emb = torch.nan_to_num(batch.edge_relation_tokens.float())
        if question_emb.dim() != 2:
            raise ValueError(
                f"question_emb must be 2D, got shape {tuple(question_emb.shape)}."
            )
        if relation_emb.dim() != 2:
            raise ValueError(
                "edge_relation_tokens must be 2D, "
                f"got shape {tuple(relation_emb.shape)}."
            )
        if question_emb.size(-1) != relation_emb.size(-1):
            raise ValueError(
                "question_emb and edge_relation_tokens must share the same width for "
                f"relation prior, got {question_emb.size(-1)} and {relation_emb.size(-1)}."
            )

        edge_question_emb = question_emb.index_select(0, batch.edge_batch)
        relation_prior_logits = F.cosine_similarity(
            edge_question_emb,
            relation_emb,
            dim=-1,
            eps=1e-8,
        )
        return torch.nan_to_num(relation_prior_logits)

    @staticmethod
    def _build_type_features(
        *,
        node_query_scores: torch.Tensor,
        relation_prior_logits: torch.Tensor,
        valid_edges_mask: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        edge_index: torch.Tensor,
        node_batch_index: torch.Tensor,
        edge_batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Graph-level hazard features for Stop/Expand."""

        def _scatter_segment_max(
            values: torch.Tensor,
            mask: torch.Tensor,
            batch_index: torch.Tensor,
        ) -> torch.Tensor:
            result = values.new_zeros(num_graphs)
            finite_mask = mask & torch.isfinite(values)
            if not bool(finite_mask.any().item()):
                return result
            max_values, _ = scatter_max(
                values[finite_mask],
                batch_index[finite_mask],
                dim=0,
                dim_size=num_graphs,
            )
            has_any = scatter_sum(
                finite_mask.int(), batch_index, dim=0, dim_size=num_graphs
            ).bool()
            return torch.where(has_any, max_values, result)

        active_node_count = scatter_sum(
            active_nodes.int(), node_batch_index, dim=0, dim_size=num_graphs
        ).float()
        frontier_edge_count = scatter_sum(
            valid_edges_mask.int(), edge_batch_index, dim=0, dim_size=num_graphs
        ).float()

        max_active_node_score = _scatter_segment_max(
            node_query_scores,
            active_nodes,
            node_batch_index,
        )
        max_frontier_rel_prior = _scatter_segment_max(
            relation_prior_logits,
            valid_edges_mask,
            edge_batch_index,
        )

        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        src_active = active_nodes[edge_src]
        dst_active = active_nodes[edge_dst]
        single_frontier = valid_edges_mask & (src_active ^ dst_active)
        tail_scores = torch.where(
            src_active & ~dst_active,
            node_query_scores[edge_dst],
            torch.where(
                dst_active & ~src_active,
                node_query_scores[edge_src],
                torch.zeros_like(relation_prior_logits),
            ),
        )
        max_frontier_tail_score = _scatter_segment_max(
            tail_scores,
            single_frontier,
            edge_batch_index,
        )

        return torch.stack(
            [
                max_frontier_rel_prior,
                max_frontier_tail_score,
                max_active_node_score,
                max_active_node_score - max_frontier_tail_score,
                torch.log1p(frontier_edge_count),
                torch.log1p(active_node_count),
            ],
            dim=-1,
        )

    def forward(
        self,
        batch: RetrievalBatch,
        state: SubgraphState,
    ) -> PolicyStepOutput:
        active_nodes = state.active_nodes
        active_edges = state.active_edges
        relation_prior_logits = self._build_relation_prior_logits(batch)
        node_query_scores = self._build_node_query_scores(batch)

        # 1. 骨干网络：active_edges 来自 rollout 状态快照，不与后续 inplace
        #    状态推进共享 version counter。
        node_h, edge_relation_h, q_h = self.backbone(batch, active_edges=active_edges)
        edge_state_h = self._build_edge_state(
            batch=batch,
            node_h=node_h,
            edge_relation_h=edge_relation_h,
            active_nodes=active_nodes,
            relation_prior_logits=relation_prior_logits,
            node_query_scores=node_query_scores,
        )
        edge_discrimination_features = self._build_edge_discrimination_features(
            batch=batch,
            node_h=node_h,
            edge_relation_h=edge_relation_h,
            q_h=q_h,
            active_nodes=active_nodes,
            relation_prior_logits=relation_prior_logits,
            node_query_scores=node_query_scores,
        )

        # 2. 子图状态摘要：消费 snapshot，不触碰 rollout 的可变状态
        subgraph_h, component_labels = self._summarize_subgraph(
            batch=batch,
            node_h=node_h,
            edge_state_h=edge_state_h,
            active_nodes=active_nodes,
            active_edges=active_edges,
        )

        # 3. 结构特征：离散状态描述符，仅作为 action_head 的条件输入
        edge_struct_features = self._build_edge_struct_features(
            edge_index=batch.edge_index,
            active_nodes=active_nodes,
            component_labels=component_labels,
        )
        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        valid_edges_mask = (
            active_nodes[edge_src] | active_nodes[edge_dst]
        ) & ~active_edges
        type_features = self._build_type_features(
            node_query_scores=node_query_scores,
            relation_prior_logits=relation_prior_logits,
            valid_edges_mask=valid_edges_mask,
            active_nodes=active_nodes,
            active_edges=active_edges,
            edge_index=batch.edge_index,
            node_batch_index=batch.batch,
            edge_batch_index=batch.edge_batch,
            num_graphs=batch.num_graphs,
        )

        # 4. 动作打分：接收带梯度的 edge_state_h / subgraph_h
        action_logits = self.action_head(
            edge_state_h=edge_state_h,
            subgraph_h=subgraph_h,
            edge_batch_index=batch.edge_batch,
            edge_struct_features=edge_struct_features,
            expand_edge_prior_logits=relation_prior_logits,
            edge_discrimination_features=edge_discrimination_features,
            type_features=type_features,
        )

        return PolicyStepOutput(
            action_logits=action_logits,
            question_h=q_h,
            subgraph_h=subgraph_h,
        )

    def root_log_z(
        self, question_h: torch.Tensor, root_subgraph_h: torch.Tensor
    ) -> torch.Tensor:
        return self.z_head(question_h=question_h, root_subgraph_h=root_subgraph_h)


__all__ = ["Policy", "PolicyStepOutput"]
