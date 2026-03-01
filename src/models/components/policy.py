# src/models/components/policy.py
"""
[系统实体] 双流策略网络 (Dual Flow Policy)
职责：
1. 静态图编码：基于 GraphEnvContext 执行异构图流形投影。
2. 动态动作提取：无锁 O(1) 提取当前智能体的可选拓扑动作。
3. 状态不可变演进：利用 GRU 进行状态转移，严防计算图撕裂与越界崩溃。
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.configs.policy import PolicyConfig
from src.models.environment import DynamicAgentState, GraphEnvContext
from .backward_prior import StructuralBackwardPrior
from .backbone import EmbeddingBackbone
from .positional_encoding import SinusoidalPositionalEncoding


class FlowHead(nn.Module):
    """状态流预测头: log F(s_t) = alpha * (q^T W h_v) + residual(h_v)"""

    def __init__(
        self,
        *,
        node_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        qcbia_alpha_init: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("flow_head.num_layers must be >= 1.")
        if qcbia_alpha_init <= 0:
            raise ValueError("flow_head.qcbia_alpha_init must be > 0.")

        self.qcbia_proj = nn.Linear(node_dim, question_dim, bias=False)
        self.qcbia_log_alpha = nn.Parameter(torch.tensor(math.log(float(qcbia_alpha_init)), dtype=torch.float32))

        layers: list[nn.Module] = []
        in_dim = int(node_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.residual = nn.Sequential(*layers)

    def forward(self, node_features: torch.Tensor, question_features: torch.Tensor) -> torch.Tensor:
        # 【核心修复 1】：引入缩放点积，消除 1024 维带来的方差爆炸
        qcbia_score = (question_features * self.qcbia_proj(node_features)).sum(dim=-1)
        scale = math.sqrt(question_features.size(-1))
        qcbia_score = qcbia_score / scale

        alpha = torch.exp(self.qcbia_log_alpha).to(device=node_features.device, dtype=node_features.dtype)
        residual = self.residual(node_features).squeeze(-1)
        return alpha * qcbia_score + residual

    def qcbia_alpha(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.exp(self.qcbia_log_alpha).to(device=device, dtype=dtype)


class NodePriorityHead(nn.Module):
    """节点优先级打分头: 输出 query-conditioned node score。"""

    def __init__(self, *, node_dim: int, question_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("priority_head.num_layers must be >= 1.")
        self.q_proj = nn.Linear(node_dim, question_dim, bias=False)
        layers: list[nn.Module] = []
        in_dim = int(node_dim + question_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.residual = nn.Sequential(*layers)

    def forward(self, node_features: torch.Tensor, question_features: torch.Tensor) -> torch.Tensor:
        bilinear = (question_features * self.q_proj(node_features)).sum(dim=-1)
        bilinear = bilinear / math.sqrt(question_features.size(-1))
        residual = self.residual(torch.cat((node_features, question_features), dim=-1)).squeeze(-1)
        return bilinear + residual


class DualFlowPolicy(nn.Module):
    """
    双流策略网络
    严格贯彻“动静分离”与“不可变状态转移”的强化学习基座
    """

    def __init__(self, config: PolicyConfig, *, backward_prior_mode: str = "uniform_in_degree") -> None:
        super().__init__()
        self.config = config
        self.backbone = EmbeddingBackbone(config.backbone)
        self.backward_prior = StructuralBackwardPrior(mode=backward_prior_mode)

        if config.backbone.use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(dim=config.backbone.hidden_dim)
        else:
            self.pos_encoder = None

        self.memory_tracker = nn.GRUCell(
            input_size=config.backbone.hidden_dim * 2,
            hidden_size=config.backbone.hidden_dim,
        )
        self.flow_head = FlowHead(
            node_dim=config.backbone.hidden_dim,
            question_dim=config.backbone.hidden_dim,
            hidden_dim=config.flow_head.hidden_dim,
            num_layers=config.flow_head.num_layers,
            dropout=config.flow_head.dropout,
            qcbia_alpha_init=config.flow_head.qcbia_alpha_init,
        )
        self.node_priority_head = NodePriorityHead(
            node_dim=config.backbone.hidden_dim,
            question_dim=config.backbone.hidden_dim,
            hidden_dim=config.priority_head.hidden_dim,
            num_layers=config.priority_head.num_layers,
            dropout=config.priority_head.dropout,
        )
        self.stop_head = nn.Linear(config.backbone.hidden_dim, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, float(config.stop_bias_init))
        self.topk_prune_train_k = int(config.topk_prune_train_k)
        self.topk_prune_train_k_final = int(config.topk_prune_train_k_final)
        self.topk_prune_warmup_steps = int(config.topk_prune_warmup_steps)
        self.topk_prune_anneal_steps = int(config.topk_prune_anneal_steps)
        self.topk_prune_eval_k = int(config.topk_prune_eval_k)
        if self.topk_prune_train_k < 0:
            raise ValueError("policy_cfg.topk_prune_train_k must be >= 0.")
        if self.topk_prune_train_k_final < 0:
            raise ValueError("policy_cfg.topk_prune_train_k_final must be >= 0.")
        if self.topk_prune_warmup_steps < 0:
            raise ValueError("policy_cfg.topk_prune_warmup_steps must be >= 0.")
        if self.topk_prune_anneal_steps < 0:
            raise ValueError("policy_cfg.topk_prune_anneal_steps must be >= 0.")
        if self.topk_prune_eval_k < 0:
            raise ValueError("policy_cfg.topk_prune_eval_k must be >= 0.")
        self._train_step = 0

    def current_qcbia_alpha(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.flow_head.qcbia_alpha(device=device, dtype=dtype)

    def encode_context(self, context: GraphEnvContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        静态环境编码
        通常在 Episode 的第一跳 (t=0) 被调用一次，后续时间步应被缓存
        """
        node_tokens = self.backbone.project_node_embeddings(context.node_embeddings)
        relation_tokens = self.backbone.project_relation_embeddings(context.relation_tokens)
        question_tokens = self.backbone.project_question_embeddings(context.question_emb)

        node_tokens = self.backbone.encode_graph(
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_index=context.edge_index,
            edge_relations=context.edge_relations,
            num_nodes=context.num_nodes_total,
            question_tokens=question_tokens,
            node_batch=context.node_batch,
        )
        return node_tokens, relation_tokens, question_tokens

    def _encode_hidden_with_time(self, hidden: torch.Tensor, step_t: int) -> torch.Tensor:
        if self.pos_encoder is None:
            return hidden
        step_tensor = torch.full((hidden.size(0),), step_t, device=hidden.device)
        return hidden + self.pos_encoder(step_tensor)

    def _predict_log_f_from_node(self, node_features: torch.Tensor, question_features: torch.Tensor) -> torch.Tensor:
        return self.flow_head(node_features, question_features)

    @staticmethod
    def _build_topk_node_keep_mask(
        *,
        node_ptr: torch.Tensor,
        node_scores: torch.Tensor,
        topk_k: int,
    ) -> torch.Tensor:
        num_graphs = int(node_ptr.numel()) - 1
        keep_mask = torch.zeros_like(node_scores, dtype=torch.bool)
        for graph_idx in range(num_graphs):
            start = int(node_ptr[graph_idx].item())
            end = int(node_ptr[graph_idx + 1].item())
            if end <= start:
                continue
            graph_scores = node_scores[start:end]
            graph_k = min(topk_k, int(graph_scores.numel()))
            if graph_k <= 0:
                continue
            if graph_k == int(graph_scores.numel()):
                keep_mask[start:end] = True
                continue
            topk_local = torch.topk(graph_scores, k=graph_k, dim=0).indices + start
            keep_mask[topk_local] = True
        return keep_mask

    def set_training_step(self, step: int) -> None:
        self._train_step = max(int(step), 0)

    def _resolve_topk_prune_k(self) -> int:
        if not self.training:
            return self.topk_prune_eval_k
        if self.topk_prune_anneal_steps <= 0:
            return self.topk_prune_train_k
        if self.topk_prune_train_k == self.topk_prune_train_k_final:
            return self.topk_prune_train_k
        effective_step = max(self._train_step - self.topk_prune_warmup_steps, 0)
        progress = min(float(effective_step) / float(self.topk_prune_anneal_steps), 1.0)
        k_float = float(self.topk_prune_train_k) + (
            float(self.topk_prune_train_k_final - self.topk_prune_train_k) * progress
        )
        return max(int(round(k_float)), 0)

    def compute_node_priority_scores(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        node_graph_ids = env_context.node_batch.to(device=node_tokens.device, dtype=torch.long).clamp(min=0)
        node_questions = question_tokens.index_select(0, node_graph_ids)
        return self.node_priority_head(node_tokens, node_questions)

    def _compute_edge_logits(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        target_nodes: torch.Tensor,
        out_degrees: torch.Tensor,
        node_priority_keep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, num_agents = agent_state.current_nodes.shape
        agent_indices = torch.arange(B * num_agents, device=target_nodes.device)
        edge_agent_batch = agent_indices.repeat_interleave(out_degrees)

        flat_curr_nodes = agent_state.current_nodes.view(-1).clamp(min=0)
        agent_batch = torch.arange(B, device=target_nodes.device).repeat_interleave(num_agents)
        edge_graph_ids = agent_batch.index_select(0, edge_agent_batch)
        source_nodes = flat_curr_nodes.index_select(0, edge_agent_batch)

        # Strict lookahead policy:
        # logit(u->v) = log P_B(u|v) + log F(v)
        next_question = question_tokens.index_select(0, edge_graph_ids)
        next_node_features = node_tokens.index_select(0, target_nodes)
        log_f_next = self._predict_log_f_from_node(next_node_features, next_question)

        log_pb_prior = self.backward_prior.log_prob_edges(
            env_context=env_context,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_graph_ids=edge_graph_ids,
            dtype=log_f_next.dtype,
        )
        edge_logits = log_pb_prior + log_f_next

        visited = agent_state.visited_mask
        if visited.dim() == 1:
            is_visited = visited[target_nodes]
        elif visited.dim() == 2:
            is_visited = visited[edge_agent_batch, target_nodes]
        else:
            raise ValueError(f"visited_mask must be 1D or 2D, got shape={tuple(visited.shape)}")

        neg_inf = torch.tensor(float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype)
        edge_logits = edge_logits.masked_fill(is_visited, neg_inf)
        if node_priority_keep_mask is not None:
            keep_targets = node_priority_keep_mask.index_select(0, target_nodes.clamp(min=0))
            edge_logits = edge_logits.masked_fill(~keep_targets, neg_inf)
        return edge_logits, edge_agent_batch

    def _compute_stop_logits(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        B, num_agents = agent_state.current_nodes.shape
        flat_hidden = agent_state.hidden_states.view(B * num_agents, -1)
        flat_hidden = self._encode_hidden_with_time(flat_hidden, agent_state.step_t)

        stop_logits = self.stop_head(flat_hidden).squeeze(-1).to(dtype=dtype)

        # Keep super-source anti-early-stop rule.
        if env_context.start_local_indices is not None:
            graph_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(num_agents)
            super_abs = env_context.start_local_indices.to(device=device) + env_context.node_ptr[:-1].to(device=device)
            super_nodes = super_abs.index_select(0, graph_ids)
            current_nodes = agent_state.current_nodes.view(-1).clamp(min=0)
            at_super = current_nodes == super_nodes
            stop_logits = stop_logits.masked_fill(
                at_super,
                torch.tensor(float("-inf"), device=stop_logits.device, dtype=stop_logits.dtype),
            )

        return stop_logits

    def compute_state_flow(
        self,
        agent_state: DynamicAgentState,
        question_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, num_agents = agent_state.current_nodes.shape
        current_nodes = agent_state.current_nodes.view(-1).clamp(min=0)
        agent_batch = torch.arange(B, device=current_nodes.device).repeat_interleave(num_agents)
        q_features = question_tokens.index_select(0, agent_batch)
        node_features = node_tokens.index_select(0, current_nodes)
        log_f = self._predict_log_f_from_node(node_features, q_features)
        return log_f.view(B, num_agents)

    def _gather_actions_from_csr_lock_free(
        self,
        adj_t: torch.Tensor,
        active_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [核心算子] O(1) 并行拓扑动作提取
        利用前缀和与张量广播，粉碎 O(E) 的遍历图开销。
        """
        crow = adj_t.crow_indices()
        col = adj_t.col_indices()
        values = adj_t.values()

        start_ptrs = crow[active_nodes]
        end_ptrs = crow[active_nodes + 1]
        out_degrees = end_ptrs - start_ptrs

        total_edges = int(out_degrees.sum().item())
        if total_edges == 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=active_nodes.device)
            return empty_idx, empty_idx, out_degrees

        base_idx = start_ptrs.repeat_interleave(out_degrees)
        segment_starts = out_degrees.cumsum(0) - out_degrees
        flat_offsets = torch.arange(total_edges, device=active_nodes.device, dtype=torch.long)
        increments = flat_offsets - segment_starts.repeat_interleave(out_degrees)
        gather_idx = base_idx + increments

        target_nodes = col[gather_idx]
        gathered_edge_ids = values[gather_idx]
        return gathered_edge_ids, target_nodes, out_degrees

    def compute_action_scores(
        self,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del relation_tokens
        B, num_agents = agent_state.current_nodes.shape
        flat_curr_nodes = agent_state.current_nodes.view(-1)
        flat_active_mask = ~agent_state.done_mask.view(-1)
        active_nodes = torch.where(flat_active_mask, flat_curr_nodes, torch.zeros_like(flat_curr_nodes))

        topk_prune_k = self._resolve_topk_prune_k()
        node_priority_keep_mask: torch.Tensor | None = None
        if topk_prune_k > 0:
            node_priority_scores = self.compute_node_priority_scores(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
            node_priority_keep_mask = self._build_topk_node_keep_mask(
                node_ptr=env_context.node_ptr,
                node_scores=node_priority_scores,
                topk_k=topk_prune_k,
            )

        edge_ids, target_nodes, out_degrees = self._gather_actions_from_csr_lock_free(
            env_context.adj_t_fwd,
            active_nodes,
        )
        if edge_ids.numel() == 0:
            return self._build_empty_output(B, num_agents, node_tokens.device)

        edge_logits, edge_agent_batch = self._compute_edge_logits(
            env_context=env_context,
            agent_state=agent_state,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            target_nodes=target_nodes,
            out_degrees=out_degrees,
            node_priority_keep_mask=node_priority_keep_mask,
        )

        stop_logits = self._compute_stop_logits(
            env_context=env_context,
            agent_state=agent_state,
            device=node_tokens.device,
            dtype=node_tokens.dtype,
        )

        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits.view(B, num_agents),
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees.view(B, num_agents),
        }

    def _build_empty_output(self, B: int, num_agents: int, device: torch.device) -> dict[str, torch.Tensor]:
        """防崩溃的安全网兜底输出"""
        return {
            "edge_logits": torch.empty(0, device=device),
            "edge_agent_batch": torch.empty(0, dtype=torch.long, device=device),
            "stop_logits": torch.zeros((B, num_agents), device=device),
            "edge_ids": torch.empty(0, dtype=torch.long, device=device),
            "target_nodes": torch.empty(0, dtype=torch.long, device=device),
            "out_degrees": torch.zeros((B, num_agents), dtype=torch.long, device=device),
        }

    def evolve_state(
        self,
        agent_state: DynamicAgentState,
        chosen_target_nodes: torch.Tensor,
        chosen_edge_relations: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        is_stop: torch.Tensor,
    ) -> DynamicAgentState:
        """
        [系统实体] 记忆更新与严格不可变的状态转移
        严防 CUDA 内存越界 (Index Out of Bounds) 和计算图原地修改破坏。
        """
        B, num_agents = agent_state.current_nodes.shape
        flat_hidden = agent_state.hidden_states.view(-1, self.config.backbone.hidden_dim)

        safe_target_nodes = torch.where(is_stop, torch.zeros_like(chosen_target_nodes), chosen_target_nodes)
        safe_edge_relations = torch.where(is_stop, torch.zeros_like(chosen_edge_relations), chosen_edge_relations)

        new_node_emb = node_tokens.index_select(0, safe_target_nodes)
        new_rel_emb = relation_tokens.index_select(0, safe_edge_relations)

        gru_input = torch.cat([new_node_emb, new_rel_emb], dim=-1)
        next_hidden = self.memory_tracker(gru_input, flat_hidden)
        next_hidden = torch.where(is_stop.unsqueeze(-1), flat_hidden, next_hidden)

        new_visited_mask = agent_state.visited_mask.clone()
        if new_visited_mask.dim() == 1:
            new_visited_mask.scatter_(0, safe_target_nodes, True)
        elif new_visited_mask.dim() == 2:
            active_move = ~is_stop
            if bool(active_move.any().item()):
                row_ids = torch.arange(is_stop.numel(), device=is_stop.device, dtype=torch.long)[active_move]
                col_ids = safe_target_nodes[active_move]
                new_visited_mask[row_ids, col_ids] = True
        else:
            raise ValueError(f"visited_mask must be 1D or 2D, got shape={tuple(new_visited_mask.shape)}")

        new_current_nodes = torch.where(
            is_stop.view(B, num_agents),
            agent_state.current_nodes,
            safe_target_nodes.view(B, num_agents),
        )

        return DynamicAgentState(
            step_t=agent_state.step_t + 1,
            current_nodes=new_current_nodes,
            hidden_states=next_hidden.view(B, num_agents, -1),
            visited_mask=new_visited_mask,
            cumulative_rewards=agent_state.cumulative_rewards,
            done_mask=agent_state.done_mask | is_stop.view(B, num_agents),
        )

    def forward(
        self,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
    ) -> dict[str, torch.Tensor]:
        """前向计算流入口"""
        node_tokens, relation_tokens, question_tokens = self.encode_context(env_context)
        return self.compute_action_scores(
            env_context=env_context,
            agent_state=agent_state,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            relation_tokens=relation_tokens,
        )


__all__ = ["DualFlowPolicy"]
