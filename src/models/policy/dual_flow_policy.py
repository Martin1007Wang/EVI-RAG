# src/models/policy/dual_flow_policy.py
"""
[系统实体] 双流策略网络 (Dual Flow Policy)
职责：
1. 静态图编码：基于 GraphEnvContext 执行异构图流形投影。
2. 动态动作提取：并行提取当前智能体可选拓扑动作。
3. 状态不可变演进：基于路径 token 历史做因果自注意力状态更新。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.configs.policy import PolicyConfig
from src.models.environment import (
    DynamicAgentState,
    GraphEnvContext,
    has_super_source_layout,
)
from .action import (
    build_empty_output as build_empty_output_helper,
    compute_stop_logits as compute_stop_logits_helper,
    gather_actions_from_csr_lock_free as gather_actions_from_csr_lock_free_helper,
)
from src.models.backbone import EmbeddingBackbone
from .edge import compute_edge_logits as compute_edge_logits_helper
from .path import (
    build_path_token_embeddings as build_path_token_embeddings_helper,
    encode_path_history as encode_path_history_helper,
    evolve_state as evolve_state_helper,
    resolve_path_state as resolve_path_state_helper,
)
from .question import (
    build_question_context_tokens as build_question_context_tokens_helper,
    build_question_padding_mask as build_question_padding_mask_helper,
    compute_agent_potentials as compute_agent_potentials_helper,
    compute_question_token_pool as compute_question_token_pool_helper,
)
from src.models.backbone import SinusoidalPositionalEncoding


class NodePriorityHead(nn.Module):
    """节点优先级打分头: 输出 query-conditioned node score。"""

    def __init__(
        self,
        *,
        node_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
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

    def forward(
        self, node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        bilinear = (question_features * self.q_proj(node_features)).sum(dim=-1)
        bilinear = bilinear / math.sqrt(question_features.size(-1))
        residual = self.residual(
            torch.cat((node_features, question_features), dim=-1)
        ).squeeze(-1)
        return bilinear + residual


class DualFlowPolicy(nn.Module):
    """
    双流策略网络
    使用路径因果自注意力 + 问题跨注意力提取状态势能向量 vecF(s_t | q)。
    """

    def __init__(
        self, config: PolicyConfig, *, backward_prior_mode: str = "uniform_in_degree"
    ) -> None:
        super().__init__()
        del backward_prior_mode
        self.config = config
        self.backbone = EmbeddingBackbone(config.backbone)
        self.stop_delta_scale = float(config.stop_delta_scale)
        self.stop_delta_temperature = float(config.stop_delta_temperature)
        if self.stop_delta_scale <= 0.0:
            raise ValueError("stop_delta_scale must be > 0.")
        if self.stop_delta_temperature <= 0.0:
            raise ValueError("stop_delta_temperature must be > 0.")
        self.doob_h_alpha = float(config.doob_h_alpha)
        self.doob_h_node_temperature = float(config.doob_h_node_temperature)
        if self.doob_h_alpha < 0.0:
            raise ValueError("doob_h_alpha must be >= 0.")
        if self.doob_h_node_temperature <= 0.0:
            raise ValueError("doob_h_node_temperature must be > 0.")

        graph_hidden_dim = int(config.backbone.hidden_dim)
        if graph_hidden_dim <= 0:
            raise ValueError("backbone.hidden_dim must be > 0.")
        policy_dim = int(config.backbone.embedding_dim)
        if policy_dim <= 0:
            raise ValueError("backbone.embedding_dim must be > 0.")
        self.graph_hidden_dim = graph_hidden_dim
        self.policy_dim = policy_dim

        if graph_hidden_dim == policy_dim:
            self.path_to_policy = nn.Identity()
            self.node_to_policy = nn.Identity()
            self.relation_to_policy = nn.Identity()
        else:
            self.path_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            self.node_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            self.relation_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            nn.init.xavier_uniform_(self.path_to_policy.weight)
            nn.init.xavier_uniform_(self.node_to_policy.weight)
            nn.init.xavier_uniform_(self.relation_to_policy.weight)
            if self.path_to_policy.bias is not None:
                nn.init.zeros_(self.path_to_policy.bias)
            if self.node_to_policy.bias is not None:
                nn.init.zeros_(self.node_to_policy.bias)
            if self.relation_to_policy.bias is not None:
                nn.init.zeros_(self.relation_to_policy.bias)
        dropout = float(config.flow_head.dropout)

        if config.backbone.use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(dim=graph_hidden_dim)
        else:
            self.pos_encoder = None

        # Path encoder: h_t = SelfAttention(path_tokens)[:, -1, :]
        self.path_self_attention = nn.MultiheadAttention(
            embed_dim=graph_hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.path_self_attention_norm = nn.LayerNorm(graph_hidden_dim)

        # Potential extractor: vecF = CrossAttention(Query=h_t, Keys/Values=C_q)
        self.question_cross_attention = nn.MultiheadAttention(
            embed_dim=policy_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.question_cross_attention_norm = nn.LayerNorm(policy_dim)
        self.question_token_scorer = nn.Sequential(
            nn.Linear(policy_dim, policy_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(policy_dim, 1),
        )
        self.question_global_proj = nn.Linear(policy_dim, policy_dim)

        # Action encoder: E_e = MLP([rel_embed || target_node_embed])
        self.edge_action_encoder = nn.Sequential(
            nn.Linear(policy_dim * 2, int(config.flow_head.hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(int(config.flow_head.hidden_dim), policy_dim),
        )
        self.edge_action_norm = nn.LayerNorm(policy_dim)
        self.relation_group_head = nn.Sequential(
            nn.Linear(policy_dim * 2, int(config.flow_head.hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(int(config.flow_head.hidden_dim), 1),
        )
        lexical_rank = int(config.flow_head.relation_low_rank)
        if lexical_rank <= 0:
            raise ValueError(
                "flow_head.relation_low_rank must be > 0 for token-level lexical alignment."
            )
        self.relation_lexical_proj = nn.Linear(policy_dim, lexical_rank, bias=False)
        self.question_lexical_proj = nn.Linear(policy_dim, lexical_rank, bias=False)
        self.lexical_bias_log_scale = nn.Parameter(torch.tensor(1.0))

        self.node_priority_head = NodePriorityHead(
            node_dim=graph_hidden_dim,
            question_dim=graph_hidden_dim,
            hidden_dim=config.priority_head.hidden_dim,
            num_layers=config.priority_head.num_layers,
            dropout=config.priority_head.dropout,
        )
        self.log_flow_head = nn.Linear(policy_dim, 1)
        nn.init.zeros_(self.log_flow_head.weight)
        nn.init.zeros_(self.log_flow_head.bias)
        self.stop_head = nn.Linear(policy_dim, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, float(config.stop_bias_init))

    def _compute_state_log_flows(
        self,
        *,
        agent_potential: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw_flow = self.log_flow_head(agent_potential.to(dtype=torch.float32)).squeeze(
            -1
        )
        raw_flow = torch.where(
            torch.isfinite(raw_flow), raw_flow, torch.zeros_like(raw_flow)
        )
        return raw_flow.to(dtype=dtype)

    def encode_context(
        self, context: GraphEnvContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        静态环境编码
        通常在 Episode 的第一跳 (t=0) 被调用一次，后续时间步应被缓存
        """
        node_tokens = self.backbone.project_node_embeddings(context.node_embeddings)
        relation_tokens = self.backbone.project_relation_embeddings(
            context.relation_tokens
        )
        question_tokens = self.backbone.project_question_embeddings(
            context.question_emb
        )

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

    def _build_question_context_tokens(
        self,
        *,
        env_context: GraphEnvContext,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return build_question_context_tokens_helper(
            env_context=env_context,
            question_tokens=question_tokens,
            policy_dim=self.policy_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            embedding_dim=int(self.config.backbone.embedding_dim),
            path_to_policy=self.path_to_policy,
        )

    @staticmethod
    def _build_question_padding_mask(
        *,
        env_context: GraphEnvContext,
        question_context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return build_question_padding_mask_helper(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )

    def compute_node_priority_scores(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        node_graph_ids = env_context.node_batch.to(
            device=node_tokens.device, dtype=torch.long
        ).clamp(min=0)
        node_questions = question_tokens.index_select(0, node_graph_ids)
        return self.node_priority_head(node_tokens, node_questions)

    def compute_node_log_h(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        node_scores = self.compute_node_priority_scores(
            env_context=env_context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        node_scores = node_scores.to(dtype=torch.float32) / self.doob_h_node_temperature
        node_log_h = F.logsigmoid(node_scores)
        node_log_h = torch.where(
            torch.isfinite(node_log_h),
            node_log_h,
            torch.full_like(node_log_h, float("-inf")),
        )
        return node_log_h.to(dtype=node_tokens.dtype)

    def build_action_cache(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        question_context_tokens = self._build_question_context_tokens(
            env_context=env_context,
            question_tokens=question_tokens,
        )
        question_padding_mask = self._build_question_padding_mask(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )
        node_log_h = self.compute_node_log_h(
            env_context=env_context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        cache = {
            "question_context_tokens": question_context_tokens,
            "question_padding_mask": question_padding_mask,
            "node_log_h": node_log_h,
        }
        if has_super_source_layout(
            node_ptr=env_context.node_ptr,
            node_global_ids=env_context.node_global_ids,
            num_nodes_total=env_context.num_nodes_total,
            device=question_tokens.device,
        ):
            cache["super_node_mask"] = (
                env_context.node_global_ids.to(
                    device=question_tokens.device, dtype=torch.long
                )
                < 0
            )
        return cache

    @staticmethod
    def _resolve_path_state(
        agent_state: DynamicAgentState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return resolve_path_state_helper(agent_state=agent_state)

    def _build_path_token_embeddings(
        self,
        *,
        path_token_ids: torch.Tensor,
        path_token_types: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return build_path_token_embeddings_helper(
            path_token_ids=path_token_ids,
            path_token_types=path_token_types,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            pos_encoder=self.pos_encoder,
        )

    def _encode_path_history(
        self,
        *,
        path_tokens: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return encode_path_history_helper(
            path_tokens=path_tokens,
            path_lengths=path_lengths,
            path_self_attention=self.path_self_attention,
            path_self_attention_norm=self.path_self_attention_norm,
        )

    def _compute_question_token_pool(
        self,
        *,
        agent_question_context: torch.Tensor,
        agent_question_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return compute_question_token_pool_helper(
            question_token_scorer=self.question_token_scorer,
            question_global_proj=self.question_global_proj,
            agent_question_context=agent_question_context,
            agent_question_padding_mask=agent_question_padding_mask,
        )

    def _compute_agent_potentials(
        self,
        *,
        env_context: GraphEnvContext,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        num_agents: int,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_agent_potentials_helper(
            env_context=env_context,
            question_tokens=question_tokens,
            agent_history=agent_history,
            num_agents=num_agents,
            question_context_tokens=question_context_tokens,
            question_padding_mask=question_padding_mask,
            policy_dim=self.policy_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            embedding_dim=int(self.config.backbone.embedding_dim),
            path_to_policy=self.path_to_policy,
            question_lexical_proj=self.question_lexical_proj,
            question_cross_attention=self.question_cross_attention,
            question_cross_attention_norm=self.question_cross_attention_norm,
            question_token_scorer=self.question_token_scorer,
            question_global_proj=self.question_global_proj,
        )

    def _compute_edge_logits(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_log_h: torch.Tensor,
        edge_agent_batch: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_relations: torch.Tensor,
        current_nodes: torch.Tensor,
        total_agents: int,
        agent_potential: torch.Tensor,
        lexical_question_tokens: torch.Tensor,
        agent_question_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_edge_logits_helper(
            env_context=env_context,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            node_log_h=node_log_h,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            edge_relations=edge_relations,
            current_nodes=current_nodes,
            total_agents=total_agents,
            agent_potential=agent_potential,
            lexical_question_tokens=lexical_question_tokens,
            agent_question_padding_mask=agent_question_padding_mask,
            relation_to_policy=self.relation_to_policy,
            node_to_policy=self.node_to_policy,
            edge_action_encoder=self.edge_action_encoder,
            edge_action_norm=self.edge_action_norm,
            relation_group_head=self.relation_group_head,
            relation_lexical_proj=self.relation_lexical_proj,
            lexical_bias_log_scale=self.lexical_bias_log_scale,
            doob_h_alpha=self.doob_h_alpha,
        )

    def _compute_stop_delta(
        self,
        *,
        agent_potential: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw_stop = self.stop_head(agent_potential.to(dtype=torch.float32)).squeeze(-1)
        raw_stop = torch.where(
            torch.isfinite(raw_stop), raw_stop, torch.zeros_like(raw_stop)
        )
        normalized = raw_stop / self.stop_delta_temperature
        bounded_delta = self.stop_delta_scale * torch.tanh(normalized)
        return bounded_delta.to(dtype=dtype)

    def _compute_stop_logits(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        stop_delta: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        edge_logits: torch.Tensor | None = None,
        edge_agent_batch: torch.Tensor | None = None,
        total_agents: int | None = None,
        super_node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_stop_logits_helper(
            env_context=env_context,
            agent_state=agent_state,
            stop_delta=stop_delta,
            device=device,
            dtype=dtype,
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            total_agents=total_agents,
            super_node_mask=super_node_mask,
        )

    def _gather_actions_from_csr_lock_free(
        self,
        adj_t,
        active_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return gather_actions_from_csr_lock_free_helper(
            adj_t=adj_t,
            active_nodes=active_nodes,
        )

    def compute_action_scores(
        self,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        action_cache: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        B, num_agents = agent_state.current_nodes.shape
        total_agents = B * num_agents
        flat_curr_nodes = agent_state.current_nodes.view(-1)
        flat_active_mask = ~agent_state.done_mask.view(-1)

        if action_cache is None:
            resolved_cache = self.build_action_cache(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
        else:
            resolved_cache = action_cache
        node_log_h = resolved_cache.get("node_log_h")
        if node_log_h is None:
            node_log_h = self.compute_node_log_h(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
        question_context_tokens = resolved_cache.get("question_context_tokens")
        if question_context_tokens is None:
            raise ValueError("action_cache must provide `question_context_tokens`.")
        question_padding_mask = resolved_cache.get("question_padding_mask")
        if question_padding_mask is None:
            raise ValueError(
                "action_cache must provide `question_padding_mask` for token-level question interaction."
            )
        super_node_mask = resolved_cache.get("super_node_mask")

        path_token_ids, path_token_types, path_lengths = self._resolve_path_state(
            agent_state
        )
        path_embeddings = self._build_path_token_embeddings(
            path_token_ids=path_token_ids,
            path_token_types=path_token_types,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
        )
        agent_history = self._encode_path_history(
            path_tokens=path_embeddings,
            path_lengths=path_lengths,
        )
        agent_potential, _, lexical_question_tokens, agent_question_padding_mask = (
            self._compute_agent_potentials(
                env_context=env_context,
                question_tokens=question_tokens,
                agent_history=agent_history,
                num_agents=num_agents,
                question_context_tokens=question_context_tokens,
                question_padding_mask=question_padding_mask,
            )
        )
        state_log_flows = self._compute_state_log_flows(
            agent_potential=agent_potential, dtype=node_tokens.dtype
        )
        stop_delta = self._compute_stop_delta(
            agent_potential=agent_potential, dtype=node_tokens.dtype
        )

        active_nodes = torch.where(
            flat_active_mask, flat_curr_nodes, torch.zeros_like(flat_curr_nodes)
        )
        edge_ids, target_nodes, out_degrees = self._gather_actions_from_csr_lock_free(
            env_context.adj_t_fwd,
            active_nodes,
        )
        if int(edge_ids.numel()) == 0:
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=super_node_mask,
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
                state_log_flows=state_log_flows.view(B, num_agents),
            )

        all_agent_rows = torch.arange(
            total_agents, device=target_nodes.device, dtype=torch.long
        )
        edge_agent_batch_full = all_agent_rows.repeat_interleave(out_degrees)
        edge_active_mask = flat_active_mask.index_select(0, edge_agent_batch_full)
        if not bool(edge_active_mask.all().item()):
            edge_ids = edge_ids[edge_active_mask]
            target_nodes = target_nodes[edge_active_mask]
            edge_agent_batch_full = edge_agent_batch_full[edge_active_mask]
            out_degrees_active = torch.zeros(
                (total_agents,), dtype=torch.long, device=node_tokens.device
            )
            if int(edge_agent_batch_full.numel()) > 0:
                out_degrees_active.scatter_add_(
                    0,
                    edge_agent_batch_full,
                    torch.ones_like(
                        edge_agent_batch_full,
                        dtype=torch.long,
                        device=node_tokens.device,
                    ),
                )
            out_degrees = out_degrees_active
        if int(edge_ids.numel()) == 0:
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=super_node_mask,
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
                state_log_flows=state_log_flows.view(B, num_agents),
            )

        visited = agent_state.visited_mask
        if visited.dim() == 1:
            is_visited = visited[target_nodes]
        elif visited.dim() == 2:
            is_visited = visited[edge_agent_batch_full, target_nodes]
        else:
            raise ValueError(
                f"visited_mask must be 1D or 2D, got shape={tuple(visited.shape)}"
            )

        keep_edge = ~is_visited
        if not bool(keep_edge.any().item()):
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=super_node_mask,
            )
            return {
                "edge_logits": torch.empty(
                    0, device=node_tokens.device, dtype=node_tokens.dtype
                ),
                "edge_agent_batch": torch.empty(
                    0, dtype=torch.long, device=node_tokens.device
                ),
                "stop_logits": stop_logits.view(B, num_agents),
                "edge_ids": torch.empty(0, dtype=torch.long, device=node_tokens.device),
                "target_nodes": torch.empty(
                    0, dtype=torch.long, device=node_tokens.device
                ),
                "out_degrees": torch.zeros(
                    (B, num_agents), dtype=torch.long, device=node_tokens.device
                ),
                "state_log_flows": state_log_flows.view(B, num_agents),
            }

        edge_ids = edge_ids[keep_edge]
        target_nodes = target_nodes[keep_edge]
        edge_agent_batch = edge_agent_batch_full[keep_edge]
        out_degrees_filtered = torch.zeros(
            (total_agents,), dtype=torch.long, device=node_tokens.device
        )
        out_degrees_filtered.scatter_add_(
            0,
            edge_agent_batch,
            torch.ones_like(
                edge_agent_batch, dtype=torch.long, device=node_tokens.device
            ),
        )

        edge_logits, edge_agent_batch = self._compute_edge_logits(
            env_context=env_context,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            node_log_h=node_log_h,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            edge_relations=env_context.edge_relations.index_select(
                0, edge_ids.clamp(min=0)
            ),
            current_nodes=flat_curr_nodes,
            total_agents=total_agents,
            agent_potential=agent_potential,
            lexical_question_tokens=lexical_question_tokens,
            agent_question_padding_mask=agent_question_padding_mask,
        )

        stop_logits = self._compute_stop_logits(
            env_context=env_context,
            agent_state=agent_state,
            stop_delta=stop_delta,
            device=node_tokens.device,
            dtype=node_tokens.dtype,
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            total_agents=total_agents,
            super_node_mask=super_node_mask,
        )

        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits.view(B, num_agents),
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees_filtered.view(B, num_agents),
            "state_log_flows": state_log_flows.view(B, num_agents),
        }

    @staticmethod
    def _build_empty_output(
        *,
        B: int,
        num_agents: int,
        device: torch.device,
        dtype: torch.dtype,
        stop_logits: torch.Tensor,
        state_log_flows: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return build_empty_output_helper(
            B=B,
            num_agents=num_agents,
            device=device,
            dtype=dtype,
            stop_logits=stop_logits,
            state_log_flows=state_log_flows,
        )

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
        [系统实体] 严格不可变的状态转移
        轨迹历史采用 token 序列维护，不依赖 GRU 递归隐状态。
        """
        del relation_tokens
        return evolve_state_helper(
            agent_state=agent_state,
            chosen_target_nodes=chosen_target_nodes,
            chosen_edge_relations=chosen_edge_relations,
            node_tokens=node_tokens,
            is_stop=is_stop,
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
