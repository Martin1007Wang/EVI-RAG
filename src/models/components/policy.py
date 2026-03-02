# src/models/components/policy.py
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
from torch import nn

from src.models.configs.policy import PolicyConfig
from src.models.environment import DynamicAgentState, GraphEnvContext
from .backward_prior import StructuralBackwardPrior
from .backbone import EmbeddingBackbone
from .positional_encoding import SinusoidalPositionalEncoding


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
    使用路径因果自注意力 + 问题跨注意力提取状态势能向量 vecF(s_t | q)。
    """

    def __init__(self, config: PolicyConfig, *, backward_prior_mode: str = "uniform_in_degree") -> None:
        super().__init__()
        self.config = config
        self.backbone = EmbeddingBackbone(config.backbone)
        self.backward_prior = StructuralBackwardPrior(mode=backward_prior_mode)
        # Theoretical contract: structural backward prior enters edge energy with fixed unit coefficient.
        self.edge_logit_pb_weight = 1.0
        self.stop_delta_scale = float(config.stop_delta_scale)
        self.stop_delta_temperature = float(config.stop_delta_temperature)
        if self.stop_delta_scale <= 0.0:
            raise ValueError("stop_delta_scale must be > 0.")
        if self.stop_delta_temperature <= 0.0:
            raise ValueError("stop_delta_temperature must be > 0.")

        hidden_dim = int(config.backbone.hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("backbone.hidden_dim must be > 0.")
        dropout = float(config.flow_head.dropout)

        if config.backbone.use_positional_encoding:
            self.pos_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        else:
            self.pos_encoder = None

        # Path encoder: h_t = SelfAttention(path_tokens)[:, -1, :]
        self.path_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.path_self_attention_norm = nn.LayerNorm(hidden_dim)

        # Potential extractor: vecF = CrossAttention(Query=h_t, Keys/Values=C_q)
        self.question_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.question_cross_attention_norm = nn.LayerNorm(hidden_dim)

        # Action encoder: E_e = MLP([rel_embed || target_node_embed])
        self.edge_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, int(config.flow_head.hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(int(config.flow_head.hidden_dim), hidden_dim),
        )
        self.edge_action_norm = nn.LayerNorm(hidden_dim)

        self.node_priority_head = NodePriorityHead(
            node_dim=hidden_dim,
            question_dim=hidden_dim,
            hidden_dim=config.priority_head.hidden_dim,
            num_layers=config.priority_head.num_layers,
            dropout=config.priority_head.dropout,
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, float(config.stop_bias_init))

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

    def _build_question_context_tokens(
        self,
        *,
        env_context: GraphEnvContext,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        question_ctx = env_context.question_ctx
        if question_ctx is None:
            return question_tokens.unsqueeze(1)
        if question_ctx.dim() != 3:
            raise ValueError(f"question_ctx must be 3D [B, L, d], got shape={tuple(question_ctx.shape)}.")
        if int(question_ctx.size(0)) != int(env_context.num_graphs):
            raise ValueError(
                "question_ctx batch mismatch with num_graphs: "
                f"question_ctx={int(question_ctx.size(0))}, num_graphs={int(env_context.num_graphs)}."
            )
        if int(question_ctx.size(1)) <= 0:
            raise ValueError("question_ctx length L must be > 0 when provided.")
        hidden_dim = int(question_tokens.size(-1))
        if int(question_ctx.size(-1)) == hidden_dim:
            return question_ctx.to(device=question_tokens.device, dtype=question_tokens.dtype)
        embedding_dim = int(self.config.backbone.embedding_dim)
        if int(question_ctx.size(-1)) != embedding_dim:
            raise ValueError(
                "question_ctx last dim mismatch with backbone dims: "
                f"question_ctx={int(question_ctx.size(-1))}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}."
            )
        return self.backbone.project_question_embeddings(question_ctx.to(device=question_tokens.device))

    @staticmethod
    def _build_question_padding_mask(
        *,
        env_context: GraphEnvContext,
        question_context_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        raw_mask = env_context.question_ctx_mask
        if raw_mask is None:
            return None
        if raw_mask.dim() != 2:
            raise ValueError(f"question_ctx_mask must be 2D [B, L], got shape={tuple(raw_mask.shape)}.")
        expected_shape = question_context_tokens.shape[:2]
        if tuple(raw_mask.shape) != tuple(expected_shape):
            raise ValueError(
                "question_ctx_mask shape mismatch with question_context_tokens: "
                f"mask={tuple(raw_mask.shape)}, context={tuple(expected_shape)}."
            )
        valid_mask = raw_mask.to(device=question_context_tokens.device, dtype=torch.bool)
        if bool((~valid_mask).all(dim=1).any().item()):
            raise ValueError("question_ctx_mask contains rows with zero valid tokens.")
        return ~valid_mask

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

    def build_action_cache(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del node_tokens
        question_context_tokens = self._build_question_context_tokens(
            env_context=env_context,
            question_tokens=question_tokens,
        )
        question_padding_mask = self._build_question_padding_mask(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )
        cache = {"question_context_tokens": question_context_tokens}
        if question_padding_mask is not None:
            cache["question_padding_mask"] = question_padding_mask
        return cache

    @staticmethod
    def _resolve_path_state(agent_state: DynamicAgentState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, num_agents = agent_state.current_nodes.shape
        total_agents = B * num_agents
        path_ids = agent_state.path_token_ids
        path_types = agent_state.path_token_types
        path_lengths = agent_state.path_lengths

        if path_ids is None and path_types is None and path_lengths is None:
            default_ids = agent_state.current_nodes.view(total_agents, 1).clone()
            default_types = torch.zeros_like(default_ids, dtype=torch.bool)
            default_lengths = torch.ones((total_agents,), dtype=torch.long, device=default_ids.device)
            return default_ids, default_types, default_lengths
        if path_ids is None or path_types is None or path_lengths is None:
            raise ValueError("path_token_ids/path_token_types/path_lengths must be all provided or all omitted.")
        if path_ids.dim() != 3 or path_types.dim() != 3:
            raise ValueError(
                "path_token_ids and path_token_types must be 3D [B, num_agents, T], "
                f"got ids={tuple(path_ids.shape)}, types={tuple(path_types.shape)}."
            )
        if path_lengths.dim() != 2:
            raise ValueError(f"path_lengths must be 2D [B, num_agents], got shape={tuple(path_lengths.shape)}.")
        if tuple(path_ids.shape) != tuple(path_types.shape):
            raise ValueError(
                "path_token_ids/path_token_types shape mismatch: "
                f"ids={tuple(path_ids.shape)}, types={tuple(path_types.shape)}."
            )
        if int(path_ids.size(0)) != B or int(path_ids.size(1)) != num_agents:
            raise ValueError(
                "path_token_ids leading dims mismatch with current_nodes: "
                f"path={tuple(path_ids.shape[:2])}, current_nodes={(B, num_agents)}."
            )
        if tuple(path_lengths.shape) != (B, num_agents):
            raise ValueError(
                "path_lengths shape mismatch with current_nodes: "
                f"path_lengths={tuple(path_lengths.shape)}, current_nodes={(B, num_agents)}."
            )
        flat_ids = path_ids.reshape(total_agents, path_ids.size(-1))
        flat_types = path_types.reshape(total_agents, path_types.size(-1)).to(dtype=torch.bool)
        flat_lengths = path_lengths.reshape(total_agents).to(device=flat_ids.device, dtype=torch.long)
        if int(flat_ids.size(-1)) <= 0:
            raise ValueError("path_token_ids must have T > 0.")
        if bool((flat_lengths <= 0).any().item()):
            raise ValueError("path_lengths must be > 0 for every agent.")
        if bool((flat_lengths > flat_ids.size(-1)).any().item()):
            raise ValueError(
                "path_lengths exceeds path_token_ids width: "
                f"max_len={int(flat_lengths.max().item())}, width={int(flat_ids.size(-1))}."
            )
        return flat_ids, flat_types, flat_lengths

    def _build_path_token_embeddings(
        self,
        *,
        path_token_ids: torch.Tensor,
        path_token_types: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if node_tokens.dim() != 2:
            raise ValueError(f"node_tokens must be 2D [N, d], got shape={tuple(node_tokens.shape)}.")
        if relation_tokens.dim() != 2:
            raise ValueError(f"relation_tokens must be 2D [R, d], got shape={tuple(relation_tokens.shape)}.")
        if int(node_tokens.size(0)) <= 0:
            raise ValueError("node_tokens must contain at least one node.")
        total_agents, token_len = path_token_ids.shape
        hidden_dim = int(node_tokens.size(-1))
        safe_node_ids = path_token_ids.clamp(min=0, max=int(node_tokens.size(0)) - 1)
        node_part = node_tokens.index_select(0, safe_node_ids.reshape(-1)).view(total_agents, token_len, hidden_dim)
        if int(relation_tokens.size(0)) == 0:
            if bool(path_token_types.any().item()):
                raise ValueError("path_token_types contains relation tokens but relation_tokens is empty.")
            relation_part = torch.zeros_like(node_part)
        else:
            safe_rel_ids = path_token_ids.clamp(min=0, max=int(relation_tokens.size(0)) - 1)
            relation_part = relation_tokens.index_select(0, safe_rel_ids.reshape(-1)).view(
                total_agents,
                token_len,
                hidden_dim,
            )
        path_tokens = torch.where(path_token_types.unsqueeze(-1), relation_part, node_part)
        if self.pos_encoder is not None:
            token_positions = torch.arange(token_len, device=path_tokens.device, dtype=torch.long)
            pos = self.pos_encoder(token_positions).to(device=path_tokens.device, dtype=path_tokens.dtype)
            path_tokens = path_tokens + pos.unsqueeze(0)
        return path_tokens

    def _encode_path_history(
        self,
        *,
        path_tokens: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> torch.Tensor:
        total_agents, token_len, hidden_dim = path_tokens.shape
        key_padding_mask = torch.arange(token_len, device=path_tokens.device, dtype=torch.long).unsqueeze(0)
        key_padding_mask = key_padding_mask >= path_lengths.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones((token_len, token_len), device=path_tokens.device, dtype=torch.bool),
            diagonal=1,
        )

        path_tokens_fp32 = path_tokens.to(dtype=torch.float32)
        attn_out, _ = self.path_self_attention(
            path_tokens_fp32,
            path_tokens_fp32,
            path_tokens_fp32,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        encoded = self.path_self_attention_norm(path_tokens_fp32 + attn_out)
        last_idx = (path_lengths - 1).clamp(min=0)
        row_idx = torch.arange(total_agents, device=path_tokens.device, dtype=torch.long)
        last_hidden = encoded[row_idx, last_idx]
        last_hidden = torch.where(torch.isfinite(last_hidden), last_hidden, torch.zeros_like(last_hidden))
        return last_hidden.to(dtype=path_tokens.dtype).view(total_agents, hidden_dim)

    def _compute_agent_potentials(
        self,
        *,
        env_context: GraphEnvContext,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        num_agents: int,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = int(env_context.num_graphs)
        total_agents = B * int(num_agents)
        if question_context_tokens is None:
            question_context_tokens = self._build_question_context_tokens(
                env_context=env_context,
                question_tokens=question_tokens,
            )
        if question_padding_mask is None:
            question_padding_mask = self._build_question_padding_mask(
                env_context=env_context,
                question_context_tokens=question_context_tokens,
            )
        agent_graph_ids = torch.arange(B, device=question_tokens.device, dtype=torch.long).repeat_interleave(num_agents)
        if int(agent_graph_ids.numel()) != total_agents:
            raise ValueError("agent_graph_ids shape mismatch in potential computation.")
        agent_question_context = question_context_tokens.index_select(0, agent_graph_ids)
        key_padding_mask = None
        if question_padding_mask is not None:
            key_padding_mask = question_padding_mask.index_select(0, agent_graph_ids).to(dtype=torch.bool)
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("question_padding_mask contains all-masked rows after agent expansion.")

        query = agent_history.unsqueeze(1).to(dtype=torch.float32)
        context_fp32 = agent_question_context.to(dtype=torch.float32)
        cross_out, _ = self.question_cross_attention(
            query,
            context_fp32,
            context_fp32,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        vec_f = self.question_cross_attention_norm(query.squeeze(1) + cross_out.squeeze(1))
        vec_f = torch.where(torch.isfinite(vec_f), vec_f, torch.zeros_like(vec_f))
        return vec_f.to(dtype=agent_history.dtype), agent_graph_ids

    def _compute_edge_logits(
        self,
        *,
        env_context: GraphEnvContext,
        current_nodes: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_agent_batch: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_relations: torch.Tensor,
        agent_potential: torch.Tensor,
        agent_graph_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_nodes = current_nodes.index_select(0, edge_agent_batch)
        edge_graph_ids = agent_graph_ids.index_select(0, edge_agent_batch)
        state_vec = agent_potential.index_select(0, edge_agent_batch).to(dtype=torch.float32)
        edge_rel_features = relation_tokens.index_select(0, edge_relations.clamp(min=0)).to(dtype=torch.float32)
        target_features = node_tokens.index_select(0, target_nodes.clamp(min=0)).to(dtype=torch.float32)
        edge_action = self.edge_action_encoder(torch.cat((edge_rel_features, target_features), dim=-1))
        edge_action = self.edge_action_norm(edge_action)
        dot_logits = (state_vec * edge_action).sum(dim=-1) / math.sqrt(float(state_vec.size(-1)))
        dot_logits = torch.where(torch.isfinite(dot_logits), dot_logits, torch.zeros_like(dot_logits))

        log_pb_prior = self.backward_prior.log_prob_edges(
            env_context=env_context,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_graph_ids=edge_graph_ids,
            dtype=node_tokens.dtype,
        )
        edge_logits = dot_logits + self.edge_logit_pb_weight * log_pb_prior.to(dtype=torch.float32)
        neg_inf = torch.tensor(float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype)
        edge_logits = torch.where(torch.isfinite(edge_logits), edge_logits, neg_inf)
        return edge_logits.to(dtype=node_tokens.dtype), edge_agent_batch

    @staticmethod
    def _segment_logsumexp(
        *,
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_segments < 0:
            raise ValueError(f"num_segments must be >= 0, got {num_segments}.")
        if num_segments == 0:
            empty = torch.empty((0,), device=values.device, dtype=dtype)
            return empty, torch.empty((0,), device=values.device, dtype=torch.bool)
        out = torch.zeros((num_segments,), device=values.device, dtype=torch.float32)
        has_finite = torch.zeros((num_segments,), device=values.device, dtype=torch.bool)
        if int(values.numel()) == 0:
            return out.to(dtype=dtype), has_finite
        if values.dim() != 1 or segment_ids.dim() != 1:
            raise ValueError("segment_logsumexp expects 1D values and segment_ids.")
        if int(values.numel()) != int(segment_ids.numel()):
            raise ValueError(
                "segment_logsumexp size mismatch between values and segment_ids: "
                f"values={int(values.numel())}, segment_ids={int(segment_ids.numel())}."
            )
        if bool((segment_ids < 0).any().item()) or bool((segment_ids >= num_segments).any().item()):
            raise ValueError("segment_ids out of range in segment_logsumexp.")

        finite_mask = torch.isfinite(values)
        if not bool(finite_mask.any().item()):
            return out.to(dtype=dtype), has_finite
        finite_ids = segment_ids[finite_mask].to(device=values.device, dtype=torch.long)
        finite_vals = values[finite_mask].to(dtype=torch.float32)
        has_finite.scatter_(0, finite_ids, True)

        neg_inf = torch.tensor(float("-inf"), device=values.device, dtype=torch.float32)
        max_per_segment = torch.full((num_segments,), fill_value=neg_inf, device=values.device, dtype=torch.float32)
        max_per_segment.scatter_reduce_(0, finite_ids, finite_vals, reduce="amax", include_self=True)
        safe_max = torch.where(has_finite, max_per_segment, torch.zeros_like(max_per_segment))
        shifted = torch.exp(finite_vals - safe_max.index_select(0, finite_ids))
        sum_per_segment = torch.zeros((num_segments,), device=values.device, dtype=torch.float32)
        sum_per_segment.scatter_add_(0, finite_ids, shifted)
        lse = safe_max + torch.log(sum_per_segment.clamp(min=torch.finfo(torch.float32).tiny))
        out = torch.where(has_finite, lse, torch.zeros_like(lse))
        return out.to(dtype=dtype), has_finite

    def _compute_stop_delta(
        self,
        *,
        agent_potential: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw_stop = self.stop_head(agent_potential.to(dtype=torch.float32)).squeeze(-1)
        raw_stop = torch.where(torch.isfinite(raw_stop), raw_stop, torch.zeros_like(raw_stop))
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
    ) -> torch.Tensor:
        B, num_agents = agent_state.current_nodes.shape
        expected_total_agents = B * num_agents
        total_agents = expected_total_agents if total_agents is None else int(total_agents)
        if total_agents != expected_total_agents:
            raise ValueError(
                "total_agents mismatch in stop-logit computation: "
                f"expected={expected_total_agents}, got={total_agents}."
            )
        if int(stop_delta.numel()) != total_agents:
            raise ValueError(
                "stop_delta size mismatch with total_agents: "
                f"stop_delta={int(stop_delta.numel())}, total_agents={total_agents}."
            )
        if edge_logits is not None and edge_agent_batch is not None:
            edge_lse, has_finite_edge = self._segment_logsumexp(
                values=edge_logits.to(dtype=torch.float32),
                segment_ids=edge_agent_batch.to(device=device, dtype=torch.long),
                num_segments=total_agents,
                dtype=dtype,
            )
            stop_logits = torch.where(has_finite_edge, edge_lse + stop_delta, stop_delta)
        else:
            stop_logits = stop_delta

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

    def _gather_actions_from_csr_lock_free(
        self,
        adj_t,
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
        question_context_tokens = resolved_cache.get("question_context_tokens")
        if question_context_tokens is None:
            raise ValueError("action_cache must provide `question_context_tokens`.")
        question_padding_mask = resolved_cache.get("question_padding_mask")

        path_token_ids, path_token_types, path_lengths = self._resolve_path_state(agent_state)
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
        agent_potential, agent_graph_ids = self._compute_agent_potentials(
            env_context=env_context,
            question_tokens=question_tokens,
            agent_history=agent_history,
            num_agents=num_agents,
            question_context_tokens=question_context_tokens,
            question_padding_mask=question_padding_mask,
        )
        stop_delta = self._compute_stop_delta(agent_potential=agent_potential, dtype=node_tokens.dtype)

        active_nodes = torch.where(flat_active_mask, flat_curr_nodes, torch.zeros_like(flat_curr_nodes))
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
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
            )

        all_agent_rows = torch.arange(total_agents, device=target_nodes.device, dtype=torch.long)
        edge_agent_batch_full = all_agent_rows.repeat_interleave(out_degrees)
        edge_active_mask = flat_active_mask.index_select(0, edge_agent_batch_full)
        if not bool(edge_active_mask.all().item()):
            edge_ids = edge_ids[edge_active_mask]
            target_nodes = target_nodes[edge_active_mask]
            edge_agent_batch_full = edge_agent_batch_full[edge_active_mask]
            out_degrees_active = torch.zeros((total_agents,), dtype=torch.long, device=node_tokens.device)
            if int(edge_agent_batch_full.numel()) > 0:
                out_degrees_active.scatter_add_(
                    0,
                    edge_agent_batch_full,
                    torch.ones_like(edge_agent_batch_full, dtype=torch.long, device=node_tokens.device),
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
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
            )

        visited = agent_state.visited_mask
        if visited.dim() == 1:
            is_visited = visited[target_nodes]
        elif visited.dim() == 2:
            is_visited = visited[edge_agent_batch_full, target_nodes]
        else:
            raise ValueError(f"visited_mask must be 1D or 2D, got shape={tuple(visited.shape)}")

        keep_edge = ~is_visited
        if not bool(keep_edge.any().item()):
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
            )
            return {
                "edge_logits": torch.empty(0, device=node_tokens.device, dtype=node_tokens.dtype),
                "edge_agent_batch": torch.empty(0, dtype=torch.long, device=node_tokens.device),
                "stop_logits": stop_logits.view(B, num_agents),
                "edge_ids": torch.empty(0, dtype=torch.long, device=node_tokens.device),
                "target_nodes": torch.empty(0, dtype=torch.long, device=node_tokens.device),
                "out_degrees": torch.zeros((B, num_agents), dtype=torch.long, device=node_tokens.device),
            }

        edge_ids = edge_ids[keep_edge]
        target_nodes = target_nodes[keep_edge]
        edge_agent_batch = edge_agent_batch_full[keep_edge]
        out_degrees_filtered = torch.zeros((total_agents,), dtype=torch.long, device=node_tokens.device)
        out_degrees_filtered.scatter_add_(
            0,
            edge_agent_batch,
            torch.ones_like(edge_agent_batch, dtype=torch.long, device=node_tokens.device),
        )

        edge_logits, edge_agent_batch = self._compute_edge_logits(
            env_context=env_context,
            current_nodes=flat_curr_nodes.clamp(min=0),
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            edge_relations=env_context.edge_relations.index_select(0, edge_ids.clamp(min=0)),
            agent_potential=agent_potential,
            agent_graph_ids=agent_graph_ids,
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
        )

        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits.view(B, num_agents),
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees_filtered.view(B, num_agents),
        }

    @staticmethod
    def _build_empty_output(
        *,
        B: int,
        num_agents: int,
        device: torch.device,
        dtype: torch.dtype,
        stop_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """防崩溃的安全网兜底输出"""
        return {
            "edge_logits": torch.empty(0, device=device, dtype=dtype),
            "edge_agent_batch": torch.empty(0, dtype=torch.long, device=device),
            "stop_logits": stop_logits,
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
        [系统实体] 严格不可变的状态转移
        轨迹历史采用 token 序列维护，不依赖 GRU 递归隐状态。
        """
        del relation_tokens
        B, num_agents = agent_state.current_nodes.shape
        total_agents = B * num_agents

        safe_target_nodes = torch.where(is_stop, torch.zeros_like(chosen_target_nodes), chosen_target_nodes)
        safe_edge_relations = torch.where(is_stop, torch.zeros_like(chosen_edge_relations), chosen_edge_relations)

        flat_hidden = agent_state.hidden_states.view(total_agents, -1)
        if int(flat_hidden.size(-1)) == int(node_tokens.size(-1)):
            moved_hidden = node_tokens.index_select(0, safe_target_nodes.clamp(min=0)).to(dtype=flat_hidden.dtype)
            next_hidden = torch.where(is_stop.unsqueeze(-1), flat_hidden, moved_hidden)
        else:
            next_hidden = flat_hidden

        new_visited_mask = agent_state.visited_mask.clone()
        if new_visited_mask.dim() == 1:
            active_move = ~is_stop
            if bool(active_move.any().item()):
                new_visited_mask.scatter_(0, safe_target_nodes[active_move], True)
        elif new_visited_mask.dim() == 2:
            active_move = ~is_stop
            if bool(active_move.any().item()):
                row_ids = torch.arange(is_stop.numel(), device=is_stop.device, dtype=torch.long)[active_move]
                col_ids = safe_target_nodes[active_move]
                new_visited_mask[row_ids, col_ids] = True
        else:
            raise ValueError(f"visited_mask must be 1D or 2D, got shape={tuple(new_visited_mask.shape)}")

        current_nodes = agent_state.current_nodes.view(total_agents)
        next_current_flat = torch.where(is_stop, current_nodes, safe_target_nodes)
        next_current_nodes = next_current_flat.view(B, num_agents)

        path_token_ids, path_token_types, path_lengths = self._resolve_path_state(agent_state)
        move_mask = ~is_stop
        next_lengths = path_lengths + move_mask.to(dtype=torch.long) * 2
        old_width = int(path_token_ids.size(1))
        next_width = max(old_width, int(next_lengths.max().item()))
        if next_width > old_width:
            next_path_ids = torch.zeros((total_agents, next_width), dtype=path_token_ids.dtype, device=path_token_ids.device)
            next_path_types = torch.zeros((total_agents, next_width), dtype=torch.bool, device=path_token_types.device)
            next_path_ids[:, :old_width] = path_token_ids
            next_path_types[:, :old_width] = path_token_types
        else:
            next_path_ids = path_token_ids.clone()
            next_path_types = path_token_types.clone()
        if bool(move_mask.any().item()):
            move_rows = torch.where(move_mask)[0]
            rel_pos = path_lengths.index_select(0, move_rows)
            node_pos = rel_pos + 1
            move_rel = safe_edge_relations.index_select(0, move_rows).to(dtype=next_path_ids.dtype)
            move_nodes = safe_target_nodes.index_select(0, move_rows).to(dtype=next_path_ids.dtype)
            next_path_ids[move_rows, rel_pos] = move_rel
            next_path_types[move_rows, rel_pos] = True
            next_path_ids[move_rows, node_pos] = move_nodes
            next_path_types[move_rows, node_pos] = False

        return DynamicAgentState(
            step_t=agent_state.step_t + 1,
            current_nodes=next_current_nodes,
            hidden_states=next_hidden.view(B, num_agents, -1),
            visited_mask=new_visited_mask,
            cumulative_rewards=agent_state.cumulative_rewards,
            done_mask=agent_state.done_mask | is_stop.view(B, num_agents),
            path_token_ids=next_path_ids.view(B, num_agents, next_width),
            path_token_types=next_path_types.view(B, num_agents, next_width),
            path_lengths=next_lengths.view(B, num_agents),
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
