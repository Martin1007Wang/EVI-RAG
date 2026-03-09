from __future__ import annotations

import math

import torch
from torch import nn

from .edge import compute_edge_logits as compute_edge_logits_helper
from .question import (
    build_question_context_tokens as build_question_context_tokens_helper,
    build_question_lexical_tokens as build_question_lexical_tokens_helper,
    build_question_padding_mask as build_question_padding_mask_helper,
    compute_agent_potentials as compute_agent_potentials_helper,
    compute_question_token_pool as compute_question_token_pool_helper,
)


class PolicyProjectionModule(nn.Module):
    def __init__(self, *, graph_hidden_dim: int, policy_dim: int) -> None:
        super().__init__()
        if graph_hidden_dim == policy_dim:
            self.node_to_policy = nn.Identity()
            self.relation_to_policy = nn.Identity()
        else:
            self.node_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            self.relation_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            nn.init.xavier_uniform_(self.node_to_policy.weight)
            nn.init.xavier_uniform_(self.relation_to_policy.weight)
            if self.node_to_policy.bias is not None:
                nn.init.zeros_(self.node_to_policy.bias)
            if self.relation_to_policy.bias is not None:
                nn.init.zeros_(self.relation_to_policy.bias)


class NodeFlowHead(nn.Module):
    """节点流量头: 输出 query-conditioned logF(node)。"""

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


class QuestionContextModule(nn.Module):
    def __init__(
        self,
        *,
        policy_dim: int,
        graph_hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        lexical_rank: int,
    ) -> None:
        super().__init__()
        if graph_hidden_dim == policy_dim:
            self.path_to_policy = nn.Identity()
        else:
            self.path_to_policy = nn.Linear(graph_hidden_dim, policy_dim)
            nn.init.xavier_uniform_(self.path_to_policy.weight)
            if self.path_to_policy.bias is not None:
                nn.init.zeros_(self.path_to_policy.bias)

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
        self.question_lexical_proj = nn.Linear(policy_dim, lexical_rank, bias=False)

        self.policy_dim = int(policy_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.embedding_dim = int(embedding_dim)

    def build_question_context_tokens(
        self,
        *,
        env_context,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return build_question_context_tokens_helper(
            env_context=env_context,
            question_tokens=question_tokens,
            policy_dim=self.policy_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            embedding_dim=self.embedding_dim,
            path_to_policy=self.path_to_policy,
        )

    @staticmethod
    def build_question_padding_mask(
        *,
        env_context,
        question_context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return build_question_padding_mask_helper(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )

    def compute_question_token_pool(
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

    def build_question_lexical_tokens(
        self, *, question_context_tokens: torch.Tensor
    ) -> torch.Tensor:
        return build_question_lexical_tokens_helper(
            question_context_tokens=question_context_tokens,
            question_lexical_proj=self.question_lexical_proj,
        )

    def compute_agent_potentials(
        self,
        *,
        env_context,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        num_agents: int,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
        lexical_question_tokens: torch.Tensor | None = None,
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
            embedding_dim=self.embedding_dim,
            path_to_policy=self.path_to_policy,
            question_lexical_proj=self.question_lexical_proj,
            question_cross_attention=self.question_cross_attention,
            question_cross_attention_norm=self.question_cross_attention_norm,
            question_token_scorer=self.question_token_scorer,
            question_global_proj=self.question_global_proj,
            lexical_question_tokens=lexical_question_tokens,
        )


class EdgeScoreModule(nn.Module):
    def __init__(
        self,
        *,
        policy_dim: int,
        hidden_dim: int,
        dropout: float,
        lexical_rank: int,
        doob_h_alpha: float,
    ) -> None:
        super().__init__()
        self.edge_action_encoder = nn.Sequential(
            nn.Linear(policy_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, policy_dim),
        )
        self.edge_action_norm = nn.LayerNorm(policy_dim)
        self.relation_group_head = nn.Sequential(
            nn.Linear(policy_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.relation_lexical_proj = nn.Linear(policy_dim, lexical_rank, bias=False)
        self.node_lexical_proj = nn.Linear(policy_dim, lexical_rank, bias=False)
        self.lexical_bias_log_scale = nn.Parameter(torch.tensor(1.0))
        self.node_lexical_bias_log_scale = nn.Parameter(torch.tensor(1.0))
        self.doob_h_alpha = float(doob_h_alpha)

    def compute_edge_logits(
        self,
        *,
        env_context,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_log_f: torch.Tensor | None = None,
        edge_next_log_f: torch.Tensor | None = None,
        edge_agent_batch: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_relations: torch.Tensor,
        current_nodes: torch.Tensor,
        total_agents: int,
        agent_potential: torch.Tensor,
        lexical_question_tokens: torch.Tensor,
        agent_question_padding_mask: torch.Tensor,
        relation_to_policy: nn.Module,
        node_to_policy: nn.Module,
        doob_h_alpha_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
        doob_h_alpha = (
            self.doob_h_alpha
            if doob_h_alpha_override is None
            else float(doob_h_alpha_override)
        )
        return compute_edge_logits_helper(
            env_context=env_context,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            node_log_f=node_log_f,
            edge_next_log_f=edge_next_log_f,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            edge_relations=edge_relations,
            current_nodes=current_nodes,
            total_agents=total_agents,
            agent_potential=agent_potential,
            lexical_question_tokens=lexical_question_tokens,
            agent_question_padding_mask=agent_question_padding_mask,
            relation_to_policy=relation_to_policy,
            node_to_policy=node_to_policy,
            edge_action_encoder=self.edge_action_encoder,
            edge_action_norm=self.edge_action_norm,
            relation_group_head=self.relation_group_head,
            relation_lexical_proj=self.relation_lexical_proj,
            lexical_bias_log_scale=self.lexical_bias_log_scale,
            node_lexical_proj=self.node_lexical_proj,
            node_lexical_bias_log_scale=self.node_lexical_bias_log_scale,
            doob_h_alpha=doob_h_alpha,
        )


class StopDeltaHead(nn.Module):
    def __init__(
        self,
        *,
        policy_dim: int,
        stop_bias_init: float,
        stop_delta_scale: float,
        stop_delta_temperature: float,
    ) -> None:
        super().__init__()
        self.stop_head = nn.Linear(policy_dim, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, float(stop_bias_init))
        self.stop_delta_scale = float(stop_delta_scale)
        self.stop_delta_temperature = float(stop_delta_temperature)

    def forward(
        self, *, agent_potential: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        raw_stop = self.stop_head(agent_potential.to(dtype=torch.float32)).squeeze(-1)
        raw_stop = torch.where(
            torch.isfinite(raw_stop), raw_stop, torch.zeros_like(raw_stop)
        )
        normalized = raw_stop / self.stop_delta_temperature
        bounded_delta = self.stop_delta_scale * torch.tanh(normalized)
        return bounded_delta.to(dtype=dtype)


class StartLogitHead(nn.Module):
    """Score candidate start nodes inside q_local_indices sets."""

    def __init__(self, *, policy_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(policy_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, *, node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        feats = torch.cat((node_features, question_features), dim=-1)
        logits = self.mlp(feats).squeeze(-1)
        return torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))


__all__ = [
    "PolicyProjectionModule",
    "NodeFlowHead",
    "QuestionContextModule",
    "EdgeScoreModule",
    "StopDeltaHead",
    "StartLogitHead",
]
