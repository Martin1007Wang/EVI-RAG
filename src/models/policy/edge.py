from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.environment import GraphEnvContext
from src.utils.segment_ops import segment_logsumexp_1d, segment_mean_1d


def compute_relation_lexical_bias(
    *,
    lexical_question_tokens: torch.Tensor,
    agent_question_padding_mask: torch.Tensor,
    relation_tokens: torch.Tensor,
    relation_to_policy: nn.Module,
    relation_lexical_proj: nn.Module,
    edge_agent_batch: torch.Tensor,
    edge_relations: torch.Tensor,
) -> torch.Tensor:
    if lexical_question_tokens.dim() != 3:
        raise ValueError(
            "lexical_question_tokens must be 3D [A, L, r] for lexical bias, "
            f"got shape={tuple(lexical_question_tokens.shape)}."
        )
    if agent_question_padding_mask.dim() != 2:
        raise ValueError(
            "agent_question_padding_mask must be 2D [A, L] for lexical bias, "
            f"got shape={tuple(agent_question_padding_mask.shape)}."
        )
    if int(agent_question_padding_mask.size(0)) != int(lexical_question_tokens.size(0)):
        raise ValueError(
            "agent_question_padding_mask/lexical_question_tokens batch mismatch in lexical bias."
        )
    num_relations = int(relation_tokens.size(0))
    if num_relations <= 0:
        raise ValueError("relation_tokens must contain at least one relation.")

    relation_features = relation_to_policy(relation_tokens.to(dtype=torch.float32))
    rel_query = relation_lexical_proj(relation_features)
    rel_query = F.normalize(rel_query, dim=-1, eps=1.0e-6)
    token_repr = F.normalize(
        lexical_question_tokens.to(dtype=torch.float32), dim=-1, eps=1.0e-6
    )
    token_similarity = torch.matmul(token_repr, rel_query.transpose(0, 1))
    token_similarity = token_similarity.masked_fill(
        agent_question_padding_mask.unsqueeze(-1),
        torch.tensor(
            float("-inf"),
            device=token_similarity.device,
            dtype=token_similarity.dtype,
        ),
    )
    lexical_bias_per_rel = token_similarity.amax(dim=1)
    lexical_bias_per_rel = torch.where(
        torch.isfinite(lexical_bias_per_rel),
        lexical_bias_per_rel,
        torch.zeros_like(lexical_bias_per_rel),
    )
    safe_rel = edge_relations.clamp(min=0, max=num_relations - 1).to(dtype=torch.long)
    edge_agent_ids = edge_agent_batch.to(dtype=torch.long)
    return lexical_bias_per_rel[edge_agent_ids, safe_rel]


def compute_node_lexical_bias(
    *,
    lexical_question_tokens: torch.Tensor,
    agent_question_padding_mask: torch.Tensor,
    node_tokens: torch.Tensor,
    node_to_policy: nn.Module,
    node_lexical_proj: nn.Module,
    edge_agent_batch: torch.Tensor,
    target_nodes: torch.Tensor,
) -> torch.Tensor:
    if lexical_question_tokens.dim() != 3:
        raise ValueError(
            "lexical_question_tokens must be 3D [A, L, r] for lexical bias, "
            f"got shape={tuple(lexical_question_tokens.shape)}."
        )
    if agent_question_padding_mask.dim() != 2:
        raise ValueError(
            "agent_question_padding_mask must be 2D [A, L] for lexical bias, "
            f"got shape={tuple(agent_question_padding_mask.shape)}."
        )
    if int(agent_question_padding_mask.size(0)) != int(lexical_question_tokens.size(0)):
        raise ValueError(
            "agent_question_padding_mask/lexical_question_tokens batch mismatch in lexical bias."
        )
    if node_tokens.dim() != 2:
        raise ValueError(
            f"node_tokens must be 2D [N, d] for lexical bias, got shape={tuple(node_tokens.shape)}."
        )
    num_nodes = int(node_tokens.size(0))
    if num_nodes <= 0:
        raise ValueError("node_tokens must contain at least one node.")

    safe_target = target_nodes.clamp(min=0, max=num_nodes - 1).to(dtype=torch.long)
    target_features = node_tokens.index_select(0, safe_target).to(dtype=torch.float32)
    target_features = node_to_policy(target_features)
    edge_query = node_lexical_proj(target_features)
    edge_query = F.normalize(edge_query, dim=-1, eps=1.0e-6)
    edge_agent_ids = edge_agent_batch.to(dtype=torch.long)

    token_repr = F.normalize(
        lexical_question_tokens.to(dtype=torch.float32), dim=-1, eps=1.0e-6
    )
    edge_tokens = token_repr.index_select(0, edge_agent_ids)
    token_mask = agent_question_padding_mask.index_select(0, edge_agent_ids)
    token_similarity = (edge_tokens * edge_query.unsqueeze(1)).sum(dim=-1)
    token_similarity = token_similarity.masked_fill(
        token_mask,
        torch.tensor(
            float("-inf"),
            device=token_similarity.device,
            dtype=token_similarity.dtype,
        ),
    )
    lexical_bias = token_similarity.amax(dim=1)
    return torch.where(
        torch.isfinite(lexical_bias), lexical_bias, torch.zeros_like(lexical_bias)
    )


def compute_edge_logits(
    *,
    env_context: GraphEnvContext,
    node_tokens: torch.Tensor,
    relation_tokens: torch.Tensor,
    node_log_f: torch.Tensor | None,
    edge_next_log_f: torch.Tensor | None,
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
    edge_action_encoder: nn.Module,
    edge_action_norm: nn.Module,
    relation_group_head: nn.Module,
    relation_lexical_proj: nn.Module,
    lexical_bias_log_scale: torch.Tensor,
    node_lexical_proj: nn.Module,
    node_lexical_bias_log_scale: torch.Tensor,
    doob_h_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
    if total_agents <= 0:
        raise ValueError(
            f"total_agents must be > 0 in edge-logit computation, got {total_agents}."
        )
    if edge_agent_batch.dim() != 1:
        raise ValueError(
            f"edge_agent_batch must be 1D, got shape={tuple(edge_agent_batch.shape)}."
        )
    if target_nodes.dim() != 1:
        raise ValueError(
            f"target_nodes must be 1D, got shape={tuple(target_nodes.shape)}."
        )
    if edge_relations.dim() != 1:
        raise ValueError(
            f"edge_relations must be 1D, got shape={tuple(edge_relations.shape)}."
        )
    if current_nodes.dim() != 1:
        raise ValueError(
            f"current_nodes must be 1D [total_agents], got shape={tuple(current_nodes.shape)}."
        )
    if int(current_nodes.numel()) != int(total_agents):
        raise ValueError(
            "current_nodes length mismatch with total_agents in edge-logit computation: "
            f"current_nodes={int(current_nodes.numel())}, total_agents={int(total_agents)}."
        )
    if int(edge_agent_batch.numel()) != int(target_nodes.numel()):
        raise ValueError(
            "edge_agent_batch/target_nodes size mismatch in edge-logit computation: "
            f"edge_agent_batch={int(edge_agent_batch.numel())}, target_nodes={int(target_nodes.numel())}."
        )
    if int(edge_agent_batch.numel()) != int(edge_relations.numel()):
        raise ValueError(
            "edge_agent_batch/edge_relations size mismatch in edge-logit computation: "
            f"edge_agent_batch={int(edge_agent_batch.numel())}, edge_relations={int(edge_relations.numel())}."
        )
    if bool((edge_agent_batch < 0).any().item()) or bool(
        (edge_agent_batch >= total_agents).any().item()
    ):
        raise ValueError(
            "edge_agent_batch contains out-of-range agent ids in edge-logit computation."
        )

    num_relations = int(relation_tokens.size(0))
    if num_relations <= 0:
        raise ValueError(
            "relation_tokens must contain at least one relation for hierarchical edge scoring."
        )
    safe_edge_relations = edge_relations.clamp(min=0, max=num_relations - 1).to(
        dtype=torch.long
    )
    agent_ids = edge_agent_batch.to(dtype=torch.long)
    state_vec = agent_potential.index_select(0, edge_agent_batch).to(
        dtype=torch.float32
    )
    edge_rel_features = relation_tokens.index_select(0, safe_edge_relations).to(
        dtype=torch.float32
    )
    edge_rel_features = relation_to_policy(edge_rel_features)
    target_features = node_tokens.index_select(0, target_nodes.clamp(min=0)).to(
        dtype=torch.float32
    )
    target_features = node_to_policy(target_features)
    edge_action = edge_action_encoder(
        torch.cat((edge_rel_features, target_features), dim=-1)
    )
    edge_action = edge_action_norm(edge_action)
    conditional_logits = (state_vec * edge_action).sum(dim=-1) / math.sqrt(
        float(state_vec.size(-1))
    )
    conditional_logits = torch.where(
        torch.isfinite(conditional_logits),
        conditional_logits,
        torch.zeros_like(conditional_logits),
    )
    relation_lexical_bias = compute_relation_lexical_bias(
        lexical_question_tokens=lexical_question_tokens,
        agent_question_padding_mask=agent_question_padding_mask,
        relation_tokens=relation_tokens,
        relation_to_policy=relation_to_policy,
        relation_lexical_proj=relation_lexical_proj,
        edge_agent_batch=edge_agent_batch,
        edge_relations=edge_relations,
    )
    relation_lexical_scale = F.softplus(lexical_bias_log_scale.to(dtype=torch.float32))
    node_lexical_bias = compute_node_lexical_bias(
        lexical_question_tokens=lexical_question_tokens,
        agent_question_padding_mask=agent_question_padding_mask,
        node_tokens=node_tokens,
        node_to_policy=node_to_policy,
        node_lexical_proj=node_lexical_proj,
        edge_agent_batch=edge_agent_batch,
        target_nodes=target_nodes,
    )
    node_lexical_scale = F.softplus(node_lexical_bias_log_scale.to(dtype=torch.float32))
    conditional_logits = (
        conditional_logits
        + relation_lexical_scale * relation_lexical_bias
        + node_lexical_scale * node_lexical_bias
    )

    # Hierarchical policy decomposition per state:
    # log P(e|s, move) = log P(r(e)|s) + log P(e|s, r(e)).
    relation_group_input = torch.cat((state_vec, edge_rel_features), dim=-1)
    relation_logits_edge = (
        relation_group_head(relation_group_input).squeeze(-1).to(dtype=torch.float32)
    )
    relation_logits_edge = torch.where(
        torch.isfinite(relation_logits_edge),
        relation_logits_edge,
        torch.zeros_like(relation_logits_edge),
    )

    group_keys = agent_ids * num_relations + safe_edge_relations
    unique_group_keys, inverse_group = torch.unique(
        group_keys, sorted=False, return_inverse=True
    )
    num_groups = int(unique_group_keys.numel())
    group_agent_ids = torch.div(unique_group_keys, num_relations, rounding_mode="floor")

    relation_logits_group = segment_mean_1d(
        values=relation_logits_edge,
        segment_ids=inverse_group,
        num_segments=num_groups,
        dtype=torch.float32,
    )
    relation_lse_per_agent, relation_has_finite = segment_logsumexp_1d(
        values=relation_logits_group,
        segment_ids=group_agent_ids,
        num_segments=total_agents,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=0.0,
    )
    if not bool(relation_has_finite.index_select(0, group_agent_ids).all().item()):
        raise ValueError(
            "Hierarchical relation scoring produced non-finite relation-agent groups."
        )
    relation_log_prob_group = (
        relation_logits_group - relation_lse_per_agent.index_select(0, group_agent_ids)
    )
    relation_log_prob_edge = relation_log_prob_group.index_select(0, inverse_group)

    conditional_lse_group, conditional_has_finite = segment_logsumexp_1d(
        values=conditional_logits,
        segment_ids=inverse_group,
        num_segments=num_groups,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=0.0,
    )
    if not bool(conditional_has_finite.all().item()):
        raise ValueError(
            "Hierarchical conditional scoring produced non-finite relation groups."
        )
    conditional_log_prob = conditional_logits - conditional_lse_group.index_select(
        0, inverse_group
    )

    edge_logits = relation_log_prob_edge + conditional_log_prob
    if edge_next_log_f is not None:
        if edge_next_log_f.dim() != 1 or int(edge_next_log_f.numel()) != int(
            target_nodes.numel()
        ):
            raise ValueError(
                "edge_next_log_f must be 1D with one value per edge candidate: "
                f"edge_next_log_f={tuple(edge_next_log_f.shape)}, edges={int(target_nodes.numel())}."
            )
        edge_target_log_f = edge_next_log_f.to(dtype=torch.float32)
    elif node_log_f is not None:
        if node_log_f.dim() != 1 or int(node_log_f.numel()) != int(
            env_context.num_nodes_total
        ):
            raise ValueError(
                "node_log_f must be 1D with num_nodes_total entries in node-flow move shift: "
                f"node_log_f={tuple(node_log_f.shape)}, num_nodes_total={int(env_context.num_nodes_total)}."
            )
        safe_target_nodes = target_nodes.clamp(
            min=0, max=max(int(env_context.num_nodes_total) - 1, 0)
        ).to(dtype=torch.long)
        edge_target_log_f = node_log_f.index_select(0, safe_target_nodes).to(
            dtype=torch.float32
        )
    else:
        edge_target_log_f = torch.zeros_like(edge_logits, dtype=torch.float32)
    edge_logits = edge_logits + doob_h_alpha * edge_target_log_f
    node_global_ids = env_context.node_global_ids.to(
        device=edge_logits.device, dtype=torch.long
    )
    if int(node_global_ids.numel()) != int(env_context.num_nodes_total):
        raise ValueError(
            "node_global_ids length mismatch with num_nodes_total in hierarchical edge scoring: "
            f"node_global_ids={int(node_global_ids.numel())}, num_nodes_total={int(env_context.num_nodes_total)}."
        )
    source_nodes = current_nodes.index_select(0, agent_ids).to(dtype=torch.long)
    safe_source_nodes = source_nodes.clamp(
        min=0, max=max(int(env_context.num_nodes_total) - 1, 0)
    )
    source_global_ids = node_global_ids.index_select(0, safe_source_nodes)
    super_source_edge = (source_nodes >= 0) & (source_global_ids < 0)
    # At virtual super-node steps relation selection is degenerate; keep entity-conditional logits only.
    super_edge_logits = conditional_log_prob + doob_h_alpha * edge_target_log_f
    edge_logits = torch.where(super_source_edge, super_edge_logits, edge_logits)
    neg_inf = torch.tensor(
        float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype
    )
    edge_logits = torch.where(torch.isfinite(edge_logits), edge_logits, neg_inf)
    edge_meta: dict[str, torch.Tensor | int] = {
        "relation_group_keys": unique_group_keys,
        "relation_log_prob_group": relation_log_prob_group,
        "edge_group_index": inverse_group,
        "num_relations": num_relations,
    }
    return edge_logits.to(dtype=node_tokens.dtype), edge_agent_batch, edge_meta


__all__ = [
    "compute_edge_logits",
    "compute_relation_lexical_bias",
    "compute_node_lexical_bias",
]
