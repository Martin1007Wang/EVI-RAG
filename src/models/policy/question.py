from __future__ import annotations

from typing import cast

import torch
from torch import nn

from src.models.environment import GraphEnvContext


def build_question_context_tokens(
    *,
    env_context: GraphEnvContext,
    question_tokens: torch.Tensor,
    policy_dim: int,
    graph_hidden_dim: int,
    embedding_dim: int,
    path_to_policy: nn.Module,
) -> torch.Tensor:
    question_ctx = env_context.question_ctx
    if question_ctx is None:
        raise ValueError(
            "question_ctx is required for token-level question interaction. "
            "Preprocessing must provide per-token question embeddings."
        )
    if question_ctx.dim() != 3:
        raise ValueError(
            f"question_ctx must be 3D [B, L, d], got shape={tuple(question_ctx.shape)}."
        )
    if int(question_ctx.size(0)) != int(env_context.num_graphs):
        raise ValueError(
            "question_ctx batch mismatch with num_graphs: "
            f"question_ctx={int(question_ctx.size(0))}, num_graphs={int(env_context.num_graphs)}."
        )
    if int(question_ctx.size(1)) <= 0:
        raise ValueError("question_ctx length L must be > 0 when provided.")
    ctx_tokens = question_ctx.to(
        device=question_tokens.device, dtype=question_tokens.dtype
    )
    if int(ctx_tokens.size(-1)) == policy_dim:
        return ctx_tokens
    if int(ctx_tokens.size(-1)) == graph_hidden_dim:
        return path_to_policy(ctx_tokens)
    if int(ctx_tokens.size(-1)) != embedding_dim:
        raise ValueError(
            "question_ctx last dim mismatch with backbone dims: "
            f"question_ctx={int(ctx_tokens.size(-1))}, "
            f"embedding_dim={embedding_dim}, "
            f"hidden_dim={graph_hidden_dim}, policy_dim={policy_dim}."
        )
    return ctx_tokens


def build_question_padding_mask(
    *,
    env_context: GraphEnvContext,
    question_context_tokens: torch.Tensor,
) -> torch.Tensor:
    raw_mask = env_context.question_ctx_mask
    if raw_mask is None:
        raise ValueError(
            "question_ctx_mask is required for token-level question interaction. "
            "Preprocessing must provide a valid-token mask."
        )
    if raw_mask.dim() != 2:
        raise ValueError(
            f"question_ctx_mask must be 2D [B, L], got shape={tuple(raw_mask.shape)}."
        )
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


def build_question_lexical_tokens(
    *,
    question_context_tokens: torch.Tensor,
    question_lexical_proj: nn.Module,
) -> torch.Tensor:
    if question_context_tokens.dim() != 3:
        raise ValueError(
            "question_context_tokens must be 3D [B, L, d] for lexical projection, "
            f"got shape={tuple(question_context_tokens.shape)}."
        )
    context_fp32 = question_context_tokens.to(dtype=torch.float32)
    return question_lexical_proj(context_fp32)


def compute_question_token_pool(
    *,
    question_token_scorer: nn.Module,
    question_global_proj: nn.Module,
    agent_question_context: torch.Tensor,
    agent_question_padding_mask: torch.Tensor,
) -> torch.Tensor:
    context_fp32 = agent_question_context.to(dtype=torch.float32)
    token_logits = question_token_scorer(context_fp32).squeeze(-1)
    if token_logits.dim() != 2:
        raise ValueError(
            f"question_token_scorer output must be 2D [A, L], got shape={tuple(token_logits.shape)}."
        )
    neg_inf = torch.tensor(
        float("-inf"), device=token_logits.device, dtype=token_logits.dtype
    )
    token_logits = token_logits.masked_fill(agent_question_padding_mask, neg_inf)
    if bool((~torch.isfinite(token_logits)).all(dim=1).any().item()):
        raise ValueError(
            "question token saliency has rows with no finite logits after masking."
        )
    token_weights = torch.softmax(token_logits, dim=-1)
    token_weights = torch.where(
        torch.isfinite(token_weights), token_weights, torch.zeros_like(token_weights)
    )
    pooled = torch.einsum("al,ald->ad", token_weights, context_fp32)
    pooled = question_global_proj(pooled)
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


def compute_agent_potentials(
    *,
    env_context: GraphEnvContext,
    question_tokens: torch.Tensor,
    agent_history: torch.Tensor,
    num_agents: int,
    question_context_tokens: torch.Tensor | None,
    question_padding_mask: torch.Tensor | None,
    policy_dim: int,
    graph_hidden_dim: int,
    embedding_dim: int,
    path_to_policy: nn.Module,
    question_lexical_proj: nn.Module,
    question_cross_attention: nn.MultiheadAttention,
    question_cross_attention_norm: nn.LayerNorm,
    question_token_scorer: nn.Module,
    question_global_proj: nn.Module,
    lexical_question_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(env_context.num_graphs)
    total_agents = B * int(num_agents)
    if question_context_tokens is None:
        question_context_tokens = build_question_context_tokens(
            env_context=env_context,
            question_tokens=question_tokens,
            policy_dim=policy_dim,
            graph_hidden_dim=graph_hidden_dim,
            embedding_dim=embedding_dim,
            path_to_policy=path_to_policy,
        )
    if question_context_tokens is None:
        raise ValueError(
            "question_context_tokens must be resolved for potential computation."
        )
    assert question_context_tokens is not None
    if question_padding_mask is None:
        question_padding_mask = build_question_padding_mask(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )
    if question_padding_mask is None:
        raise ValueError(
            "question_padding_mask must be resolved for potential computation."
        )
    assert question_padding_mask is not None
    agent_graph_ids = torch.arange(
        B, device=question_tokens.device, dtype=torch.long
    ).repeat_interleave(num_agents)
    if int(agent_graph_ids.numel()) != total_agents:
        raise ValueError("agent_graph_ids shape mismatch in potential computation.")
    agent_question_context = question_context_tokens.index_select(0, agent_graph_ids)
    agent_question_padding_mask = question_padding_mask.index_select(
        0, agent_graph_ids
    ).to(dtype=torch.bool)
    if bool(agent_question_padding_mask.all(dim=1).any().item()):
        raise ValueError(
            "question_padding_mask contains all-masked rows after agent expansion."
        )

    query = path_to_policy(agent_history).unsqueeze(1).to(dtype=torch.float32)
    context_fp32 = agent_question_context.to(dtype=torch.float32)
    if lexical_question_tokens is None:
        lexical_question_tokens = question_lexical_proj(context_fp32)
    else:
        if lexical_question_tokens.dim() != 3:
            raise ValueError(
                "lexical_question_tokens must be 3D [B, L, r] when provided, "
                f"got shape={tuple(lexical_question_tokens.shape)}."
            )
        if int(lexical_question_tokens.size(0)) != B:
            raise ValueError(
                "lexical_question_tokens batch mismatch with num_graphs: "
                f"lexical={int(lexical_question_tokens.size(0))}, num_graphs={B}."
            )
        if int(lexical_question_tokens.size(1)) != int(agent_question_context.size(1)):
            raise ValueError(
                "lexical_question_tokens length mismatch with question context: "
                f"lexical={int(lexical_question_tokens.size(1))}, "
                f"context={int(agent_question_context.size(1))}."
            )
        lexical_tokens = cast(torch.Tensor, lexical_question_tokens)
        lexical_tokens = lexical_tokens.to(
            device=context_fp32.device, dtype=torch.float32
        )
        lexical_question_tokens = lexical_tokens.index_select(0, agent_graph_ids)
    cross_out, _ = question_cross_attention(
        query,
        context_fp32,
        context_fp32,
        key_padding_mask=agent_question_padding_mask,
        need_weights=False,
    )
    pooled_question = compute_question_token_pool(
        question_token_scorer=question_token_scorer,
        question_global_proj=question_global_proj,
        agent_question_context=agent_question_context,
        agent_question_padding_mask=agent_question_padding_mask,
    )
    vec_f = question_cross_attention_norm(
        query.squeeze(1) + cross_out.squeeze(1) + pooled_question
    )
    vec_f = torch.where(torch.isfinite(vec_f), vec_f, torch.zeros_like(vec_f))
    return (
        vec_f.to(dtype=agent_history.dtype),
        agent_graph_ids,
        lexical_question_tokens,
        agent_question_padding_mask,
    )


__all__ = [
    "build_question_context_tokens",
    "build_question_padding_mask",
    "build_question_lexical_tokens",
    "compute_agent_potentials",
    "compute_question_token_pool",
]
