from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


def build_path_token_embeddings(
    *,
    path_token_ids: torch.Tensor,
    path_token_types: torch.Tensor,
    node_tokens: torch.Tensor,
    relation_tokens: torch.Tensor,
    pos_encoder: nn.Module | None,
) -> torch.Tensor:
    if node_tokens.dim() != 2:
        raise ValueError(
            f"node_tokens must be 2D [N, d], got shape={tuple(node_tokens.shape)}."
        )
    if relation_tokens.dim() != 2:
        raise ValueError(
            "relation_tokens must be 2D [R, d], "
            f"got shape={tuple(relation_tokens.shape)}."
        )
    if int(node_tokens.size(0)) <= 0:
        raise ValueError("node_tokens must contain at least one node.")

    total_agents, token_len = path_token_ids.shape
    hidden_dim = int(node_tokens.size(-1))
    safe_node_ids = path_token_ids.clamp(min=0, max=int(node_tokens.size(0)) - 1)
    node_part = node_tokens.index_select(0, safe_node_ids.reshape(-1)).view(
        total_agents, token_len, hidden_dim
    )
    if int(relation_tokens.size(0)) == 0:
        if bool(path_token_types.any().item()):
            raise ValueError(
                "path_token_types contains relation tokens but relation_tokens is empty."
            )
        relation_part = torch.zeros_like(node_part)
    else:
        safe_rel_ids = path_token_ids.clamp(min=0, max=int(relation_tokens.size(0)) - 1)
        relation_part = relation_tokens.index_select(0, safe_rel_ids.reshape(-1)).view(
            total_agents,
            token_len,
            hidden_dim,
        )
    path_tokens = torch.where(path_token_types.unsqueeze(-1), relation_part, node_part)
    if pos_encoder is not None:
        token_positions = torch.arange(
            token_len, device=path_tokens.device, dtype=torch.long
        )
        pos = pos_encoder(token_positions).to(
            device=path_tokens.device, dtype=path_tokens.dtype
        )
        path_tokens = path_tokens + pos.unsqueeze(0)
    return path_tokens


def encode_path_history(
    *,
    path_tokens: torch.Tensor,
    path_lengths: torch.Tensor,
    path_self_attention: nn.MultiheadAttention,
    path_self_attention_norm: nn.LayerNorm,
) -> torch.Tensor:
    total_agents, token_len, hidden_dim = path_tokens.shape
    key_padding_mask = torch.arange(
        token_len, device=path_tokens.device, dtype=torch.long
    ).unsqueeze(0)
    key_padding_mask = key_padding_mask >= path_lengths.unsqueeze(1)
    causal_mask = torch.triu(
        torch.ones((token_len, token_len), device=path_tokens.device, dtype=torch.bool),
        diagonal=1,
    )

    path_tokens_fp32 = path_tokens.to(dtype=torch.float32)
    sdp_ctx = nullcontext()
    if path_tokens.device.type == "cuda":
        sdp_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    with sdp_ctx:
        attn_out, _ = path_self_attention(
            path_tokens_fp32,
            path_tokens_fp32,
            path_tokens_fp32,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
    encoded = path_self_attention_norm(path_tokens_fp32 + attn_out)
    last_idx = (path_lengths - 1).clamp(min=0)
    row_idx = torch.arange(total_agents, device=path_tokens.device, dtype=torch.long)
    last_hidden = encoded[row_idx, last_idx]
    last_hidden = torch.where(
        torch.isfinite(last_hidden), last_hidden, torch.zeros_like(last_hidden)
    )
    return last_hidden.to(dtype=path_tokens.dtype).view(total_agents, hidden_dim)


__all__ = ["build_path_token_embeddings", "encode_path_history"]
