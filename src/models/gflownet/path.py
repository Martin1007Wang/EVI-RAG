from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


def max_path_tokens(*, max_steps: int) -> int:
    return (2 * int(max_steps)) + 1


def initialize_path_token_ids(
    *, start_nodes: torch.Tensor, max_steps: int
) -> torch.Tensor:
    token_ids = torch.zeros(
        (*start_nodes.shape, max_path_tokens(max_steps=max_steps)),
        device=start_nodes.device,
        dtype=torch.long,
    )
    token_ids[..., 0] = start_nodes
    return token_ids


def append_relation_and_node_tokens(
    *,
    path_token_ids: torch.Tensor,
    num_steps: torch.Tensor,
    relation_ids: torch.Tensor,
    target_nodes: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    updated = path_token_ids.clone()
    flat_updated = updated.view(-1, int(updated.size(-1)))
    flat_num_steps = num_steps.reshape(-1)
    flat_relation_ids = relation_ids.reshape(-1)
    flat_target_nodes = target_nodes.reshape(-1)
    if active_mask is None:
        active_indices = torch.arange(
            int(flat_num_steps.numel()), device=flat_num_steps.device, dtype=torch.long
        )
    else:
        active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).view(-1)
    if int(active_indices.numel()) == 0:
        return updated
    relation_positions = (2 * flat_num_steps.index_select(0, active_indices) + 1).to(
        dtype=torch.long
    )
    node_positions = relation_positions + 1
    flat_updated[active_indices, relation_positions] = flat_relation_ids.index_select(
        0, active_indices
    )
    flat_updated[active_indices, node_positions] = flat_target_nodes.index_select(
        0, active_indices
    )
    return updated.view_as(path_token_ids)


def derive_parent_path_token_ids(
    *,
    child_path_token_ids: torch.Tensor,
    child_num_steps: torch.Tensor,
    parent_nodes: torch.Tensor,
) -> torch.Tensor:
    parent_num_steps = (child_num_steps - 1).clamp_min(0)
    updated = child_path_token_ids.clone()
    token_length = int(updated.size(-1))
    positions = torch.arange(token_length, device=updated.device, dtype=torch.long)
    parent_token_lengths = 2 * parent_num_steps.reshape(-1) + 1
    flat_updated = updated.view(-1, token_length)
    flat_updated = flat_updated.masked_fill(
        positions.unsqueeze(0) >= parent_token_lengths.unsqueeze(1),
        0,
    )
    row_idx = torch.arange(
        int(flat_updated.size(0)), device=updated.device, dtype=torch.long
    )
    last_node_positions = 2 * parent_num_steps.reshape(-1)
    flat_updated[row_idx, last_node_positions] = parent_nodes.reshape(-1)
    return flat_updated.view_as(child_path_token_ids)


def build_path_token_embeddings(
    *,
    path_token_ids: torch.Tensor,
    path_lengths: torch.Tensor,
    node_tokens: torch.Tensor,
    relation_tokens: torch.Tensor,
    position_embedding: nn.Embedding,
) -> torch.Tensor:
    if path_token_ids.dim() != 2:
        raise ValueError(
            f"path_token_ids must be 2D [N, T], got shape={tuple(path_token_ids.shape)}."
        )
    total_agents, token_len = path_token_ids.shape
    hidden_dim = int(node_tokens.size(-1))
    safe_node_ids = path_token_ids.clamp(
        min=0, max=max(int(node_tokens.size(0)) - 1, 0)
    )
    node_part = node_tokens.index_select(0, safe_node_ids.reshape(-1)).view(
        total_agents, token_len, hidden_dim
    )
    if int(relation_tokens.size(0)) == 0:
        relation_part = torch.zeros_like(node_part)
    else:
        safe_rel_ids = path_token_ids.clamp(
            min=0, max=max(int(relation_tokens.size(0)) - 1, 0)
        )
        relation_part = relation_tokens.index_select(0, safe_rel_ids.reshape(-1)).view(
            total_agents, token_len, hidden_dim
        )
    token_types = (
        torch.arange(token_len, device=path_token_ids.device, dtype=torch.long) % 2 == 1
    )
    path_tokens = torch.where(
        token_types.view(1, token_len, 1), relation_part, node_part
    )
    pos = position_embedding(
        torch.arange(token_len, device=path_token_ids.device, dtype=torch.long)
    ).to(dtype=path_tokens.dtype)
    path_tokens = path_tokens + pos.unsqueeze(0)
    key_padding_mask = torch.arange(
        token_len, device=path_token_ids.device, dtype=torch.long
    ).unsqueeze(0) >= path_lengths.reshape(-1, 1)
    return torch.where(
        key_padding_mask.unsqueeze(-1), torch.zeros_like(path_tokens), path_tokens
    )


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
    ).unsqueeze(0) >= path_lengths.reshape(-1, 1)
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
    last_idx = (path_lengths.reshape(-1) - 1).clamp_min(0)
    row_idx = torch.arange(total_agents, device=path_tokens.device, dtype=torch.long)
    last_hidden = encoded[row_idx, last_idx]
    last_hidden = torch.where(
        torch.isfinite(last_hidden),
        last_hidden,
        torch.zeros_like(last_hidden),
    )
    return last_hidden.to(dtype=path_tokens.dtype).view(total_agents, hidden_dim)


def reconstruct_trace_path_token_ids(
    *,
    start_nodes: torch.Tensor,
    trace_edge_ids: torch.Tensor,
    trace_num_steps: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    path_token_ids = initialize_path_token_ids(
        start_nodes=start_nodes, max_steps=max_steps
    )
    trace_path_token_ids = torch.zeros(
        (*trace_edge_ids.shape, int(path_token_ids.size(-1))),
        device=trace_edge_ids.device,
        dtype=torch.long,
    )
    current_path_token_ids = path_token_ids
    for step_idx in range(int(trace_edge_ids.size(-1))):
        trace_path_token_ids[:, :, step_idx] = current_path_token_ids
        chosen_edge_ids = trace_edge_ids[:, :, step_idx]
        active_mask = chosen_edge_ids >= 0
        if not bool(active_mask.any().item()):
            continue
        flat_edges = chosen_edge_ids.reshape(-1)
        safe_edge_ids = flat_edges.clamp(min=0)
        relation_ids = edge_type.index_select(0, safe_edge_ids).view_as(chosen_edge_ids)
        target_nodes = (
            edge_index[1].index_select(0, safe_edge_ids).view_as(chosen_edge_ids)
        )
        current_path_token_ids = append_relation_and_node_tokens(
            path_token_ids=current_path_token_ids,
            num_steps=trace_num_steps[:, :, step_idx],
            relation_ids=relation_ids,
            target_nodes=target_nodes,
            active_mask=active_mask,
        )
    return trace_path_token_ids


__all__ = [
    "append_relation_and_node_tokens",
    "build_path_token_embeddings",
    "derive_parent_path_token_ids",
    "encode_path_history",
    "initialize_path_token_ids",
    "max_path_tokens",
    "reconstruct_trace_path_token_ids",
]
