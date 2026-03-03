from __future__ import annotations

import torch
from torch import nn

from src.models.environment import DynamicAgentState


def resolve_path_state(
    *, agent_state: DynamicAgentState
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, num_agents = agent_state.current_nodes.shape
    total_agents = B * num_agents
    path_ids = agent_state.path_token_ids
    path_types = agent_state.path_token_types
    path_lengths = agent_state.path_lengths

    if path_ids is None and path_types is None and path_lengths is None:
        default_ids = agent_state.current_nodes.view(total_agents, 1).clone()
        default_types = torch.zeros_like(default_ids, dtype=torch.bool)
        default_lengths = torch.ones(
            (total_agents,), dtype=torch.long, device=default_ids.device
        )
        return default_ids, default_types, default_lengths
    if path_ids is None or path_types is None or path_lengths is None:
        raise ValueError(
            "path_token_ids/path_token_types/path_lengths must be all provided or all omitted."
        )
    if path_ids.dim() != 3 or path_types.dim() != 3:
        raise ValueError(
            "path_token_ids and path_token_types must be 3D [B, num_agents, T], "
            f"got ids={tuple(path_ids.shape)}, types={tuple(path_types.shape)}."
        )
    if path_lengths.dim() != 2:
        raise ValueError(
            f"path_lengths must be 2D [B, num_agents], got shape={tuple(path_lengths.shape)}."
        )
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
    flat_types = path_types.reshape(total_agents, path_types.size(-1)).to(
        dtype=torch.bool
    )
    flat_lengths = path_lengths.reshape(total_agents).to(
        device=flat_ids.device, dtype=torch.long
    )
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
            f"relation_tokens must be 2D [R, d], got shape={tuple(relation_tokens.shape)}."
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


def evolve_state(
    *,
    agent_state: DynamicAgentState,
    chosen_target_nodes: torch.Tensor,
    chosen_edge_relations: torch.Tensor,
    node_tokens: torch.Tensor,
    is_stop: torch.Tensor,
) -> DynamicAgentState:
    B, num_agents = agent_state.current_nodes.shape
    total_agents = B * num_agents

    safe_target_nodes = torch.where(
        is_stop, torch.zeros_like(chosen_target_nodes), chosen_target_nodes
    )
    safe_edge_relations = torch.where(
        is_stop, torch.zeros_like(chosen_edge_relations), chosen_edge_relations
    )

    flat_hidden = agent_state.hidden_states.view(total_agents, -1)
    if int(flat_hidden.size(-1)) == int(node_tokens.size(-1)):
        moved_hidden = node_tokens.index_select(0, safe_target_nodes.clamp(min=0)).to(
            dtype=flat_hidden.dtype
        )
        next_hidden = torch.where(is_stop.unsqueeze(-1), flat_hidden, moved_hidden)
    else:
        next_hidden = flat_hidden

    was_active = ~agent_state.done_mask.view(-1)
    active_move = was_active & (~is_stop)
    new_visited_mask = agent_state.visited_mask.clone()
    if new_visited_mask.dim() == 1:
        if bool(active_move.any().item()):
            new_visited_mask.scatter_(0, safe_target_nodes[active_move], True)
    elif new_visited_mask.dim() == 2:
        if bool(active_move.any().item()):
            row_ids = torch.arange(
                is_stop.numel(), device=is_stop.device, dtype=torch.long
            )[active_move]
            col_ids = safe_target_nodes[active_move]
            new_visited_mask[row_ids, col_ids] = True
    else:
        raise ValueError(
            f"visited_mask must be 1D or 2D, got shape={tuple(new_visited_mask.shape)}"
        )

    current_nodes = agent_state.current_nodes.view(total_agents)
    next_current_flat = torch.where(is_stop, current_nodes, safe_target_nodes)
    next_current_nodes = next_current_flat.view(B, num_agents)
    next_num_moves_flat = agent_state.num_moves.view(total_agents).to(
        dtype=torch.long
    ) + active_move.to(dtype=torch.long)
    next_num_moves = next_num_moves_flat.view(B, num_agents)

    path_token_ids, path_token_types, path_lengths = resolve_path_state(
        agent_state=agent_state
    )
    next_lengths = path_lengths + active_move.to(dtype=torch.long) * 2
    old_width = int(path_token_ids.size(1))
    next_width = max(old_width, int(next_lengths.max().item()))
    if next_width > old_width:
        next_path_ids = torch.zeros(
            (total_agents, next_width),
            dtype=path_token_ids.dtype,
            device=path_token_ids.device,
        )
        next_path_types = torch.zeros(
            (total_agents, next_width), dtype=torch.bool, device=path_token_types.device
        )
        next_path_ids[:, :old_width] = path_token_ids
        next_path_types[:, :old_width] = path_token_types
    else:
        next_path_ids = path_token_ids.clone()
        next_path_types = path_token_types.clone()
    if bool(active_move.any().item()):
        move_rows = torch.where(active_move)[0]
        rel_pos = path_lengths.index_select(0, move_rows)
        node_pos = rel_pos + 1
        move_rel = safe_edge_relations.index_select(0, move_rows).to(
            dtype=next_path_ids.dtype
        )
        move_nodes = safe_target_nodes.index_select(0, move_rows).to(
            dtype=next_path_ids.dtype
        )
        next_path_ids[move_rows, rel_pos] = move_rel
        next_path_types[move_rows, rel_pos] = True
        next_path_ids[move_rows, node_pos] = move_nodes
        next_path_types[move_rows, node_pos] = False

    return DynamicAgentState(
        step_t=agent_state.step_t + 1,
        current_nodes=next_current_nodes,
        flow_direction=agent_state.flow_direction,
        hidden_states=next_hidden.view(B, num_agents, -1),
        visited_mask=new_visited_mask,
        cumulative_rewards=agent_state.cumulative_rewards,
        done_mask=agent_state.done_mask | is_stop.view(B, num_agents),
        num_moves=next_num_moves,
        path_token_ids=next_path_ids.view(B, num_agents, next_width),
        path_token_types=next_path_types.view(B, num_agents, next_width),
        path_lengths=next_lengths.view(B, num_agents),
    )


__all__ = [
    "build_path_token_embeddings",
    "encode_path_history",
    "evolve_state",
    "resolve_path_state",
]
