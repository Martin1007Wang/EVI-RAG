from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.environment import DynamicAgentState, GraphEnvContext
from src.utils.segment_ops import segment_logsumexp_1d, segment_mean_1d
from .edge import compute_relation_lexical_bias


def compute_stop_logits(
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
        edge_lse, has_finite_edge = segment_logsumexp_1d(
            values=edge_logits.to(dtype=torch.float32),
            segment_ids=edge_agent_batch.to(device=device, dtype=torch.long),
            num_segments=total_agents,
            dtype=dtype,
            ignore_non_finite=True,
            empty_value=0.0,
        )
        stop_logits = torch.where(has_finite_edge, edge_lse + stop_delta, stop_delta)
    else:
        stop_logits = stop_delta

    del super_node_mask

    return stop_logits


def apply_visited_mask_to_edge_logits(
    *,
    edge_logits: torch.Tensor,
    edge_agent_batch: torch.Tensor,
    target_nodes: torch.Tensor,
    visited_mask: torch.Tensor,
) -> torch.Tensor:
    if int(edge_logits.numel()) == 0:
        return edge_logits

    visited_mask = visited_mask.to(device=edge_logits.device, dtype=torch.long)
    if visited_mask.dim() != 2:
        raise ValueError(
            "visited_mask must be 2D [total_agents, window], "
            f"got shape={tuple(visited_mask.shape)}"
        )
    if bool((edge_agent_batch < 0).any().item()) or bool(
        (edge_agent_batch >= int(visited_mask.size(0))).any().item()
    ):
        raise ValueError("edge_agent_batch out of range for visited_mask.")
    recent_nodes = visited_mask.index_select(0, edge_agent_batch)
    valid_recent = recent_nodes >= 0
    visited_edges = (target_nodes.unsqueeze(1) == recent_nodes) & valid_recent
    visited_edges = visited_edges.any(dim=1)

    if not bool(visited_edges.any().item()):
        return edge_logits

    neg_inf = edge_logits.new_full((), float("-inf"))
    return edge_logits.masked_fill(visited_edges, neg_inf)


def gather_actions_from_csr_lock_free(
    *,
    adj_t,
    active_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    O(1) parallel topology action extraction from CSR adjacency.
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
    flat_offsets = torch.arange(
        total_edges, device=active_nodes.device, dtype=torch.long
    )
    increments = flat_offsets - segment_starts.repeat_interleave(out_degrees)
    gather_idx = base_idx + increments

    target_nodes = col[gather_idx]
    gathered_edge_ids = values[gather_idx]
    return gathered_edge_ids, target_nodes, out_degrees


def gather_parent_edges_from_csr(
    *,
    adj_t,
    current_nodes: torch.Tensor,
    active_move: torch.Tensor,
    num_nodes_total: int,
    exclude_parents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_agents = int(current_nodes.numel())
    if int(active_move.numel()) != total_agents:
        raise ValueError("active_move length mismatch in backward parent gathering.")
    if exclude_parents is not None:
        exclude_parents = exclude_parents.to(
            device=current_nodes.device, dtype=torch.long
        ).view(-1)
        if int(exclude_parents.numel()) != total_agents:
            raise ValueError(
                "exclude_parents length mismatch in backward parent gathering."
            )

    active_nodes = torch.where(
        active_move, current_nodes, torch.zeros_like(current_nodes)
    )
    parent_edge_ids, parent_nodes, out_degrees = gather_actions_from_csr_lock_free(
        adj_t=adj_t,
        active_nodes=active_nodes,
    )
    invalid_rows = active_move.clone()
    if int(parent_nodes.numel()) == 0:
        return parent_edge_ids, parent_nodes, parent_nodes, invalid_rows

    agent_rows = torch.arange(
        total_agents, device=parent_nodes.device, dtype=torch.long
    )
    edge_agent_batch = agent_rows.repeat_interleave(out_degrees)
    edge_active_mask = active_move.index_select(0, edge_agent_batch)
    if not bool(edge_active_mask.all().item()):
        parent_nodes = parent_nodes[edge_active_mask]
        edge_agent_batch = edge_agent_batch[edge_active_mask]
        parent_edge_ids = parent_edge_ids[edge_active_mask]
    if int(parent_nodes.numel()) == 0:
        return parent_edge_ids, parent_nodes, edge_agent_batch, invalid_rows

    current_nodes_by_edge = current_nodes.index_select(0, edge_agent_batch)
    parent_is_current = parent_nodes == current_nodes_by_edge
    allowed_edges = ~parent_is_current
    if exclude_parents is not None:
        excluded = parent_nodes == exclude_parents.index_select(0, edge_agent_batch)
        allowed_edges = allowed_edges & (~excluded)
    if not bool(allowed_edges.all().item()):
        parent_nodes = parent_nodes[allowed_edges]
        edge_agent_batch = edge_agent_batch[allowed_edges]
        parent_edge_ids = parent_edge_ids[allowed_edges]
    allowed_counts = torch.zeros(
        (total_agents,), device=current_nodes.device, dtype=torch.long
    )
    if int(parent_nodes.numel()) > 0:
        allowed_counts.scatter_add_(
            0,
            edge_agent_batch,
            torch.ones_like(edge_agent_batch, dtype=torch.long),
        )
    invalid_rows = active_move & (allowed_counts == 0)
    if int(parent_nodes.numel()) == 0:
        return parent_edge_ids, parent_nodes, edge_agent_batch, invalid_rows

    return parent_edge_ids, parent_nodes, edge_agent_batch, invalid_rows


def compute_parent_log_probs(
    *,
    agent_potential: torch.Tensor,
    node_tokens: torch.Tensor,
    relation_tokens: torch.Tensor,
    node_to_policy: nn.Module,
    relation_to_policy: nn.Module,
    edge_action_encoder: nn.Module,
    edge_action_norm: nn.Module,
    relation_group_head: nn.Module,
    relation_lexical_proj: nn.Module,
    lexical_bias_log_scale: torch.Tensor,
    parent_nodes: torch.Tensor,
    parent_edge_ids: torch.Tensor,
    edge_relations: torch.Tensor,
    edge_agent_batch: torch.Tensor,
    lexical_question_tokens: torch.Tensor,
    agent_question_padding_mask: torch.Tensor,
    total_agents: int,
    num_edges_total: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(parent_nodes.numel()) != int(parent_edge_ids.numel()):
        raise ValueError(
            "parent_nodes/parent_edge_ids size mismatch in backward log prob computation."
        )
    if int(edge_agent_batch.numel()) != int(parent_nodes.numel()):
        raise ValueError(
            "edge_agent_batch/parent_nodes size mismatch in backward log prob computation."
        )
    if int(edge_relations.numel()) != int(parent_edge_ids.numel()):
        raise ValueError(
            "edge_relations/parent_edge_ids size mismatch in backward log prob computation."
        )

    num_relations = int(relation_tokens.size(0))
    if num_relations <= 0:
        raise ValueError("relation_tokens must contain at least one relation.")

    state_vec = agent_potential.index_select(0, edge_agent_batch).to(
        dtype=torch.float32
    )
    parent_features = node_tokens.index_select(0, parent_nodes).to(dtype=torch.float32)
    parent_features = node_to_policy(parent_features)
    safe_rel = edge_relations.clamp(min=0, max=num_relations - 1).to(dtype=torch.long)
    relation_features = relation_tokens.index_select(0, safe_rel).to(
        dtype=torch.float32
    )
    relation_features = relation_to_policy(relation_features)
    parent_edge_features = parent_features + relation_features

    edge_action = edge_action_encoder(
        torch.cat((relation_features, parent_features), dim=-1)
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

    lexical_bias = compute_relation_lexical_bias(
        lexical_question_tokens=lexical_question_tokens,
        agent_question_padding_mask=agent_question_padding_mask,
        relation_tokens=relation_tokens,
        relation_to_policy=relation_to_policy,
        relation_lexical_proj=relation_lexical_proj,
        edge_agent_batch=edge_agent_batch,
        edge_relations=edge_relations,
    )
    lexical_scale = F.softplus(lexical_bias_log_scale.to(dtype=torch.float32))
    conditional_logits = conditional_logits + lexical_scale * lexical_bias

    relation_group_input = torch.cat((state_vec, relation_features), dim=-1)
    relation_logits_edge = (
        relation_group_head(relation_group_input).squeeze(-1).to(dtype=torch.float32)
    )
    relation_logits_edge = torch.where(
        torch.isfinite(relation_logits_edge),
        relation_logits_edge,
        torch.zeros_like(relation_logits_edge),
    )

    group_keys = edge_agent_batch * num_relations + safe_rel
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
    relation_lse, relation_has_finite = segment_logsumexp_1d(
        values=relation_logits_group,
        segment_ids=group_agent_ids,
        num_segments=total_agents,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=0.0,
    )
    if not bool(relation_has_finite.index_select(0, group_agent_ids).all().item()):
        raise ValueError("Backward relation scoring produced non-finite groups.")
    relation_log_prob_group = relation_logits_group - relation_lse.index_select(
        0, group_agent_ids
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
        raise ValueError("Backward conditional scoring produced non-finite groups.")
    conditional_log_prob = conditional_logits - conditional_lse_group.index_select(
        0, inverse_group
    )
    edge_logits = relation_log_prob_edge + conditional_log_prob
    edge_keys = edge_agent_batch * int(num_edges_total) + parent_edge_ids
    return edge_logits, inverse_group, unique_group_keys, edge_keys


def select_parent_log_prob(
    *,
    parent_log_prob: torch.Tensor,
    unique_keys: torch.Tensor,
    chosen_edge_ids: torch.Tensor,
    active_rows: torch.Tensor,
    num_edges_total: int,
    total_agents: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(chosen_edge_ids.numel()) != int(total_agents):
        raise ValueError("chosen_edge_ids length mismatch in backward log prob lookup.")
    agent_ids = torch.arange(
        total_agents, device=chosen_edge_ids.device, dtype=torch.long
    )
    chosen_keys = agent_ids * num_edges_total + chosen_edge_ids
    active_keys = chosen_keys.index_select(0, active_rows)
    invalid_rows = torch.zeros(
        (total_agents,), device=chosen_edge_ids.device, dtype=torch.bool
    )
    if int(unique_keys.numel()) == 0:
        invalid_rows.index_fill_(0, active_rows, True)
        return (
            parent_log_prob.new_empty((0,)),
            active_rows.new_empty((0,)),
            invalid_rows,
        )
    search_idx = torch.searchsorted(unique_keys, active_keys)
    in_range = search_idx < int(unique_keys.numel())
    safe_idx = search_idx.clamp(max=max(int(unique_keys.numel()) - 1, 0))
    matched = in_range & (unique_keys.index_select(0, safe_idx) == active_keys)
    valid_rows = active_rows[matched]
    invalid_rows.index_fill_(0, active_rows[~matched], True)
    if int(valid_rows.numel()) == 0:
        return (
            parent_log_prob.new_empty((0,)),
            valid_rows,
            invalid_rows,
        )
    chosen_log_prob = parent_log_prob.index_select(0, safe_idx[matched])
    return chosen_log_prob, valid_rows, invalid_rows


def build_empty_output(
    *,
    B: int,
    num_agents: int,
    device: torch.device,
    dtype: torch.dtype,
    stop_logits: torch.Tensor,
    state_log_flows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "edge_logits": torch.empty(0, device=device, dtype=dtype),
        "edge_agent_batch": torch.empty(0, dtype=torch.long, device=device),
        "stop_logits": stop_logits,
        "edge_ids": torch.empty(0, dtype=torch.long, device=device),
        "target_nodes": torch.empty(0, dtype=torch.long, device=device),
        "out_degrees": torch.zeros((B, num_agents), dtype=torch.long, device=device),
        "state_log_flows": state_log_flows,
    }


__all__ = [
    "build_empty_output",
    "apply_visited_mask_to_edge_logits",
    "compute_parent_log_probs",
    "compute_stop_logits",
    "gather_parent_edges_from_csr",
    "gather_actions_from_csr_lock_free",
    "select_parent_log_prob",
]
