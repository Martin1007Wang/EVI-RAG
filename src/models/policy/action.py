from __future__ import annotations

import torch

from src.models.environment import DynamicAgentState, GraphEnvContext
from src.utils.segment_ops import segment_logsumexp_1d


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

    # Keep super-source anti-early-stop rule: stop is forbidden while standing on any virtual node.
    if super_node_mask is not None:
        if super_node_mask.dim() != 1 or int(super_node_mask.numel()) != int(
            env_context.num_nodes_total
        ):
            raise ValueError(
                "super_node_mask shape mismatch in stop-logit computation: "
                f"mask_shape={tuple(super_node_mask.shape)}, num_nodes_total={int(env_context.num_nodes_total)}."
            )
        super_mask = super_node_mask.to(device=device, dtype=torch.bool)
        current_nodes = agent_state.current_nodes.view(-1)
        safe_nodes = current_nodes.clamp(
            min=0, max=max(int(env_context.num_nodes_total) - 1, 0)
        )
        at_super = (current_nodes >= 0) & super_mask.index_select(0, safe_nodes)
        stop_logits = stop_logits.masked_fill(
            at_super,
            torch.tensor(
                float("-inf"), device=stop_logits.device, dtype=stop_logits.dtype
            ),
        )

    return stop_logits


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
    "compute_stop_logits",
    "gather_actions_from_csr_lock_free",
]
