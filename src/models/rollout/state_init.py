from __future__ import annotations

import torch

from src.models.environment import (
    DynamicAgentState,
    FlowDirection,
    GraphEnvContext,
    has_super_source_layout,
    infer_super_source_absolute_indices,
)


def expand_grouped_start_nodes(
    *,
    q_local_indices: torch.Tensor,
    q_ptr: torch.Tensor,
    num_agents: int,
    deterministic: bool,
) -> torch.Tensor:
    num_graphs = int(q_ptr.numel()) - 1
    counts = (q_ptr[1:] - q_ptr[:-1]).clamp(min=0)
    if int(counts.numel()) != num_graphs:
        raise ValueError("q_ptr shape mismatch in grouped start expansion.")
    if bool((counts <= 0).any().item()):
        raise ValueError("q_local_indices has empty groups; fail-fast by contract.")
    base = q_ptr[:-1].unsqueeze(1)
    slots = torch.arange(
        num_agents, device=q_local_indices.device, dtype=torch.long
    ).unsqueeze(0)
    slots = slots.expand(num_graphs, -1)
    if deterministic:
        offsets = torch.remainder(slots, counts.unsqueeze(1))
    else:
        random_shift = torch.floor(
            torch.rand((num_graphs, 1), device=q_local_indices.device)
            * counts.unsqueeze(1).to(dtype=torch.float32)
        ).to(dtype=torch.long)
        offsets = torch.remainder(slots + random_shift, counts.unsqueeze(1))
    gather_idx = (base + offsets).reshape(-1)
    return q_local_indices.index_select(0, gather_idx).view(num_graphs, num_agents)


def resolve_start_nodes_absolute(
    *,
    env_context: GraphEnvContext,
    num_agents: int,
    deterministic: bool,
    flow_direction: FlowDirection,
    backward_without_super_error: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device = env_context.node_embeddings.device
    num_graphs = int(env_context.num_graphs)
    if has_super_source_layout(
        node_ptr=env_context.node_ptr,
        node_global_ids=env_context.node_global_ids,
        num_nodes_total=env_context.num_nodes_total,
        device=device,
    ):
        forward_super_abs, backward_super_abs = infer_super_source_absolute_indices(
            node_ptr=env_context.node_ptr,
            node_global_ids=env_context.node_global_ids,
            num_nodes_total=env_context.num_nodes_total,
            device=device,
        )
        super_nodes_abs = torch.stack((forward_super_abs, backward_super_abs), dim=1)
        if flow_direction == "forward":
            start_nodes_absolute = forward_super_abs.unsqueeze(1).expand(
                num_graphs, num_agents
            )
        elif flow_direction == "backward":
            start_nodes_absolute = backward_super_abs.unsqueeze(1).expand(
                num_graphs, num_agents
            )
        else:
            raise ValueError(f"Unsupported flow_direction: {flow_direction!r}.")
        return start_nodes_absolute, super_nodes_abs
    if flow_direction == "backward":
        raise ValueError(backward_without_super_error)
    start_local = expand_grouped_start_nodes(
        q_local_indices=env_context.q_local_indices,
        q_ptr=env_context.q_ptr,
        num_agents=num_agents,
        deterministic=deterministic,
    )
    start_nodes_absolute = start_local + env_context.node_ptr[:-1].unsqueeze(1)
    return start_nodes_absolute, None


def initialize_agent_state(
    *,
    env_context: GraphEnvContext,
    num_agents: int,
    deterministic: bool,
    flow_direction: FlowDirection,
    backward_without_super_error: str,
) -> DynamicAgentState:
    start_nodes_absolute, super_nodes_abs = resolve_start_nodes_absolute(
        env_context=env_context,
        num_agents=num_agents,
        deterministic=deterministic,
        flow_direction=flow_direction,
        backward_without_super_error=backward_without_super_error,
    )
    device = env_context.node_embeddings.device
    num_graphs = int(env_context.num_graphs)
    current_nodes = start_nodes_absolute.clone()
    hidden_states = (
        env_context.question_emb.unsqueeze(1).expand(num_graphs, num_agents, -1).clone()
    )
    path_token_ids = current_nodes.unsqueeze(-1).clone()
    path_token_types = torch.zeros_like(path_token_ids, dtype=torch.bool)
    path_lengths = torch.ones((num_graphs, num_agents), dtype=torch.long, device=device)
    visited_mask = torch.zeros(
        (num_graphs * num_agents, env_context.num_nodes_total),
        dtype=torch.bool,
        device=device,
    )
    row_ids = torch.arange(num_graphs * num_agents, device=device, dtype=torch.long)
    visited_mask[row_ids, current_nodes.view(-1)] = True
    if super_nodes_abs is not None:
        super_cols = (
            super_nodes_abs.unsqueeze(1)
            .expand(num_graphs, num_agents, 2)
            .reshape(-1, 2)
        )
        visited_mask[row_ids.unsqueeze(1), super_cols] = True
    return DynamicAgentState(
        step_t=0,
        current_nodes=current_nodes,
        flow_direction=flow_direction,
        hidden_states=hidden_states,
        visited_mask=visited_mask,
        cumulative_rewards=torch.zeros((num_graphs, num_agents), device=device),
        done_mask=torch.zeros(
            (num_graphs, num_agents), dtype=torch.bool, device=device
        ),
        num_moves=torch.zeros(
            (num_graphs, num_agents), dtype=torch.long, device=device
        ),
        path_token_ids=path_token_ids,
        path_token_types=path_token_types,
        path_lengths=path_lengths,
    )


def compute_min_required_moves(
    *,
    env_context: GraphEnvContext,
    start_nodes_abs: torch.Tensor,
    base_stop_min_steps: int,
    flatten: bool,
) -> torch.Tensor:
    if base_stop_min_steps < 0:
        raise ValueError("sampling.stop_min_steps must be >= 0.")
    offsets = compute_start_virtual_step_offsets(
        env_context=env_context,
        start_nodes_abs=start_nodes_abs,
        flatten=True,
    )
    required_flat = (
        torch.full_like(offsets, fill_value=base_stop_min_steps, dtype=torch.long)
        + offsets
    )
    if flatten:
        return required_flat
    return required_flat.view_as(start_nodes_abs)


def compute_max_allowed_moves(
    *,
    env_context: GraphEnvContext,
    start_nodes_abs: torch.Tensor,
    base_max_steps: int,
    flatten: bool,
) -> torch.Tensor:
    if base_max_steps < 0:
        raise ValueError("sampling.max_steps must be >= 0.")
    offsets = compute_start_virtual_step_offsets(
        env_context=env_context,
        start_nodes_abs=start_nodes_abs,
        flatten=True,
    )
    allowed_flat = (
        torch.full_like(offsets, fill_value=base_max_steps, dtype=torch.long) + offsets
    )
    if flatten:
        return allowed_flat
    return allowed_flat.view_as(start_nodes_abs)


def compute_effective_max_steps(
    *,
    env_context: GraphEnvContext,
    start_nodes_abs: torch.Tensor,
    base_max_steps: int,
) -> int:
    if base_max_steps < 0:
        raise ValueError("max_steps must be >= 0.")
    offsets_flat = compute_start_virtual_step_offsets(
        env_context=env_context,
        start_nodes_abs=start_nodes_abs,
        flatten=True,
    )
    if int(offsets_flat.numel()) == 0:
        return base_max_steps + 1
    first_offset = offsets_flat[0]
    if bool((offsets_flat != first_offset).any().item()):
        raise ValueError(
            "Mixed virtual/non-virtual start nodes are not supported under real-hop max_steps semantics."
        )
    return int(base_max_steps + int(first_offset.item()) + 1)


def compute_start_virtual_step_offsets(
    *,
    env_context: GraphEnvContext,
    start_nodes_abs: torch.Tensor,
    flatten: bool,
) -> torch.Tensor:
    start_nodes_flat = start_nodes_abs.view(-1).to(dtype=torch.long)
    if int(start_nodes_flat.numel()) == 0:
        empty = start_nodes_abs.new_zeros(start_nodes_abs.shape, dtype=torch.long)
        return empty.view(-1) if flatten else empty
    node_global_ids = env_context.node_global_ids.to(
        device=start_nodes_flat.device,
        dtype=torch.long,
    )
    if int(node_global_ids.numel()) != int(env_context.num_nodes_total):
        raise ValueError(
            "node_global_ids length mismatch with num_nodes_total in start-step offset resolution: "
            f"node_global_ids={int(node_global_ids.numel())}, num_nodes_total={int(env_context.num_nodes_total)}."
        )
    start_global_ids = node_global_ids.index_select(0, start_nodes_flat)
    offsets_flat = (start_global_ids < 0).to(dtype=torch.long)
    if flatten:
        return offsets_flat
    return offsets_flat.view_as(start_nodes_abs)


__all__ = [
    "expand_grouped_start_nodes",
    "resolve_start_nodes_absolute",
    "initialize_agent_state",
    "compute_min_required_moves",
    "compute_max_allowed_moves",
    "compute_effective_max_steps",
    "compute_start_virtual_step_offsets",
]
