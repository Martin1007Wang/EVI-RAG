from __future__ import annotations

import torch

from .types import SearchState


def _normalize_node_ids(
    *,
    node_ids: torch.Tensor | None,
    num_nodes: int | None,
    device: torch.device,
) -> torch.Tensor:
    if node_ids is None:
        raise ValueError(
            "SearchState.observation is missing node_ids required for no-repeat constraints."
        )
    normalized = node_ids.to(device=device, dtype=torch.long)
    if normalized.dim() != 1:
        raise ValueError(
            "SearchState.observation.node_ids must be a 1D torch.long tensor for no-repeat constraints."
        )
    if num_nodes is not None and int(normalized.numel()) != int(num_nodes):
        raise ValueError(
            "SearchState.observation.node_ids length mismatch with topology.num_nodes for no-repeat constraints. "
            f"node_ids={int(normalized.numel())} num_nodes={int(num_nodes)}."
        )
    return normalized


def _validate_node_index_tensor(
    *,
    values: torch.Tensor,
    upper_bound: int,
    name: str,
) -> torch.Tensor:
    normalized = values.to(dtype=torch.long)
    if bool((normalized < 0).any().item()) or bool(
        (normalized >= int(upper_bound)).any().item()
    ):
        raise ValueError(f"{name} out of range for no-repeat constraints.")
    return normalized


def build_full_no_repeat_mask_from_flat_state(
    *,
    flat_current_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_ids: torch.Tensor | None,
    num_nodes: int | None = None,
    candidate_target_nodes: torch.Tensor,
    candidate_edge_agent_batch: torch.Tensor,
) -> torch.Tensor:
    if tuple(candidate_target_nodes.shape) != tuple(candidate_edge_agent_batch.shape):
        raise ValueError(
            "candidate_target_nodes and candidate_edge_agent_batch must share the same shape for no-repeat constraints. "
            f"candidate_target_nodes={tuple(candidate_target_nodes.shape)} "
            f"candidate_edge_agent_batch={tuple(candidate_edge_agent_batch.shape)}."
        )
    if int(candidate_target_nodes.numel()) == 0:
        return torch.zeros_like(candidate_target_nodes, dtype=torch.bool)

    device = candidate_target_nodes.device
    normalized_num_steps = flat_num_steps.to(device=device, dtype=torch.long).view(-1)
    normalized_current_nodes = flat_current_nodes.to(
        device=device, dtype=torch.long
    ).view(-1)
    if tuple(normalized_current_nodes.shape) != tuple(normalized_num_steps.shape):
        raise ValueError(
            "flat_current_nodes and flat_num_steps must align for no-repeat constraints. "
            f"flat_current_nodes={tuple(normalized_current_nodes.shape)} "
            f"flat_num_steps={tuple(normalized_num_steps.shape)}."
        )
    normalized_node_ids = _normalize_node_ids(
        node_ids=node_ids,
        num_nodes=num_nodes,
        device=device,
    )
    num_nodes = int(normalized_node_ids.numel())
    normalized_current_nodes = _validate_node_index_tensor(
        values=normalized_current_nodes,
        upper_bound=num_nodes,
        name="flat_current_nodes",
    )
    normalized_target_nodes = _validate_node_index_tensor(
        values=candidate_target_nodes.to(device=device),
        upper_bound=num_nodes,
        name="candidate_target_nodes",
    )
    normalized_edge_agent_batch = candidate_edge_agent_batch.to(
        device=device, dtype=torch.long
    )
    if bool((normalized_edge_agent_batch < 0).any().item()) or bool(
        (normalized_edge_agent_batch >= int(normalized_num_steps.numel())).any().item()
    ):
        raise ValueError(
            "candidate_edge_agent_batch out of range for no-repeat constraints."
        )

    candidate_entities = normalized_node_ids.index_select(0, normalized_target_nodes)
    if flat_path_token_ids is None:
        if bool((normalized_num_steps != 0).any().item()):
            raise ValueError(
                "Full no-repeat constraints require exact path_token_ids for non-root SearchState instances."
            )
        current_entities = normalized_node_ids.index_select(0, normalized_current_nodes)
        return candidate_entities == current_entities.index_select(
            0, normalized_edge_agent_batch
        )

    normalized_path_token_ids = flat_path_token_ids.to(device=device, dtype=torch.long)
    if int(normalized_path_token_ids.size(0)) != int(normalized_num_steps.numel()):
        raise ValueError(
            "flat_path_token_ids must align with flat_num_steps for no-repeat constraints. "
            f"flat_path_token_ids={tuple(normalized_path_token_ids.shape)} "
            f"flat_num_steps={tuple(normalized_num_steps.shape)}."
        )
    agent_num_steps = normalized_num_steps.index_select(0, normalized_edge_agent_batch)
    max_visited_steps = (
        int(agent_num_steps.max().item()) if int(agent_num_steps.numel()) > 0 else 0
    )
    # path_token_ids interleave [node_0, rel_0, node_1, rel_1, ..., node_t, STOP?],
    # so the visited node slots are exactly the even positions 0, 2, 4, ...
    visited_positions = 2 * torch.arange(
        max_visited_steps + 1, device=device, dtype=torch.long
    )
    visited_nodes = normalized_path_token_ids.index_select(1, visited_positions)
    visited_nodes = _validate_node_index_tensor(
        values=visited_nodes.reshape(-1),
        upper_bound=num_nodes,
        name="path_token_ids node positions",
    ).view_as(visited_nodes)
    visited_entities = normalized_node_ids.index_select(
        0, visited_nodes.reshape(-1)
    ).view_as(visited_nodes)
    visited_step_mask = torch.arange(
        max_visited_steps + 1, device=device, dtype=torch.long
    ).unsqueeze(0) <= normalized_num_steps.unsqueeze(1)
    candidate_visited_entities = visited_entities.index_select(
        0, normalized_edge_agent_batch
    )
    candidate_visited_step_mask = visited_step_mask.index_select(
        0, normalized_edge_agent_batch
    )
    return (
        (candidate_visited_entities == candidate_entities.unsqueeze(1))
        & candidate_visited_step_mask
    ).any(dim=1)


def build_full_no_repeat_mask(
    *,
    state: SearchState,
    candidate_target_nodes: torch.Tensor,
    candidate_edge_agent_batch: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    flat_path_token_ids = None
    if state.path_token_ids is not None:
        flat_path_token_ids = state.flatten_path_token_ids(max_steps=max_steps)
    return build_full_no_repeat_mask_from_flat_state(
        flat_current_nodes=state.flatten_current_nodes(),
        flat_num_steps=state.flatten_num_steps(),
        flat_path_token_ids=flat_path_token_ids,
        node_ids=_normalize_node_ids(
            node_ids=getattr(state.observation, "node_ids", None),
            num_nodes=int(state.topology.num_nodes),
            device=state.current_nodes.device,
        ),
        num_nodes=int(state.topology.num_nodes),
        candidate_target_nodes=candidate_target_nodes,
        candidate_edge_agent_batch=candidate_edge_agent_batch,
    )


__all__ = ["build_full_no_repeat_mask", "build_full_no_repeat_mask_from_flat_state"]
