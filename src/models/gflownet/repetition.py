from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import SearchState


ABSOLUTE_NODE_TOKEN_STRIDE = 2


@dataclass(frozen=True)
class FlatEntityPrefixState:
    """Flat prefix state view with explicit absolute-node and entity-id semantics."""

    current_abs_node_indices: torch.Tensor
    num_steps: torch.Tensor
    path_token_ids: torch.Tensor | None
    node_entity_ids_by_abs_node: torch.Tensor


def _normalize_node_entity_ids(
    *,
    node_entity_ids_by_abs_node: torch.Tensor | None,
    num_nodes: int | None,
    device: torch.device,
) -> torch.Tensor:
    if node_entity_ids_by_abs_node is None:
        raise ValueError(
            "SearchState.observation is missing node_entity_ids required for no-repeat constraints."
        )
    normalized = node_entity_ids_by_abs_node.to(device=device, dtype=torch.long)
    if normalized.dim() != 1:
        raise ValueError(
            "SearchState.observation.node_entity_ids must be a 1D torch.long tensor for no-repeat constraints."
        )
    if num_nodes is not None and int(normalized.numel()) != int(num_nodes):
        raise ValueError(
            "SearchState.observation.node_entity_ids length mismatch with topology.num_nodes for no-repeat constraints. "
            f"node_entity_ids={int(normalized.numel())} num_nodes={int(num_nodes)}."
        )
    return normalized


def _normalize_abs_node_indices(
    *,
    values: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    normalized = values.to(device=device, dtype=torch.long).view(-1)
    if bool((normalized < 0).any().item()) or bool(
        (normalized >= int(num_nodes)).any().item()
    ):
        raise ValueError(f"{name} out of range for no-repeat constraints.")
    return normalized


def _normalize_agent_indices(
    *,
    candidate_agent_indices: torch.Tensor,
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    normalized = candidate_agent_indices.to(device=device, dtype=torch.long).view(-1)
    if bool((normalized < 0).any().item()) or bool(
        (normalized >= int(num_agents)).any().item()
    ):
        raise ValueError(
            "candidate_agent_indices out of range for no-repeat constraints."
        )
    return normalized


def _normalize_flat_entity_prefix_state(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor | None,
    num_nodes: int | None,
    device: torch.device,
) -> FlatEntityPrefixState:
    normalized_num_steps = flat_num_steps.to(device=device, dtype=torch.long).view(-1)
    normalized_entity_ids = _normalize_node_entity_ids(
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
        device=device,
    )
    resolved_num_nodes = int(normalized_entity_ids.numel())
    normalized_current_abs_nodes = _normalize_abs_node_indices(
        values=flat_current_abs_nodes,
        num_nodes=resolved_num_nodes,
        device=device,
        name="flat_current_abs_nodes",
    )
    if tuple(normalized_current_abs_nodes.shape) != tuple(normalized_num_steps.shape):
        raise ValueError(
            "flat_current_abs_nodes and flat_num_steps must align for no-repeat constraints. "
            f"flat_current_abs_nodes={tuple(normalized_current_abs_nodes.shape)} "
            f"flat_num_steps={tuple(normalized_num_steps.shape)}."
        )
    normalized_path_token_ids = None
    if flat_path_token_ids is not None:
        normalized_path_token_ids = flat_path_token_ids.to(
            device=device, dtype=torch.long
        )
        if int(normalized_path_token_ids.size(0)) != int(normalized_num_steps.numel()):
            raise ValueError(
                "flat_path_token_ids must align with flat_num_steps for no-repeat constraints. "
                f"flat_path_token_ids={tuple(normalized_path_token_ids.shape)} "
                f"flat_num_steps={tuple(normalized_num_steps.shape)}."
            )
    return FlatEntityPrefixState(
        current_abs_node_indices=normalized_current_abs_nodes,
        num_steps=normalized_num_steps,
        path_token_ids=normalized_path_token_ids,
        node_entity_ids_by_abs_node=normalized_entity_ids,
    )


def _extract_prefix_abs_node_history(
    state_view: FlatEntityPrefixState,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = state_view.current_abs_node_indices.device
    num_nodes = int(state_view.node_entity_ids_by_abs_node.numel())
    if state_view.path_token_ids is None:
        if bool((state_view.num_steps != 0).any().item()):
            raise ValueError(
                "Full no-repeat constraints require exact path_token_ids for non-root SearchState instances."
            )
        return state_view.current_abs_node_indices.unsqueeze(1), torch.ones(
            (int(state_view.num_steps.numel()), 1),
            device=device,
            dtype=torch.bool,
        )

    max_prefix_steps = (
        int(state_view.num_steps.max().item())
        if int(state_view.num_steps.numel()) > 0
        else 0
    )
    # path_token_ids interleave [node_0, rel_0, node_1, rel_1, ..., node_t, STOP?],
    # so visited absolute-node slots are exactly the even positions 0, 2, 4, ...
    prefix_abs_node_positions = ABSOLUTE_NODE_TOKEN_STRIDE * torch.arange(
        max_prefix_steps + 1,
        device=device,
        dtype=torch.long,
    )
    prefix_abs_node_history = state_view.path_token_ids.index_select(
        1, prefix_abs_node_positions
    )
    prefix_abs_node_history = _normalize_abs_node_indices(
        values=prefix_abs_node_history,
        num_nodes=num_nodes,
        device=device,
        name="path_token_ids node positions",
    ).view_as(prefix_abs_node_history)
    valid_history_mask = torch.arange(
        max_prefix_steps + 1,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0) <= state_view.num_steps.unsqueeze(1)
    return prefix_abs_node_history, valid_history_mask


def build_entity_revisit_mask_from_flat_state(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor | None,
    num_nodes: int | None = None,
    candidate_target_abs_nodes: torch.Tensor,
    candidate_agent_indices: torch.Tensor,
) -> torch.Tensor:
    if tuple(candidate_target_abs_nodes.shape) != tuple(candidate_agent_indices.shape):
        raise ValueError(
            "candidate_target_abs_nodes and candidate_agent_indices must share the same shape for no-repeat constraints. "
            f"candidate_target_abs_nodes={tuple(candidate_target_abs_nodes.shape)} "
            f"candidate_agent_indices={tuple(candidate_agent_indices.shape)}."
        )
    if int(candidate_target_abs_nodes.numel()) == 0:
        return torch.zeros_like(candidate_target_abs_nodes, dtype=torch.bool)

    device = candidate_target_abs_nodes.device
    state_view = _normalize_flat_entity_prefix_state(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
        device=device,
    )
    resolved_num_nodes = int(state_view.node_entity_ids_by_abs_node.numel())
    normalized_candidate_target_abs_nodes = _normalize_abs_node_indices(
        values=candidate_target_abs_nodes,
        num_nodes=resolved_num_nodes,
        device=device,
        name="candidate_target_abs_nodes",
    )
    normalized_candidate_agent_indices = _normalize_agent_indices(
        candidate_agent_indices=candidate_agent_indices,
        num_agents=int(state_view.num_steps.numel()),
        device=device,
    )
    candidate_target_entity_ids = state_view.node_entity_ids_by_abs_node.index_select(
        0, normalized_candidate_target_abs_nodes
    )
    prefix_abs_node_history, valid_history_mask = _extract_prefix_abs_node_history(
        state_view
    )
    prefix_entity_history = state_view.node_entity_ids_by_abs_node.index_select(
        0, prefix_abs_node_history.reshape(-1)
    ).view_as(prefix_abs_node_history)
    candidate_prefix_entity_history = prefix_entity_history.index_select(
        0, normalized_candidate_agent_indices
    )
    candidate_valid_history_mask = valid_history_mask.index_select(
        0, normalized_candidate_agent_indices
    )
    return (
        (candidate_prefix_entity_history == candidate_target_entity_ids.unsqueeze(1))
        & candidate_valid_history_mask
    ).any(dim=1)


def build_entity_revisit_mask(
    *,
    state: SearchState,
    candidate_target_abs_nodes: torch.Tensor,
    candidate_agent_indices: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    flat_path_token_ids = None
    if state.path_token_ids is not None:
        flat_path_token_ids = state.flatten_path_token_ids(max_steps=max_steps)
    return build_entity_revisit_mask_from_flat_state(
        flat_current_abs_nodes=state.flatten_current_nodes(),
        flat_num_steps=state.flatten_num_steps(),
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=state.observation.node_entity_ids,
        num_nodes=int(state.topology.num_nodes),
        candidate_target_abs_nodes=candidate_target_abs_nodes,
        candidate_agent_indices=candidate_agent_indices,
    )


__all__ = [
    "FlatEntityPrefixState",
    "build_entity_revisit_mask",
    "build_entity_revisit_mask_from_flat_state",
]
