from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch import nn

STOP_RELATION = -1
_EDGE_HEAD_INDEX = 0
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_NODE_NONE = -1


@dataclass
class GraphState:
    batch: Dict[str, Any]
    curr_nodes: torch.Tensor
    step_counts: torch.Tensor
    done: torch.Tensor
    max_steps: int
    mode: str
    target_mask: torch.Tensor
    traj_log_pf: list[torch.Tensor] = field(default_factory=list)
    traj_log_pb: list[torch.Tensor] = field(default_factory=list)
    traj_actions: list[torch.Tensor] = field(default_factory=list)


class GraphEnv(nn.Module):
    """Minimal path environment with edge-level actions and forward/backward modes."""

    def __init__(
        self,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}.")

    def reset(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        mode: str = "forward",
        init_node_locals: torch.Tensor | None = None,
        max_steps_override: int | None = None,
    ) -> GraphState:
        edge_index = batch["edge_index"]
        if edge_index.device != device:
            raise ValueError("GraphEnv.reset expects batch tensors already on the target device.")
        node_ptr = batch["node_ptr"]
        start_node_locals = batch["start_node_locals"]
        start_ptr = batch["start_ptr"]
        dummy_mask = batch["dummy_mask"]
        num_graphs = int(node_ptr.numel() - 1)
        start_counts = start_ptr[1:] - start_ptr[:-1]
        missing_start = start_counts == 0
        mode_val = str(mode).lower()
        if init_node_locals is None:
            if mode_val == "forward":
                init_node_locals = start_node_locals
            elif mode_val == "backward":
                init_node_locals = batch["target_node_locals"]
            else:
                raise ValueError(f"Unsupported mode for GraphEnv.reset: {mode!r}")
        init_ptr = start_ptr if mode_val == "forward" else batch["target_ptr"]
        start_nodes, has_start = GFlowNetBatchProcessor.compute_single_start_nodes(
            start_node_locals=init_node_locals,
            start_ptr=init_ptr,
            num_graphs=num_graphs,
            device=device,
        )
        curr_nodes = torch.where(
            has_start,
            start_nodes,
            torch.full_like(start_nodes, _NODE_NONE),
        )
        done = missing_start | dummy_mask.to(device=device, dtype=torch.bool)
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        if mode_val == "forward":
            target_mask = batch["node_is_target"].to(device=device, dtype=torch.bool)
        else:
            target_mask = batch["node_is_start"].to(device=device, dtype=torch.bool)

        state_max_steps = int(self.max_steps)
        if max_steps_override is not None:
            override = int(max_steps_override)
            if override <= 0:
                raise ValueError(f"max_steps_override must be > 0, got {override}.")
            state_max_steps = min(state_max_steps, override)
        return GraphState(
            batch=batch,
            curr_nodes=curr_nodes,
            step_counts=step_counts,
            done=done,
            max_steps=state_max_steps,
            mode=mode_val,
            target_mask=target_mask,
        )

    def forward_edge_mask(self, state: GraphState) -> torch.Tensor:
        edge_index = state.batch["edge_index"]
        edge_batch = state.batch["edge_batch"]
        if edge_batch.numel() == _ZERO:
            return torch.zeros((0,), device=edge_index.device, dtype=torch.bool)
        horizon_exhausted = state.step_counts[edge_batch] >= int(state.max_steps)
        base = (~state.done[edge_batch]) & (~horizon_exhausted)
        heads = edge_index[_EDGE_HEAD_INDEX]
        current_nodes = state.curr_nodes.index_select(0, edge_batch)
        return base & (heads == current_nodes)

    def step(
        self,
        state: GraphState,
        actions: torch.Tensor,
        *,
        step_index: int,
    ) -> GraphState:
        _ = step_index
        device = state.curr_nodes.device
        num_graphs = int(state.batch["node_ptr"].numel() - 1)
        actions = actions.to(device=device, dtype=torch.long)
        if actions.numel() != num_graphs:
            raise ValueError("actions length mismatch with batch size.")

        stop_action = actions == STOP_RELATION
        invalid_action = actions < STOP_RELATION
        if invalid_action.any():
            active_invalid = invalid_action & (~state.done)
            if active_invalid.any():
                raise ValueError("Invalid action encountered in GraphEnv.step.")

        move_mask = actions >= _ZERO
        if move_mask.any():
            edge_ids = actions[move_mask]
            edge_index = state.batch["edge_index"]
            edge_batch = state.batch["edge_batch"]
            edge_pairs = edge_index.index_select(1, edge_ids)
            heads = edge_pairs[_EDGE_HEAD_INDEX]
            tails = edge_pairs[_EDGE_TAIL_INDEX]
            edge_graph_ids = edge_batch.index_select(0, edge_ids)
            current_nodes = state.curr_nodes.index_select(0, edge_graph_ids)
            head_match = heads == current_nodes
            if head_match.numel() > _ZERO and not bool(head_match.all().detach().item()):
                raise ValueError("Directed env selected edge whose head is not current.")
            state.curr_nodes.index_copy_(0, edge_graph_ids, tails)

        active = ~state.done
        increment = (active & (~stop_action)).to(dtype=state.step_counts.dtype)
        state.step_counts = state.step_counts + increment
        horizon_exhausted = state.step_counts >= int(state.max_steps)
        state.done = state.done | stop_action | horizon_exhausted
        return state


__all__ = [
    "GraphEnv",
    "GraphState",
    "STOP_RELATION",
]
