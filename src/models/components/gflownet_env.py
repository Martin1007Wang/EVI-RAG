from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

STOP_RELATION = -1
_EDGE_TAIL_INDEX = 1
_NODE_NONE = -1
_ZERO = 0
_ONE = 1


@dataclass
class GraphState:
    batch: Dict[str, Any]
    curr_nodes: torch.Tensor
    step_counts: torch.Tensor
    stopped: torch.Tensor
    max_steps: int
    mode: str

    def clone(self) -> GraphState:
        return GraphState(
            batch=self.batch,
            curr_nodes=self.curr_nodes.clone(),
            step_counts=self.step_counts.clone(),
            stopped=self.stopped.clone(),
            max_steps=self.max_steps,
            mode=self.mode,
        )


class GraphEnv(nn.Module):

    def __init__(self, max_steps: int) -> None:
        super().__init__()
        self.max_steps = int(max_steps)

    def reset(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        mode: str = "forward",
        init_node_locals: torch.Tensor | None = None,
        init_ptr: torch.Tensor | None = None,
        max_steps_override: int | None = None,
    ) -> GraphState:
        node_ptr = batch["node_ptr"]
        num_graphs = int(node_ptr.numel() - 1)
        mode_val = str(mode).lower()
        if init_node_locals is None:
            if mode_val not in {"forward", "backward"}:
                raise ValueError(f"Unsupported mode: {mode}")
            resolved_node_locals = batch["start_node_locals"]
            resolved_ptr = batch["start_ptr"]
        else:
            resolved_node_locals = init_node_locals
            resolved_ptr = init_ptr

        if resolved_ptr is None:
            raise ValueError("init_ptr must be provided when init_node_locals is overridden.")
        start_node_locals = resolved_node_locals.to(device=device, dtype=torch.long).view(-1)
        start_ptr = resolved_ptr.to(device=device, dtype=torch.long).view(-1)
        if start_ptr.numel() != num_graphs + _ONE:
            raise ValueError("start_ptr length mismatch with batch size.")
        if start_ptr.numel() > _ZERO:
            if int(start_ptr[_ZERO].item()) != _ZERO:
                raise ValueError("start_ptr must start at 0.")
            total = int(start_ptr[-_ONE].item())
            if total != start_node_locals.numel():
                raise ValueError("start_node_locals length mismatch with start_ptr.")
        start_deltas = start_ptr[_ONE:] - start_ptr[:-_ONE]
        if bool((start_deltas < _ZERO).any().item()):
            raise ValueError("start_ptr must be non-decreasing.")
        start_counts = start_deltas
        has_start = start_counts > _ZERO
        if bool((start_counts > _ONE).any().item()):
            multi_count = int((start_counts > _ONE).sum().item())
            raise ValueError(f"Multiple start nodes per graph ({multi_count} graphs); enforce single-start upstream.")
        start_nodes = torch.full((num_graphs,), _NODE_NONE, device=device, dtype=torch.long)
        if start_node_locals.numel() > _ZERO:
            start_indices = start_ptr[:-_ONE]
            safe_indices = torch.where(has_start, start_indices, torch.zeros_like(start_indices))
            chosen = start_node_locals.index_select(0, safe_indices)
            start_nodes = torch.where(has_start, chosen, start_nodes)

        curr_nodes = torch.where(
            has_start,
            start_nodes,
            torch.full_like(start_nodes, _NODE_NONE),
        )
        missing_start = ~has_start
        dummy_mask = batch["dummy_mask"].to(device=device, dtype=torch.bool)
        stopped = missing_start | dummy_mask
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        state_max_steps = self.max_steps
        if max_steps_override is not None:
            override = int(max_steps_override)
            if override > 0:
                state_max_steps = min(state_max_steps, override)
        return GraphState(
            batch=batch,
            curr_nodes=curr_nodes,
            step_counts=step_counts,
            stopped=stopped,
            max_steps=state_max_steps,
            mode=mode_val,
        )

    def step(
        self,
        state: GraphState,
        actions: torch.Tensor,
        *,
        step_index: int = 0,
    ) -> GraphState:
        next_curr_nodes = state.curr_nodes.clone()
        next_step_counts = state.step_counts.clone()
        next_stopped = state.stopped.clone()
        stop_actions = actions == STOP_RELATION
        move_actions = ~stop_actions
        next_stopped = next_stopped | stop_actions
        if move_actions.any():
            edge_index = state.batch["edge_index"]
            valid_moves = actions[move_actions]
            target_nodes = edge_index[_EDGE_TAIL_INDEX, valid_moves]
            active_graph_indices = torch.nonzero(move_actions, as_tuple=True)[0]
            next_curr_nodes.index_put_((active_graph_indices,), target_nodes)
        active_before = ~state.stopped
        next_step_counts = next_step_counts + active_before.long()

        # 6. Check Horizon
        horizon_reached = next_step_counts >= state.max_steps
        next_stopped = next_stopped | horizon_reached

        return GraphState(
            batch=state.batch,
            curr_nodes=next_curr_nodes,
            step_counts=next_step_counts,
            stopped=next_stopped,
            max_steps=state.max_steps,
            mode=state.mode,
        )


__all__ = [
    "GraphEnv",
    "GraphState",
    "STOP_RELATION",
]
