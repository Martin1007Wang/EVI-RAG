from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

STOP_RELATION = -1  # Sentinel action id for graph-level stop.
STOP_EDGE_RELATION = STOP_RELATION  # Legacy stop-edge relation id (should not appear in data).
DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1
_ZERO = 0


@dataclass
class GraphBatch:
    edge_index: torch.Tensor          # [2, E_total]
    edge_batch: torch.Tensor          # [E_total]
    edge_relations: torch.Tensor      # [E_total]
    question_tokens: torch.Tensor     # [B, H]
    node_tokens: torch.Tensor         # [N_total, H]
    node_ptr: torch.Tensor            # [B+1]
    edge_ptr: torch.Tensor            # [B+1]
    start_node_locals: torch.Tensor   # [S_total] batched node indices
    start_ptr: torch.Tensor           # [B+1]
    answer_node_locals: torch.Tensor  # [A_total] batched node indices
    answer_ptr: torch.Tensor          # [B+1]
    node_is_start: torch.Tensor       # [N_total] bool
    node_is_answer: torch.Tensor      # [N_total] bool
    node_batch: torch.Tensor          # [N_total]
    node_in_degree: torch.Tensor      # [N_total]


@dataclass
class GraphState:
    graph: GraphBatch
    active_nodes: torch.Tensor        # [N_total] bool
    visited_nodes: torch.Tensor       # [N_total] bool
    used_edge_mask: torch.Tensor      # [E_total] bool
    actions: torch.Tensor             # [B, max_steps+1] long (edge id or STOP_RELATION sentinel)
    directions: torch.Tensor          # [B, max_steps+1] long (DIRECTION_*)
    done: torch.Tensor                # [B] bool
    step_counts: torch.Tensor         # [B] long
    answer_hits: torch.Tensor         # [B] bool
    answer_node_hit: torch.Tensor     # [B] long (local node id, -1 if none)
    start_node_hit: torch.Tensor      # [B] long (local node id, -1 if none)


class GraphEnv(nn.Module):
    """Set-based graph environment with edge-level actions."""

    def __init__(
        self,
        max_steps: int,
        stop_on_answer: bool = False,
        min_stop_steps: int = 0,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.stop_on_answer = bool(stop_on_answer)
        self.min_stop_steps = int(min_stop_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.min_stop_steps < 0:
            raise ValueError(f"min_stop_steps must be >= 0, got {self.min_stop_steps}")
        if self.min_stop_steps > self.max_steps:
            raise ValueError(
                f"min_stop_steps must be <= max_steps ({self.max_steps}), got {self.min_stop_steps}"
            )

    def reset(self, batch: Dict[str, Any], *, device: torch.device) -> GraphState:
        edge_index = batch["edge_index"].to(device)
        edge_batch = batch["edge_batch"].to(device)
        edge_relations = batch["edge_relations"].to(device=device, dtype=torch.long)
        question_tokens = batch["question_tokens"].to(device=device, dtype=torch.float32)
        node_tokens = batch["node_tokens"].to(device=device, dtype=torch.float32)
        node_ptr = batch["node_ptr"].to(device)
        edge_ptr = batch["edge_ptr"].to(device)
        start_node_locals = batch["start_node_locals"].to(device=device, dtype=torch.long)
        start_ptr = batch["start_ptr"].to(device)
        answer_node_locals = batch["answer_node_locals"].to(device=device, dtype=torch.long)
        answer_ptr = batch["answer_ptr"].to(device)
        num_edges = edge_index.size(1)
        num_graphs = int(node_ptr.numel() - 1)
        start_counts = start_ptr[1:] - start_ptr[:-1]
        missing_start = start_counts == 0
        dummy_mask = batch["dummy_mask"].to(device=device, dtype=torch.bool)

        num_nodes_total = int(node_ptr[-1].item())
        node_batch = batch["node_batch"].to(device=device, dtype=torch.long)
        node_in_degree = batch["node_in_degree"].to(device=device, dtype=torch.long)
        node_is_start = batch["node_is_start"].to(device=device, dtype=torch.bool)
        node_is_answer = batch["node_is_answer"].to(device=device, dtype=torch.bool)

        active_nodes = node_is_start.clone()
        visited_nodes = active_nodes.clone()

        answer_hits = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        answer_node_hit = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        if active_nodes.any():
            hit_nodes = active_nodes & node_is_answer
            if hit_nodes.any():
                hit_idx = torch.nonzero(hit_nodes, as_tuple=False).view(-1)
                hit_batch = node_batch[hit_idx]
                local_idx = hit_idx - node_ptr[hit_batch]
                sentinel = int(num_nodes_total) + 1
                min_local = torch.full((num_graphs,), sentinel, device=device, dtype=torch.long)
                min_local.scatter_reduce_(0, hit_batch, local_idx, reduce="amin", include_self=True)
                has_hit = min_local != sentinel
                answer_hits = has_hit
                answer_node_hit = torch.where(has_hit, min_local, answer_node_hit)

        start_node_hit = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        if answer_hits.any():
            start_node_hit = torch.where(answer_hits, answer_node_hit, start_node_hit)

        graph = GraphBatch(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            question_tokens=question_tokens,
            node_tokens=node_tokens,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
            answer_node_locals=answer_node_locals,
            answer_ptr=answer_ptr,
            node_is_start=node_is_start,
            node_is_answer=node_is_answer,
            node_batch=node_batch,
            node_in_degree=node_in_degree,
        )

        state_cache = batch.get("state_cache")
        if isinstance(state_cache, GraphState):
            reuse = (
                state_cache.actions.shape == (num_graphs, self.max_steps + 1)
                and state_cache.directions.shape == (num_graphs, self.max_steps + 1)
                and state_cache.active_nodes.numel() == num_nodes_total
                and state_cache.used_edge_mask.numel() == num_edges
                and state_cache.step_counts.numel() == num_graphs
            )
        else:
            reuse = False

        if not reuse:
            done = missing_start.to(device=device) | dummy_mask
            if self.stop_on_answer:
                done = done | answer_hits
            step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
            actions = torch.full((num_graphs, self.max_steps + 1), STOP_RELATION, dtype=torch.long, device=device)
            directions = torch.full(
                (num_graphs, self.max_steps + 1),
                DIRECTION_FORWARD,
                dtype=torch.long,
                device=device,
            )
            used_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
            return GraphState(
                graph=graph,
                active_nodes=active_nodes,
                visited_nodes=visited_nodes,
                used_edge_mask=used_edge_mask,
                actions=actions,
                directions=directions,
                done=done,
                step_counts=step_counts,
                answer_hits=answer_hits,
                answer_node_hit=answer_node_hit,
                start_node_hit=start_node_hit,
            )

        state = state_cache
        state.graph = graph
        if state.active_nodes.shape == active_nodes.shape:
            state.active_nodes.copy_(active_nodes)
        else:
            state.active_nodes = active_nodes.clone()
        if state.visited_nodes.shape == visited_nodes.shape:
            state.visited_nodes.copy_(visited_nodes)
        else:
            state.visited_nodes = visited_nodes.clone()
        state.used_edge_mask.zero_()
        state.actions.fill_(STOP_RELATION)
        state.directions.fill_(DIRECTION_FORWARD)
        done = missing_start.to(device=device) | dummy_mask
        if self.stop_on_answer:
            done = done | answer_hits
        state.done.copy_(done)
        state.step_counts.zero_()
        if state.answer_hits.shape == answer_hits.shape:
            state.answer_hits.copy_(answer_hits)
        else:
            state.answer_hits = answer_hits.clone()
        if state.answer_node_hit.shape == answer_node_hit.shape:
            state.answer_node_hit.copy_(answer_node_hit)
        else:
            state.answer_node_hit = answer_node_hit.clone()
        if state.start_node_hit.shape == start_node_hit.shape:
            state.start_node_hit.copy_(start_node_hit)
        else:
            state.start_node_hit = start_node_hit.clone()
        return state

    def candidate_edge_mask(self, state: GraphState) -> torch.Tensor:
        forward_mask, backward_mask = self.candidate_edge_masks(state)
        return forward_mask | backward_mask

    def candidate_edge_masks(self, state: GraphState) -> tuple[torch.Tensor, torch.Tensor]:
        edge_batch = state.graph.edge_batch
        horizon_exhausted = state.step_counts[edge_batch] >= self.max_steps
        base = (~state.done[edge_batch]) & (~horizon_exhausted)
        heads = state.graph.edge_index[0]
        tails = state.graph.edge_index[1]
        active = state.active_nodes
        visited = state.visited_nodes
        head_active = active[heads]
        next_from_head = head_active & (~visited[tails])
        move_mask = base & next_from_head
        # Directed action space: only forward edges are valid.
        return move_mask, torch.zeros_like(move_mask, dtype=torch.bool)

    def step(
        self,
        state: GraphState,
        actions: torch.Tensor,
        *,
        step_index: int,
    ) -> GraphState:
        device = state.graph.edge_index.device
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        actions = actions.to(device=device, dtype=torch.long)
        valid_action = actions >= _ZERO
        is_stop_edge = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        if valid_action.any():
            edge_ids = actions[valid_action]
            rel_ids = state.graph.edge_relations[edge_ids]
            is_stop_edge[valid_action] = rel_ids == STOP_EDGE_RELATION
        is_stop = state.done | is_stop_edge | (~valid_action)

        edge_batch = state.graph.edge_batch
        edge_index = state.graph.edge_index
        num_edges = int(edge_index.size(1))

        move_action = valid_action & (~is_stop_edge)
        act_edges = actions[move_action]
        edge_selected = torch.zeros(num_edges, dtype=torch.bool, device=device)
        edge_selected[act_edges] = True
        state.used_edge_mask = state.used_edge_mask | edge_selected

        step_directions = torch.full((num_graphs,), DIRECTION_FORWARD, device=device, dtype=torch.long)
        next_active = torch.zeros_like(state.active_nodes)
        heads = edge_index[0, edge_selected]
        tails = edge_index[1, edge_selected]
        head_active = state.active_nodes[heads]
        if bool((~head_active).any().item()):
            raise ValueError("Directed env selected edge whose head is not active.")
        if step_index == 0:
            chosen_start = heads
            edge_graph_ids = edge_batch[edge_selected]
            local_start = chosen_start - state.graph.node_ptr[edge_graph_ids]
            state.start_node_hit[edge_graph_ids] = local_start
        move_from_head = head_active
        next_active[tails[move_from_head]] = True

        node_batch = state.graph.node_batch
        replace_graphs = (~is_stop) & (~state.done)
        if replace_graphs.any():
            replace_nodes = replace_graphs[node_batch]
            state.active_nodes = torch.where(replace_nodes, next_active, state.active_nodes)

        state.visited_nodes = state.visited_nodes | state.active_nodes

        hit_nodes = state.active_nodes & state.graph.node_is_answer
        hit_idx = torch.nonzero(hit_nodes, as_tuple=False).view(-1)
        hit_batch = state.graph.node_batch[hit_idx]
        local_idx = hit_idx - state.graph.node_ptr[hit_batch]
        sentinel = int(state.graph.node_ptr[-1].item()) + 1
        min_local = torch.full((num_graphs,), sentinel, device=device, dtype=torch.long)
        min_local.scatter_reduce_(0, hit_batch, local_idx, reduce="amin", include_self=True)
        has_hit = min_local != sentinel
        newly_hit = (~state.answer_hits) & has_hit
        state.answer_node_hit = torch.where(newly_hit, min_local, state.answer_node_hit)
        state.answer_hits = state.answer_hits | has_hit

        state.actions[:, step_index] = actions
        state.directions[:, step_index] = step_directions
        increment = (~is_stop).to(dtype=state.step_counts.dtype)
        state.step_counts = state.step_counts + increment
        horizon_exhausted = state.step_counts >= self.max_steps
        done = state.done | is_stop | horizon_exhausted
        if self.stop_on_answer:
            done = done | state.answer_hits
        state.done = done
        return state


__all__ = [
    "GraphEnv",
    "GraphBatch",
    "GraphState",
    "STOP_RELATION",
    "STOP_EDGE_RELATION",
    "DIRECTION_FORWARD",
    "DIRECTION_BACKWARD",
]
