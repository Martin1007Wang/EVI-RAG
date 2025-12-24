from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

STOP_RELATION = -1
DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1


@dataclass
class GraphBatch:
    edge_index: torch.Tensor          # [2, E_total]
    edge_batch: torch.Tensor          # [E_total]
    edge_relations: torch.Tensor      # [E_total]
    edge_scores_norm: torch.Tensor    # [E_total]
    node_ptr: torch.Tensor            # [B+1]
    edge_ptr: torch.Tensor            # [B+1]
    start_node_locals: torch.Tensor   # [S_total]
    start_ptr: torch.Tensor           # [B+1]
    answer_node_locals: torch.Tensor  # [A_total]
    answer_ptr: torch.Tensor          # [B+1]
    node_is_start: torch.Tensor       # [N_total] bool
    node_is_answer: torch.Tensor      # [N_total] bool
    node_batch: torch.Tensor          # [N_total]


@dataclass
class GraphState:
    graph: GraphBatch
    active_nodes: torch.Tensor        # [N_total] bool
    visited_nodes: torch.Tensor       # [N_total] bool
    used_edge_mask: torch.Tensor      # [E_total] bool
    actions: torch.Tensor             # [B, max_steps+1] long (edge id or STOP_RELATION)
    directions: torch.Tensor          # [B, max_steps+1] long (DIRECTION_*)
    action_embeddings: torch.Tensor   # [B, max_steps, H]
    done: torch.Tensor                # [B] bool
    step_counts: torch.Tensor         # [B] long
    answer_hits: torch.Tensor         # [B] bool
    selection_order: torch.Tensor     # [E_total] long (step index or -1)


class GraphEnv(nn.Module):
    """Set-based graph environment with edge-level actions."""

    def __init__(
        self,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

    def reset(self, batch: Dict[str, torch.Tensor], *, device: torch.device) -> GraphState:
        edge_index = batch["edge_index"].to(device)
        edge_batch = batch["edge_batch"].to(device)
        edge_relations = batch["edge_relations"].to(device=device, dtype=torch.long)
        node_ptr = batch["node_ptr"].to(device)
        edge_ptr = batch["edge_ptr"].to(device)
        if "edge_scores" not in batch:
            raise ValueError("edge_scores missing in graph batch; g_agent cache must provide per-edge scores.")
        edge_scores = batch["edge_scores"].to(device=device, dtype=torch.float32).view(-1)
        start_node_locals = batch["start_node_locals"].to(device=device, dtype=torch.long)
        start_ptr = batch["start_ptr"].to(device)
        answer_node_locals = batch["answer_node_locals"].to(device=device, dtype=torch.long)
        answer_ptr = batch["answer_ptr"].to(device)
        if (start_node_locals < 0).any() or (answer_node_locals < 0).any():
            raise ValueError("node locals contain negative values; packed batching forbids padding.")

        num_edges = edge_index.size(1)
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("Graph batch must have positive num_graphs.")

        start_counts = start_ptr[1:] - start_ptr[:-1]
        if start_counts.numel() != num_graphs:
            raise ValueError("start_ptr has inconsistent length; ensure g_agent cache uses PyG slicing.")
        if (start_counts < 0).any():
            raise ValueError("start_ptr must be non-decreasing.")
        missing_start = start_counts == 0

        if edge_relations.numel() != num_edges:
            raise ValueError(f"edge_relations length {edge_relations.numel()} != num_edges {num_edges}")
        if edge_scores.numel() != num_edges:
            raise ValueError(f"edge_scores length {edge_scores.numel()} != num_edges {num_edges}")
        edge_scores_norm = edge_scores

        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= 0:
            raise ValueError("GraphEnv.reset received empty node_ptr; g_agent cache may be corrupt.")

        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)
        if node_batch.numel() != num_nodes_total:
            raise ValueError("node_batch length mismatch with node_ptr.")

        node_is_start = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals] = True

        active_nodes = node_is_start.clone()
        visited_nodes = active_nodes.clone()

        answer_hits = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        if active_nodes.any():
            hit_nodes = active_nodes & node_is_answer
            if hit_nodes.any():
                hit_batch = node_batch[hit_nodes]
                answer_hits = torch.bincount(hit_batch, minlength=num_graphs) > 0

        done = missing_start.to(device=device)
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        actions = torch.full((num_graphs, self.max_steps + 1), STOP_RELATION, dtype=torch.long, device=device)
        directions = torch.full(
            (num_graphs, self.max_steps + 1),
            DIRECTION_FORWARD,
            dtype=torch.long,
            device=device,
        )
        used_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        action_embeddings = torch.zeros(num_graphs, self.max_steps, 0, device=device)
        selection_order = torch.full((num_edges,), -1, dtype=torch.long, device=device)

        graph = GraphBatch(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            edge_scores_norm=edge_scores_norm,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
            answer_node_locals=answer_node_locals,
            answer_ptr=answer_ptr,
            node_is_start=node_is_start,
            node_is_answer=node_is_answer,
            node_batch=node_batch,
        )
        return GraphState(
            graph=graph,
            active_nodes=active_nodes,
            visited_nodes=visited_nodes,
            used_edge_mask=used_edge_mask,
            actions=actions,
            directions=directions,
            action_embeddings=action_embeddings,
            done=done,
            step_counts=step_counts,
            answer_hits=answer_hits,
            selection_order=selection_order,
        )

    def candidate_edge_mask(self, state: GraphState) -> torch.Tensor:
        forward_mask, backward_mask = self.candidate_edge_masks(state)
        return forward_mask | backward_mask

    def candidate_edge_masks(self, state: GraphState) -> tuple[torch.Tensor, torch.Tensor]:
        edge_batch = state.graph.edge_batch
        horizon_exhausted = state.step_counts[edge_batch] >= self.max_steps
        base = (~state.done[edge_batch]) & (~horizon_exhausted)
        heads = state.graph.edge_index[0]
        tails = state.graph.edge_index[1]
        forward_incident = state.active_nodes[heads]
        backward_incident = state.active_nodes[tails]
        forward_mask = base & forward_incident
        backward_mask = base & backward_incident
        return forward_mask, backward_mask

    def potential(self, state: GraphState, *, valid_edges_override: torch.Tensor | None = None) -> torch.Tensor:
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        if num_graphs <= 0:
            return torch.zeros(0, device=state.graph.edge_scores_norm.device, dtype=state.graph.edge_scores_norm.dtype)
        return torch.zeros(num_graphs, device=state.graph.edge_scores_norm.device, dtype=state.graph.edge_scores_norm.dtype)

    def step(
        self,
        state: GraphState,
        actions: torch.Tensor,
        action_embeddings: torch.Tensor,
        *,
        step_index: int,
    ) -> GraphState:
        device = state.graph.edge_index.device
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        if actions.numel() != num_graphs:
            raise ValueError(f"Actions length {actions.numel()} != num_graphs {num_graphs}")
        if action_embeddings.shape[0] != num_graphs:
            raise ValueError("action_embeddings batch mismatch with actions.")

        actions = actions.to(device=device, dtype=torch.long)
        is_stop = actions == STOP_RELATION
        is_stop = is_stop | state.done

        edge_batch = state.graph.edge_batch
        edge_index = state.graph.edge_index
        num_edges = int(edge_index.size(1))

        valid_action = ~is_stop
        if bool(valid_action.any().item()):
            if ((actions[valid_action] < 0) | (actions[valid_action] >= num_edges)).any():
                raise ValueError("Actions contain out-of-range edge ids.")
            graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
            act_graph = graph_ids[valid_action]
            act_edges = actions[valid_action]
            if (edge_batch[act_edges] != act_graph).any():
                raise ValueError("Actions must point to edges within their respective graphs.")

        candidate_mask = self.candidate_edge_mask(state)
        edge_selected = torch.zeros(num_edges, dtype=torch.bool, device=device)
        if bool(valid_action.any().item()):
            act_edges = actions[valid_action]
            if (~candidate_mask[act_edges]).any():
                raise ValueError("Actions contain edges that are not valid candidates for the current state.")
            edge_selected[act_edges] = True

        if edge_selected.any():
            state.used_edge_mask[edge_selected] = True
            state.selection_order[edge_selected] = int(step_index)

        step_directions = torch.full(
            (num_graphs,),
            DIRECTION_FORWARD,
            device=device,
            dtype=torch.long,
        )
        next_active = torch.zeros_like(state.active_nodes)
        if edge_selected.any():
            heads = edge_index[0, edge_selected]
            tails = edge_index[1, edge_selected]
            head_active = state.active_nodes[heads]
            tail_active = state.active_nodes[tails]
            if not bool((head_active | tail_active).any().item()):
                raise ValueError("Selected directed edge is not incident to any active node.")
            is_backward = (~head_active) & tail_active
            if is_backward.any():
                backward_graphs = edge_batch[edge_selected][is_backward]
                step_directions[backward_graphs] = DIRECTION_BACKWARD
            if head_active.any():
                next_active[tails[head_active]] = True
            if tail_active.any():
                next_active[heads[tail_active]] = True

        node_batch = state.graph.node_batch
        replace_graphs = (~is_stop) & (~state.done)
        if replace_graphs.any():
            replace_nodes = replace_graphs[node_batch]
            state.active_nodes = torch.where(replace_nodes, next_active, state.active_nodes)

        state.visited_nodes = state.visited_nodes | state.active_nodes

        if state.active_nodes.any():
            hit_nodes = state.active_nodes & state.graph.node_is_answer
            if hit_nodes.any():
                hit_batch = state.graph.node_batch[hit_nodes]
                hits = torch.bincount(hit_batch, minlength=num_graphs) > 0
                state.answer_hits = state.answer_hits | hits

        state.actions[:, step_index] = actions
        state.directions[:, step_index] = step_directions

        if state.action_embeddings.numel() == 0:
            state.action_embeddings = torch.zeros(
                num_graphs,
                self.max_steps,
                action_embeddings.size(-1),
                device=device,
                dtype=action_embeddings.dtype,
            )
        write_pos = state.step_counts.clamp(max=self.max_steps - 1)
        if (~is_stop).any():
            idx = torch.nonzero(~is_stop, as_tuple=False).view(-1)
            state.action_embeddings[idx, write_pos[idx]] = action_embeddings[idx]

        increment = (~is_stop).to(dtype=state.step_counts.dtype)
        state.step_counts = state.step_counts + increment
        horizon_exhausted = state.step_counts >= self.max_steps
        state.done = state.done | is_stop | horizon_exhausted
        return state


__all__ = [
    "GraphEnv",
    "GraphBatch",
    "GraphState",
    "STOP_RELATION",
    "DIRECTION_FORWARD",
    "DIRECTION_BACKWARD",
]
