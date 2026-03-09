from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.environment.state import DynamicAgentState, RECENT_NODE_WINDOW

_PAD_VALUE = -1


def _flat_prefix_lengths(num_moves: torch.Tensor) -> torch.Tensor:
    return num_moves.view(-1).to(dtype=torch.long) + 1


def _ensure_prefix_tensors(
    *,
    current_node: torch.Tensor,
    num_moves: torch.Tensor,
    path_nodes: torch.Tensor | None,
    path_edge_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path_nodes is None or path_edge_ids is None:
        if bool((num_moves != 0).any().item()):
            raise ValueError(
                "TrajectoryState with num_moves > 0 must carry explicit prefix tensors."
            )
        total_agents = int(current_node.numel())
        node_width = 2
        prefix_nodes = torch.full(
            (total_agents, node_width),
            fill_value=_PAD_VALUE,
            device=current_node.device,
            dtype=torch.long,
        )
        prefix_nodes[:, 0] = current_node.view(-1)
        prefix_edges = torch.full(
            (total_agents, node_width - 1),
            fill_value=_PAD_VALUE,
            device=current_node.device,
            dtype=torch.long,
        )
        return prefix_nodes, prefix_edges
    flat_nodes = path_nodes.view(-1, int(path_nodes.size(-1))).to(dtype=torch.long)
    flat_edges = path_edge_ids.view(-1, int(path_edge_ids.size(-1))).to(
        dtype=torch.long
    )
    return flat_nodes, flat_edges


def _recent_visited_from_prefix(
    *, prefix_nodes: torch.Tensor, prefix_lengths: torch.Tensor
) -> torch.Tensor:
    total_agents = int(prefix_nodes.size(0))
    recent = torch.full(
        (total_agents, RECENT_NODE_WINDOW),
        fill_value=_PAD_VALUE,
        device=prefix_nodes.device,
        dtype=torch.long,
    )
    if total_agents == 0:
        return recent
    row_ids = torch.arange(total_agents, device=prefix_nodes.device, dtype=torch.long)
    for offset in range(RECENT_NODE_WINDOW):
        token_pos = prefix_lengths - 1 - offset
        valid = token_pos >= 0
        if not bool(valid.any().item()):
            continue
        safe_pos = token_pos.clamp(min=0)
        values = prefix_nodes[row_ids, safe_pos]
        recent[:, offset] = torch.where(valid, values, recent[:, offset])
    return recent


@dataclass(frozen=True)
class TrajectoryState:
    step_t: int
    current_node: torch.Tensor
    done_mask: torch.Tensor
    num_moves: torch.Tensor
    path_nodes: torch.Tensor | None = None
    path_edge_ids: torch.Tensor | None = None

    @classmethod
    def initialize(
        cls, *, start_nodes: torch.Tensor, max_steps: int
    ) -> "TrajectoryState":
        if start_nodes.dim() != 2:
            raise ValueError(
                "start_nodes must be 2D [num_graphs, num_rollouts], "
                f"got shape={tuple(start_nodes.shape)}."
            )
        prefix_nodes = torch.full(
            (*start_nodes.shape, max_steps + 1),
            fill_value=_PAD_VALUE,
            device=start_nodes.device,
            dtype=torch.long,
        )
        prefix_nodes[..., 0] = start_nodes
        prefix_edges = torch.full(
            (*start_nodes.shape, max_steps),
            fill_value=_PAD_VALUE,
            device=start_nodes.device,
            dtype=torch.long,
        )
        return cls(
            step_t=0,
            current_node=start_nodes.clone(),
            done_mask=torch.zeros_like(start_nodes, dtype=torch.bool),
            num_moves=torch.zeros_like(start_nodes, dtype=torch.long),
            path_nodes=prefix_nodes,
            path_edge_ids=prefix_edges,
        )

    @classmethod
    def from_edge_path(
        cls,
        *,
        start_node: int,
        edge_ids: tuple[int, ...],
        edge_index: torch.Tensor,
        max_steps: int,
        device: torch.device,
    ) -> "TrajectoryState":
        num_moves = len(edge_ids)
        if num_moves > max_steps:
            raise ValueError("edge path length cannot exceed max_steps.")
        prefix_nodes = torch.full(
            (1, 1, max_steps + 1),
            fill_value=_PAD_VALUE,
            device=device,
            dtype=torch.long,
        )
        prefix_edges = torch.full(
            (1, 1, max_steps),
            fill_value=_PAD_VALUE,
            device=device,
            dtype=torch.long,
        )
        prefix_nodes[0, 0, 0] = int(start_node)
        current_node = int(start_node)
        for move_idx, edge_id in enumerate(edge_ids):
            edge_id_int = int(edge_id)
            edge_src = int(edge_index[0, edge_id_int].item())
            edge_dst = int(edge_index[1, edge_id_int].item())
            if edge_src != current_node:
                raise ValueError("edge path is not source-contiguous.")
            prefix_edges[0, 0, move_idx] = edge_id_int
            prefix_nodes[0, 0, move_idx + 1] = edge_dst
            current_node = edge_dst
        return cls(
            step_t=num_moves,
            current_node=torch.tensor(
                [[current_node]], device=device, dtype=torch.long
            ),
            done_mask=torch.zeros((1, 1), device=device, dtype=torch.bool),
            num_moves=torch.tensor([[num_moves]], device=device, dtype=torch.long),
            path_nodes=prefix_nodes,
            path_edge_ids=prefix_edges,
        )

    def flatten_current(self) -> torch.Tensor:
        return self.current_node.view(-1)

    def flatten_done(self) -> torch.Tensor:
        return self.done_mask.view(-1)

    def flatten_num_moves(self) -> torch.Tensor:
        return self.num_moves.view(-1)

    def flatten_path_nodes(self) -> torch.Tensor:
        prefix_nodes, _ = _ensure_prefix_tensors(
            current_node=self.current_node,
            num_moves=self.num_moves,
            path_nodes=self.path_nodes,
            path_edge_ids=self.path_edge_ids,
        )
        return prefix_nodes

    def flatten_path_edge_ids(self) -> torch.Tensor:
        _, prefix_edges = _ensure_prefix_tensors(
            current_node=self.current_node,
            num_moves=self.num_moves,
            path_nodes=self.path_nodes,
            path_edge_ids=self.path_edge_ids,
        )
        return prefix_edges

    def flatten_prefix_lengths(self) -> torch.Tensor:
        return _flat_prefix_lengths(self.num_moves)

    def flatten_previous_nodes(self) -> torch.Tensor:
        prefix_nodes = self.flatten_path_nodes()
        total_agents = int(prefix_nodes.size(0))
        prev_nodes = torch.full(
            (total_agents,),
            fill_value=_PAD_VALUE,
            device=prefix_nodes.device,
            dtype=torch.long,
        )
        flat_num_moves = self.flatten_num_moves()
        valid = flat_num_moves > 0
        if bool(valid.any().item()):
            row_ids = torch.arange(
                total_agents, device=prefix_nodes.device, dtype=torch.long
            )
            prev_nodes[valid] = prefix_nodes[row_ids[valid], flat_num_moves[valid] - 1]
        return prev_nodes

    def flatten_incoming_edge_ids(self) -> torch.Tensor:
        prefix_edges = self.flatten_path_edge_ids()
        total_agents = int(prefix_edges.size(0))
        incoming = torch.full(
            (total_agents,),
            fill_value=_PAD_VALUE,
            device=prefix_edges.device,
            dtype=torch.long,
        )
        flat_num_moves = self.flatten_num_moves()
        valid = flat_num_moves > 0
        if bool(valid.any().item()):
            row_ids = torch.arange(
                total_agents, device=prefix_edges.device, dtype=torch.long
            )
            incoming[valid] = prefix_edges[row_ids[valid], flat_num_moves[valid] - 1]
        return incoming

    def build_recent_visited_mask(self) -> torch.Tensor:
        return _recent_visited_from_prefix(
            prefix_nodes=self.flatten_path_nodes(),
            prefix_lengths=self.flatten_prefix_lengths(),
        )

    def build_child_prefix_tensors(
        self,
        *,
        edge_agent_batch: torch.Tensor,
        target_nodes: torch.Tensor,
        chosen_edge_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_nodes = self.flatten_path_nodes().index_select(0, edge_agent_batch)
        prefix_edges = self.flatten_path_edge_ids().index_select(0, edge_agent_batch)
        next_num_moves = self.flatten_num_moves().index_select(0, edge_agent_batch) + 1
        row_ids = torch.arange(
            int(edge_agent_batch.numel()), device=target_nodes.device
        )
        prefix_edges[row_ids, next_num_moves - 1] = chosen_edge_ids.to(dtype=torch.long)
        prefix_nodes[row_ids, next_num_moves] = target_nodes.to(dtype=torch.long)
        return prefix_nodes, prefix_edges, next_num_moves

    def as_dynamic_agent_state(
        self, *, question_emb: torch.Tensor
    ) -> DynamicAgentState:
        if question_emb.dim() != 2:
            raise ValueError(
                "question_emb must be 2D [num_graphs, d] when converting state."
            )
        num_graphs, num_rollouts = self.current_node.shape
        if int(question_emb.size(0)) != num_graphs:
            raise ValueError(
                "question_emb batch mismatch with TrajectoryState: "
                f"question_emb={int(question_emb.size(0))}, num_graphs={num_graphs}."
            )
        device = self.current_node.device
        return DynamicAgentState(
            step_t=self.step_t,
            current_nodes=self.current_node,
            flow_direction="forward",
            hidden_states=question_emb.unsqueeze(1).expand(
                num_graphs, num_rollouts, -1
            ),
            visited_mask=self.build_recent_visited_mask(),
            cumulative_rewards=torch.zeros(
                (num_graphs, num_rollouts), device=device, dtype=torch.float32
            ),
            done_mask=self.done_mask,
            num_moves=self.num_moves,
            path_token_ids=None,
            path_token_types=None,
            path_lengths=None,
        )
