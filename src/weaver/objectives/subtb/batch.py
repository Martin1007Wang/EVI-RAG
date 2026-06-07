from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.weaver.context import GraphContext
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.state import StateBatch


@dataclass(frozen=True, slots=True)
class SubTBTermTable:
    traj_ids: torch.Tensor
    start_steps: torch.Tensor
    end_steps: torch.Tensor
    start_state_ids: torch.Tensor
    end_state_ids: torch.Tensor
    lambda_exponent: torch.Tensor

    @property
    def num_terms(self) -> int:
        return int(self.traj_ids.numel())


@dataclass(frozen=True, slots=True)
class SubTBBatch:
    trajectories: TrajectoryBatch
    states: StateBatch
    prefix_state_ids: torch.Tensor
    valid_steps: torch.Tensor
    step_traj_ids: torch.Tensor
    step_ids: torch.Tensor
    step_parent_state_ids: torch.Tensor
    step_edge_ids: torch.Tensor
    terminal_state_ids: torch.Tensor
    terminal_trainable_stop_mask: torch.Tensor
    transition_terms: SubTBTermTable
    terminal_terms: SubTBTermTable
    step_child_state_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    onpolicy_transition_terms: SubTBTermTable | None = None
    replay_transition_terms: SubTBTermTable | None = None
    onpolicy_terminal_terms: SubTBTermTable | None = None
    replay_terminal_terms: SubTBTermTable | None = None

    def __post_init__(self) -> None:
        if self.onpolicy_transition_terms is None:
            object.__setattr__(self, "onpolicy_transition_terms", self.transition_terms)
        if self.replay_transition_terms is None:
            object.__setattr__(self, "replay_transition_terms", _empty_terms(device=self.prefix_state_ids.device))
        if self.onpolicy_terminal_terms is None:
            object.__setattr__(self, "onpolicy_terminal_terms", self.terminal_terms)
        if self.replay_terminal_terms is None:
            object.__setattr__(self, "replay_terminal_terms", _empty_terms(device=self.prefix_state_ids.device))
        if int(self.step_child_state_ids.numel()) == 0 and int(self.step_traj_ids.numel()) > 0:
            object.__setattr__(
                self,
                "step_child_state_ids",
                self.prefix_state_ids[self.step_traj_ids, self.step_ids + 1],
            )


def prepare_subtb_batch(
    *,
    trajectories: TrajectoryBatch,
    graph_context: GraphContext,
) -> SubTBBatch:
    budget = int(trajectories.budget)
    device = trajectories.device
    trajectory_count = trajectories.num_trajectories
    depth = torch.arange(budget + 1, device=device)
    prefix_valid = depth.view(1, -1).le(trajectories.edge_count.view(-1, 1))
    prefix_edges = trajectories.edge_ids.view(trajectory_count, 1, budget).expand(-1, budget + 1, -1)
    edge_pos = torch.arange(budget, device=device).view(1, 1, -1)
    prefix_edges = torch.where(edge_pos.lt(depth.view(1, -1, 1)), prefix_edges, -1)
    sentinel = max(int(graph_context.num_edges), 1)
    prefix_edges = torch.sort(torch.where(prefix_edges.lt(0), sentinel, prefix_edges), dim=2).values
    prefix_edges = torch.where(prefix_edges.eq(sentinel), -1, prefix_edges)
    prefix_graph_ids = trajectories.graph_ids.view(-1, 1).expand(-1, budget + 1)
    prefix_count = depth.view(1, -1).expand(trajectory_count, -1)
    state_keys = torch.cat(
        [prefix_graph_ids[prefix_valid].view(-1, 1), prefix_count[prefix_valid].view(-1, 1), prefix_edges[prefix_valid]],
        dim=1,
    )
    unique_state_keys, inverse_state = torch.unique(state_keys, dim=0, return_inverse=True)
    prefix_state_ids = torch.full((trajectory_count, budget + 1), -1, dtype=torch.long, device=device)
    prefix_state_ids[prefix_valid] = inverse_state
    states = StateBatch.from_selected_edges(
        graph_ids=unique_state_keys[:, 0],
        edge_ids=unique_state_keys[:, 2:],
        edge_count=unique_state_keys[:, 1],
        budget=budget,
        graph_context=graph_context,
    )
    valid_steps = trajectories.valid_edge_mask()
    step_positions = valid_steps.nonzero(as_tuple=False)
    if int(step_positions.numel()) == 0:
        step_traj_ids = torch.empty(0, dtype=torch.long, device=device)
        step_ids = torch.empty(0, dtype=torch.long, device=device)
        step_parent_state_ids = torch.empty(0, dtype=torch.long, device=device)
        step_child_state_ids = torch.empty(0, dtype=torch.long, device=device)
        step_edge_ids = torch.empty(0, dtype=torch.long, device=device)
    else:
        step_traj_ids = step_positions[:, 0]
        step_ids = step_positions[:, 1]
        step_parent_state_ids = prefix_state_ids[step_traj_ids, step_ids]
        step_child_state_ids = prefix_state_ids[step_traj_ids, step_ids + 1]
        step_edge_ids = trajectories.edge_ids[step_traj_ids, step_ids]
        order = torch.argsort(step_parent_state_ids)
        step_traj_ids = step_traj_ids.index_select(0, order)
        step_ids = step_ids.index_select(0, order)
        step_parent_state_ids = step_parent_state_ids.index_select(0, order)
        step_child_state_ids = step_child_state_ids.index_select(0, order)
        step_edge_ids = step_edge_ids.index_select(0, order)

    terminal_step_by_traj = trajectories.edge_count.to(dtype=torch.long)
    traj_rows = torch.arange(trajectory_count, device=device)
    terminal_state_ids = prefix_state_ids[traj_rows, terminal_step_by_traj]
    trajectory_rows = trajectories.is_policy | trajectories.is_replay
    terminal_trainable_stop_mask = states.edge_count.index_select(0, terminal_state_ids).gt(0)

    transition_terms = _build_transition_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectory_rows,
    )
    onpolicy_transition_terms = _build_transition_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectories.is_policy,
    )
    replay_transition_terms = _build_transition_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectories.is_replay,
    )
    terminal_terms = _build_terminal_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectory_rows,
    )
    onpolicy_terminal_terms = _build_terminal_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectories.is_policy,
    )
    replay_terminal_terms = _build_terminal_terms(
        prefix_state_ids=prefix_state_ids,
        lengths=terminal_step_by_traj,
        row_mask=trajectories.is_replay,
    )

    return SubTBBatch(
        trajectories=trajectories,
        states=states,
        prefix_state_ids=prefix_state_ids,
        valid_steps=valid_steps,
        step_traj_ids=step_traj_ids,
        step_ids=step_ids,
        step_parent_state_ids=step_parent_state_ids,
        step_child_state_ids=step_child_state_ids,
        step_edge_ids=step_edge_ids,
        terminal_state_ids=terminal_state_ids,
        terminal_trainable_stop_mask=terminal_trainable_stop_mask,
        transition_terms=transition_terms,
        terminal_terms=terminal_terms,
        onpolicy_transition_terms=onpolicy_transition_terms,
        replay_transition_terms=replay_transition_terms,
        onpolicy_terminal_terms=onpolicy_terminal_terms,
        replay_terminal_terms=replay_terminal_terms,
    )


def _build_transition_terms(
    *,
    prefix_state_ids: torch.Tensor,
    lengths: torch.Tensor,
    row_mask: torch.Tensor,
) -> SubTBTermTable:
    width = int(prefix_state_ids.size(1))
    steps = torch.arange(width, dtype=torch.long, device=prefix_state_ids.device)
    start_grid = steps.view(1, width, 1)
    end_grid = steps.view(1, 1, width)
    span = end_grid - start_grid
    valid = row_mask.view(-1, 1, 1) & span.gt(0) & end_grid.le(lengths.view(-1, 1, 1))

    traj_ids, start_steps, end_steps = valid.nonzero(as_tuple=True)
    return SubTBTermTable(
        traj_ids=traj_ids,
        start_steps=start_steps,
        end_steps=end_steps,
        start_state_ids=prefix_state_ids[traj_ids, start_steps],
        end_state_ids=prefix_state_ids[traj_ids, end_steps],
        lambda_exponent=(end_steps - start_steps - 1).clamp_min(0),
    )


def _build_terminal_terms(
    *,
    prefix_state_ids: torch.Tensor,
    lengths: torch.Tensor,
    row_mask: torch.Tensor,
) -> SubTBTermTable:
    width = int(prefix_state_ids.size(1))
    steps = torch.arange(width, dtype=torch.long, device=prefix_state_ids.device)
    start_grid = steps.view(1, width, 1)
    end_grid = steps.view(1, 1, width)
    span = end_grid - start_grid
    valid = row_mask.view(-1, 1, 1) & span.ge(0) & end_grid.le(lengths.view(-1, 1, 1)) & end_grid.gt(0)

    traj_ids, start_steps, end_steps = valid.nonzero(as_tuple=True)
    return SubTBTermTable(
        traj_ids=traj_ids,
        start_steps=start_steps,
        end_steps=end_steps,
        start_state_ids=prefix_state_ids[traj_ids, start_steps],
        end_state_ids=prefix_state_ids[traj_ids, end_steps],
        lambda_exponent=(end_steps - start_steps).clamp_min(1) - 1,
    )


def _empty_terms(*, device: torch.device) -> SubTBTermTable:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return SubTBTermTable(
        traj_ids=empty,
        start_steps=empty,
        end_steps=empty,
        start_state_ids=empty,
        end_state_ids=empty,
        lambda_exponent=empty,
    )
