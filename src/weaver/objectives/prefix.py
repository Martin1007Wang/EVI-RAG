from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches


@dataclass(frozen=True, slots=True)
class ExpansionPrefixBatch:
    """
    Ordered-prefix expansion events derived from completed trajectories.

    For item i:

        parent[i] --edge_ids[i]--> child[i]
    """

    parent: StateBatch
    child: StateBatch
    edge_ids: torch.Tensor
    traj_ids: torch.Tensor
    step_ids: torch.Tensor
    source: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.edge_ids.device

    @property
    def num_items(self) -> int:
        return int(self.edge_ids.numel())


@dataclass(frozen=True, slots=True)
class TerminalPrefixBatch:
    """
    Terminal prefix states for completed trajectories.

    Every trajectory contributes exactly one terminal state. ``reason`` keeps
    policy stop, no-frontier, and budget boundary terminals distinct for
    diagnostics; the objective treats each row as a terminal boundary.
    """

    state: StateBatch
    traj_ids: torch.Tensor
    step_ids: torch.Tensor
    reason: torch.Tensor
    source: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.traj_ids.device

    @property
    def num_items(self) -> int:
        return int(self.traj_ids.numel())


@dataclass(frozen=True, slots=True)
class PrefixBatch:
    """
    Loss-facing ordered-prefix view derived from trajectory records.

    This object is not a rollout record. It exists only to feed objective
    construction.
    """

    expansions: ExpansionPrefixBatch
    terminals: TerminalPrefixBatch

    @property
    def num_expansions(self) -> int:
        return self.expansions.num_items

    @property
    def num_terminals(self) -> int:
        return self.terminals.num_items

    @property
    def num_items(self) -> int:
        return int(self.num_expansions + self.num_terminals)


def build_prefix_batch(
    trajectories: TrajectoryBatch,
) -> PrefixBatch:
    """
    Reconstruct ordered prefix states from completed trajectories.

    Legality is owned by rollout/replay construction. This function only
    replays recorded physical edge ids into StateBatch prefixes.
    """

    device = trajectories.device
    budget = trajectories.budget

    state = StateBatch.initial(
        graph_ids=trajectories.graph_ids,
        budget=budget,
    )

    traj_ids = torch.arange(
        trajectories.num_trajectories,
        dtype=torch.long,
        device=device,
    )

    parent_states: list[StateBatch] = []
    child_states: list[StateBatch] = []
    action_edge_ids: list[torch.Tensor] = []
    action_traj_ids: list[torch.Tensor] = []
    action_step_ids: list[torch.Tensor] = []
    action_source: list[torch.Tensor] = []

    for step in range(budget):
        rows = trajectories.edge_count.gt(step).nonzero(as_tuple=False).flatten()

        if int(rows.numel()) == 0:
            continue

        edge_ids = trajectories.edge_ids.index_select(0, rows)[:, step]

        parent = state.take(rows)
        local_rows = torch.arange(
            parent.num_states,
            dtype=torch.long,
            device=device,
        )
        child = parent.advance(
            ExpansionBatch(
                state_ids=local_rows,
                edge_ids=edge_ids,
            )
        )

        parent_states.append(parent)
        child_states.append(child)
        action_edge_ids.append(edge_ids)
        action_traj_ids.append(traj_ids.index_select(0, rows))
        action_step_ids.append(
            torch.full(
                (int(rows.numel()),),
                int(step),
                dtype=torch.long,
                device=device,
            )
        )
        action_source.append(trajectories.source.index_select(0, rows))

        state = state.advance(
            ExpansionBatch(
                state_ids=rows,
                edge_ids=edge_ids,
            )
        )

    return PrefixBatch(
        expansions=ExpansionPrefixBatch(
            parent=(cat_state_batches(parent_states) if parent_states else _empty_state(device=device, budget=budget)),
            child=(cat_state_batches(child_states) if child_states else _empty_state(device=device, budget=budget)),
            edge_ids=(torch.cat(action_edge_ids, dim=0) if action_edge_ids else _empty_long(device)),
            traj_ids=(torch.cat(action_traj_ids, dim=0) if action_traj_ids else _empty_long(device)),
            step_ids=(torch.cat(action_step_ids, dim=0) if action_step_ids else _empty_long(device)),
            source=(torch.cat(action_source, dim=0) if action_source else _empty_long(device)),
        ),
        terminals=TerminalPrefixBatch(
            state=state.take(traj_ids),
            traj_ids=traj_ids,
            step_ids=trajectories.edge_count,
            reason=trajectories.stop_reason,
            source=trajectories.source,
        ),
    )


def _empty_state(
    *,
    device: torch.device,
    budget: int,
) -> StateBatch:
    return StateBatch.initial(
        graph_ids=torch.empty(
            0,
            dtype=torch.long,
            device=device,
        ),
        budget=int(budget),
    )


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )


__all__ = [
    "ExpansionPrefixBatch",
    "PrefixBatch",
    "TerminalPrefixBatch",
    "build_prefix_batch",
]
