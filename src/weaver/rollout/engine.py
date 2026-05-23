from __future__ import annotations

import torch

from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy import ForwardPolicy
from src.weaver.state import Frontier, State
from src.weaver.transition import ExpansionBatch, SampleMeta, SRC_UNKNOWN, TerminalBatch, TrainingBatch

from .action import StepAction, sample_step
from .result import RolloutResult
from .tape import RolloutTape


class RolloutEngine:
    """
    Vectorized finite-horizon rollout engine.
    """

    def __init__(self, expand_budget: int) -> None:
        self.expand_budget = int(expand_budget)

    def sample_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        num_rollouts: int,
    ) -> tuple[list[RolloutResult], TrainingBatch | None]:
        with torch.no_grad():
            fused, training = self.sample_fused_rollouts(
                policy=policy,
                context=context,
                features=features,
                rollouts_per_graph=int(num_rollouts),
            )

        return (
            split_fused_rollouts(
                fused=fused,
                rollouts_per_graph=int(num_rollouts),
                num_graphs=int(context.num_graphs),
            ),
            training,
        )

    def sample_fused_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        rollouts_per_graph: int,
    ) -> tuple[RolloutResult, TrainingBatch | None]:
        graph_ids = torch.arange(
            int(context.num_graphs),
            dtype=torch.long,
            device=context.device,
        ).repeat_interleave(int(rollouts_per_graph))
        state = State.initial(
            graph=context,
            graph_ids=graph_ids,
            expand_budget=self.expand_budget,
        )
        tape = RolloutTape(
            R=state.num_rows,
            T=self.expand_budget + 1,
            device=context.device,
        )
        trajectory_ids = torch.arange(
            state.num_rows,
            dtype=torch.long,
            device=context.device,
        )
        expansion_parts: list[ExpansionBatch] = []
        terminal_parts: list[TerminalBatch] = []

        for t in range(self.expand_budget + 1):
            active_rows = (~tape.is_stopped).nonzero(as_tuple=False).flatten()
            if active_rows.numel() == 0:
                break

            active_state = state.select_rows(active_rows)
            frontier = active_state.frontier(
                context,
                expand_budget=self.expand_budget,
            )
            policy_out = policy(
                features=features,
                state=active_state,
                context=context,
                frontier=frontier,
            )
            forced_local = forced_terminal_rows(
                state=active_state,
                frontier=frontier,
                expand_budget=self.expand_budget,
            )
            sample_rows = rows_without_forced(
                num_rows=active_state.num_rows,
                forced_rows=forced_local,
                device=context.device,
            )
            actions: list[StepAction] = []
            if sample_rows.numel() > 0:
                actions.append(
                    sample_step(
                        policy_out=policy_out,
                        rows=sample_rows,
                    )
                )
            if forced_local.numel() > 0:
                actions.append(
                    StepAction.forced_terminal(
                        rows=forced_local,
                        dtype=policy_out.stop_log_flow.dtype,
                        reason=forced_stop_reason(
                            state=active_state,
                            frontier=frontier,
                            rows=forced_local,
                            expand_budget=self.expand_budget,
                        ),
                        device=context.device,
                    )
                )
            sampled = StepAction.concat(actions)
            action = StepAction(
                row_ids=active_rows.index_select(0, sampled.row_ids),
                edge_ids=sampled.edge_ids,
                policy_log_prob=sampled.policy_log_prob,
                behavior_log_prob=sampled.behavior_log_prob,
                forced=sampled.forced,
                stop_reason=sampled.stop_reason,
            )
            tape.write(t, action)

            if bool(action.expand_mask.any()):
                expand_rows = action.expand_rows
                expand_edge_ids = action.expand_edge_ids
                parent = state.select_rows(expand_rows)
                child = parent.expand(
                    graph=context,
                    rows=torch.arange(
                        parent.num_rows,
                        dtype=torch.long,
                        device=context.device,
                    ),
                    edge_ids=expand_edge_ids,
                    expand_budget=self.expand_budget,
                )
                expansion_parts.append(
                    ExpansionBatch(
                        parent=parent,
                        child=child,
                        edge_ids=expand_edge_ids,
                        meta=SampleMeta(
                            trajectory_ids=trajectory_ids.index_select(0, expand_rows),
                            step_ids=torch.full(
                                (expand_rows.numel(),),
                                int(t),
                                dtype=torch.long,
                                device=context.device,
                            ),
                            source_ids=torch.full(
                                (expand_rows.numel(),),
                                SRC_UNKNOWN,
                                dtype=torch.long,
                                device=context.device,
                            ),
                        ),
                    )
                )
                state = state.expand(
                    graph=context,
                    rows=expand_rows,
                    edge_ids=expand_edge_ids,
                    expand_budget=self.expand_budget,
                )
            if bool(action.terminal_mask.any()):
                terminal_rows = action.terminal_rows
                if terminal_rows.numel() > 0:
                    terminal_parts.append(
                        TerminalBatch(
                            state=state.select_rows(terminal_rows),
                            meta=SampleMeta(
                                trajectory_ids=trajectory_ids.index_select(0, terminal_rows),
                                step_ids=torch.full(
                                    (terminal_rows.numel(),),
                                    int(t),
                                    dtype=torch.long,
                                    device=context.device,
                                ),
                                source_ids=torch.full(
                                    (terminal_rows.numel(),),
                                    SRC_UNKNOWN,
                                    dtype=torch.long,
                                    device=context.device,
                                ),
                            ),
                            stop_reason=action.stop_reason[action.terminal_mask],
                        )
                    )

        terminal_step = tape.terminal_step.clone()
        unstopped = terminal_step.lt(0)
        if bool(unstopped.any()):
            terminal_step[unstopped] = self.expand_budget

        empty_state = initial_empty_state(context=context)
        training = None
        if expansion_parts or terminal_parts:
            training = TrainingBatch(
                expansions=(
                    ExpansionBatch.concat(expansion_parts)
                    if expansion_parts
                    else ExpansionBatch.empty_like(graph_like=empty_state)
                ),
                terminals=(
                    TerminalBatch.concat(terminal_parts)
                    if terminal_parts
                    else TerminalBatch.empty_like(graph_like=empty_state)
                ),
            )

        return (
            RolloutResult(
                source_graph_id=graph_ids,
                selected_edge_ids=tape.selected_edge_ids,
                policy_action_log_prob=tape.policy_action_log_prob,
                behavior_action_log_prob=tape.behavior_action_log_prob,
                terminal_step=terminal_step,
                stop_reason=tape.stop_reason,
                expand_budget=self.expand_budget,
                terminal_state=state,
            ),
            training,
        )


def forced_terminal_rows(
    *,
    state: State,
    frontier: Frontier,
    expand_budget: int,
) -> torch.Tensor:
    num_rows = state.num_rows
    has_frontier = torch.zeros(
        num_rows,
        dtype=torch.bool,
        device=state.device,
    )
    if frontier.row_ids.numel() > 0:
        has_frontier.index_fill_(0, frontier.row_ids, True)
    del expand_budget
    exhausted = state.remaining_budget.le(0)
    return (~has_frontier | exhausted).nonzero(as_tuple=False).flatten()


def forced_stop_reason(
    *,
    state: State,
    frontier: Frontier,
    rows: torch.Tensor,
    expand_budget: int,
) -> torch.Tensor:
    rows = rows.to(device=state.device, dtype=torch.long).view(-1)
    if rows.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=state.device)
    has_frontier = torch.zeros(state.num_rows, dtype=torch.bool, device=state.device)
    if frontier.row_ids.numel() > 0:
        has_frontier.index_fill_(0, frontier.row_ids, True)
    no_frontier = ~has_frontier.index_select(0, rows)
    del expand_budget
    exhausted = state.remaining_budget.le(0).index_select(0, rows)
    out = torch.full(
        (rows.numel(),),
        int(RolloutResult.NO_FRONTIER_STOP),
        dtype=torch.long,
        device=state.device,
    )
    out[exhausted] = int(RolloutResult.BUDGET_TRUNCATED)
    if bool((~(no_frontier | exhausted)).any()):
        raise RuntimeError("forced_stop_reason requires rows that are actually forced to stop.")
    return out


def rows_without_forced(
    *,
    num_rows: int,
    forced_rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(int(num_rows), dtype=torch.long, device=device)
    if forced_rows.numel() == 0:
        return rows
    keep = torch.ones(int(num_rows), dtype=torch.bool, device=device)
    keep[forced_rows.to(device=device, dtype=torch.long)] = False
    return rows[keep]


def split_fused_rollouts(
    *,
    fused: RolloutResult,
    rollouts_per_graph: int,
    num_graphs: int,
) -> list[RolloutResult]:
    out: list[RolloutResult] = []
    for rollout_id in range(int(rollouts_per_graph)):
        rows = torch.arange(
            int(num_graphs),
            dtype=torch.long,
            device=fused.device,
        ) * int(rollouts_per_graph) + int(rollout_id)
        out.append(fused.select_rows(rows))
    return out


def initial_empty_state(*, context: GraphContext) -> State:
    return State.initial(
        graph=context,
        graph_ids=torch.empty(0, dtype=torch.long, device=context.device),
        expand_budget=0,
    )


__all__ = [
    "RolloutEngine",
    "forced_terminal_rows",
]
