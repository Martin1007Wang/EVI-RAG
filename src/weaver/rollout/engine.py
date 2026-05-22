from __future__ import annotations

import torch

from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy import ForwardPolicy
from src.weaver.state import Frontier, State

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
    ) -> list[RolloutResult]:
        with torch.no_grad():
            fused = self.sample_fused_rollouts(
                policy=policy,
                context=context,
                features=features,
                rollouts_per_graph=int(num_rollouts),
            )

        return split_fused_rollouts(
            fused=fused,
            rollouts_per_graph=int(num_rollouts),
            num_graphs=int(context.num_graphs),
        )

    def sample_fused_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        rollouts_per_graph: int,
    ) -> RolloutResult:
        graph_ids = torch.arange(
            int(context.num_graphs),
            dtype=torch.long,
            device=context.device,
        ).repeat_interleave(int(rollouts_per_graph))
        state = State.initial(
            graph=context,
            graph_ids=graph_ids,
        )
        tape = RolloutTape(
            R=state.num_rows,
            T=self.expand_budget + 1,
            device=context.device,
        )

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
            forced_local = forced_stop_rows(
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
                    StepAction.forced_stop(
                        rows=forced_local,
                        dtype=policy_out.stop_logit.dtype,
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
            )
            tape.write(t, action)

            if bool(action.expand_mask.any()):
                state = state.expand(
                    graph=context,
                    rows=action.expand_rows,
                    edge_ids=action.expand_edge_ids,
                    expand_budget=self.expand_budget,
                )

        stop_step = tape.stop_step.clone()
        unstopped = stop_step.lt(0)
        if bool(unstopped.any()):
            stop_step[unstopped] = self.expand_budget

        return RolloutResult(
            source_graph_id=graph_ids,
            selected_edge_ids=tape.selected_edge_ids,
            policy_action_log_prob=tape.policy_action_log_prob,
            behavior_action_log_prob=tape.behavior_action_log_prob,
            stop_step=stop_step,
            forced_stop=tape.forced_stop,
            expand_budget=self.expand_budget,
        )


def forced_stop_rows(
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
    exhausted = state.depth.ge(int(expand_budget))
    return (~has_frontier | exhausted).nonzero(as_tuple=False).flatten()


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


__all__ = [
    "RolloutEngine",
]
