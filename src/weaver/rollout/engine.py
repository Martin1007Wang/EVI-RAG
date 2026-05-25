from __future__ import annotations

import torch

from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import STOP_EDGE_ID, ForwardPolicy
from src.weaver.state import ExpansionBatch, StateBatch

from .trajectory import (
    BUDGET,
    NO_FRONTIER,
    POLICY_STOP,
    SRC_POLICY,
    TrajectoryBatch,
)


class RolloutEngine:
    """
    Vectorized finite-horizon rollout engine.

    It samples trajectories only.
    Training transitions should be derived later from TrajectoryBatch.
    """

    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        if self.budget < 0:
            raise ValueError("budget must be nonnegative.")

    @torch.no_grad()
    def sample(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeatureBank,
        graph_ids: torch.Tensor,
    ) -> TrajectoryBatch:
        graph_ids = graph_ids.to(
            device=context.device,
            dtype=torch.long,
        ).view(-1)

        state = StateBatch.initial(
            graph_ids=graph_ids,
            budget=self.budget,
        )

        num_rows = state.num_states

        edge_ids = torch.full(
            (num_rows, self.budget),
            -1,
            dtype=torch.long,
            device=context.device,
        )

        edge_logp = torch.zeros(
            (num_rows, self.budget),
            dtype=torch.float32,
            device=context.device,
        )

        edge_count = torch.zeros(
            num_rows,
            dtype=torch.long,
            device=context.device,
        )

        stop_reason = torch.full(
            (num_rows,),
            -1,
            dtype=torch.long,
            device=context.device,
        )

        stop_logp = torch.zeros(
            num_rows,
            dtype=torch.float32,
            device=context.device,
        )

        done = torch.zeros(
            num_rows,
            dtype=torch.bool,
            device=context.device,
        )

        for _ in range(self.budget + 1):
            active_rows = (~done).nonzero(as_tuple=False).flatten()
            if int(active_rows.numel()) == 0:
                break

            active_edge_count = state.edge_count.index_select(0, active_rows)
            budget_mask = active_edge_count.ge(self.budget)

            budget_rows = active_rows[budget_mask]
            if int(budget_rows.numel()) > 0:
                stop_reason[budget_rows] = int(BUDGET)
                done[budget_rows] = True

            decision_rows = active_rows[~budget_mask]
            if int(decision_rows.numel()) == 0:
                continue

            decision_state = state.take(decision_rows)
            action_space = decision_state.action_space(context)

            local_rows = torch.arange(
                decision_state.num_states,
                dtype=torch.long,
                device=context.device,
            )

            policy_out = policy(
                features=features,
                state=decision_state,
                context=context,
                action_space=action_space,
            )

            action_edge_ids = policy_out.sample(rows=local_rows)

            action_logp = policy_out.gather_log_prob(
                row_ids=local_rows,
                edge_ids=action_edge_ids,
            ).float()

            no_frontier = action_space.expand_count.eq(0)

            invalid_expand = no_frontier & action_edge_ids.ge(0)
            if bool(invalid_expand.any()):
                raise RuntimeError("Policy sampled EXPAND for a row with no legal expansion.")

            stop_mask = action_edge_ids.eq(int(STOP_EDGE_ID))
            stopped_rows = decision_rows[stop_mask]

            if int(stopped_rows.numel()) > 0:
                stopped_local = stop_mask.nonzero(as_tuple=False).flatten()

                stopped_no_frontier = no_frontier.index_select(0, stopped_local)
                no_frontier_rows = stopped_rows[stopped_no_frontier]
                policy_stop_rows = stopped_rows[~stopped_no_frontier]

                if int(no_frontier_rows.numel()) > 0:
                    stop_reason[no_frontier_rows] = int(NO_FRONTIER)

                if int(policy_stop_rows.numel()) > 0:
                    stop_reason[policy_stop_rows] = int(POLICY_STOP)

                stop_logp[stopped_rows] = action_logp[stop_mask]
                done[stopped_rows] = True

            expand_mask = action_edge_ids.ge(0)
            expand_rows = decision_rows[expand_mask]

            if int(expand_rows.numel()) == 0:
                continue

            expand_edges = action_edge_ids[expand_mask]
            expand_logp = action_logp[expand_mask]

            pos = state.edge_count.index_select(0, expand_rows)

            edge_ids[expand_rows, pos] = expand_edges
            edge_logp[expand_rows, pos] = expand_logp
            edge_count[expand_rows] = edge_count[expand_rows] + 1

            state = state.advance(
                ExpansionBatch(
                    state_ids=expand_rows,
                    edge_ids=expand_edges,
                )
            )

        unfinished = stop_reason.lt(0)
        if bool(unfinished.any()):
            stop_reason[unfinished] = int(BUDGET)

        return TrajectoryBatch(
            graph_ids=graph_ids,
            edge_ids=edge_ids,
            edge_logp=edge_logp,
            edge_count=edge_count,
            stop_reason=stop_reason,
            stop_logp=stop_logp,
            source=torch.full(
                (num_rows,),
                int(SRC_POLICY),
                dtype=torch.long,
                device=context.device,
            ),
        )
