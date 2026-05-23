from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy.latent_prefix import EdgeOnlyProposalPolicy
from src.weaver.state import State
from src.weaver.transition import ExpansionBatch, SampleMeta, SRC_UNKNOWN


@dataclass(frozen=True, slots=True)
class PrefixBatch:
    state: State
    trajectory_ids: torch.Tensor
    prefix_step: torch.Tensor

    @property
    def num_items(self) -> int:
        return int(self.trajectory_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.trajectory_ids.device


@dataclass(frozen=True, slots=True)
class LatentPrefixRollout:
    prefixes: PrefixBatch
    expansions: ExpansionBatch
    trajectory_graph_ids: torch.Tensor
    trajectory_log_prob: torch.Tensor
    dead_end: torch.Tensor
    expand_budget: int

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectory_graph_ids.numel())


class LatentPrefixRolloutEngine:
    """
    Fixed-horizon edge-only proposal rollout for latent-prefix training.
    """

    def __init__(self, expand_budget: int) -> None:
        self.expand_budget = int(expand_budget)

    @torch.no_grad()
    def sample_rollouts(
        self,
        *,
        policy: EdgeOnlyProposalPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        rollouts_per_graph: int,
    ) -> LatentPrefixRollout:
        graph_ids = torch.arange(
            int(context.num_graphs),
            dtype=torch.long,
            device=context.device,
        ).repeat_interleave(int(rollouts_per_graph))
        state = State.initial(graph=context, graph_ids=graph_ids, expand_budget=self.expand_budget)
        trajectory_ids = torch.arange(state.num_rows, dtype=torch.long, device=context.device)
        prefix_states: list[State] = []
        prefix_trajectory_ids: list[torch.Tensor] = []
        prefix_steps: list[torch.Tensor] = []
        expansion_parts: list[ExpansionBatch] = []
        trajectory_log_prob = torch.zeros(state.num_rows, dtype=torch.float32, device=context.device)
        dead_end = torch.zeros(state.num_rows, dtype=torch.bool, device=context.device)

        for step in range(self.expand_budget):
            prefix_states.append(state.clone())
            prefix_trajectory_ids.append(trajectory_ids)
            prefix_steps.append(torch.full((state.num_rows,), step, dtype=torch.long, device=context.device))

            active_rows = (~dead_end).nonzero(as_tuple=False).flatten()
            if active_rows.numel() == 0:
                continue

            active_state = state.select_rows(active_rows)
            frontier = active_state.frontier(context, expand_budget=self.expand_budget)
            policy_out = policy(features=features, state=active_state, context=context, frontier=frontier)
            has_frontier = policy_out.has_frontier()
            local_expand_rows = has_frontier.nonzero(as_tuple=False).flatten()
            local_dead_rows = (~has_frontier).nonzero(as_tuple=False).flatten()

            if local_dead_rows.numel() > 0:
                dead_end[active_rows.index_select(0, local_dead_rows)] = True
            if local_expand_rows.numel() == 0:
                continue

            edge_ids = policy_out.sample(rows=local_expand_rows)
            log_prob = policy_out.gather_log_prob(row_ids=local_expand_rows, edge_ids=edge_ids)
            expand_rows = active_rows.index_select(0, local_expand_rows)
            trajectory_log_prob.index_add_(0, expand_rows, log_prob)

            parent = state.select_rows(expand_rows)
            child = parent.expand(
                graph=context,
                rows=torch.arange(parent.num_rows, dtype=torch.long, device=context.device),
                edge_ids=edge_ids,
                expand_budget=self.expand_budget,
            )
            expansion_parts.append(
                ExpansionBatch(
                    parent=parent,
                    child=child,
                    edge_ids=edge_ids,
                    meta=SampleMeta(
                        trajectory_ids=trajectory_ids.index_select(0, expand_rows),
                        step_ids=torch.full((expand_rows.numel(),), step, dtype=torch.long, device=context.device),
                        source_ids=torch.full((expand_rows.numel(),), SRC_UNKNOWN, dtype=torch.long, device=context.device),
                    ),
                )
            )
            state = state.expand(
                graph=context,
                rows=expand_rows,
                edge_ids=edge_ids,
                expand_budget=self.expand_budget,
            )

        prefix_states.append(state.clone())
        prefix_trajectory_ids.append(trajectory_ids)
        prefix_steps.append(torch.full((state.num_rows,), self.expand_budget, dtype=torch.long, device=context.device))
        empty_state = State.initial(
            graph=context,
            graph_ids=torch.empty(0, dtype=torch.long, device=context.device),
            expand_budget=self.expand_budget,
        )
        return LatentPrefixRollout(
            prefixes=PrefixBatch(
                state=State.concat(prefix_states),
                trajectory_ids=torch.cat(prefix_trajectory_ids, dim=0),
                prefix_step=torch.cat(prefix_steps, dim=0),
            ),
            expansions=(
                ExpansionBatch.concat(expansion_parts)
                if expansion_parts
                else ExpansionBatch.empty_like(graph_like=empty_state)
            ),
            trajectory_graph_ids=graph_ids,
            trajectory_log_prob=trajectory_log_prob,
            dead_end=dead_end,
            expand_budget=self.expand_budget,
        )


__all__ = [
    "LatentPrefixRollout",
    "LatentPrefixRolloutEngine",
    "PrefixBatch",
]
