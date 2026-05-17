from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import Policy, PolicyOutput
from src.weaver.state import FrontierBuilder, State

from .result import RolloutResult
from .sampling import SampledAction, sample_action
from .trace import RolloutTrace


@dataclass(frozen=True, slots=True)
class RolloutContext:
    graph_context: GraphContext
    features: FeatureBank
    frontier_builder: FrontierBuilder

    def __post_init__(self) -> None:
        if self.features.edge_h.requires_grad:
            raise ValueError(
                "RolloutContext.features must be detached. "
                "Call _detach_feature_bank() before constructing RolloutContext."
            )

    @property
    def device(self) -> torch.device:
        return self.graph_context.device


class RolloutEngine:
    """
    Label-free vectorized finite-horizon rollout engine.
    """

    def __init__(self, expand_budget: int) -> None:
        self.expand_budget = int(expand_budget)

    def prepare_context(
        self,
        *,
        batch: RetrievalBatch,
        features: FeatureBank,
    ) -> RolloutContext:
        features = _detach_feature_bank(features)
        device = features.edge_h.device
        graph_context = GraphContext.from_batch(batch, device=device)
        frontier_builder = FrontierBuilder.from_graph_context(graph_context)
        return RolloutContext(
            graph_context=graph_context,
            features=features,
            frontier_builder=frontier_builder,
        )

    def sample_rollouts(
        self,
        *,
        policy: Policy,
        context: RolloutContext,
        num_rollouts: int,
        temperature: float = 1.0,
    ) -> list[RolloutResult]:
        with torch.no_grad():
            fused = self._sample_fused_rollouts(
                policy=policy,
                context=context,
                rollouts_per_graph=int(num_rollouts),
                temperature=float(temperature),
            )
        return fused.split_by_rollout_id(
            rollouts_per_graph=int(num_rollouts),
        )

    def _sample_fused_rollouts(
        self,
        *,
        policy: Policy,
        context: RolloutContext,
        rollouts_per_graph: int,
        temperature: float,
    ) -> RolloutResult:
        graph_context = context.graph_context
        device = context.device

        state = State.initial_from_graph_context(
            graph_context,
            budget=self.expand_budget,
            rollouts_per_graph=rollouts_per_graph,
        )
        source_graph_id = state.row_to_graph.to(device=device, dtype=torch.long)
        num_rows = int(source_graph_id.numel())
        trace = RolloutTrace(
            R=num_rows,
            T=self.expand_budget + 1,
            device=device,
        )
        alive = torch.ones(num_rows, dtype=torch.bool, device=device)

        for t in range(self.expand_budget + 1):
            active_rows = alive.nonzero(as_tuple=False).flatten()
            if active_rows.numel() == 0:
                break

            active_state = state.select_rows(active_rows)
            policy_out = policy(
                context=graph_context,
                state=active_state,
                features=context.features,
                frontier_builder=context.frontier_builder,
            )
            action = sample_action(
                policy_out=policy_out,
                temperature=temperature,
            )

            trace.write_state(t=t, rows=active_rows)
            self._write_terminal_rows(
                t=t,
                active_rows=active_rows,
                policy_out=policy_out,
                action=action,
                trace=trace,
                alive=alive,
            )
            self._expand_rows(
                t=t,
                state=state,
                active_rows=active_rows,
                action=action,
                graph_context=graph_context,
                trace=trace,
            )

        return RolloutResult.from_trace(
            trace=trace,
            source_graph_id=source_graph_id,
            expand_budget=self.expand_budget,
        )

    @staticmethod
    def _write_terminal_rows(
        *,
        t: int,
        active_rows: torch.Tensor,
        policy_out: PolicyOutput,
        action: SampledAction,
        trace: RolloutTrace,
        alive: torch.Tensor,
    ) -> None:
        if action.stop_rows.numel() == 0:
            return
        terminal_rows = active_rows.index_select(0, action.stop_rows)
        trace.write_terminal(
            t=t,
            rows=terminal_rows,
            stop_log_prob=policy_out.stop_log_prob.index_select(0, action.stop_rows),
            forced=action.forced_stop,
        )
        alive[terminal_rows] = False

    @staticmethod
    def _expand_rows(
        *,
        t: int,
        state: State,
        active_rows: torch.Tensor,
        action: SampledAction,
        graph_context: GraphContext,
        trace: RolloutTrace,
    ) -> None:
        if action.expand_rows.numel() == 0:
            return
        rows = active_rows.index_select(0, action.expand_rows)
        trace.write_expand(
            t=t,
            rows=rows,
            edge_ids=action.expand_edge_ids,
        )
        state.apply_edges_(
            edge_index=graph_context.edge_index,
            rows=rows,
            edge_ids=action.expand_edge_ids,
        )


def _detach_feature_bank(features: FeatureBank) -> FeatureBank:
    return FeatureBank(
        node_h=features.node_h.detach(),
        edge_h=features.edge_h.detach(),
        query_h=features.query_h.detach(),
        node_is_non_text=features.node_is_non_text.detach(),
        node_sem_h=features.node_sem_h.detach(),
        rel_sem_h=features.rel_sem_h.detach(),
        query_sem_h=features.query_sem_h.detach(),
        rel_h=features.rel_h.detach(),
    )


__all__ = [
    "RolloutEngine",
    "RolloutContext",
]
