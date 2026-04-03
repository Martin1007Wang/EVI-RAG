from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .policy import SubgraphPolicy
from .rollout_engine import HierarchicalRolloutEngine, SubgraphTrajectorySampleBatch
from .subgraph_batch import SubgraphBatch


class SubgraphSampler:
    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = int(max_steps)
        self.engine = HierarchicalRolloutEngine(max_steps=int(max_steps))

    def sample(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphBatch,
        rollouts_per_graph: int,
        temperature: float,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphTrajectorySampleBatch:
        return self.engine.sample(
            policy=policy,
            prepared_batch=prepared_batch,
            rollouts_per_graph=int(rollouts_per_graph),
            temperature=float(temperature),
            action_pruning=action_pruning,
        )

    def teacher_force(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphBatch,
        edge_sequences: tuple[tuple[int, ...], ...],
    ) -> SubgraphTrajectorySampleBatch:
        return self.engine.teacher_force(
            policy=policy,
            prepared_batch=prepared_batch,
            edge_sequences=edge_sequences,
        )


__all__ = ["SubgraphSampler", "SubgraphTrajectorySampleBatch"]
