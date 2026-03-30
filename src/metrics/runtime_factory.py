from __future__ import annotations

from typing import Any

from src.metrics.search_eval_utils import RUNTIME_ANSWER_TASK, search_eval_runtime_task
from src.metrics.protocol import MetricRuntimeProtocol
from src.models.gflownet.policy import SubgraphPolicy
from src.models.gflownet.sampler import SubgraphSampler

from .subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime


class GraphTaskRuntimeFactory:
    def build_runtime(
        self,
        *,
        horizon_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        eval_cfg: dict[str, Any],
        policy: object,
    ) -> MetricRuntimeProtocol:
        del training_cfg
        if not isinstance(policy, SubgraphPolicy):
            raise TypeError("GraphTaskRuntimeFactory requires SubgraphPolicy.")
        if search_eval_runtime_task(eval_cfg) != RUNTIME_ANSWER_TASK:
            raise ValueError(
                "Subgraph mode currently supports only answer_search evaluation."
            )
        return SubgraphAnswerSearchRuntime(
            eval_cfg=eval_cfg,
            policy=policy,
            sampler=SubgraphSampler(max_steps=int(horizon_cfg["max_steps"])),
        )


__all__ = ["GraphTaskRuntimeFactory"]
