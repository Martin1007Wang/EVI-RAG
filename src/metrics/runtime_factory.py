from __future__ import annotations

from src.metrics.protocol import MetricRuntimeProtocol
from src.models.configs import GFlowNetTrainingConfig, HorizonConfig, SearchEvalConfig
from src.models.gflownet.subgraph.policy import SubgraphPolicy
from src.models.gflownet.subgraph.sampler import SubgraphSampler

from .subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime


class GraphTaskRuntimeFactory:
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: object,
    ) -> MetricRuntimeProtocol:
        del training_cfg
        if not isinstance(policy, SubgraphPolicy):
            raise TypeError("GraphTaskRuntimeFactory requires SubgraphPolicy.")
        if eval_cfg.runtime_task != "answer_search":
            raise ValueError(
                "Subgraph mode currently supports only answer_search evaluation."
            )
        return SubgraphAnswerSearchRuntime(
            eval_cfg=eval_cfg,
            policy=policy,
            sampler=SubgraphSampler(max_steps=int(horizon_cfg.max_steps)),
        )


__all__ = ["GraphTaskRuntimeFactory"]
