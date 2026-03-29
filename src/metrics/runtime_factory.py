from __future__ import annotations

from src.metrics.answer_metrics import (
    AnswerReachabilityEvaluator,
    AnswerReachabilityRuntime,
)
from src.metrics.edge_metrics import EdgeRetrievalRuntime
from src.metrics.protocol import MetricRuntimeProtocol
from src.models.configs import GFlowNetTrainingConfig, HorizonConfig, SearchEvalConfig
from src.models.configs.policy import SUBGRAPH_STATE_MODE
from src.models.gflownet.prefix_sampler import (
    AnswerReachabilityTrajectorySupervisor,
    ForwardTrajectoryGFNSampler,
)
from src.models.gflownet.prefix_state import SearchPolicyProtocol
from src.models.gflownet.subgraph.policy import SubgraphPolicy
from src.models.gflownet.subgraph.sampler import SubgraphSampler

from .search_backends import FlowFrontierBackend, MonteCarloBackend
from .subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime


def _build_trajectory_sampler(
    *,
    horizon_cfg: HorizonConfig,
    training_cfg: GFlowNetTrainingConfig,
) -> ForwardTrajectoryGFNSampler:
    trajectory_supervisor = AnswerReachabilityTrajectorySupervisor(
        non_gold_terminal_log_reward=float(training_cfg.terminal_failure_log_reward),
        gold_reward_mode=str(training_cfg.answer_quotient.gold_reward_mode),
    )
    return ForwardTrajectoryGFNSampler(
        max_steps=int(horizon_cfg.max_steps),
        trajectory_supervisor=trajectory_supervisor,
        force_stop_on_answer_hit=bool(training_cfg.force_stop_on_answer_hit),
        step_log_penalty=float(training_cfg.step_log_penalty),
        answer_stop_log_reward_bonus=float(training_cfg.answer_stop_log_reward_bonus),
    )


def _build_answer_backend(
    *,
    horizon_cfg: HorizonConfig,
    eval_cfg: SearchEvalConfig,
) -> FlowFrontierBackend | MonteCarloBackend:
    if eval_cfg.answer_posterior_backend == "flow_frontier":
        return FlowFrontierBackend(
            max_steps=int(horizon_cfg.max_steps),
            eval_cfg=eval_cfg,
        )
    return MonteCarloBackend(
        max_steps=int(horizon_cfg.max_steps),
        eval_cfg=eval_cfg,
    )


class GraphTaskRuntimeFactory:
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
    ) -> MetricRuntimeProtocol:
        if getattr(policy, "state_mode", None) == SUBGRAPH_STATE_MODE:
            if not isinstance(policy, SubgraphPolicy):
                raise TypeError("subgraph runtime requires SubgraphPolicy.")
            if eval_cfg.runtime_task != "answer_search":
                raise ValueError(
                    "subgraph mode currently supports only answer_search evaluation."
                )
            return SubgraphAnswerSearchRuntime(
                eval_cfg=eval_cfg,
                policy=policy,
                sampler=SubgraphSampler(max_steps=int(horizon_cfg.max_steps)),
            )
        sampler = _build_trajectory_sampler(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
        )
        if eval_cfg.runtime_task == "edge_retrieval":
            return EdgeRetrievalRuntime(
                eval_cfg=eval_cfg,
                policy=policy,
                backend=MonteCarloBackend(
                    max_steps=int(horizon_cfg.max_steps),
                    eval_cfg=eval_cfg,
                ),
                sampler=sampler,
            )
        search_backend = _build_answer_backend(
            horizon_cfg=horizon_cfg,
            eval_cfg=eval_cfg,
        )
        return AnswerReachabilityRuntime(
            eval_cfg=eval_cfg,
            evaluator=AnswerReachabilityEvaluator(
                eval_cfg=eval_cfg,
                policy=policy,
                backend=search_backend,
            ),
            sampler=sampler,
            search_backend=search_backend,
        )


__all__ = ["GraphTaskRuntimeFactory"]
