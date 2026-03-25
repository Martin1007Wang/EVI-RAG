from __future__ import annotations

from src.metrics.answer_metrics import (
    AnswerReachabilityEvaluator,
    AnswerReachabilityRuntime,
)
from src.metrics.edge_metrics import EdgeRetrievalRuntime
from src.metrics.protocol import MetricRuntimeProtocol
from src.models.configs import GFlowNetTrainingConfig, HorizonConfig, SearchEvalConfig
from src.models.gflownet import (
    AnswerReachabilityTrajectorySupervisor,
    ForwardTrajectoryGFNSampler,
    SearchPolicyProtocol,
)

from .search_backends import FlowFrontierBackend, MonteCarloBackend


def _build_trajectory_sampler(
    *,
    horizon_cfg: HorizonConfig,
    training_cfg: GFlowNetTrainingConfig,
) -> ForwardTrajectoryGFNSampler:
    trajectory_supervisor = AnswerReachabilityTrajectorySupervisor(
        non_gold_terminal_log_reward=float(training_cfg.terminal_failure_log_reward)
    )
    return ForwardTrajectoryGFNSampler(
        max_steps=int(horizon_cfg.max_steps),
        trajectory_supervisor=trajectory_supervisor,
        force_stop_on_answer_hit=bool(training_cfg.force_stop_on_answer_hit),
        step_log_penalty=float(training_cfg.step_log_penalty),
    )


def _build_answer_backend(
    *,
    horizon_cfg: HorizonConfig,
    eval_cfg: SearchEvalConfig,
) -> FlowFrontierBackend | MonteCarloBackend:
    if eval_cfg.support_search_method == "flow_frontier":
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
        sampler = _build_trajectory_sampler(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
        )
        if eval_cfg.task == "edge_retrieval":
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
