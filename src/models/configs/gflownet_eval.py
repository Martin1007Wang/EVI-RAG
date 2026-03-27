from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


ANSWER_TASKS = frozenset({"answer_ranking"})
EDGE_RETRIEVAL_TASK = "edge_retrieval"
RUNTIME_ANSWER_TASK = "answer_search"
FLOW_FRONTIER_BACKEND = "flow_frontier"
MONTE_CARLO_BACKEND = "monte_carlo"
FULL_REPORT = "full"
RANK_ONLY_REPORT = "rank_only"


@dataclass(frozen=True)
class HorizonConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("horizon.max_steps must be >= 1.")


@dataclass(frozen=True)
class FlowFrontierEvalConfig:
    prune_epsilon: float = 1.0e-3
    max_expansions: int = 20000
    max_frontier_size: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 <= self.prune_epsilon <= 1.0:
            raise ValueError("eval_cfg.flow_frontier.prune_epsilon must be in [0, 1].")
        if self.max_expansions < 1:
            raise ValueError("eval_cfg.flow_frontier.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("eval_cfg.flow_frontier.max_frontier_size must be >= 1.")


@dataclass(frozen=True)
class MonteCarloEvalConfig:
    rollouts: int = 4096
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.rollouts < 1:
            raise ValueError("eval_cfg.monte_carlo.rollouts must be >= 1.")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("eval_cfg.monte_carlo.confidence must be in (0, 1).")


@dataclass(frozen=True)
class SearchEvalConfig:
    """Evaluation settings for graph-search tasks.

    `task` stays intentionally small: answer ranking/search or edge retrieval.
    `answer_posterior_backend` decides how answer probabilities are estimated,
    while `report_profile` only decides how much support/window detail to emit.
    `runtime_task` exposes the even smaller runtime dispatch surface.
    """

    report_profile: str = FULL_REPORT
    task: str = "answer_ranking"
    answer_mass_threshold: float = 0.9
    support_mass_threshold: float = 0.9
    support_path_overlap_penalty: float = 0.25
    answer_top_ks: tuple[int, ...] = (1, 5, 10)
    edge_top_ks: tuple[int, ...] = (1, 5, 10, 25, 50)
    edge_emit_top_k: int = 25
    answer_posterior_backend: str = FLOW_FRONTIER_BACKEND
    flow_frontier: FlowFrontierEvalConfig = field(
        default_factory=FlowFrontierEvalConfig
    )
    monte_carlo: MonteCarloEvalConfig = field(default_factory=MonteCarloEvalConfig)

    def __post_init__(self) -> None:
        flow_frontier_cfg = self.flow_frontier
        if isinstance(flow_frontier_cfg, Mapping):
            flow_frontier_cfg = FlowFrontierEvalConfig(**dict(flow_frontier_cfg))
            object.__setattr__(self, "flow_frontier", flow_frontier_cfg)

        monte_carlo_cfg = self.monte_carlo
        if isinstance(monte_carlo_cfg, Mapping):
            monte_carlo_cfg = MonteCarloEvalConfig(**dict(monte_carlo_cfg))
            object.__setattr__(self, "monte_carlo", monte_carlo_cfg)

        if self.report_profile not in {FULL_REPORT, RANK_ONLY_REPORT}:
            raise ValueError(
                "eval_cfg.report_profile must be one of {'full', 'rank_only'}."
            )
        if self.task not in {*ANSWER_TASKS, EDGE_RETRIEVAL_TASK}:
            raise ValueError(
                "eval_cfg.task must be one of {'answer_ranking', 'edge_retrieval'}."
            )
        if self.task == EDGE_RETRIEVAL_TASK and self.report_profile != RANK_ONLY_REPORT:
            raise ValueError(
                "edge_retrieval only supports eval_cfg.report_profile='rank_only'."
            )
        if not 0.0 < self.answer_mass_threshold <= 1.0:
            raise ValueError("eval_cfg.answer_mass_threshold must be in (0, 1].")
        if not 0.0 < self.support_mass_threshold <= 1.0:
            raise ValueError("eval_cfg.support_mass_threshold must be in (0, 1].")
        if self.support_path_overlap_penalty < 0.0:
            raise ValueError("eval_cfg.support_path_overlap_penalty must be >= 0.")
        if len(self.answer_top_ks) == 0:
            raise ValueError("eval_cfg.answer_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.answer_top_ks):
            raise ValueError("eval_cfg.answer_top_ks values must be >= 1.")
        if len(self.edge_top_ks) == 0:
            raise ValueError("eval_cfg.edge_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.edge_top_ks):
            raise ValueError("eval_cfg.edge_top_ks values must be >= 1.")
        if self.edge_emit_top_k < 1:
            raise ValueError("eval_cfg.edge_emit_top_k must be >= 1.")
        if self.answer_posterior_backend not in {
            MONTE_CARLO_BACKEND,
            FLOW_FRONTIER_BACKEND,
        }:
            raise ValueError(
                "eval_cfg.answer_posterior_backend must be one of {'monte_carlo', 'flow_frontier'}."
            )
        if (
            self.task == EDGE_RETRIEVAL_TASK
            and self.answer_posterior_backend != MONTE_CARLO_BACKEND
        ):
            raise ValueError(
                "edge_retrieval only supports eval_cfg.answer_posterior_backend='monte_carlo'."
            )

    @property
    def runtime_task(self) -> str:
        if self.task == EDGE_RETRIEVAL_TASK:
            return EDGE_RETRIEVAL_TASK
        return RUNTIME_ANSWER_TASK

    @property
    def is_answer_task(self) -> bool:
        return self.runtime_task == RUNTIME_ANSWER_TASK

    @property
    def include_answer_support(self) -> bool:
        return self.report_profile != RANK_ONLY_REPORT


__all__ = [
    "ANSWER_TASKS",
    "EDGE_RETRIEVAL_TASK",
    "FLOW_FRONTIER_BACKEND",
    "FULL_REPORT",
    "HorizonConfig",
    "FlowFrontierEvalConfig",
    "MONTE_CARLO_BACKEND",
    "MonteCarloEvalConfig",
    "RANK_ONLY_REPORT",
    "RUNTIME_ANSWER_TASK",
    "SearchEvalConfig",
]
