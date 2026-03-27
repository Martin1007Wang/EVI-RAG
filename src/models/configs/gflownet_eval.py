from __future__ import annotations

from dataclasses import dataclass


ANSWER_TASKS = frozenset({"answer_ranking"})
EDGE_RETRIEVAL_TASK = "edge_retrieval"
RUNTIME_ANSWER_TASK = "answer_search"


@dataclass(frozen=True)
class HorizonConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("horizon.max_steps must be >= 1.")


@dataclass(frozen=True)
class SearchEvalConfig:
    """Evaluation settings for graph-search tasks.

    `task` stays intentionally small: answer ranking/search or edge retrieval.
    `runtime_task` exposes the even smaller runtime dispatch surface.
    """

    metrics_profile: str = "full"
    task: str = "answer_ranking"
    answer_mass_threshold: float = 0.9
    support_mass_threshold: float = 0.9
    support_path_overlap_penalty: float = 0.25
    answer_top_ks: tuple[int, ...] = (1, 5, 10)
    edge_top_ks: tuple[int, ...] = (1, 5, 10, 25, 50)
    edge_emit_top_k: int = 25
    support_search_method: str = "flow_frontier"
    flow_prune_epsilon: float = 1.0e-3
    monte_carlo_rollouts: int = 4096
    monte_carlo_confidence: float = 0.95
    max_expansions: int = 20000
    max_frontier_size: int = 4096

    def __post_init__(self) -> None:
        if self.metrics_profile not in {"full", "rank_only"}:
            raise ValueError(
                "eval_cfg.metrics_profile must be one of {'full', 'rank_only'}."
            )
        if self.task not in {*ANSWER_TASKS, EDGE_RETRIEVAL_TASK}:
            raise ValueError(
                "eval_cfg.task must be one of {'answer_ranking', 'edge_retrieval'}."
            )
        if self.task == EDGE_RETRIEVAL_TASK and self.metrics_profile != "rank_only":
            raise ValueError(
                "edge_retrieval only supports eval_cfg.metrics_profile='rank_only'."
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
        if self.support_search_method not in {"monte_carlo", "flow_frontier"}:
            raise ValueError(
                "eval_cfg.support_search_method must be one of {'monte_carlo', 'flow_frontier'}."
            )
        if (
            self.task == EDGE_RETRIEVAL_TASK
            and self.support_search_method != "monte_carlo"
        ):
            raise ValueError(
                "edge_retrieval only supports eval_cfg.support_search_method='monte_carlo'."
            )
        if not 0.0 <= self.flow_prune_epsilon <= 1.0:
            raise ValueError("eval_cfg.flow_prune_epsilon must be in [0, 1].")
        if self.monte_carlo_rollouts < 1:
            raise ValueError("eval_cfg.monte_carlo_rollouts must be >= 1.")
        if not 0.0 < self.monte_carlo_confidence < 1.0:
            raise ValueError("eval_cfg.monte_carlo_confidence must be in (0, 1).")
        if self.max_expansions < 1:
            raise ValueError("eval_cfg.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("eval_cfg.max_frontier_size must be >= 1.")

    @property
    def runtime_task(self) -> str:
        if self.task == EDGE_RETRIEVAL_TASK:
            return EDGE_RETRIEVAL_TASK
        return RUNTIME_ANSWER_TASK

    @property
    def is_answer_task(self) -> bool:
        return self.runtime_task == RUNTIME_ANSWER_TASK


__all__ = [
    "ANSWER_TASKS",
    "EDGE_RETRIEVAL_TASK",
    "HorizonConfig",
    "RUNTIME_ANSWER_TASK",
    "SearchEvalConfig",
]
