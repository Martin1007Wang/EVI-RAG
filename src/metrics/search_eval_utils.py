from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


ANSWER_TASKS = frozenset({"answer_ranking"})
EDGE_RETRIEVAL_TASK = "edge_retrieval"
RUNTIME_ANSWER_TASK = "answer_search"
FLOW_FRONTIER_BACKEND = "flow_frontier"
MONTE_CARLO_BACKEND = "monte_carlo"
FULL_REPORT = "full"
RANK_ONLY_REPORT = "rank_only"

_DEFAULT_SEARCH_EVAL_CFG: dict[str, Any] = {
    "report_profile": FULL_REPORT,
    "task": "answer_ranking",
    "answer_mass_threshold": 0.9,
    "support_mass_threshold": 0.9,
    "support_path_overlap_penalty": 0.25,
    "answer_top_ks": (1, 5, 10),
    "edge_top_ks": (1, 5, 10, 25, 50),
    "edge_emit_top_k": 25,
    "answer_posterior_backend": FLOW_FRONTIER_BACKEND,
    "flow_frontier": {
        "prune_epsilon": 1.0e-3,
        "max_expansions": 20000,
        "max_frontier_size": 4096,
    },
    "monte_carlo": {
        "rollouts": 4096,
        "confidence": 0.95,
    },
}


def _to_plain_mapping(node: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(node, DictConfig):
        container = OmegaConf.to_container(node, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Expected {field_name} to resolve to a mapping.")
        return dict(container)
    if isinstance(node, Mapping):
        return {str(key): value for key, value in node.items()}
    raise TypeError(f"Expected {field_name} to be a mapping, got {type(node)!r}.")


def _deep_merge(
    *, base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(base=merged[key], override=value)
            continue
        merged[key] = deepcopy(value)
    return merged


def normalize_search_eval_cfg(eval_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_SEARCH_EVAL_CFG,
        override=_to_plain_mapping(eval_cfg, field_name="eval_cfg"),
    )
    flow_frontier = _to_plain_mapping(
        cfg.get("flow_frontier", {}), field_name="eval_cfg.flow_frontier"
    )
    monte_carlo = _to_plain_mapping(
        cfg.get("monte_carlo", {}), field_name="eval_cfg.monte_carlo"
    )
    cfg["flow_frontier"] = flow_frontier
    cfg["monte_carlo"] = monte_carlo

    report_profile = str(cfg.get("report_profile", FULL_REPORT))
    if report_profile not in {FULL_REPORT, RANK_ONLY_REPORT}:
        raise ValueError(
            "evaluation.report_profile must be one of {'full', 'rank_only'}."
        )
    cfg["report_profile"] = report_profile

    task = str(cfg.get("task", "answer_ranking"))
    if task not in {*ANSWER_TASKS, EDGE_RETRIEVAL_TASK}:
        raise ValueError(
            "evaluation.task must be one of {'answer_ranking', 'edge_retrieval'}."
        )
    cfg["task"] = task

    answer_posterior_backend = str(
        cfg.get("answer_posterior_backend", FLOW_FRONTIER_BACKEND)
    )
    if answer_posterior_backend not in {FLOW_FRONTIER_BACKEND, MONTE_CARLO_BACKEND}:
        raise ValueError(
            "evaluation.answer_posterior_backend must be one of {'flow_frontier', 'monte_carlo'}."
        )
    cfg["answer_posterior_backend"] = answer_posterior_backend

    if task == EDGE_RETRIEVAL_TASK and report_profile != RANK_ONLY_REPORT:
        raise ValueError(
            "edge_retrieval only supports evaluation.report_profile='rank_only'."
        )
    if task == EDGE_RETRIEVAL_TASK and answer_posterior_backend != MONTE_CARLO_BACKEND:
        raise ValueError(
            "edge_retrieval only supports evaluation.answer_posterior_backend='monte_carlo'."
        )

    answer_mass_threshold = float(cfg.get("answer_mass_threshold", 0.9))
    if not 0.0 < answer_mass_threshold <= 1.0:
        raise ValueError("evaluation.answer_mass_threshold must be in (0, 1].")
    cfg["answer_mass_threshold"] = answer_mass_threshold

    support_mass_threshold = float(cfg.get("support_mass_threshold", 0.9))
    if not 0.0 < support_mass_threshold <= 1.0:
        raise ValueError("evaluation.support_mass_threshold must be in (0, 1].")
    cfg["support_mass_threshold"] = support_mass_threshold

    support_path_overlap_penalty = float(cfg.get("support_path_overlap_penalty", 0.25))
    if support_path_overlap_penalty < 0.0:
        raise ValueError("evaluation.support_path_overlap_penalty must be >= 0.")
    cfg["support_path_overlap_penalty"] = support_path_overlap_penalty

    answer_top_ks = tuple(int(k) for k in cfg.get("answer_top_ks", (1, 5, 10)))
    if not answer_top_ks or any(k < 1 for k in answer_top_ks):
        raise ValueError("evaluation.answer_top_ks must be non-empty and >= 1.")
    cfg["answer_top_ks"] = answer_top_ks

    edge_top_ks = tuple(int(k) for k in cfg.get("edge_top_ks", (1, 5, 10, 25, 50)))
    if not edge_top_ks or any(k < 1 for k in edge_top_ks):
        raise ValueError("evaluation.edge_top_ks must be non-empty and >= 1.")
    cfg["edge_top_ks"] = edge_top_ks

    edge_emit_top_k = int(cfg.get("edge_emit_top_k", 25))
    if edge_emit_top_k < 1:
        raise ValueError("evaluation.edge_emit_top_k must be >= 1.")
    cfg["edge_emit_top_k"] = edge_emit_top_k

    prune_epsilon = float(flow_frontier.get("prune_epsilon", 1.0e-3))
    if not 0.0 <= prune_epsilon <= 1.0:
        raise ValueError("evaluation.flow_frontier.prune_epsilon must be in [0, 1].")
    flow_frontier["prune_epsilon"] = prune_epsilon

    max_expansions = int(flow_frontier.get("max_expansions", 20000))
    if max_expansions < 1:
        raise ValueError("evaluation.flow_frontier.max_expansions must be >= 1.")
    flow_frontier["max_expansions"] = max_expansions

    max_frontier_size = int(flow_frontier.get("max_frontier_size", 4096))
    if max_frontier_size < 1:
        raise ValueError("evaluation.flow_frontier.max_frontier_size must be >= 1.")
    flow_frontier["max_frontier_size"] = max_frontier_size

    rollouts = int(monte_carlo.get("rollouts", 4096))
    if rollouts < 1:
        raise ValueError("evaluation.monte_carlo.rollouts must be >= 1.")
    monte_carlo["rollouts"] = rollouts

    confidence = float(monte_carlo.get("confidence", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("evaluation.monte_carlo.confidence must be in (0, 1).")
    monte_carlo["confidence"] = confidence

    return cfg


def search_eval_runtime_task(eval_cfg: Mapping[str, Any]) -> str:
    if str(eval_cfg.get("task", "answer_ranking")) == EDGE_RETRIEVAL_TASK:
        return EDGE_RETRIEVAL_TASK
    return RUNTIME_ANSWER_TASK


def search_eval_is_answer_task(eval_cfg: Mapping[str, Any]) -> bool:
    return search_eval_runtime_task(eval_cfg) == RUNTIME_ANSWER_TASK


def search_eval_include_answer_support(eval_cfg: Mapping[str, Any]) -> bool:
    return str(eval_cfg.get("report_profile", FULL_REPORT)) != RANK_ONLY_REPORT


__all__ = [
    "EDGE_RETRIEVAL_TASK",
    "FLOW_FRONTIER_BACKEND",
    "FULL_REPORT",
    "MONTE_CARLO_BACKEND",
    "RANK_ONLY_REPORT",
    "RUNTIME_ANSWER_TASK",
    "normalize_search_eval_cfg",
    "search_eval_include_answer_support",
    "search_eval_is_answer_task",
    "search_eval_runtime_task",
]
