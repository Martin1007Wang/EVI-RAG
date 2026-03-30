from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


ANSWER_TASKS = frozenset({"answer_ranking"})
EDGE_RETRIEVAL_TASK = "edge_retrieval"
RUNTIME_ANSWER_TASK = "answer_search"
FULL_REPORT = "full"
RANK_ONLY_REPORT = "rank_only"

_DEFAULT_SEARCH_EVAL_CFG: dict[str, Any] = {
    "report_profile": FULL_REPORT,
    "task": "answer_ranking",
    "support_mass_threshold": 0.9,
    "support_path_overlap_penalty": 0.25,
    "answer_top_ks": (1, 5, 10),
    "edge_top_ks": (1, 5, 10, 25, 50),
    "edge_emit_top_k": 25,
    "monte_carlo": {
        "rollouts": 4096,
        "batch_rollouts": 256,
        "temperature": 1.0,
        "confidence": 0.95,
        "early_stop": {
            "enabled": True,
            "min_rollouts": 512,
            "stability_top_k": 1,
        },
        "action_pruning": {
            "per_node_top_k": 100,
            "per_state_top_k": 256,
        },
    },
}
_LEGACY_SEARCH_EVAL_KEYS = {
    "answer_posterior_backend": "Remove eval_cfg.answer_posterior_backend; Monte Carlo is now the only posterior estimator.",
    "flow_frontier": "Remove eval_cfg.flow_frontier; exact frontier search is no longer supported.",
}
_REMOVED_SEARCH_EVAL_KEYS = {
    "answer_mass_threshold": (
        "Remove eval_cfg.answer_mass_threshold; full-vote answer marginals are not a "
        "normalized posterior mass budget, so this threshold is no longer well-defined."
    ),
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


def _assert_no_legacy_search_eval_keys(eval_cfg: Mapping[str, Any]) -> None:
    legacy_messages = [
        f"{key}: {message}"
        for key, message in _LEGACY_SEARCH_EVAL_KEYS.items()
        if key in eval_cfg
    ]
    removed_messages = [
        f"{key}: {message}"
        for key, message in _REMOVED_SEARCH_EVAL_KEYS.items()
        if key in eval_cfg
    ]
    if legacy_messages:
        raise ValueError(
            "Legacy exact answer-posterior config detected. "
            + " ".join(legacy_messages)
        )
    if removed_messages:
        raise ValueError(
            "Removed search-eval config detected. " + " ".join(removed_messages)
        )


def normalize_search_eval_cfg(eval_cfg: Any) -> dict[str, Any]:
    plain_eval_cfg = _to_plain_mapping(eval_cfg, field_name="eval_cfg")
    _assert_no_legacy_search_eval_keys(plain_eval_cfg)
    cfg = _deep_merge(base=_DEFAULT_SEARCH_EVAL_CFG, override=plain_eval_cfg)
    monte_carlo = _to_plain_mapping(
        cfg.get("monte_carlo", {}), field_name="eval_cfg.monte_carlo"
    )
    cfg["monte_carlo"] = monte_carlo
    early_stop = _to_plain_mapping(
        monte_carlo.get("early_stop", {}), field_name="eval_cfg.monte_carlo.early_stop"
    )
    monte_carlo["early_stop"] = early_stop
    action_pruning = _to_plain_mapping(
        monte_carlo.get("action_pruning", {}),
        field_name="eval_cfg.monte_carlo.action_pruning",
    )
    monte_carlo["action_pruning"] = action_pruning

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

    if task == EDGE_RETRIEVAL_TASK and report_profile != RANK_ONLY_REPORT:
        raise ValueError(
            "edge_retrieval only supports evaluation.report_profile='rank_only'."
        )

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

    rollouts = int(monte_carlo.get("rollouts", 4096))
    if rollouts < 1:
        raise ValueError("evaluation.monte_carlo.rollouts must be >= 1.")
    monte_carlo["rollouts"] = rollouts

    batch_rollouts = int(monte_carlo.get("batch_rollouts", 256))
    if batch_rollouts < 1:
        raise ValueError("evaluation.monte_carlo.batch_rollouts must be >= 1.")
    monte_carlo["batch_rollouts"] = batch_rollouts

    temperature = float(monte_carlo.get("temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("evaluation.monte_carlo.temperature must be > 0.")
    monte_carlo["temperature"] = temperature

    confidence = float(monte_carlo.get("confidence", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("evaluation.monte_carlo.confidence must be in (0, 1).")
    monte_carlo["confidence"] = confidence

    early_stop_enabled = bool(early_stop.get("enabled", True))
    early_stop["enabled"] = early_stop_enabled

    early_stop_min_rollouts = int(early_stop.get("min_rollouts", 512))
    if early_stop_min_rollouts < 1:
        raise ValueError("evaluation.monte_carlo.early_stop.min_rollouts must be >= 1.")
    early_stop["min_rollouts"] = early_stop_min_rollouts

    stability_top_k = int(early_stop.get("stability_top_k", 1))
    if stability_top_k < 1:
        raise ValueError(
            "evaluation.monte_carlo.early_stop.stability_top_k must be >= 1."
        )
    early_stop["stability_top_k"] = stability_top_k

    per_node_top_k = int(action_pruning.get("per_node_top_k", 100))
    if per_node_top_k < 1:
        raise ValueError(
            "evaluation.monte_carlo.action_pruning.per_node_top_k must be >= 1."
        )
    action_pruning["per_node_top_k"] = per_node_top_k

    per_state_top_k = int(action_pruning.get("per_state_top_k", 256))
    if per_state_top_k < 1:
        raise ValueError(
            "evaluation.monte_carlo.action_pruning.per_state_top_k must be >= 1."
        )
    action_pruning["per_state_top_k"] = per_state_top_k

    return cfg


def search_eval_monte_carlo_cfg(eval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_search_eval_cfg(eval_cfg)
    monte_carlo = normalized["monte_carlo"]
    if not isinstance(monte_carlo, dict):
        raise TypeError(
            "normalize_search_eval_cfg must return a mapping for monte_carlo."
        )
    return monte_carlo


def format_search_eval_answer_posterior(eval_cfg: Mapping[str, Any]) -> str:
    monte_carlo_cfg = search_eval_monte_carlo_cfg(eval_cfg)
    early_stop_cfg = monte_carlo_cfg["early_stop"]
    action_pruning_cfg = monte_carlo_cfg["action_pruning"]
    return (
        "monte_carlo("
        f"rollouts={int(monte_carlo_cfg['rollouts'])}, "
        f"batch_rollouts={int(monte_carlo_cfg['batch_rollouts'])}, "
        f"temperature={float(monte_carlo_cfg['temperature'])}, "
        f"early_stop={bool(early_stop_cfg['enabled'])}@{float(monte_carlo_cfg['confidence'])}/"
        f"min={int(early_stop_cfg['min_rollouts'])}/topk={int(early_stop_cfg['stability_top_k'])}, "
        f"prune=node:{int(action_pruning_cfg['per_node_top_k'])},"
        f"state:{int(action_pruning_cfg['per_state_top_k'])}"
        ")"
    )


def search_eval_answer_posterior_signature(
    eval_cfg: Mapping[str, Any],
) -> tuple[Any, ...]:
    monte_carlo_cfg = search_eval_monte_carlo_cfg(eval_cfg)
    early_stop_cfg = monte_carlo_cfg["early_stop"]
    action_pruning_cfg = monte_carlo_cfg["action_pruning"]
    return (
        "monte_carlo",
        int(monte_carlo_cfg["rollouts"]),
        int(monte_carlo_cfg["batch_rollouts"]),
        float(monte_carlo_cfg["temperature"]),
        float(monte_carlo_cfg["confidence"]),
        bool(early_stop_cfg["enabled"]),
        int(early_stop_cfg["min_rollouts"]),
        int(early_stop_cfg["stability_top_k"]),
        int(action_pruning_cfg["per_node_top_k"]),
        int(action_pruning_cfg["per_state_top_k"]),
    )


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
    "FULL_REPORT",
    "RANK_ONLY_REPORT",
    "RUNTIME_ANSWER_TASK",
    "format_search_eval_answer_posterior",
    "normalize_search_eval_cfg",
    "search_eval_answer_posterior_signature",
    "search_eval_include_answer_support",
    "search_eval_is_answer_task",
    "search_eval_monte_carlo_cfg",
    "search_eval_runtime_task",
]
