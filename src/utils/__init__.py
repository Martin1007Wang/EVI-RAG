"""Public utils API (keep import-lightweight).

This package is imported by many modules and scripts. Avoid importing optional
Hydra/OmegaConf/Rich dependencies at module import time; expose them lazily via
`__getattr__` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "infer_batch_size",
    "log_metric",
    "normalize_k_values",
    "extract_sample_ids",
    "extract_answer_entity_ids",
    "compute_ranking_metrics",
    "compute_answer_recall",
    "compute_answer_hit",
    "summarize_uncertainty",
    "RankingStats",
    "setup_optimizer",
    "RankedLogger",
    "Registry",
    "enforce_tags",
    "print_config_tree",
    "extras",
    "get_metric_value",
    "task_wrapper",
]

if TYPE_CHECKING:  # pragma: no cover
    from .instantiators import instantiate_callbacks, instantiate_loggers
    from .logging_utils import infer_batch_size, log_hyperparameters, log_metric
    from .metrics import (
        RankingStats,
        compute_answer_hit,
        compute_answer_recall,
        compute_ranking_metrics,
        extract_answer_entity_ids,
        extract_sample_ids,
        normalize_k_values,
        summarize_uncertainty,
    )
    from .optimization import setup_optimizer
    from .pylogger import RankedLogger
    from .registry import Registry
    from .rich_utils import enforce_tags, print_config_tree
    from .utils import extras, get_metric_value, task_wrapper


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in ("instantiate_callbacks", "instantiate_loggers"):
        from .instantiators import instantiate_callbacks, instantiate_loggers

        return {"instantiate_callbacks": instantiate_callbacks, "instantiate_loggers": instantiate_loggers}[name]

    if name in ("log_hyperparameters", "infer_batch_size", "log_metric"):
        from .logging_utils import infer_batch_size, log_hyperparameters, log_metric

        return {"log_hyperparameters": log_hyperparameters, "infer_batch_size": infer_batch_size, "log_metric": log_metric}[name]

    if name in (
        "normalize_k_values",
        "extract_sample_ids",
        "extract_answer_entity_ids",
        "compute_ranking_metrics",
        "compute_answer_recall",
        "compute_answer_hit",
        "summarize_uncertainty",
        "RankingStats",
    ):
        from .metrics import (
            RankingStats,
            compute_answer_hit,
            compute_answer_recall,
            compute_ranking_metrics,
            extract_answer_entity_ids,
            extract_sample_ids,
            normalize_k_values,
            summarize_uncertainty,
        )

        return {
            "normalize_k_values": normalize_k_values,
            "extract_sample_ids": extract_sample_ids,
            "extract_answer_entity_ids": extract_answer_entity_ids,
            "compute_ranking_metrics": compute_ranking_metrics,
            "compute_answer_recall": compute_answer_recall,
            "compute_answer_hit": compute_answer_hit,
            "summarize_uncertainty": summarize_uncertainty,
            "RankingStats": RankingStats,
        }[name]

    if name == "setup_optimizer":
        from .optimization import setup_optimizer

        return setup_optimizer

    if name == "RankedLogger":
        from .pylogger import RankedLogger

        return RankedLogger

    if name == "Registry":
        from .registry import Registry

        return Registry

    if name in ("enforce_tags", "print_config_tree"):
        from .rich_utils import enforce_tags, print_config_tree

        return {"enforce_tags": enforce_tags, "print_config_tree": print_config_tree}[name]

    if name in ("extras", "get_metric_value", "task_wrapper"):
        from .utils import extras, get_metric_value, task_wrapper

        return {"extras": extras, "get_metric_value": get_metric_value, "task_wrapper": task_wrapper}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)

