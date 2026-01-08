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
    "compute_answer_recall",
    "compute_answer_hit",
    "summarize_uncertainty",
    "setup_optimizer",
    "RankedLogger",
    "Registry",
    "enforce_tags",
    "print_config_tree",
    "extras",
    "get_metric_value",
    "task_wrapper",
    "resolve_run_name",
    "apply_run_name",
]

if TYPE_CHECKING:  # pragma: no cover
    from .config import (
        apply_run_name,
        enforce_tags,
        extras,
        get_metric_value,
        instantiate_callbacks,
        instantiate_loggers,
        print_config_tree,
        resolve_run_name,
        task_wrapper,
    )
    from .logging import RankedLogger, infer_batch_size, log_hyperparameters, log_metric
    from .metrics import (
        compute_answer_hit,
        compute_answer_recall,
        extract_answer_entity_ids,
        extract_sample_ids,
        normalize_k_values,
        summarize_uncertainty,
    )
    from .optimization import setup_optimizer
    from .registry import Registry


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in (
        "instantiate_callbacks",
        "instantiate_loggers",
        "enforce_tags",
        "print_config_tree",
        "extras",
        "get_metric_value",
        "task_wrapper",
        "resolve_run_name",
        "apply_run_name",
    ):
        from .config import (
            apply_run_name,
            enforce_tags,
            extras,
            get_metric_value,
            instantiate_callbacks,
            instantiate_loggers,
            print_config_tree,
            resolve_run_name,
            task_wrapper,
        )

        return {
            "instantiate_callbacks": instantiate_callbacks,
            "instantiate_loggers": instantiate_loggers,
            "enforce_tags": enforce_tags,
            "print_config_tree": print_config_tree,
            "extras": extras,
            "get_metric_value": get_metric_value,
            "task_wrapper": task_wrapper,
            "resolve_run_name": resolve_run_name,
            "apply_run_name": apply_run_name,
        }[name]

    if name in ("log_hyperparameters", "infer_batch_size", "log_metric", "RankedLogger"):
        from .logging import RankedLogger, infer_batch_size, log_hyperparameters, log_metric

        return {
            "log_hyperparameters": log_hyperparameters,
            "infer_batch_size": infer_batch_size,
            "log_metric": log_metric,
            "RankedLogger": RankedLogger,
        }[name]

    if name in (
        "normalize_k_values",
        "extract_sample_ids",
        "extract_answer_entity_ids",
        "compute_answer_recall",
        "compute_answer_hit",
        "summarize_uncertainty",
    ):
        from .metrics import (
            compute_answer_hit,
            compute_answer_recall,
            extract_answer_entity_ids,
            extract_sample_ids,
            normalize_k_values,
            summarize_uncertainty,
        )

        return {
            "normalize_k_values": normalize_k_values,
            "extract_sample_ids": extract_sample_ids,
            "extract_answer_entity_ids": extract_answer_entity_ids,
            "compute_answer_recall": compute_answer_recall,
            "compute_answer_hit": compute_answer_hit,
            "summarize_uncertainty": summarize_uncertainty,
        }[name]

    if name == "setup_optimizer":
        from .optimization import setup_optimizer

        return setup_optimizer

    if name == "Registry":
        from .registry import Registry

        return Registry

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
