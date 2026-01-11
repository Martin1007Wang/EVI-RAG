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
    "log_metric",
    "setup_optimizer",
    "RankedLogger",
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
    from .logging_utils import RankedLogger, log_hyperparameters, log_metric
    from .optimization import setup_optimizer


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

    if name in ("log_hyperparameters", "log_metric", "RankedLogger"):
        from .logging_utils import RankedLogger, log_hyperparameters, log_metric

        return {
            "log_hyperparameters": log_hyperparameters,
            "log_metric": log_metric,
            "RankedLogger": RankedLogger,
        }[name]

    if name == "setup_optimizer":
        from .optimization import setup_optimizer

        return setup_optimizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
