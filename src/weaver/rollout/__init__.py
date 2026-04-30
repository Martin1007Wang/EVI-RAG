from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RolloutBatch",
    "RolloutRunner",
]

_EXPORTS = {
    "RolloutBatch": ("src.weaver.rollout.schema", "RolloutBatch"),
    "RolloutRunner": ("src.weaver.rollout.runner", "RolloutRunner"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
