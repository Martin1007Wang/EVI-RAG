from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LossOutput",
    "Policy",
    "PolicyContext",
    "PolicyOutput",
    "RewardModel",
    "State",
    "WeaverModule",
]

_EXPORTS = {
    "LossOutput": ("src.weaver.loss", "LossOutput"),
    "Policy": ("src.weaver.policy", "Policy"),
    "PolicyContext": ("src.weaver.policy", "PolicyContext"),
    "PolicyOutput": ("src.weaver.policy", "PolicyOutput"),
    "RewardModel": ("src.weaver.reward", "RewardModel"),
    "State": ("src.weaver.state", "State"),
    # REMOVED: SubTB export — see methodology.md §3.9
    "WeaverModule": ("src.weaver.module", "WeaverModule"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
