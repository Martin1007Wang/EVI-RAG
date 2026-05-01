from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LossOutput",
    "Policy",
    "PolicyOutput",
    "RewardModel",
    "State",
    "SubTrajectoryBalanceLoss",
    "WeaverModule",
]

_EXPORTS = {
    "LossOutput": ("src.weaver.loss", "LossOutput"),
    "Policy": ("src.weaver.policy", "Policy"),
    "PolicyOutput": ("src.weaver.policy", "PolicyOutput"),
    "RewardModel": ("src.weaver.reward", "RewardModel"),
    "State": ("src.weaver.state", "State"),
    "SubTrajectoryBalanceLoss": ("src.weaver.loss", "SubTrajectoryBalanceLoss"),
    "WeaverModule": ("src.weaver.module", "WeaverModule"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
