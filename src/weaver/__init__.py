from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CandidateEdges",
    "CoveragePathGuide",
    "LossOutput",
    "MinimalSufficiencyTeacher",
    "Policy",
    "PolicyOutput",
    "PolicyStepOutput",
    "RewardModel",
    "State",
    "SubTrajectoryBalanceLoss",
    "WeaverModule",
]

_EXPORTS = {
    "CandidateEdges": ("src.weaver.policy", "CandidateEdges"),
    "CoveragePathGuide": ("src.weaver.proposal", "CoveragePathGuide"),
    "LossOutput": ("src.weaver.losses", "LossOutput"),
    "MinimalSufficiencyTeacher": (
        "src.weaver.proposal",
        "MinimalSufficiencyTeacher",
    ),
    "Policy": ("src.weaver.policy", "Policy"),
    "PolicyOutput": ("src.weaver.policy", "PolicyOutput"),
    "PolicyStepOutput": ("src.weaver.policy", "PolicyStepOutput"),
    "RewardModel": ("src.weaver.reward", "RewardModel"),
    "State": ("src.weaver.state", "State"),
    "SubTrajectoryBalanceLoss": ("src.weaver.losses", "SubTrajectoryBalanceLoss"),
    "WeaverModule": ("src.weaver.module", "WeaverModule"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
