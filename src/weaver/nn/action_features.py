from __future__ import annotations

from .transition_features import TransitionFeatureBuilder, TransitionFeatureOutput

ActionFeatureBuilder = TransitionFeatureBuilder
ActionFeatureOutput = TransitionFeatureOutput

__all__ = [
    "ActionFeatureBuilder",
    "ActionFeatureOutput",
    "TransitionFeatureBuilder",
    "TransitionFeatureOutput",
]
