from __future__ import annotations

from .analysis import EdgeSupportAnalysis, ReachabilityAnalysis


ExactReachabilityAnalysis = ReachabilityAnalysis
ExactEdgeSupportAnalysis = EdgeSupportAnalysis


__all__ = [
    "ExactEdgeSupportAnalysis",
    "ExactReachabilityAnalysis",
]
