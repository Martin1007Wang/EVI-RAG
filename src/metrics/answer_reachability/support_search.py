from __future__ import annotations

from typing import Protocol

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import PreparedSearchBatch, SearchPolicyProtocol

from .analysis import ReachabilityAnalysis
from .schema import SupportWindowResult


class SupportSearchProtocol(Protocol):
    requires_analysis: bool

    def generate_window(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: ReachabilityAnalysis | None = None,
        include_answer_support: bool = True,
    ) -> SupportWindowResult: ...

    def generate_windows_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        analysis: list[ReachabilityAnalysis] | None = None,
        include_answer_support: bool = True,
    ) -> list[SupportWindowResult]: ...


__all__ = ["SupportSearchProtocol"]
