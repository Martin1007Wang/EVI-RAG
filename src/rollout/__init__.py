"""Legacy rollout namespace kept only for metrics IO helpers."""

from __future__ import annotations

from .io.metrics_io import to_serializable, write_metrics_jsonl

__all__ = [
    "to_serializable",
    "write_metrics_jsonl",
]
