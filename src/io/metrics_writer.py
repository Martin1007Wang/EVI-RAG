from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .serialization import write_metrics_jsonl


def append_stage_metrics(
    output_dir: Path,
    *,
    stage: str,
    step: int | None,
    metrics: Mapping[str, Any],
    epoch: int | None = None,
    record_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    file_name: str | None = None,
) -> Path:
    path = output_dir / (file_name or f"{stage}.jsonl")
    return write_metrics_jsonl(
        path=path,
        stage=stage,
        metrics={str(name): value for name, value in metrics.items()},
        step=step,
        epoch=epoch,
        metadata=metadata,
        record_kind=record_kind,
    )


__all__ = ["append_stage_metrics"]
