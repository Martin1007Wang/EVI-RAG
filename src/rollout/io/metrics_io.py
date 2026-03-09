from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def to_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.reshape(()).item())
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def write_metrics_jsonl(
    *,
    path: Path,
    stage: str,
    metrics: Dict[str, Any],
    step: int,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {k: to_serializable(v) for k, v in metrics.items()},
    }
    if metadata:
        record["metadata"] = {k: to_serializable(v) for k, v in metadata.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


__all__ = ["to_serializable", "write_metrics_jsonl"]
