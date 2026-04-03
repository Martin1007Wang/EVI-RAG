from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _serialize_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): to_serializable(value) for key, value in values.items()}


def write_metrics_json(
    *,
    path: Path,
    metrics: Mapping[str, Any],
    indent: int = 2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_mapping(metrics)
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=True), encoding="utf-8"
    )
    return path


def write_metrics_jsonl(
    *,
    path: Path,
    stage: str,
    metrics: Mapping[str, Any],
    step: int,
    epoch: int | None = None,
    record_kind: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": _serialize_mapping(metrics),
    }
    if record_kind is not None:
        record["record_kind"] = str(record_kind)
    if metadata:
        record["metadata"] = _serialize_mapping(metadata)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


__all__ = ["to_serializable", "write_metrics_json", "write_metrics_jsonl"]
