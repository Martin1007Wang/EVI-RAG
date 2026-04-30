from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: nocover
    torch = None


def _json_ready(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_metrics_json(*, path: Path, metrics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_metrics_jsonl(
    *,
    path: Path,
    stage: str,
    metrics: dict[str, Any],
    step: int | None,
    epoch: int | None,
    metadata: dict[str, Any] | None = None,
    record_kind: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "step": None if step is None else int(step),
        "epoch": None if epoch is None else int(epoch),
        "record_kind": str(record_kind) if record_kind is not None else "metrics",
        "metrics": _json_ready(metrics),
    }
    if metadata:
        payload["metadata"] = _json_ready(metadata)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


__all__ = ["write_metrics_json", "write_metrics_jsonl"]
