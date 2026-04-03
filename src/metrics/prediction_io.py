from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from src.metrics.serialization import to_serializable


class PredictionCodecProtocol(Protocol):
    kind: str

    def serialize_result(self, result: Any) -> dict[str, Any]: ...

    def serialize_label(self, label: Any) -> dict[str, Any]: ...

    def deserialize_result(self, record: dict[str, Any]) -> Any: ...

    def deserialize_label(self, record: dict[str, Any]) -> Any: ...


def append_jsonl_records(path: str | Path, *, records: Iterable[Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = record
            if is_dataclass(record):
                payload = asdict(record)  # type: ignore[arg-type]
            handle.write(json.dumps(to_serializable(payload), ensure_ascii=True) + "\n")


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield dict(json.loads(text))


def jsonl_has_records(path: str | Path | None) -> bool:
    if path is None:
        return False
    resolved = Path(path)
    if not resolved.exists():
        return False
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


__all__ = [
    "PredictionCodecProtocol",
    "append_jsonl_records",
    "iter_jsonl_records",
    "jsonl_has_records",
]
