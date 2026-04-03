from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from src.utils.logging_utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

_PROFILE_ENV = "RANKFLOW_PROFILE_CUDA_MEMORY"
_MEMORY_RECORDS: list[dict[str, Any]] = []


def format_cuda_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_idx = 0
    while abs(value) >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def cuda_memory_profiling_enabled() -> bool:
    raw = str(os.getenv(_PROFILE_ENV, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def reset_cuda_memory_records() -> None:
    _MEMORY_RECORDS.clear()


def get_cuda_memory_records() -> list[dict[str, Any]]:
    return [dict(record) for record in _MEMORY_RECORDS]


def _resolve_cuda_device(device: torch.device | None = None) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _safe_cuda_snapshot(device: torch.device) -> dict[str, int | None]:
    try:
        torch.cuda.synchronize(device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception:
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
        }


@contextmanager
def profile_cuda_memory(
    label: str,
    *,
    device: torch.device | None = None,
    extra: str | None = None,
    reset_peak: bool = True,
) -> Iterator[None]:
    resolved_device = _resolve_cuda_device(device)
    if not cuda_memory_profiling_enabled() or resolved_device is None:
        yield
        return

    if reset_peak:
        try:
            torch.cuda.reset_peak_memory_stats(resolved_device)
        except Exception:
            pass

    before = _safe_cuda_snapshot(resolved_device)
    start_time = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        after = _safe_cuda_snapshot(resolved_device)
        elapsed_s = time.perf_counter() - start_time
        record = {
            "label": str(label),
            "device": str(resolved_device),
            "extra": None if extra in (None, "") else str(extra),
            "status": status,
            "elapsed_s": float(elapsed_s),
            **{f"before_{key}": value for key, value in before.items()},
            **{f"after_{key}": value for key, value in after.items()},
        }
        before_allocated = before.get("allocated_bytes")
        after_allocated = after.get("allocated_bytes")
        before_reserved = before.get("reserved_bytes")
        after_reserved = after.get("reserved_bytes")
        peak_allocated = after.get("max_allocated_bytes")
        peak_reserved = after.get("max_reserved_bytes")
        record["delta_allocated_bytes"] = (
            None
            if before_allocated is None or after_allocated is None
            else int(after_allocated) - int(before_allocated)
        )
        record["delta_reserved_bytes"] = (
            None
            if before_reserved is None or after_reserved is None
            else int(after_reserved) - int(before_reserved)
        )
        record["peak_delta_allocated_bytes"] = (
            None
            if before_allocated is None or peak_allocated is None
            else int(peak_allocated) - int(before_allocated)
        )
        record["peak_delta_reserved_bytes"] = (
            None
            if before_reserved is None or peak_reserved is None
            else int(peak_reserved) - int(before_reserved)
        )
        _MEMORY_RECORDS.append(record)
        log.info(
            "[cuda-mem] label=%s status=%s elapsed_s=%.3f alloc=%s->%s delta=%s peak_delta=%s reserved=%s->%s delta=%s peak_delta=%s extra=%s",
            label,
            status,
            elapsed_s,
            format_cuda_bytes(before_allocated),
            format_cuda_bytes(after_allocated),
            format_cuda_bytes(record["delta_allocated_bytes"]),
            format_cuda_bytes(record["peak_delta_allocated_bytes"]),
            format_cuda_bytes(before_reserved),
            format_cuda_bytes(after_reserved),
            format_cuda_bytes(record["delta_reserved_bytes"]),
            format_cuda_bytes(record["peak_delta_reserved_bytes"]),
            extra,
        )


__all__ = [
    "cuda_memory_profiling_enabled",
    "format_cuda_bytes",
    "get_cuda_memory_records",
    "profile_cuda_memory",
    "reset_cuda_memory_records",
]
