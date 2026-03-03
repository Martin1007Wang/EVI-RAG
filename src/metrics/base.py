from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch

_DEFAULT_QUANTILE = 0.95


def _to_iterable(raw: Any) -> Iterable[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set, range)):
        return raw
    if isinstance(raw, str):
        return [raw]
    try:
        return list(raw)
    except TypeError:
        return [raw]


def normalize_k_values(
    raw_values: Any, default: Optional[Sequence[int]] = None
) -> List[int]:
    normalized: List[int] = []
    seen = set()
    for item in _to_iterable(raw_values):
        try:
            k = int(item)
        except (TypeError, ValueError):
            continue
        if k <= 0 or k in seen:
            continue
        normalized.append(k)
        seen.add(k)
    if not normalized and default is not None:
        return normalize_k_values(default, default=None)
    normalized.sort()
    return normalized


def summarize_uncertainty(
    values: Iterable[torch.Tensor], quantile: float = _DEFAULT_QUANTILE
) -> Tuple[float, float]:
    tensors = [v for v in values if isinstance(v, torch.Tensor) and v.numel() > 0]
    if not tensors:
        return 0.0, 0.0
    concat = torch.cat(tensors)
    mean = float(concat.mean().detach().tolist())
    quant = float(concat.quantile(quantile).detach().tolist())
    return mean, quant


__all__ = ["normalize_k_values", "summarize_uncertainty"]
