from __future__ import annotations

from typing import Dict, List, Sequence

from src.data.schema.constants import _REL_LABEL_SAMPLE_LIMIT


def _init_split_counters(
    splits: Sequence[str], keys: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    return {split: {key: 0 for key in keys} for split in splits}


def _safe_div(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _sample_labels(labels: set[str], limit: int = _REL_LABEL_SAMPLE_LIMIT) -> List[str]:
    if limit <= 0:
        return []
    return sorted(labels)[:limit]
