from __future__ import annotations

from typing import List, Sequence

from src.data.schema.constants import _ALLOWED_SPLITS


def _validate_split_names(splits: Sequence[str], *, context: str) -> List[str]:
    normalized = [str(split) for split in splits]
    unknown = sorted(set(normalized) - set(_ALLOWED_SPLITS))
    if unknown:
        raise ValueError(f"{context} contains unsupported split names: {unknown}. Expected one of {_ALLOWED_SPLITS}.")
    return normalized


def _assert_allowed_split_name(split: str) -> str:
    split = str(split)
    _validate_split_names([split], context="split")
    return split
