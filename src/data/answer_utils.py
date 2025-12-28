from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List


def normalize_answer_texts(raw: Any, *, field_name: str, sample_id: str) -> List[str]:
    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty list (sample_id={sample_id})")

    if isinstance(raw, (list, tuple)):
        seq: List[Any] = list(raw)
    elif hasattr(raw, "tolist"):
        seq = raw.tolist()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        seq = list(raw)
    else:
        raise ValueError(f"{field_name} must be a non-empty list (sample_id={sample_id})")

    if not isinstance(seq, list) or not seq:
        raise ValueError(f"{field_name} must be a non-empty list (sample_id={sample_id})")

    answers: List[str] = []
    for idx, ans in enumerate(seq):
        if not isinstance(ans, str):
            raise ValueError(
                f"{field_name}[{idx}] must be string (sample_id={sample_id}), got {type(ans).__name__}"
            )
        text = ans.strip()
        if not text:
            raise ValueError(f"{field_name}[{idx}] is empty (sample_id={sample_id})")
        answers.append(text)
    return answers


__all__ = ["normalize_answer_texts"]
