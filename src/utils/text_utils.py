from __future__ import annotations

from typing import Optional


def count_tokens(text: str, *, encoding: Optional[str] = "cl100k_base") -> int:
    """
    Lightweight token counter (strict).
    """
    raw = text or ""
    if encoding is None:
        raise ValueError("encoding must be provided for token counting.")
    try:  # pragma: no cover - dependency check
        import tiktoken
    except ModuleNotFoundError as exc:
        raise ImportError("tiktoken is required for token counting.") from exc
    enc = tiktoken.get_encoding(encoding)
    return int(len(enc.encode(raw)))


__all__ = ["count_tokens"]
