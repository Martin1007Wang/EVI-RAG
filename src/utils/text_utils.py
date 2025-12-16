from __future__ import annotations

from typing import Optional


def count_tokens(text: str, *, encoding: Optional[str] = "cl100k_base") -> int:
    """
    Lightweight token counter with graceful fallback.

    - If tiktoken is available and an encoding is specified, use it.
    - Otherwise, fall back to whitespace-split length to avoid hard dependency.
    """
    raw = text or ""
    if encoding is not None:
        try:  # pragma: no cover - optional dependency path
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            return int(len(enc.encode(raw)))
        except Exception:
            pass
    return int(len(raw.split()))


__all__ = ["count_tokens"]
