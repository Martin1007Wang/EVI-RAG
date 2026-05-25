from __future__ import annotations


def row_key(row_idx: int) -> bytes:
    row = int(row_idx)
    if row < 0:
        raise ValueError(f"row_idx must be nonnegative, got {row}")
    return row.to_bytes(8, byteorder="big", signed=False)


__all__ = ["row_key"]
