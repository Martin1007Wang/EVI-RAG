from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_dtype = torch.float32
_dtype_name = "float32"
_item_size = torch.empty((), dtype=_dtype).element_size()


@dataclass(frozen=True)
class TensorTable:
    path: Path
    shape: tuple[int, int]

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]


class TensorTableWriter:
    def __init__(self, path: str | Path, *, rows: int, dim: int) -> None:
        self.path = Path(path)
        self.rows = int(rows)
        self.dim = int(dim)

        _check_shape(self.rows, self.dim)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w+b")
        self._handle.truncate(self.rows * self.dim * _item_size)

    @property
    def table(self) -> TensorTable:
        return TensorTable(path=self.path, shape=(self.rows, self.dim))

    def write(self, start: int, rows: torch.Tensor) -> None:
        start = int(start)
        data = rows.to(dtype=_dtype, device="cpu").contiguous()

        if data.ndim != 2:
            raise ValueError(f"rows must be 2D, got shape={tuple(data.shape)}")
        if int(data.size(1)) != self.dim:
            raise ValueError(f"row dim mismatch: got {data.size(1)}, expected {self.dim}")

        end = start + int(data.size(0))
        if start < 0 or end > self.rows:
            raise ValueError(
                f"row write out of bounds: start={start}, end={end}, table_rows={self.rows}"
            )

        self._handle.seek(start * self.dim * _item_size)
        self._handle.write(data.numpy().tobytes(order="C"))

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def __enter__(self) -> TensorTableWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def write_table(path: str | Path, tensor: torch.Tensor) -> TensorTable:
    data = tensor.to(dtype=_dtype, device="cpu").contiguous()
    if data.ndim != 2:
        raise ValueError(f"tensor table must be 2D, got {tuple(data.shape)}")

    with TensorTableWriter(path, rows=int(data.size(0)), dim=int(data.size(1))) as writer:
        writer.write(0, data)
        return writer.table


def read_table(table: TensorTable) -> torch.Tensor:
    validate_file(table)
    rows, dim = table.shape
    data = torch.from_file(
        str(table.path),
        shared=False,
        size=rows * dim,
        dtype=_dtype,
    )
    return data.view(rows, dim)


def manifest_entry(*, path: str | Path, shape: tuple[int, int]) -> dict[str, Any]:
    rows, dim = shape
    _check_shape(int(rows), int(dim))
    return {
        "path": str(path),
        "dtype": _dtype_name,
        "shape": [int(rows), int(dim)],
    }


def from_manifest(*, entry: Mapping[str, Any], metadata_dir: str | Path) -> TensorTable:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("tensor table manifest entry requires non-empty 'path'")

    dtype = str(entry.get("dtype", "")).strip()
    if dtype != _dtype_name:
        raise ValueError(f"unsupported tensor table dtype: {dtype!r}")

    raw_shape = entry.get("shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 2
        or not all(isinstance(value, int) for value in raw_shape)
    ):
        raise ValueError("tensor table manifest entry requires shape [rows, dim]")

    rows, dim = int(raw_shape[0]), int(raw_shape[1])
    _check_shape(rows, dim)

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(metadata_dir) / path

    return TensorTable(path=path, shape=(rows, dim))


def validate_file(table: TensorTable) -> None:
    rows, dim = table.shape
    _check_shape(rows, dim)

    if not table.path.is_file():
        raise FileNotFoundError(f"tensor table file not found: {table.path}")

    expected = rows * dim * _item_size
    actual = table.path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"tensor table size mismatch for {table.path}: got {actual} bytes, expected {expected}"
        )


def _check_shape(rows: int, dim: int) -> None:
    if rows < 0:
        raise ValueError(f"rows must be nonnegative, got {rows}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
