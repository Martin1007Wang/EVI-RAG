from __future__ import annotations

from pathlib import Path

import lmdb


def row_key(row_idx: int) -> bytes:
    row = int(row_idx)
    if row < 0:
        raise ValueError(f"row_idx must be nonnegative, got {row}")
    return row.to_bytes(8, byteorder="big", signed=False)


class SplitIndexWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int,
        commit_frequency: int,
    ) -> None:
        self.path = Path(path)
        self.commit_frequency = int(commit_frequency)
        if self.commit_frequency <= 0:
            raise ValueError("commit_frequency must be positive")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            map_size=int(map_size),
            subdir=True,
            lock=True,
            create=True,
            max_dbs=1,
            sync=True,
            writemap=False,
        )
        self.txn: lmdb.Transaction | None = self.env.begin(write=True)
        self.uncommitted = 0

    def put(self, row_idx: int, storage_key: str) -> None:
        if self.txn is None:
            raise RuntimeError("SplitIndexWriter is closed")

        key = str(storage_key)
        if not key:
            raise ValueError("storage_key must be non-empty")

        inserted = self.txn.put(
            row_key(row_idx),
            key.encode("utf-8"),
            overwrite=False,
        )
        if not inserted:
            raise ValueError(f"duplicate split index row: {row_idx}")

        self.uncommitted += 1
        if self.uncommitted >= self.commit_frequency:
            self.commit()

    def commit(self) -> None:
        if self.txn is None:
            raise RuntimeError("SplitIndexWriter is closed")

        self.txn.commit()
        self.txn = self.env.begin(write=True)
        self.uncommitted = 0

    def close(self) -> None:
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        self.env.sync(True)
        self.env.close()

    def __enter__(self) -> SplitIndexWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class SplitIndexReader:
    def __init__(
        self,
        path: str | Path,
        *,
        readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Split index path does not exist: {self.path}")

        self.env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=max_readers,
            subdir=True,
        )

    def get(self, row_idx: int) -> str:
        with self.env.begin(write=False) as txn:
            payload = txn.get(row_key(row_idx))

        if payload is None:
            raise KeyError(f"Split index row not found in {self.path}: {row_idx}")

        return bytes(payload).decode("utf-8")

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> SplitIndexReader:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
