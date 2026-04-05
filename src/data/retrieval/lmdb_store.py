from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, cast
import lmdb
from src.utils.lmdb_utils import deserialize_sample


class LMDBSampleStore:
    """LMDB wrapper for reading graph samples (read-only, multiprocessing-safe)."""

    def __init__(self, lmdb_path: Path, *, readahead: bool = False):
        lmdb_path = Path(lmdb_path)  # 确保是Path对象
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")

        self.path = lmdb_path
        self._readahead = bool(readahead)
        self.env: Optional[lmdb.Environment] = lmdb.open(
            str(self.path),  # lmdb.open需要字符串参数
            readonly=True,
            lock=False,
            readahead=self._readahead,
            meminit=False,
            max_readers=256,
        )

    def load_sample(self, sample_id: str) -> Dict:
        if self.env is None:
            raise RuntimeError(
                "Cannot load sample: LMDB environment is closed or uninitialized."
            )

        with self.env.begin(write=False) as txn:
            data = txn.get(sample_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Sample {sample_id} not found in {self.path}")
            return deserialize_sample(cast(bytes, data))

    def close(self) -> None:
        if self.env:
            self.env.close()
            self.env = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
