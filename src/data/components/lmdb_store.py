from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import lmdb

from src.data.io.lmdb_utils import _deserialize_sample

_LMDB_MAX_READERS = 256


class EmbeddingStore:
    """LMDB wrapper for reading graph samples (read-only, multiprocessing-safe)."""

    def __init__(self, lmdb_path: Union[str, Path], *, readahead: bool = False):
        self.path = str(lmdb_path)
        self._readahead = bool(readahead)
        self.env: Optional[lmdb.Environment] = None
        if not Path(self.path).exists():
            raise FileNotFoundError(f"LMDB not found: {self.path}")

    def _init_env(self) -> None:
        if self.env is None:
            self.env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=self._readahead,
                meminit=False,
                max_readers=_LMDB_MAX_READERS,
            )

    def load_sample(self, sample_id: str) -> Dict:
        self._init_env()
        with self.env.begin(write=False) as txn:
            data = txn.get(sample_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Sample {sample_id} not found in {self.path}")
            return _deserialize_sample(data)

    def load_samples(self, sample_ids: List[str]) -> List[Dict]:
        self._init_env()
        if not sample_ids:
            return []
        with self.env.begin(write=False) as txn:
            out: List[Dict] = []
            for sample_id in sample_ids:
                data = txn.get(sample_id.encode("utf-8"))
                if data is None:
                    raise KeyError(f"Sample {sample_id} not found in {self.path}")
                out.append(_deserialize_sample(data))
            return out

    def get_sample_ids(self) -> List[str]:
        self._init_env()
        keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(values=False):
                keys.append(key.decode("utf-8"))
        return keys

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
