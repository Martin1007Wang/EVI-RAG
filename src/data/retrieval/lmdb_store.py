# src/data/retrieval/lmdb_store.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, cast
import lmdb
from src.utils.lmdb_utils import deserialize_sample
_ENV_REGISTRY: Dict[str, lmdb.Environment] = {}
def _get_or_open_env(path: Path, *, readahead: bool) -> lmdb.Environment:
    key = str(path.resolve())
    if key not in _ENV_REGISTRY:
        _ENV_REGISTRY[key] = lmdb.open(
            key,
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=256,
        )
    return _ENV_REGISTRY[key]
class LMDBSampleStore:
    def __init__(self, lmdb_path: Path, *, readahead: bool = False):
        self.path = Path(lmdb_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"LMDB not found: {self.path}")
        self._readahead = bool(readahead)
    @property
    def env(self) -> lmdb.Environment:
        return _get_or_open_env(self.path, readahead=self._readahead)
    def load_sample(self, sample_id: str) -> Dict:
        with self.env.begin(write=False) as txn:
            data = txn.get(sample_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Sample {sample_id!r} not found in {self.path}")
            return deserialize_sample(cast(bytes, data))
    def close(self) -> None:
        pass
    def __getstate__(self) -> dict:
        return {"path": self.path, "_readahead": self._readahead}
    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)