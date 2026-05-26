from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lmdb
import torch
from torch_geometric.data import Dataset

from src.data.keys import row_key
from src.data.tensor_table import read_table
from src.utils.lmdb_utils import deserialize_sample

from .artifacts import MaterializationArtifact
from .schema.batch import RetrievalData
from .schema.fields import SampleFields


class LMDBSampleStore:
    def __init__(
        self,
        path: str | Path,
        *,
        readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.path}")

        self.env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=max_readers,
            subdir=True,
        )

    def load_sample(self, row_idx: int) -> dict[str, torch.Tensor]:
        with self.env.begin(write=False) as txn:
            payload = txn.get(row_key(row_idx))

        if payload is None:
            raise KeyError(f"Sample row not found in {self.path}: {row_idx}")

        return deserialize_sample(payload)

    def close(self) -> None:
        self.env.close()


class RetrievalDataset(Dataset):
    """
    Dataset over materialized LMDB samples.

    Responsibilities:
    - open the single LMDB path for the split
    - deserialize the sample payload
    - convert the storage record into RetrievalData

    Non-responsibilities:
    - no secondary row-to-sample indirection
    - no runtime sample filtering
    - no storage field fallback
    - no path recomputation
    - no anchor/target mask materialization
    - no entity/relation embedding loading
    """

    def __init__(
        self,
        *,
        materialization: MaterializationArtifact,
        split: str,
        lmdb_readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        super().__init__()

        self.materialization = materialization
        self.split = str(split).strip()

        if not self.split:
            raise ValueError("split must be non-empty")

        self.lmdb_readahead = bool(lmdb_readahead)
        self.max_readers = int(max_readers)
        if self.max_readers <= 0:
            raise ValueError("max_readers must be positive")

        paths = self.materialization.require_split(self.split)
        self.lmdb_path = paths.lmdb
        self.num_samples = int(paths.num_samples)
        self._question_embeddings = read_table(paths.question_embeddings)
        if int(self._question_embeddings.size(0)) != self.num_samples:
            raise ValueError(
                f"question embedding rows mismatch for split {self.split}: "
                f"got {int(self._question_embeddings.size(0))}, expected {self.num_samples}"
            )

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.lmdb_path}")

        self._store: LMDBSampleStore | None = None

    def len(self) -> int:
        return self.num_samples

    def get(self, idx: int) -> RetrievalData:
        question_emb = self._question_embeddings[idx]
        raw = self._get_store().load_sample(int(idx))
        sample_id = _sample_id(raw, row_idx=int(idx))
        return _build_retrieval_data(
            raw=raw,
            sample_id=sample_id,
            question_emb=question_emb,
        )

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()
            self._store = None

    def _get_store(self) -> LMDBSampleStore:
        if self._store is None:
            self._store = LMDBSampleStore(
                self.lmdb_path,
                readahead=self.lmdb_readahead,
                max_readers=self.max_readers,
            )

        return self._store

    def __del__(self) -> None:
        self.close()


def _build_retrieval_data(
    *,
    raw: Mapping[str, Any],
    sample_id: str,
    question_emb: torch.Tensor | None,
) -> RetrievalData:
    raw = dict(raw)
    num_nodes = _scalar_int(raw[SampleFields.NUM_NODES], SampleFields.NUM_NODES)
    num_edges = _scalar_int(raw[SampleFields.NUM_EDGES], SampleFields.NUM_EDGES)

    edge_index = _tensor(raw[SampleFields.EDGE_INDEX], dtype=torch.long)

    node_entity_catalog_ids = _tensor(
        raw[SampleFields.NODE_ENTITY_CATALOG_IDS],
        dtype=torch.long,
    )

    edge_relation_catalog_ids = _tensor(
        raw[SampleFields.EDGE_RELATION_CATALOG_IDS],
        dtype=torch.long,
    )

    if question_emb is None:
        raise KeyError(f"{sample_id}: missing external question embedding")
    question_emb = _tensor(question_emb, dtype=torch.float32)

    anchor_node_ids = _tensor(
        raw[SampleFields.ANCHOR_NODE_IDS],
        dtype=torch.long,
    )

    target_node_ids = _tensor(
        raw[SampleFields.TARGET_NODE_IDS],
        dtype=torch.long,
    )

    reachable_target_node_ids = _tensor(
        raw[SampleFields.REACHABLE_TARGET_NODE_IDS],
        dtype=torch.long,
    )

    node_target_distance = _tensor(
        raw[SampleFields.NODE_TARGET_DISTANCE],
        dtype=torch.long,
    )

    weak_replay_edge_ids, weak_replay_edge_weight = _weak_replay_fields(
        raw=raw,
        sample_id=sample_id,
    )
    return RetrievalData(
        sample_id=sample_id,
        edge_index=edge_index,
        node_entity_catalog_ids=node_entity_catalog_ids,
        edge_relation_catalog_ids=edge_relation_catalog_ids,
        num_nodes=num_nodes,
        num_edges=num_edges,
        question_emb=question_emb,
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        reachable_target_node_ids=reachable_target_node_ids,
        node_target_distance=node_target_distance,
        weak_replay_edge_ids=weak_replay_edge_ids,
        weak_replay_edge_weight=weak_replay_edge_weight,
    )


def _sample_id(raw: Mapping[str, Any], *, row_idx: int) -> str:
    value = raw.get(SampleFields.SAMPLE_ID)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"row {row_idx}: missing {SampleFields.SAMPLE_ID}")
    if value.dtype != torch.uint8:
        raise TypeError(
            f"row {row_idx}: {SampleFields.SAMPLE_ID} must be uint8, got {value.dtype}"
        )
    if value.ndim != 1:
        raise ValueError(
            f"row {row_idx}: {SampleFields.SAMPLE_ID} must be 1D, got shape={tuple(value.shape)}"
        )
    return bytes(value.tolist()).decode("utf-8")


def _tensor(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == dtype:
            return value.contiguous()
        return value.to(dtype=dtype).contiguous()

    return torch.as_tensor(value, dtype=dtype).contiguous()


def _optional_tensor(
    raw: Mapping[str, Any],
    key: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = raw.get(key)
    if value is None:
        return torch.empty((0,), dtype=dtype)
    return _tensor(value, dtype=dtype)


def _weak_replay_fields(
    *,
    raw: Mapping[str, Any],
    sample_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_ids = raw.get(SampleFields.WEAK_REPLAY_EDGE_IDS)
    weight = raw.get(SampleFields.WEAK_REPLAY_EDGE_WEIGHT)
    if edge_ids is None or weight is None:
        raise KeyError(f"{sample_id}: missing weak replay edge labels")
    edge_tensor = _tensor(edge_ids, dtype=torch.long).view(-1)
    weight_tensor = _tensor(weight, dtype=torch.float32).view(-1)
    if int(edge_tensor.numel()) != int(weight_tensor.numel()):
        raise ValueError(
            f"{sample_id}: weak replay edge fields length mismatch: "
            f"edge_ids={int(edge_tensor.numel())}, weight={int(weight_tensor.numel())}"
        )
    return (
        edge_tensor.contiguous(),
        weight_tensor.contiguous(),
    )


def _scalar_int(value: object, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(
                f"{name} must be a scalar tensor, got {tuple(value.shape)}"
            )
        return int(value.item())

    if isinstance(value, int):
        return value

    raise TypeError(
        f"{name} must be an int or scalar tensor, got {type(value).__name__}"
    )
