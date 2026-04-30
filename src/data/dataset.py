from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lmdb
import torch
from torch_geometric.data import Dataset

from src.utils.lmdb_utils import deserialize_sample

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

    def load_sample(self, sample_id: str) -> dict[str, torch.Tensor]:
        key = sample_id.encode("utf-8")

        with self.env.begin(write=False) as txn:
            payload = txn.get(key)

        if payload is None:
            raise KeyError(f"Sample not found in {self.path}: {sample_id}")

        return deserialize_sample(payload)

    def close(self) -> None:
        self.env.close()


class RetrievalDataset(Dataset):
    """
    Dataset over materialized LMDB samples.

    Responsibilities:
    - read split index from metadata_dir/{split}.index.pt
    - open the single LMDB file for the split
    - deserialize the sample payload
    - convert the storage record into RetrievalData

    Non-responsibilities:
    - no manifest reading
    - no LMDB key scanning
    - no runtime sample filtering
    - no legacy field fallback
    - no path recomputation
    - no anchor/target mask materialization
    - no entity/relation embedding loading
    """

    def __init__(
        self,
        *,
        lmdb_dir: str | Path,
        metadata_dir: str | Path,
        split: str,
        lmdb_readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        super().__init__()

        self.lmdb_dir = Path(lmdb_dir)
        self.metadata_dir = Path(metadata_dir)
        self.split = str(split).strip()

        if not self.split:
            raise ValueError("split must be non-empty")

        self.lmdb_readahead = bool(lmdb_readahead)
        self.max_readers = int(max_readers)
        if self.max_readers <= 0:
            raise ValueError("max_readers must be positive")

        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"Metadata directory does not exist: {self.metadata_dir}"
            )

        if not self.lmdb_dir.exists():
            raise FileNotFoundError(f"LMDB directory does not exist: {self.lmdb_dir}")

        self.sample_ids = _load_split_index(
            metadata_dir=self.metadata_dir,
            split=self.split,
        )

        self.lmdb_path = _lmdb_path(
            lmdb_dir=self.lmdb_dir,
            split=self.split,
        )
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.lmdb_path}")

        self._store: LMDBSampleStore | None = None

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> RetrievalData:
        sample_id = self.sample_ids[idx]
        raw = self._get_store().load_sample(sample_id)
        return _build_retrieval_data(raw=raw, sample_id=sample_id)

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is None:
            return

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
) -> RetrievalData:
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

    question_emb = _tensor(
        raw[SampleFields.QUESTION_EMB],
        dtype=torch.float32,
    )

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

    anchor_node_forward_distances_flat = _tensor(
        raw[SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT],
        dtype=torch.long,
    )

    anchor_node_backward_distances_flat = _tensor(
        raw[SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT],
        dtype=torch.long,
    )

    node_target_distance = _tensor(
        raw[SampleFields.NODE_TARGET_DISTANCE],
        dtype=torch.long,
    )

    target_node_distances_flat = _tensor(
        raw[SampleFields.TARGET_NODE_DISTANCE_FLAT],
        dtype=torch.long,
    )

    target_shortest_path_count_flat = _tensor(
        raw[SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT],
        dtype=torch.float32,
    )

    target_shortest_path_edge_mask_flat = _tensor(
        raw[SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT],
        dtype=torch.bool,
    )

    _validate_runtime_shapes(
        sample_id=sample_id,
        num_nodes=num_nodes,
        num_edges=num_edges,
        edge_index=edge_index,
        node_entity_catalog_ids=node_entity_catalog_ids,
        edge_relation_catalog_ids=edge_relation_catalog_ids,
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        reachable_target_node_ids=reachable_target_node_ids,
        anchor_node_forward_distances_flat=anchor_node_forward_distances_flat,
        anchor_node_backward_distances_flat=anchor_node_backward_distances_flat,
        node_target_distance=node_target_distance,
        target_node_distances_flat=target_node_distances_flat,
        target_shortest_path_count_flat=target_shortest_path_count_flat,
        target_shortest_path_edge_mask_flat=target_shortest_path_edge_mask_flat,
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
        anchor_node_forward_distances_flat=anchor_node_forward_distances_flat,
        anchor_node_backward_distances_flat=anchor_node_backward_distances_flat,
        node_target_distance=node_target_distance,
        target_node_distances_flat=target_node_distances_flat,
        target_shortest_path_count_flat=target_shortest_path_count_flat,
        target_shortest_path_edge_mask_flat=target_shortest_path_edge_mask_flat,
    )


def _validate_runtime_shapes(
    *,
    sample_id: str,
    num_nodes: int,
    num_edges: int,
    edge_index: torch.Tensor,
    node_entity_catalog_ids: torch.Tensor,
    edge_relation_catalog_ids: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    reachable_target_node_ids: torch.Tensor,
    anchor_node_forward_distances_flat: torch.Tensor,
    anchor_node_backward_distances_flat: torch.Tensor,
    node_target_distance: torch.Tensor,
    target_node_distances_flat: torch.Tensor,
    target_shortest_path_count_flat: torch.Tensor,
    target_shortest_path_edge_mask_flat: torch.Tensor,
) -> None:
    _require_shape(
        sample_id=sample_id,
        name=SampleFields.EDGE_INDEX,
        actual=tuple(edge_index.shape),
        expected=(2, num_edges),
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.NODE_ENTITY_CATALOG_IDS,
        tensor=node_entity_catalog_ids,
        expected=num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.EDGE_RELATION_CATALOG_IDS,
        tensor=edge_relation_catalog_ids,
        expected=num_edges,
    )

    _require_node_ids(
        sample_id=sample_id,
        name=SampleFields.ANCHOR_NODE_IDS,
        tensor=anchor_node_ids,
        num_nodes=num_nodes,
    )

    _require_node_ids(
        sample_id=sample_id,
        name=SampleFields.TARGET_NODE_IDS,
        tensor=target_node_ids,
        num_nodes=num_nodes,
    )

    _require_node_ids(
        sample_id=sample_id,
        name=SampleFields.REACHABLE_TARGET_NODE_IDS,
        tensor=reachable_target_node_ids,
        num_nodes=num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT,
        tensor=anchor_node_forward_distances_flat,
        expected=num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT,
        tensor=anchor_node_backward_distances_flat,
        expected=num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.NODE_TARGET_DISTANCE,
        tensor=node_target_distance,
        expected=num_nodes,
    )

    num_reachable_targets = int(reachable_target_node_ids.numel())

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.TARGET_NODE_DISTANCE_FLAT,
        tensor=target_node_distances_flat,
        expected=num_reachable_targets * num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT,
        tensor=target_shortest_path_count_flat,
        expected=num_reachable_targets * num_nodes,
    )

    _require_numel(
        sample_id=sample_id,
        name=SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT,
        tensor=target_shortest_path_edge_mask_flat,
        expected=num_reachable_targets * num_edges,
    )


def _load_split_index(
    *,
    metadata_dir: Path,
    split: str,
) -> list[str]:
    path = metadata_dir / f"{split}.index.pt"

    if not path.exists():
        raise FileNotFoundError(f"Split index file not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, Mapping):
        sample_ids = payload.get("sample_ids")
    else:
        sample_ids = payload

    if isinstance(sample_ids, str) or not isinstance(sample_ids, Sequence):
        raise TypeError(
            f"{path} must contain a sequence of sample ids "
            "or a mapping with key 'sample_ids'"
        )

    return [str(sample_id) for sample_id in sample_ids]


def _tensor(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == dtype:
            return value.contiguous()
        return value.to(dtype=dtype).contiguous()

    return torch.as_tensor(value, dtype=dtype).contiguous()


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


def _require_shape(
    *,
    sample_id: str,
    name: str,
    actual: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    if actual != expected:
        raise ValueError(
            f"{sample_id}: {name} shape mismatch, got {actual}, expected {expected}"
        )


def _require_numel(
    *,
    sample_id: str,
    name: str,
    tensor: torch.Tensor,
    expected: int,
) -> None:
    actual = int(tensor.numel())
    if actual != expected:
        raise ValueError(
            f"{sample_id}: {name} length mismatch, got {actual}, expected {expected}"
        )


def _require_node_ids(
    *,
    sample_id: str,
    name: str,
    tensor: torch.Tensor,
    num_nodes: int,
) -> None:
    if tensor.ndim != 1:
        raise ValueError(
            f"{sample_id}: {name} must be 1D, got shape {tuple(tensor.shape)}"
        )

    if tensor.numel() == 0:
        return

    min_id = int(tensor.min().item())
    max_id = int(tensor.max().item())

    if min_id < 0 or max_id >= num_nodes:
        raise ValueError(
            f"{sample_id}: {name} contains node ids outside [0, {num_nodes})"
        )


def _lmdb_path(
    *,
    lmdb_dir: Path,
    split: str,
) -> Path:
    return lmdb_dir / f"{split}.lmdb"
