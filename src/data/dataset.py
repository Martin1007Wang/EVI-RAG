# src/data/dataset.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch_geometric.data import Dataset
from src.data.schema import RetrievalData, SampleFields, StorageSchema
from src.utils.lmdb_utils import assign_lmdb_shard
from .retrieval import LMDBSampleStore


class RetrievalDataset(Dataset):
    def __init__(
        self,
        sample_ids: List[str],
        lmdb_paths: List[Path],
        split: str = "train",
        lmdb_readahead: bool = False,
        sample_num_nodes: Optional[Sequence[int]] = None,
        sample_num_edges: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.sample_ids = sample_ids
        self.split = split
        self._lmdb_paths = [Path(p) for p in lmdb_paths]
        self._num_shards = len(lmdb_paths)
        self._lmdb_readahead = lmdb_readahead
        self._sample_stores: Optional[List[LMDBSampleStore]] = None
        self.sample_num_nodes = self._coerce_optional_sizes(
            sample_num_nodes, name="sample_num_nodes"
        )
        self.sample_num_edges = self._coerce_optional_sizes(
            sample_num_edges, name="sample_num_edges"
        )

    def _coerce_optional_sizes(
        self,
        values: Optional[Sequence[int]],
        *,
        name: str,
    ) -> Optional[List[int]]:
        if values is None:
            return None
        coerced = [max(int(v), 1) for v in values]
        if len(coerced) != len(self.sample_ids):
            raise ValueError(
                f"{name} length mismatch: expected {len(self.sample_ids)}, got {len(coerced)}."
            )
        return coerced

    def _get_stores(self) -> List[LMDBSampleStore]:
        if self._sample_stores is None:
            self._sample_stores = [
                LMDBSampleStore(p, readahead=self._lmdb_readahead)
                for p in self._lmdb_paths
            ]
        return self._sample_stores

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> RetrievalData:
        sample_id = self.sample_ids[idx]
        shard_id = assign_lmdb_shard(sample_id, self._num_shards)
        raw = self._get_stores()[shard_id].load_sample(sample_id)
        StorageSchema.validate(raw)
        return self._build_sample(raw, sample_id)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_sample_stores"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _build_sample(self, raw: Dict[str, Any], sample_id: str) -> RetrievalData:
        question_emb = torch.as_tensor(
            raw[SampleFields.QUESTION_EMB], dtype=torch.float32
        )
        if not bool(torch.isfinite(question_emb).all().item()):
            raise ValueError(
                f"Sample {sample_id!r} has non-finite question_emb. "
                "The materialized preprocessing artifacts are corrupted; rebuild preprocess outputs."
            )
        train_target_mask = torch.as_tensor(
            raw[SampleFields.TRAIN_TARGET_MASK], dtype=torch.bool
        )
        sample_kwargs: Dict[str, Any] = {
            "sample_id": sample_id,
            "num_nodes": int(raw[SampleFields.NUM_NODES]),
            "edge_index": raw[SampleFields.EDGE_INDEX],
            "edge_relation_ids_global": raw[SampleFields.EDGE_RELATION_IDS_GLOBAL],
            "node_entity_ids_global": raw[SampleFields.NODE_ENTITY_IDS_GLOBAL],
            "question_emb": question_emb,
            "is_anchor_mask": raw[SampleFields.IS_ANCHOR_MASK],
            "train_target_mask": train_target_mask,
            "anchor_signed_distance": raw[SampleFields.ANCHOR_SIGNED_DISTANCE],
            "answer_entity_ids_global": raw[SampleFields.ANSWER_ENTITY_IDS_GLOBAL],
            "gold_answer_in_graph": bool(train_target_mask.any().item()),
        }
        optional_tensor_fields = {
            SampleFields.TRAIN_TARGET_NODE_IDS: torch.long,
            SampleFields.TARGET_NODE_DISTANCE_FLAT: torch.long,
            SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT: torch.float32,
            SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT: torch.bool,
            SampleFields.shortest_path_edge_mask: torch.bool,
            SampleFields.NODE_TO_TARGET_DISTANCE: torch.long,
            SampleFields.shortest_path_count: torch.float32,
            SampleFields.MIN_TARGET_DIST: torch.long,
            SampleFields.MAX_PATH_LENGTH: torch.long,
        }
        for field_name, dtype in optional_tensor_fields.items():
            value = raw.get(field_name)
            if value is not None:
                sample_kwargs[field_name] = torch.as_tensor(value, dtype=dtype)
        return RetrievalData(**sample_kwargs)

    def close(self) -> None:
        if self._sample_stores is not None:
            for store in self._sample_stores:
                store.close()
            self._sample_stores = None

    def __del__(self) -> None:
        self.close()
