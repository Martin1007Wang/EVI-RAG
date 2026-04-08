# src/data/dataset.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.data.schema import SampleFields, StorageSchema
from torch_geometric.data import Dataset, Data
from .retrieval import LMDBSampleStore
from src.utils.lmdb_utils import assign_lmdb_shard


class RetrievalDataset(Dataset):
    def __init__(
        self,
        sample_ids: List[str],
        lmdb_paths: List[Path],
        split: str = "train",
        lmdb_readahead: bool = False,
    ):
        super().__init__()
        self.sample_ids = sample_ids
        self.split = split
        self._lmdb_paths = [Path(p) for p in lmdb_paths]  # 确保是Path对象列表
        self._num_shards = len(lmdb_paths)
        self._lmdb_readahead = lmdb_readahead
        self._sample_stores: Optional[List[LMDBSampleStore]] = None

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> Data:
        sample_id = self.sample_ids[idx]
        if self._sample_stores is None:
            self._sample_stores = [
                LMDBSampleStore(p, readahead=self._lmdb_readahead)
                for p in self._lmdb_paths
            ]
        shard_id = assign_lmdb_shard(sample_id, self._num_shards)
        raw = self._sample_stores[shard_id].load_sample(sample_id)
        StorageSchema.validate(raw)  # ← 在加载时校验，比训练中途崩溃友好得多
        return self._build_sample(raw, sample_id)

    def _build_sample(self, raw: Dict[str, Any], sample_id: str) -> Data:
        sample = Data(
            sample_id=sample_id,
            num_nodes=int(raw[SampleFields.NUM_NODES]),
            edge_index=raw[SampleFields.EDGE_INDEX],
            edge_relation_ids_global=raw[SampleFields.EDGE_RELATION_IDS_GLOBAL],
            node_entity_ids_global=raw[SampleFields.NODE_ENTITY_IDS_GLOBAL],
            question_emb=torch.as_tensor(
                raw[SampleFields.QUESTION_EMB], dtype=torch.float32
            ),
            is_anchor_mask=raw[SampleFields.IS_ANCHOR_MASK],
            is_target_mask=raw[SampleFields.IS_TARGET_MASK],
            answer_entity_ids_global=raw[SampleFields.ANSWER_ENTITY_IDS_GLOBAL],
            gold_answer_in_graph=bool(raw[SampleFields.IS_TARGET_MASK].any().item()),
        )
        return sample

    def close(self) -> None:
        if self._sample_stores is not None:
            for store in self._sample_stores:
                store.close()
            self._sample_stores = None

    def __del__(self):
        self.close()
