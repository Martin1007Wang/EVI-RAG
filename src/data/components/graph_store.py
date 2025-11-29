"""Minimal vocabulary-backed graph store used by retrieval datasets (G_global)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import lmdb

logger = logging.getLogger(__name__)


class GraphStore:
    """Thin wrapper around the vocabulary LMDB (G_global metadata).

    The previous implementation attempted to materialize adjacency matrices
    and legacy pickle formats. Those code paths added branching, memory use and
    implicit fallbacks that hid configuration issues. ``GraphStore`` now only
    exposes the global entity/relation mappings required by the retrieval
    pipeline, forcing datasets to rely on the single supported LMDB format.
    """

    def __init__(self, vocabulary_path: Union[str, Path]):
        self.vocabulary_path = Path(vocabulary_path).expanduser().resolve()
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        if not self.vocabulary_path.exists():
            raise FileNotFoundError(f"Vocabulary LMDB not found: {self.vocabulary_path}")

        env = lmdb.open(str(self.vocabulary_path), readonly=True, lock=False)
        try:
            with env.begin() as txn:
                entity_to_id_data = txn.get(b"entity_to_id")
                id_to_entity_data = txn.get(b"id_to_entity")
                relation_to_id_data = txn.get(b"relation_to_id")
                id_to_relation_data = txn.get(b"id_to_relation")

            if not all([entity_to_id_data, id_to_entity_data, relation_to_id_data, id_to_relation_data]):
                raise ValueError("Vocabulary LMDB missing required mappings")

            self.entity2id = pickle.loads(entity_to_id_data)
            raw_id_to_entity = pickle.loads(id_to_entity_data)
            self.id2entity = {int(k): v for k, v in raw_id_to_entity.items()}

            self.relation2id = pickle.loads(relation_to_id_data)
            raw_id_to_relation = pickle.loads(id_to_relation_data)
            self.id2relation = {int(k): v for k, v in raw_id_to_relation.items()}

            logger.info(
                "Loaded vocabulary: %d entities / %d relations",
                len(self.entity2id),
                len(self.relation2id),
            )
        finally:
            env.close()

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    def get_entity_id(self, name: str) -> Optional[int]:
        return self.entity2id.get(name)

    def get_relation_id(self, name: str) -> Optional[int]:
        return self.relation2id.get(name)
