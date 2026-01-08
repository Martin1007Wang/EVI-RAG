"""Minimal vocabulary-backed graph store used by retrieval datasets (G_global)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import lmdb

logger = logging.getLogger(__name__)


class GraphStore:
    """Thin wrapper around the vocabulary LMDB (G_global metadata).

    ``GraphStore`` exposes the global entity/relation mappings required by the
    retrieval pipeline, backed by a single LMDB format (label lists).
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
                entity_labels_data = txn.get(b"entity_labels")
                relation_labels_data = txn.get(b"relation_labels")

            if not all([entity_labels_data, relation_labels_data]):
                raise ValueError("Vocabulary LMDB missing required label lists")

            entity_labels = json.loads(entity_labels_data.decode("utf-8"))
            relation_labels = json.loads(relation_labels_data.decode("utf-8"))
            self.entity2id = {label: idx for idx, label in enumerate(entity_labels)}
            self.id2entity = {idx: label for idx, label in enumerate(entity_labels)}
            self.relation2id = {label: idx for idx, label in enumerate(relation_labels)}
            self.id2relation = {idx: label for idx, label in enumerate(relation_labels)}

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
