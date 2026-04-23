from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class EntityCatalog:
    entity_embedding_map: torch.Tensor  # [num_entities] long
    cvt_mask: torch.Tensor  # [num_entities] bool
    entity_labels: list[str]
    text_labels: list[str]
    text_embedding_ids: list[int]
    max_embedding_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_embedding_map": self.entity_embedding_map,
            "cvt_mask": self.cvt_mask,
            "entity_labels": list(self.entity_labels),
            "text_labels": list(self.text_labels),
            "text_embedding_ids": list(self.text_embedding_ids),
            "max_embedding_id": int(self.max_embedding_id),
        }


class EntityVocab:
    def __init__(self) -> None:
        self._entity_to_id: dict[str, int] = {}

    def add(self, entity: str) -> None:
        if entity not in self._entity_to_id:
            self._entity_to_id[entity] = len(self._entity_to_id)

    def entity_id(self, entity: str) -> int:
        return self._entity_to_id[entity]

    def __contains__(self, entity: str) -> bool:
        return entity in self._entity_to_id

    def __len__(self) -> int:
        return len(self._entity_to_id)

    def labels(self) -> list[str]:
        return list(self._entity_to_id.keys())


class RelationVocab:
    def __init__(self) -> None:
        self._relation_to_id: dict[str, int] = {}

    def add(self, relation: str) -> None:
        if relation not in self._relation_to_id:
            self._relation_to_id[relation] = len(self._relation_to_id)

    def relation_id(self, relation: str) -> int:
        return self._relation_to_id[relation]

    def __contains__(self, relation: str) -> bool:
        return relation in self._relation_to_id

    def __len__(self) -> int:
        return len(self._relation_to_id)

    def labels(self) -> list[str]:
        return list(self._relation_to_id.keys())


def build_entity_catalog(
    entity_vocab: EntityVocab,
    *,
    is_text_entity: Callable[[str], bool],
    is_cvt_entity: Callable[[str], bool],
) -> EntityCatalog:
    entity_labels = entity_vocab.labels()
    text_labels = sorted(entity for entity in entity_labels if is_text_entity(entity))
    text_embedding_ids = {entity: idx + 1 for idx, entity in enumerate(text_labels)}
    entity_embedding_map_list: list[int] = []
    cvt_mask_list: list[bool] = []
    for entity in entity_labels:
        entity_embedding_map_list.append(text_embedding_ids.get(entity, 0))
        cvt_mask_list.append(bool(is_cvt_entity(entity)))
    text_emb_ids = [text_embedding_ids[entity] for entity in text_labels]
    return EntityCatalog(
        entity_embedding_map=torch.tensor(entity_embedding_map_list, dtype=torch.long),
        cvt_mask=torch.tensor(cvt_mask_list, dtype=torch.bool),
        entity_labels=entity_labels,
        text_labels=text_labels,
        text_embedding_ids=text_emb_ids,
        max_embedding_id=int(max(text_emb_ids, default=0)),
    )