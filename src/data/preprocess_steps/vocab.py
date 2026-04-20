from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class EntityCatalog:
    """EntityVocab 最终产出的静态资产目录"""

    entity_embedding_map: torch.Tensor  # [num_entities] long
    cvt_mask: torch.Tensor  # [num_entities] bool
    entity_labels: list[str]
    text_labels: list[str]
    text_embedding_ids: list[int]
    max_embedding_id: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class EntityVocab:
    def __init__(
        self,
        *,
        is_text_entity: Callable[[str], bool],
        is_cvt_entity: Callable[[str], bool],
    ) -> None:
        self._is_text_entity = is_text_entity
        self._is_cvt_entity = is_cvt_entity
        self._entity_to_id: dict[str, int] = {}
        self._text_entities: set[str] = set()

    def add(self, entity: str) -> None:
        if entity not in self._entity_to_id:
            self._entity_to_id[entity] = len(self._entity_to_id)
            if self._is_text_entity(entity):
                self._text_entities.add(entity)

    def entity_id(self, entity: str) -> int:
        return self._entity_to_id[entity]

    def build_catalog(self) -> EntityCatalog:
        """高性能构建实体目录，移除 O(N log N) 的排序和冗余 Tensor 赋值"""
        text_entities = sorted(self._text_entities)
        text_embedding_ids = {entity: idx + 1 for idx, entity in enumerate(text_entities)}

        entity_embedding_map_list: list[int] = []
        cvt_mask_list: list[bool] = []
        text_labels: list[str] = []
        text_emb_ids: list[int] = []

        # 利用 dict 插入序 (0 -> N)
        entity_labels = list(self._entity_to_id.keys())

        for entity in entity_labels:
            emb_id = text_embedding_ids.get(entity, 0)
            entity_embedding_map_list.append(emb_id)
            cvt_mask_list.append(self._is_cvt_entity(entity))

            if emb_id != 0:
                text_labels.append(entity)
                text_emb_ids.append(emb_id)

        return EntityCatalog(
            entity_embedding_map=torch.tensor(entity_embedding_map_list, dtype=torch.long),
            cvt_mask=torch.tensor(cvt_mask_list, dtype=torch.bool),
            entity_labels=entity_labels,
            text_labels=text_labels,
            text_embedding_ids=text_emb_ids,
            max_embedding_id=int(max(text_emb_ids, default=0)),
        )


class RelationVocab:
    def __init__(self) -> None:
        self._relation_to_id: dict[str, int] = {}

    def add(self, relation: str) -> None:
        if relation not in self._relation_to_id:
            self._relation_to_id[relation] = len(self._relation_to_id)

    def relation_id(self, relation: str) -> int:
        return self._relation_to_id[relation]

    def labels(self) -> list[str]:
        return list(self._relation_to_id.keys())
