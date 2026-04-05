from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class RawSample:
    dataset: str
    split: str
    question_id: str
    kb: str
    question: str
    graph: tuple[tuple[str, str, str], ...]
    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]
    answer_texts: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSample:
    sample: RawSample
    sample_id: str
    kept_edges: list[tuple[str, str, str]]
    question_entities_in_graph: tuple[str, ...]
    legal_answer_entities: tuple[str, ...]
    positive_edge_ids: torch.Tensor


@dataclass(frozen=True)
class SplitFilter:
    skip_no_question_entity: bool
    skip_no_ans: bool
    skip_no_path: bool


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
        self._text_entities: list[str] = []

    def add(self, entity: str) -> None:
        if entity in self._entity_to_id:
            return
        self._entity_to_id[entity] = len(self._entity_to_id)
        if self._is_text_entity(entity):
            self._text_entities.append(entity)

    def entity_id(self, entity: str) -> int:
        return self._entity_to_id[entity]

    def build_entity_metadata(self) -> dict[str, object]:
        text_entities = sorted(set(self._text_entities))
        text_embedding_ids = {entity: idx + 1 for idx, entity in enumerate(text_entities)}
        num_entities = len(self._entity_to_id)
        entity_embedding_map = torch.zeros((num_entities,), dtype=torch.long)
        cvt_mask = torch.zeros((num_entities,), dtype=torch.bool)
        entity_labels: list[str] = [""] * num_entities
        text_labels: list[str] = []
        text_emb_ids: list[int] = []
        for entity, entity_id in sorted(self._entity_to_id.items(), key=lambda item: item[1]):
            entity_labels[entity_id] = entity
            emb_id = text_embedding_ids.get(entity, 0)
            entity_embedding_map[entity_id] = int(emb_id)
            cvt_mask[entity_id] = bool(self._is_cvt_entity(entity))
            if emb_id != 0:
                text_labels.append(entity)
                text_emb_ids.append(int(emb_id))
        return {
            "entity_embedding_map": entity_embedding_map,
            "cvt_mask": cvt_mask,
            "entity_labels": entity_labels,
            "text_labels": text_labels,
            "text_embedding_ids": text_emb_ids,
            "max_embedding_id": int(max(text_emb_ids, default=0)),
        }


class RelationVocab:
    def __init__(self) -> None:
        self._relation_to_id: dict[str, int] = {}

    def add(self, relation: str) -> None:
        if relation not in self._relation_to_id:
            self._relation_to_id[relation] = len(self._relation_to_id)

    def relation_id(self, relation: str) -> int:
        return self._relation_to_id[relation]

    def labels(self) -> list[str]:
        return [relation for relation, _ in sorted(self._relation_to_id.items(), key=lambda item: item[1])]
