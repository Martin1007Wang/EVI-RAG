from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch


class EntityVocab:
    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}

    def add(self, entity: str) -> int:
        if entity not in self._to_id:
            self._to_id[entity] = len(self._to_id)
        return self._to_id[entity]

    def id(self, entity: str) -> int:
        return self._to_id[entity]

    def labels(self) -> list[str]:
        return list(self._to_id.keys())

    def __len__(self) -> int:
        return len(self._to_id)


class RelationVocab:
    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}

    def add(self, relation: str) -> int:
        if relation not in self._to_id:
            self._to_id[relation] = len(self._to_id)
        return self._to_id[relation]

    def id(self, relation: str) -> int:
        return self._to_id[relation]

    def labels(self) -> list[str]:
        return list(self._to_id.keys())

    def __len__(self) -> int:
        return len(self._to_id)


@dataclass(frozen=True)
class EntityTyping:
    non_text_prefixes: tuple[str, ...] = ("m.", "g.")

    def has_text_embedding_source(self, entity: str) -> bool:
        return not entity.startswith(self.non_text_prefixes)

    def is_non_text_entity(self, entity: str) -> bool:
        return entity.startswith(self.non_text_prefixes)


@dataclass(frozen=True)
class EntityCatalog:
    entity_labels: list[str]  # entity_labels[global_entity_id] = 原始实体字符串
    entity_text_embedding_ids: torch.Tensor  # entity_text_embedding_ids[global_entity_id] = text embedding table 中的行号
    non_text_entity_mask: torch.Tensor  # non_text_entity_mask[global_entity_id] = 这个实体是否没有文本 embedding 来源
    entity_text_labels: list[str]  # entity_text_labels[text_embedding_id - 1] = 拥有文本 embedding 的实体字符串

    @classmethod
    def build(
        cls,
        vocab: EntityVocab,
        *,
        typing: EntityTyping | None = None,
        sort_text_entities: bool = True,
    ) -> EntityCatalog:
        typing = typing or EntityTyping()
        entity_labels = vocab.labels()
        text_labels = [e for e in entity_labels if typing.has_text_embedding_source(e)]
        if sort_text_entities:
            text_labels = sorted(text_labels)
        text_id = {e: i + 1 for i, e in enumerate(text_labels)}
        embedding_ids = torch.tensor(
            [text_id.get(e, 0) for e in entity_labels],
            dtype=torch.long,
        )
        non_text_mask = torch.tensor(
            [typing.is_non_text_entity(e) for e in entity_labels],
            dtype=torch.bool,
        )
        return cls(
            entity_labels=entity_labels,
            entity_text_embedding_ids=embedding_ids,
            non_text_entity_mask=non_text_mask,
            entity_text_labels=text_labels,
        )

    @property
    def num_entities(self) -> int:
        return len(self.entity_labels)

    @property
    def num_text_entities(self) -> int:
        return len(self.entity_text_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_labels": list(self.entity_labels),
            "entity_text_embedding_ids": self.entity_text_embedding_ids,
            "non_text_entity_mask": self.non_text_entity_mask,
            "entity_text_labels": list(self.entity_text_labels),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> EntityCatalog:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            entity_labels=[str(x) for x in payload["entity_labels"]],
            entity_text_embedding_ids=payload["entity_text_embedding_ids"].long(),
            non_text_entity_mask=payload["non_text_entity_mask"].bool(),
            entity_text_labels=[str(x) for x in payload["entity_text_labels"]],
        )


@dataclass(frozen=True)
class RelationCatalog:
    relation_labels: list[str]  # relation_labels[global_relation_id] = 原始 relation 字符串
    relation_text_labels: list[str]  # relation_text_labels[global_relation_id] = 用于文本编码器的 relation 文本

    @classmethod
    def build(cls, vocab: RelationVocab) -> "RelationCatalog":
        relation_labels = vocab.labels()
        return cls(
            relation_labels=relation_labels,
            relation_text_labels=[r.replace("/", " ").replace("_", " ").strip() for r in relation_labels],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_labels": list(self.relation_labels),
            "relation_text_labels": list(self.relation_text_labels),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "RelationCatalog":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            relation_labels=[str(x) for x in payload["relation_labels"]],
            relation_text_labels=[str(x) for x in payload["relation_text_labels"]],
        )
