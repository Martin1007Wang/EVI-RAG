from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class _StringVocab:
    item_name = "item"

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}

    def add(self, value: str) -> int:
        value = str(value)
        if not value:
            raise ValueError(f"{self.item_name} must be non-empty")
        if value not in self._to_id:
            self._to_id[value] = len(self._to_id)
        return self._to_id[value]

    def id(self, value: str) -> int:
        try:
            return self._to_id[value]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.item_name}: {value!r}") from exc

    def labels(self) -> list[str]:
        return list(self._to_id.keys())

    def __len__(self) -> int:
        return len(self._to_id)


class EntityVocab(_StringVocab):
    item_name = "entity"


class RelationVocab(_StringVocab):
    item_name = "relation"


@dataclass(frozen=True)
class EntityTextPolicy:
    non_text_prefixes: tuple[str, ...] = ()

    def has_text(self, entity: str) -> bool:
        return not entity.startswith(self.non_text_prefixes)

    def is_non_text(self, entity: str) -> bool:
        return entity.startswith(self.non_text_prefixes)


@dataclass(frozen=True)
class EntityCatalog:
    # entity_labels[global_entity_id] = original entity string.
    entity_labels: list[str]
    # 1-based text embedding id per global entity; 0 means no text embedding.
    entity_text_embedding_ids: torch.Tensor
    # non_text_entity_mask[global_entity_id] = entity has no text embedding source.
    non_text_entity_mask: torch.Tensor
    # entity_text_labels[text_id - 1] = text label for positive text_id.
    entity_text_labels: list[str]

    @classmethod
    def build(
        cls,
        vocab: EntityVocab,
        *,
        text_policy: EntityTextPolicy | None = None,
        sort_text_entities: bool = True,
    ) -> EntityCatalog:
        policy = text_policy or EntityTextPolicy()
        entity_labels = vocab.labels()
        entity_text_labels = [e for e in entity_labels if policy.has_text(e)]
        if sort_text_entities:
            entity_text_labels = sorted(entity_text_labels)

        text_id_by_entity = {e: i + 1 for i, e in enumerate(entity_text_labels)}
        entity_text_embedding_ids = torch.tensor(
            [text_id_by_entity.get(e, 0) for e in entity_labels],
            dtype=torch.long,
        )
        non_text_entity_mask = torch.tensor(
            [policy.is_non_text(e) for e in entity_labels],
            dtype=torch.bool,
        )
        return cls(
            entity_labels=entity_labels,
            entity_text_embedding_ids=entity_text_embedding_ids,
            non_text_entity_mask=non_text_entity_mask,
            entity_text_labels=entity_text_labels,
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
            "entity_text_embedding_ids": (
                self.entity_text_embedding_ids.long().contiguous()
            ),
            "non_text_entity_mask": self.non_text_entity_mask.bool().contiguous(),
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


def relation_text_label(relation: str) -> str:
    raw = str(relation).strip()
    text = raw.replace("/", " ").replace("_", " ").strip()
    return text or raw


@dataclass(frozen=True)
class RelationCatalog:
    # relation_labels[global_relation_id] = original relation string.
    relation_labels: list[str]
    # relation_text_labels[global_relation_id] = text label for text encoder.
    relation_text_labels: list[str]

    @classmethod
    def build(cls, vocab: RelationVocab) -> "RelationCatalog":
        relation_labels = vocab.labels()
        return cls(
            relation_labels=relation_labels,
            relation_text_labels=[
                relation_text_label(relation) for relation in relation_labels
            ],
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


__all__ = [
    "EntityCatalog",
    "EntityTextPolicy",
    "EntityVocab",
    "RelationCatalog",
    "RelationVocab",
    "relation_text_label",
]
