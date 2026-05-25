from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class EntityTextPolicy:
    non_text_prefixes: tuple[str, ...] = ()

    def has_text(self, entity: str) -> bool:
        return not entity.startswith(self.non_text_prefixes)


def relation_text_label(relation: str) -> str:
    raw = str(relation).strip()
    text = raw.replace("/", " ").replace("_", " ").strip()
    return text or raw


@dataclass(frozen=True, slots=True)
class Catalog:
    entity_labels: list[str]
    relation_labels: list[str]
    entity_text_labels: list[str]
    entity_text_row_by_entity_id: torch.Tensor
    relation_text_labels: list[str]
    _entity_to_id: Mapping[str, int] = field(init=False, repr=False)
    _relation_to_id: Mapping[str, int] = field(init=False, repr=False)


    @property
    def num_entities(self) -> int:
        return len(self.entity_labels)

    @property
    def num_relations(self) -> int:
        return len(self.relation_labels)

    @property
    def num_text_entities(self) -> int:
        return len(self.entity_text_labels)

    @property
    def entity_to_id(self) -> dict[str, int]:
        return dict(self._entity_to_id)

    @property
    def relation_to_id(self) -> dict[str, int]:
        return dict(self._relation_to_id)

    def entity_id(self, entity: str) -> int:
        return self._entity_to_id[entity]

    def relation_id(self, relation: str) -> int:
        return self._relation_to_id[relation]

    def entity_label(self, entity_id: int) -> str:
        return _label(self.entity_labels, int(entity_id), name="entity_labels")

    def relation_label(self, relation_id: int) -> str:
        return _label(self.relation_labels, int(relation_id), name="relation_labels")

    def validate_embeddings(
        self,
        *,
        entity_text_semantic_table: torch.Tensor,
        relation_semantic_table: torch.Tensor,
    ) -> None:
        if entity_text_semantic_table.ndim != 2:
            raise ValueError("entity_text_semantic_table must be 2D")
        if relation_semantic_table.ndim != 2:
            raise ValueError("relation_semantic_table must be 2D")
        if int(entity_text_semantic_table.size(0)) != len(self.entity_text_labels):
            raise ValueError("entity_text_semantic_table rows must equal len(entity_text_labels)")
        if int(relation_semantic_table.size(0)) != len(self.relation_labels):
            raise ValueError("relation_semantic_table rows must equal len(relation_labels)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_labels": list(self.entity_labels),
            "relation_labels": list(self.relation_labels),
            "entity_text_labels": list(self.entity_text_labels),
            "entity_text_row_by_entity_id": self.entity_text_row_by_entity_id.long().contiguous().cpu(),
            "relation_text_labels": list(self.relation_text_labels),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            entity_labels=[str(x) for x in payload["entity_labels"]],
            relation_labels=[str(x) for x in payload["relation_labels"]],
            entity_text_labels=[str(x) for x in payload["entity_text_labels"]],
            entity_text_row_by_entity_id=payload["entity_text_row_by_entity_id"].long(),
            relation_text_labels=[str(x) for x in payload["relation_text_labels"]],
        )


@dataclass(slots=True)
class CatalogBuilder:
    _entity_to_id: dict[str, int] = field(default_factory=dict)
    _relation_to_id: dict[str, int] = field(default_factory=dict)

    def add_entity(self, entity: str) -> int:
        return self._add(self._entity_to_id, entity, "entity")

    def add_relation(self, relation: str) -> int:
        return self._add(self._relation_to_id, relation, "relation")

    def entity_id(self, entity: str) -> int:
        return self._entity_to_id[entity]

    def relation_id(self, relation: str) -> int:
        return self._relation_to_id[relation]

    def build(
        self,
        *,
        text_policy: EntityTextPolicy | None = None,
        sort_text_entities: bool = True,
    ) -> Catalog:
        policy = text_policy or EntityTextPolicy()
        entity_labels = list(self._entity_to_id.keys())
        relation_labels = list(self._relation_to_id.keys())
        entity_text_labels = [label for label in entity_labels if policy.has_text(label)]
        if sort_text_entities:
            entity_text_labels = sorted(entity_text_labels)
        text_row_by_entity = {label: idx for idx, label in enumerate(entity_text_labels)}
        entity_text_row_by_entity_id = torch.tensor(
            [text_row_by_entity.get(label, -1) for label in entity_labels],
            dtype=torch.long,
        )
        return Catalog(
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            entity_text_labels=entity_text_labels,
            entity_text_row_by_entity_id=entity_text_row_by_entity_id,
            relation_text_labels=[relation_text_label(label) for label in relation_labels],
        )

    @staticmethod
    def _add(table: dict[str, int], value: str, name: str) -> int:
        value = str(value)
        if not value:
            raise ValueError(f"{name} must be non-empty")
        item_id = table.get(value)
        if item_id is None:
            item_id = len(table)
            table[value] = item_id
        return item_id


@dataclass(frozen=True, slots=True)
class DebugLookup:
    catalog: Catalog
    question_text_by_sample_id: Mapping[str, str] | None = None

    @classmethod
    def from_question_text_json(
        cls,
        *,
        catalog: Catalog,
        path: str | Path,
    ) -> DebugLookup:
        return cls(
            catalog=catalog,
            question_text_by_sample_id=load_question_text_by_sample_id(path),
        )

    def entity_label(self, entity_id: int) -> str:
        return self.catalog.entity_label(entity_id)

    def relation_label(self, relation_id: int) -> str:
        return self.catalog.relation_label(relation_id)

    def local_node_global_id(self, graph: object, local_node_id: int) -> int:
        ids = _graph_tensor(graph, name="node_entity_catalog_ids")
        return _local_id(ids, int(local_node_id), name="local_node_id")

    def local_edge_global_id(self, graph: object, local_edge_id: int) -> int:
        ids = _graph_tensor(graph, name="edge_relation_catalog_ids")
        return _local_id(ids, int(local_edge_id), name="local_edge_id")

    def local_node_label(self, graph: object, local_node_id: int) -> str:
        return self.entity_label(self.local_node_global_id(graph, local_node_id))

    def local_edge_label(self, graph: object, local_edge_id: int) -> str:
        return self.relation_label(self.local_edge_global_id(graph, local_edge_id))

    def question_text(self, sample_id: str) -> str:
        question_texts = _require_question_texts(self.question_text_by_sample_id)
        key = str(sample_id)
        try:
            return str(question_texts[key])
        except KeyError as exc:
            raise KeyError(f"Unknown sample_id: {key!r}") from exc

    def graph_sample_id(self, graph: object, graph_id: int) -> str:
        sample_ids = getattr(graph, "sample_id", None)
        if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
            raise AttributeError("graph must define sequence attribute 'sample_id'.")
        index = int(graph_id)
        if index < 0 or index >= len(sample_ids):
            raise IndexError(f"graph_id out of range: {graph_id}")
        return str(sample_ids[index])

    def graph_question_text(self, graph: object, graph_id: int) -> str:
        return self.question_text(self.graph_sample_id(graph, graph_id))

    def debug_feature_view(
        self,
        *,
        batch: object,
        node_ids: Sequence[int] | None = None,
        edge_ids: Sequence[int] | None = None,
        graph_ids: Sequence[int] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        node_ids = _default_range(node_ids, graph=batch, tensor_name="node_entity_catalog_ids")
        edge_ids = _default_range(edge_ids, graph=batch, tensor_name="edge_relation_catalog_ids")
        graph_ids = _default_graph_ids(graph_ids, graph=batch)
        return {
            "entities": [
                {
                    "node_id": int(node_id),
                    "entity_id": self.local_node_global_id(batch, int(node_id)),
                    "entity_text": self.local_node_label(batch, int(node_id)),
                }
                for node_id in node_ids
            ],
            "relations": [
                {
                    "edge_id": int(edge_id),
                    "relation_id": self.local_edge_global_id(batch, int(edge_id)),
                    "relation_text": self.local_edge_label(batch, int(edge_id)),
                }
                for edge_id in edge_ids
            ],
            "questions": [
                {
                    "graph_id": int(graph_id),
                    "sample_id": self.graph_sample_id(batch, int(graph_id)),
                    "question_text": self.graph_question_text(batch, int(graph_id)),
                }
                for graph_id in graph_ids
            ],
        }

    def debug_rollout_view(
        self,
        *,
        batch: object,
        rollout: object,
        rollout_row_ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        rows = _default_rollout_rows(rollout_row_ids, rollout=rollout)
        selected_edge_ids = getattr(rollout, "selected_edge_ids", None)
        terminal_state = getattr(rollout, "terminal_state", None)
        if not isinstance(selected_edge_ids, torch.Tensor):
            raise AttributeError("rollout must define tensor attribute 'selected_edge_ids'.")
        if terminal_state is None:
            raise AttributeError("rollout must define 'terminal_state' for rollout debug view.")
        graph_ids = getattr(terminal_state, "graph_ids", None)
        if not isinstance(graph_ids, torch.Tensor):
            raise AttributeError("rollout.terminal_state must define tensor attribute 'graph_ids'.")

        out: list[dict[str, Any]] = []
        for row_id in rows:
            graph_id = int(_local_id(graph_ids, int(row_id), name="rollout_row_id"))
            edge_path = [
                {
                    "edge_id": int(edge_id),
                    "relation_id": self.local_edge_global_id(batch, int(edge_id)),
                    "relation_text": self.local_edge_label(batch, int(edge_id)),
                }
                for edge_id in selected_edge_ids[int(row_id)].tolist()
                if int(edge_id) >= 0
            ]
            terminal_nodes = _non_padded_row_ids(terminal_state.active_node_ids, int(row_id))
            out.append(
                {
                    "rollout_row_id": int(row_id),
                    "graph_id": graph_id,
                    "sample_id": self.graph_sample_id(batch, graph_id),
                    "question_text": self.graph_question_text(batch, graph_id),
                    "selected_relations": edge_path,
                    "active_entities": [
                        {
                            "node_id": int(node_id),
                            "entity_id": self.local_node_global_id(batch, int(node_id)),
                            "entity_text": self.local_node_label(batch, int(node_id)),
                        }
                        for node_id in terminal_nodes
                    ],
                }
            )
        return out


def load_question_text_by_sample_id(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"question text lookup must be a mapping, got {type(payload).__name__}.")
    return {
        str(sample_id): str(question_text)
        for sample_id, question_text in payload.items()
    }


def _require_question_texts(
    question_text_by_sample_id: Mapping[str, str] | None,
) -> Mapping[str, str]:
    if question_text_by_sample_id is None:
        raise RuntimeError("question texts are not configured for this materialization.")
    return question_text_by_sample_id


def _graph_tensor(graph: object, *, name: str) -> torch.Tensor:
    value = getattr(graph, name, None)
    if not isinstance(value, torch.Tensor):
        raise AttributeError(f"graph must define tensor attribute {name!r}.")
    return value.to(dtype=torch.long).view(-1)


def _local_id(ids: torch.Tensor, index: int, *, name: str) -> int:
    if index < 0 or index >= int(ids.numel()):
        raise IndexError(f"{name} out of range: {index}")
    return int(ids[index].item())


def _label(labels: Sequence[str], index: int, *, name: str) -> str:
    if index < 0 or index >= len(labels):
        raise IndexError(f"{name} id out of range: {index}")
    return str(labels[index])


def _default_range(
    values: Sequence[int] | None,
    *,
    graph: object,
    tensor_name: str,
) -> list[int]:
    if values is not None:
        return [int(value) for value in values]
    return list(range(int(_graph_tensor(graph, name=tensor_name).numel())))


def _default_graph_ids(values: Sequence[int] | None, *, graph: object) -> list[int]:
    if values is not None:
        return [int(value) for value in values]
    sample_ids = getattr(graph, "sample_id", None)
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise AttributeError("graph must define sequence attribute 'sample_id'.")
    return list(range(len(sample_ids)))


def _default_rollout_rows(values: Sequence[int] | None, *, rollout: object) -> list[int]:
    if values is not None:
        return [int(value) for value in values]
    selected_edge_ids = getattr(rollout, "selected_edge_ids", None)
    if not isinstance(selected_edge_ids, torch.Tensor):
        raise AttributeError("rollout must define tensor attribute 'selected_edge_ids'.")
    return list(range(int(selected_edge_ids.size(0))))


def _non_padded_row_ids(padded_ids: torch.Tensor, row_id: int) -> list[int]:
    row = padded_ids[int(row_id)].to(dtype=torch.long).view(-1)
    return [int(value) for value in row.tolist() if int(value) >= 0]


__all__ = [
    "Catalog",
    "CatalogBuilder",
    "DebugLookup",
    "EntityTextPolicy",
    "load_question_text_by_sample_id",
    "relation_text_label",
]
