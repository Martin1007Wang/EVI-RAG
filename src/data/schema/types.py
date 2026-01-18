from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.data.schema.constants import _NON_TEXT_EMBEDDING_ID


@dataclass(frozen=True)
class SplitFilter:
    skip_no_topic: bool
    skip_no_ans: bool
    skip_no_path: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    encoder: str
    device: str
    batch_size: int
    fp16: bool
    progress_bar: bool
    embeddings_out_dir: Path
    precompute_entities: bool
    precompute_relations: bool
    precompute_questions: bool
    canonicalize_relations: bool
    cosine_eps: float


@dataclass(frozen=True)
class Sample:
    dataset: str
    split: str
    question_id: str
    kb: str
    question: str
    graph: List[Tuple[str, str, str]]
    q_entity: List[str]
    a_entity: List[str]
    answer_texts: List[str]
    graph_iso_type: Optional[str] = None
    redundant: Optional[bool] = None
    test_type: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SampleFilterOutcome:
    keep: bool
    has_topic: bool
    has_answer: bool
    has_path: bool


@dataclass(frozen=True)
class TextEntityConfig:
    mode: str
    prefixes: Tuple[str, ...]
    regex: Optional[re.Pattern]

    def is_text(self, entity: str) -> bool:
        if self.mode == "prefix_allowlist":
            return any(entity.startswith(prefix) for prefix in self.prefixes)
        if self.mode == "regex":
            return bool(self.regex.match(entity)) if self.regex is not None else False
        raise ValueError(f"Unsupported entity_text_mode: {self.mode}")


@dataclass(frozen=True)
class CvtEntityConfig:
    mode: str
    prefixes: Tuple[str, ...]
    regex: Optional[re.Pattern]

    def is_cvt(self, entity: str) -> bool:
        if self.mode == "prefix_allowlist":
            return any(entity.startswith(prefix) for prefix in self.prefixes)
        if self.mode == "regex":
            return bool(self.regex.match(entity)) if self.regex is not None else False
        raise ValueError(f"Unsupported entity_cvt_mode: {self.mode}")


@dataclass(frozen=True)
class TimeRelationConfig:
    mode: str
    relation_regex: Optional[re.Pattern]
    question_regex: Optional[re.Pattern]

    def is_time_relation(self, relation: str) -> bool:
        if self.relation_regex is None:
            return False
        return bool(self.relation_regex.search(relation))

    def is_time_question(self, question: str) -> bool:
        if self.question_regex is None:
            return False
        return bool(self.question_regex.search(question))


@dataclass
class GraphRecord:
    graph_id: str
    node_entity_ids: List[int]
    node_embedding_ids: List[int]
    node_labels: List[str]
    edge_src: List[int]
    edge_dst: List[int]
    edge_relation_ids: List[int]


@dataclass(frozen=True)
class EntityLookup:
    entity_to_struct: Dict[str, int]
    text_kg_id_to_embed_id: Dict[str, int]

    def entity_id(self, ent: str) -> int:
        idx = self.entity_to_struct.get(ent)
        if idx is None:
            raise KeyError(f"Unknown entity id: {ent}")
        return idx

    def embedding_id(self, ent: str) -> int:
        return self.text_kg_id_to_embed_id.get(ent, _NON_TEXT_EMBEDDING_ID)


@dataclass(frozen=True)
class RelationLookup:
    rel_to_id: Dict[str, int]

    def relation_id(self, rel: str) -> int:
        idx = self.rel_to_id.get(rel)
        if idx is None:
            raise KeyError(f"Unknown relation id: {rel}")
        return idx


class EntityVocab:
    """Assign structural IDs and embedding IDs; separate text vs non-text."""

    def __init__(self, kb: str, text_cfg: TextEntityConfig, cvt_cfg: Optional[CvtEntityConfig] = None) -> None:
        self.kb = kb
        self._text_cfg = text_cfg
        self._cvt_cfg = cvt_cfg
        self._entity_to_struct: Dict[str, int] = {}
        self._struct_records: List[Dict[str, object]] = []
        self._embedding_records: List[Dict[str, object]] = []
        self._text_entities: List[str] = []
        self._finalized = False
        self._text_kg_id_to_embed_id: Dict[str, int] = {}

    def add_entity(self, ent: str) -> int:
        if ent in self._entity_to_struct:
            return self._entity_to_struct[ent]
        if self._finalized:
            raise RuntimeError("Cannot add entities after finalize")
        idx = len(self._entity_to_struct)
        self._entity_to_struct[ent] = idx
        if self._text_cfg.is_text(ent):
            self._text_entities.append(ent)
        return idx

    def finalize(self) -> None:
        if self._finalized:
            return
        self._text_entities = sorted(self._text_entities)
        text_embedding_offset = 1
        text_embedding_ids: Dict[str, int] = {
            ent: text_embedding_offset + idx for idx, ent in enumerate(self._text_entities)
        }
        for ent in sorted(self._entity_to_struct, key=self._entity_to_struct.get):
            struct_id = self._entity_to_struct[ent]
            is_text = self._text_cfg.is_text(ent)
            is_cvt = self._cvt_cfg.is_cvt(ent) if self._cvt_cfg is not None else False
            if is_text and is_cvt:
                raise ValueError(f"Entity cannot be both text and CVT: {ent}")
            embedding_id = text_embedding_ids.get(ent, 0)
            record = {
                "entity_id": struct_id,
                "kb": self.kb,
                "kg_id": ent,
                "label": ent,
                "is_text": is_text,
                "is_cvt": is_cvt,
                "embedding_id": embedding_id,
            }
            self._struct_records.append(record)
            if is_text:
                self._embedding_records.append(
                    {
                        "embedding_id": embedding_id,
                        "kb": self.kb,
                        "kg_id": ent,
                        "label": ent,
                    }
                )
                self._text_kg_id_to_embed_id[ent] = embedding_id
        self._finalized = True

    @property
    def struct_records(self) -> List[Dict[str, object]]:
        if not self._finalized:
            self.finalize()
        return self._struct_records

    @property
    def embedding_records(self) -> List[Dict[str, object]]:
        if not self._finalized:
            self.finalize()
        return self._embedding_records

    def entity_id(self, ent: str) -> int:
        return self._entity_to_struct[ent]

    def embedding_id(self, ent: str) -> int:
        return _NON_TEXT_EMBEDDING_ID if not self._text_cfg.is_text(ent) else self._embedding_id_for_text(ent)

    def _embedding_id_for_text(self, ent: str) -> int:
        if not self._finalized:
            self.finalize()
        return self._text_kg_id_to_embed_id.get(ent, _NON_TEXT_EMBEDDING_ID)

    def to_lookup(self) -> EntityLookup:
        if not self._finalized:
            self.finalize()
        return EntityLookup(
            entity_to_struct=dict(self._entity_to_struct),
            text_kg_id_to_embed_id=dict(self._text_kg_id_to_embed_id),
        )


class RelationVocab:
    def __init__(self, kb: str) -> None:
        self.kb = kb
        self._rel_to_id: Dict[str, int] = {}
        self._records: List[Dict[str, object]] = []

    def relation_id(self, rel: str, *, label: Optional[str] = None) -> int:
        idx = self._rel_to_id.get(rel)
        resolved_label = rel if label is None else str(label)
        if idx is None:
            idx = len(self._rel_to_id)
            self._rel_to_id[rel] = idx
            self._records.append(
                {
                    "relation_id": idx,
                    "kb": self.kb,
                    "kg_id": rel,
                    "label": resolved_label,
                }
            )
            return idx
        existing_label = self._records[idx].get("label")
        if existing_label != resolved_label:
            raise ValueError(
                f"Relation label mismatch for {rel!r}: existing={existing_label!r} new={resolved_label!r}"
            )
        return idx

    def add_relations(self, relations: Sequence[str]) -> None:
        for rel in relations:
            self.relation_id(rel)

    @property
    def records(self) -> List[Dict[str, object]]:
        return self._records

    def to_lookup(self) -> RelationLookup:
        return RelationLookup(rel_to_id=dict(self._rel_to_id))
