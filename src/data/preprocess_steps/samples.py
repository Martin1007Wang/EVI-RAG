from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
from .vocab import EntityCatalog


@dataclass(frozen=True)
class SplitFilter:
    skip_no_question_entity: bool
    skip_no_ans: bool
    skip_no_path: bool
    skip_no_train_target: bool


@dataclass(frozen=True)
class RawSample:
    dataset: str
    split: str
    question_id: str
    question: str
    graph: tuple[tuple[str, str, str], ...]
    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSample:
    dataset: str
    split: str
    question_id: str
    question: str
    graph: tuple[tuple[str, str, str], ...]

    # cleaned/aligned entities in graph order
    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]

    # local node ids aligned with the entity tuples above
    anchor_node_ids: torch.Tensor
    target_node_ids: torch.Tensor

    # flattened tensor with shape semantics (num_targets, num_nodes)
    target_node_distances_flat: torch.Tensor


@dataclass(frozen=True)
class EncodedPayload:
    entity_catalog: EntityCatalog
    relation_labels: list[str]
    entity_embeddings: torch.Tensor
    relation_embeddings: torch.Tensor
    question_embeddings: torch.Tensor
