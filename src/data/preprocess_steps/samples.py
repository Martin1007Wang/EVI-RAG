from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

# 🎯 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from .vocab import EntityCatalog


@dataclass(frozen=True)
class SplitFilter:
    """样本过滤规则：定义在不同数据切分（train/val/test）下跳过样本的阈值"""

    skip_no_question_entity: bool
    skip_no_ans: bool
    skip_no_path: bool


@dataclass(frozen=True)
class RawSample:
    """原始样本：直接从数据集（HF/STARK）提取的未处理状态"""

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
    """就绪样本：完成了图过滤和上帝视角（最短路径）标注的状态"""

    sample: RawSample
    sample_id: str
    kept_edges: list[tuple[str, str, str]]
    question_entities_in_graph: tuple[str, ...]
    legal_answer_entities: tuple[str, ...]
    positive_edge_mask: torch.Tensor
    node_to_target_distance: torch.Tensor
    shortest_suffix_count: torch.Tensor
    bounded_suffix_count: torch.Tensor
    max_path_length: int | None


@dataclass(frozen=True)
class EncodedPayload:
    """编码包裹：完成了特征编码（Text Encoding）后的最终交付物"""

    entity_catalog: EntityCatalog
    relation_labels: list[str]
    entity_embeddings: torch.Tensor
    relation_embeddings: torch.Tensor
    question_embeddings: torch.Tensor
