from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from src.utils.path_utils import compute_shortest_path_teacher_targets

# 🎯 架构对齐：从不同文件按职责导入
from .samples import PreparedSample, RawSample, SplitFilter
from .vocab import EntityVocab, RelationVocab

log = logging.getLogger(__name__)


def _build_positive_edge_mask(
    *,
    edge_index: torch.Tensor,
    positive_edge_ids: torch.Tensor,
) -> torch.Tensor:
    """Materialize the full shortest-path teacher edge union.

    ``compute_shortest_path_teacher_targets()`` already returns the union of all
    edges that lie on at least one shortest anchor→target path. That full set is
    the intended ``positive_edge_mask`` semantics for shaping / supervision.
    """

    num_edges = int(edge_index.size(1))
    mask = torch.zeros((num_edges,), dtype=torch.bool)
    if positive_edge_ids.numel() == 0 or num_edges == 0:
        return mask

    mask[positive_edge_ids.long()] = True
    return mask


def collect_and_filter_graphs(
    sample_iter: Iterable[RawSample],
    *,
    out_dir: Path,
    dataset_name: str,
    split_filters: dict[str, SplitFilter],
    is_text_entity_fn: Callable[[str], bool],
    is_cvt_entity_fn: Callable[[str], bool],
    blacklist: set[str],
    path_mode: str = "undirected",
    teacher_budget_max_steps: int = 0,
    dedup_edges: bool = True,
    remove_self_loops: bool = True,
    emit_sub_filter: bool = True,
    sub_filter_filename: str = "sub_filter.json",
) -> tuple[list[PreparedSample], EntityVocab, RelationVocab]:
    """
    预处理流水线第一阶段：收集原始样本，执行拓扑清洗，并计算上帝视角（最短路径）标签。
    """
    path_mode = str(path_mode or "undirected").strip().lower()
    if path_mode not in {"undirected", "qa_directed"}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of undirected or qa_directed."
        )
    if teacher_budget_max_steps < 0:
        raise ValueError(
            "teacher_budget_max_steps must be >= 0, got "
            f"{teacher_budget_max_steps}."
        )

    entity_vocab = EntityVocab(
        is_text_entity=is_text_entity_fn,
        is_cvt_entity=is_cvt_entity_fn,
    )
    relation_vocab = RelationVocab()

    prepared_samples: list[PreparedSample] = []
    sub_sample_ids: list[str] = []
    dropped_missing_anchor = 0
    dropped_missing_answer = 0
    dropped_missing_path = 0

    log.info("Starting graph collection and topology analysis...")

    for sample in sample_iter:
        split = str(sample.split)

        # 1. 黑名单过滤（保持高效的元组重建）
        if blacklist:
            filtered_q_ents = tuple(
                ent for ent in sample.question_entities if ent not in blacklist
            )
            if len(filtered_q_ents) != len(sample.question_entities):
                sample = RawSample(
                    dataset=sample.dataset,
                    split=sample.split,
                    question_id=sample.question_id,
                    kb=sample.kb,
                    question=sample.question,
                    graph=sample.graph,
                    question_entities=filtered_q_ents,
                    answer_entities=sample.answer_entities,
                    answer_texts=sample.answer_texts,
                )

        # 2. 边清洗（信任输入契约，移除多余 str()）
        kept_edges = _prepare_graph_edges(
            sample.graph,
            remove_self_loops=remove_self_loops,
            dedup_edges=dedup_edges,
        )
        if not kept_edges:
            continue

        # 3. 局部索引构建与全局词表注册
        node_index: dict[str, int] = {}
        edge_src: list[int] = []
        edge_dst: list[int] = []

        # 利用 setdefault 提升 20% 以上的哈希查找性能
        for head, _, tail in kept_edges:
            edge_src.append(node_index.setdefault(head, len(node_index)))
            edge_dst.append(node_index.setdefault(tail, len(node_index)))

        # 4. 实体对齐与合法性校验
        # 预先去重，确保后续 materialize 阶段无需二次处理
        question_entities_in_graph = tuple(
            ent
            for ent in _dedup_preserve_order(sample.question_entities)
            if ent in node_index
        )
        answer_entities_in_graph = tuple(
            ent
            for ent in _dedup_preserve_order(sample.answer_entities)
            if ent in node_index
        )

        split_filter = split_filters[split]
        if not question_entities_in_graph:
            dropped_missing_anchor += 1
            continue
        if split_filter.skip_no_ans and not answer_entities_in_graph:
            dropped_missing_answer += 1
            continue

        # 5. 上帝视角计算：最短路径与势能标注
        anchor_local_indices = [node_index[ent] for ent in question_entities_in_graph]
        answer_local_indices = [node_index[ent] for ent in answer_entities_in_graph]

        num_nodes_in_graph = len(node_index)
        # 使用 torch.as_tensor 减少拷贝开销
        edge_index_tensor = torch.as_tensor([edge_src, edge_dst], dtype=torch.long)

        tmp_anchor_mask = torch.zeros(num_nodes_in_graph, dtype=torch.bool)
        tmp_anchor_mask[anchor_local_indices] = True

        tmp_target_mask = torch.zeros(num_nodes_in_graph, dtype=torch.bool)
        tmp_target_mask[answer_local_indices] = True

        # 调用核心算法工具包
        teacher_targets = compute_shortest_path_teacher_targets(
            edge_index=edge_index_tensor,
            is_anchor_mask=tmp_anchor_mask,
            is_target_mask=tmp_target_mask,
            num_nodes=num_nodes_in_graph,
            path_mode=path_mode,
            budget_max_steps=teacher_budget_max_steps,
        )

        # 6. 路径存在性过滤
        reachable_indices = teacher_targets.reachable_target_node_ids.tolist()
        if split_filter.skip_no_path and not reachable_indices:
            dropped_missing_path += 1
            continue

        # 7. 幸存样本封装
        reachable_set = set(reachable_indices)
        legal_answer_entities = tuple(
            ent for ent in answer_entities_in_graph if node_index[ent] in reachable_set
        )

        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"
        if legal_answer_entities:
            sub_sample_ids.append(graph_id)

        # 构造正边掩码
        positive_edge_mask = _build_positive_edge_mask(
            edge_index=edge_index_tensor,
            positive_edge_ids=teacher_targets.positive_edge_ids,
        )

        prepared_samples.append(
            PreparedSample(
                sample=sample,
                sample_id=graph_id,
                kept_edges=kept_edges,
                question_entities_in_graph=question_entities_in_graph,
                legal_answer_entities=legal_answer_entities,
                positive_edge_mask=positive_edge_mask,
                node_to_target_distance=teacher_targets.node_to_target_distance,
                shortest_suffix_count=teacher_targets.shortest_suffix_count,
                bounded_suffix_count=teacher_targets.bounded_suffix_count,
                max_path_length=teacher_targets.max_path_length,
            )
        )

        # 8. 最终向全局词表注册（仅幸存样本的实体和关系）
        for head, relation, tail in kept_edges:
            entity_vocab.add(head)
            entity_vocab.add(tail)
            relation_vocab.add(relation)

    if not prepared_samples:
        raise RuntimeError(
            "No samples remained after preprocessing; nothing to materialize."
        )

    if emit_sub_filter:
        _write_sample_filter(
            out_dir / sub_filter_filename,
            dataset=dataset_name,
            sample_ids=sub_sample_ids,
        )

    log.info(
        "Collected %s valid samples (dropped: no_anchor=%s, no_answer=%s, no_path=%s).",
        len(prepared_samples),
        dropped_missing_anchor,
        dropped_missing_answer,
        dropped_missing_path,
    )
    return prepared_samples, entity_vocab, relation_vocab


def load_question_entity_blacklist(
    *, inline_list: Sequence[str] | None, file_path: Path | None
) -> set[str]:
    """加载黑名单实体。"""
    blacklist = {
        str(value).strip() for value in (inline_list or []) if str(value).strip()
    }
    if file_path is None:
        return blacklist

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"question_entity_blacklist_path not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(
                "question_entity_blacklist_path JSON must contain a list of entities."
            )
        blacklist.update(str(value).strip() for value in payload if str(value).strip())
        return blacklist

    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            blacklist.add(item)
    return blacklist


def _prepare_graph_edges(
    graph: Sequence[tuple[str, str, str]],
    *,
    remove_self_loops: bool,
    dedup_edges: bool,
) -> list[tuple[str, str, str]]:
    """清洗图边，极致扁平化的核心循环。"""
    if not remove_self_loops and not dedup_edges:
        return list(graph)

    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in graph:
        # 直接使用 edge[0] 和 edge[2]，无需重建 tuple
        if remove_self_loops and edge[0] == edge[2]:
            continue
        if dedup_edges and edge in seen:
            continue
        if dedup_edges:
            seen.add(edge)
        edges.append(edge)

    return edges


def _dedup_preserve_order(values: Sequence[str]) -> list[str]:
    """保留顺序的去重。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _write_sample_filter(
    path: Path, *, dataset: str, sample_ids: Sequence[str]
) -> None:
    """写入子过滤器 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "sample_ids": sorted(str(sample_id) for sample_id in sample_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "collect_and_filter_graphs",
    "load_question_entity_blacklist",
]
