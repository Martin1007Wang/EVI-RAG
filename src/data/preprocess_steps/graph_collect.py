from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from src.utils.path_utils import compute_shortest_path_labels

from .sample_types import (
    EntityVocab,
    PreparedSample,
    RawSample,
    RelationVocab,
    SplitFilter,
)


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
    dedup_edges: bool = True,
    remove_self_loops: bool = True,
    emit_sub_filter: bool = True,
    sub_filter_filename: str = "sub_filter.json",
) -> tuple[list[PreparedSample], EntityVocab, RelationVocab]:
    path_mode = str(path_mode or "undirected").strip().lower()
    if path_mode not in {"undirected", "qa_directed"}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of undirected or qa_directed."
        )

    entity_vocab = EntityVocab(
        is_text_entity=is_text_entity_fn,
        is_cvt_entity=is_cvt_entity_fn,
    )
    relation_vocab = RelationVocab()

    prepared_samples: list[PreparedSample] = []
    sub_sample_ids: list[str] = []

    for sample in sample_iter:
        split = str(sample.split)
        if blacklist:
            filtered_question_entities = tuple(
                entity for entity in sample.question_entities if entity not in blacklist
            )
            if filtered_question_entities != sample.question_entities:
                sample = RawSample(
                    dataset=sample.dataset,
                    split=sample.split,
                    question_id=sample.question_id,
                    kb=sample.kb,
                    question=sample.question,
                    graph=sample.graph,
                    question_entities=filtered_question_entities,
                    answer_entities=sample.answer_entities,
                    answer_texts=sample.answer_texts,
                )

        kept_edges = _prepare_graph_edges(
            sample.graph,
            remove_self_loops=remove_self_loops,
            dedup_edges=dedup_edges,
        )
        if not kept_edges:
            continue

        node_index: dict[str, int] = {}
        edge_src: list[int] = []
        edge_dst: list[int] = []
        for head, _, tail in kept_edges:
            if head not in node_index:
                node_index[head] = len(node_index)
            if tail not in node_index:
                node_index[tail] = len(node_index)
            edge_src.append(node_index[head])
            edge_dst.append(node_index[tail])

        question_entities_in_graph = tuple(
            entity
            for entity in _dedup_preserve_order(sample.question_entities)
            if entity in node_index
        )
        answer_entities_in_graph = tuple(
            entity
            for entity in _dedup_preserve_order(sample.answer_entities)
            if entity in node_index
        )

        split_filter = split_filters[split]
        if split_filter.skip_no_question_entity and not question_entities_in_graph:
            continue
        if split_filter.skip_no_ans and not answer_entities_in_graph:
            continue

        anchor_local_indices = [
            node_index[entity] for entity in question_entities_in_graph
        ]
        answer_local_indices = [
            node_index[entity] for entity in answer_entities_in_graph
        ]
        graph_id = f"{sample.dataset}/{sample.split}/{sample.question_id}"

        num_nodes_in_graph = len(node_index)
        edge_index_tensor = torch.tensor([edge_src, edge_dst], dtype=torch.long)

        # 构造临时的布尔掩码
        tmp_anchor_mask = torch.zeros(num_nodes_in_graph, dtype=torch.bool)
        tmp_anchor_mask[anchor_local_indices] = True

        tmp_target_mask = torch.zeros(num_nodes_in_graph, dtype=torch.bool)
        tmp_target_mask[answer_local_indices] = True

        sp_labels = compute_shortest_path_labels(
            edge_index=edge_index_tensor,
            is_anchor_mask=tmp_anchor_mask,
            is_target_mask=tmp_target_mask,
            num_nodes=num_nodes_in_graph,
            path_mode=path_mode,
        )

        legal_answer_local_indices = sp_labels.reachable_target_node_ids.tolist()
        if split_filter.skip_no_path and not legal_answer_local_indices:
            continue

        legal_answer_local_set = set(legal_answer_local_indices)
        legal_answer_entities = tuple(
            entity
            for entity in answer_entities_in_graph
            if node_index[entity] in legal_answer_local_set
        )
        if legal_answer_entities:
            sub_sample_ids.append(graph_id)

        prepared_samples.append(
            PreparedSample(
                sample=sample,
                sample_id=graph_id,
                kept_edges=kept_edges,
                question_entities_in_graph=question_entities_in_graph,
                legal_answer_entities=legal_answer_entities,
                positive_edge_ids=sp_labels.positive_edge_ids,
            )
        )
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
    return prepared_samples, entity_vocab, relation_vocab


def load_question_entity_blacklist(
    *, inline_list: Sequence[str] | None, file_path: Path | None
) -> set[str]:
    blacklist = {
        str(value).strip() for value in list(inline_list or []) if str(value).strip()
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
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for head, relation, tail in graph:
        edge = (str(head), str(relation), str(tail))
        if remove_self_loops and edge[0] == edge[2]:
            continue
        if dedup_edges and edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges


def _dedup_preserve_order(values: Sequence[str]) -> list[str]:
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
