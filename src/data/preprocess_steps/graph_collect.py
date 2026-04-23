from __future__ import annotations

import json
import logging
from pathlib import Path
from collections.abc import Iterable, Sequence

import torch

from src.utils.path_utils import compute_shortest_path_teacher_targets

from .entity_typing import EntityTyping
from .samples import PreparedSample, RawSample, SplitFilter
from .vocab import EntityVocab, RelationVocab
from data.preprocess_steps import entity_typing

log = logging.getLogger(__name__)


def collect_and_filter_graphs(
    sample_iter: Iterable[RawSample],
    *,
    out_dir: Path,
    dataset_name: str,
    split_filters: dict[str, SplitFilter],
    path_mode: str = "qa_directed",
    dedup_edges: bool = True,
    remove_self_loops: bool = True,
    emit_sub_filter: bool = True,
    sub_filter_filename: str = "sub_filter.json",
) -> tuple[list[PreparedSample], EntityVocab, RelationVocab]:
    if str(path_mode).strip().lower() != "qa_directed":
        raise ValueError(f"Unsupported path_mode={path_mode!r}; expected 'qa_directed'.")

    entity_vocab = EntityVocab()
    relation_vocab = RelationVocab()
    entity_typing = EntityTyping()
    prepared_samples: list[PreparedSample] = []
    answer_reachable_subset_ids: list[str] = []
    dropped_missing_question_entities = 0
    dropped_missing_answer_entities = 0
    dropped_missing_path = 0
    for sample in sample_iter:
        graph_edges = _prepare_graph_edges(sample.graph,remove_self_loops=remove_self_loops,dedup_edges=dedup_edges,)
        if not graph_edges:
            continue
        node_index, edge_index = _build_local_graph(graph_edges)
        question_entities = tuple(ent for ent in _dedup_preserve_order(sample.question_entities) if ent in node_index)
        answer_entities_in_graph = tuple(ent for ent in _dedup_preserve_order(sample.answer_entities) if ent in node_index)
        split_filter = split_filters[sample.split]
        if not question_entities:
            dropped_missing_question_entities += 1
            continue
        if split_filter.skip_no_ans and not answer_entities_in_graph:
            dropped_missing_answer_entities += 1
            continue
        anchor_node_ids = torch.as_tensor([node_index[ent] for ent in question_entities],dtype=torch.long,)
        answer_node_ids = torch.as_tensor([node_index[ent] for ent in answer_entities_in_graph],dtype=torch.long,)
        teacher_targets = _discover_reachable_targets(
            edge_index=edge_index,
            anchor_node_ids=anchor_node_ids,
            answer_node_ids=answer_node_ids,
            num_nodes=len(node_index),
            path_mode=path_mode,
        )

        target_node_ids = teacher_targets.target_node_ids.long()

        if split_filter.skip_no_path and target_node_ids.numel() == 0:
            dropped_missing_path += 1
            continue

        target_node_set = set(target_node_ids.tolist())
        reachable_answer_entities = tuple(
            ent for ent in answer_entities_in_graph if node_index[ent] in target_node_set
        )

        sample_id = _build_sample_id(sample)
        if reachable_answer_entities:
            answer_reachable_subset_ids.append(sample_id)

        prepared_samples.append(
            PreparedSample(
                dataset=sample.dataset,
                split=sample.split,
                question_id=sample.question_id,
                question=sample.question,
                graph_edges=tuple(graph_edges),
                question_entities=question_entities,
                answer_entities=reachable_answer_entities,
                anchor_node_ids=anchor_node_ids,
                target_node_ids=target_node_ids,
                target_node_distances_flat=teacher_targets.target_node_distance_flat,
            )
        )

        for head, relation, tail in graph_edges:
            entity_vocab.add(head)
            entity_vocab.add(tail)
            relation_vocab.add(relation)

    if not prepared_samples:
        raise RuntimeError("No samples remained after graph collection.")

    if emit_sub_filter:
        _write_sample_filter(
            out_dir / sub_filter_filename,
            dataset=dataset_name,
            sample_ids=answer_reachable_subset_ids,
            prepared_sample_count=len(prepared_samples),
        )

    log.info(
        "Collected %s valid samples (dropped: no_anchor=%s, no_answer=%s, no_path=%s).",
        len(prepared_samples),
        dropped_missing_question_entities,
        dropped_missing_answer_entities,
        dropped_missing_path,
    )

    return prepared_samples, entity_vocab, relation_vocab


def _build_sample_id(sample: RawSample) -> str:
    return f"{sample.dataset}/{sample.split}/{sample.question_id}"


def _prepare_graph_edges(
    graph: Sequence[tuple[str, str, str]],
    *,
    remove_self_loops: bool,
    dedup_edges: bool,
) -> list[tuple[str, str, str]]:
    if not remove_self_loops and not dedup_edges:
        return list(graph)

    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in graph:
        if remove_self_loops and edge[0] == edge[2]:
            continue
        if dedup_edges and edge in seen:
            continue
        if dedup_edges:
            seen.add(edge)
        edges.append(edge)

    return edges


def _build_local_graph(
    graph_edges: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, int], torch.Tensor]:
    node_index: dict[str, int] = {}
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for head, _, tail in graph_edges:
        edge_src.append(node_index.setdefault(head, len(node_index)))
        edge_dst.append(node_index.setdefault(tail, len(node_index)))
    edge_index = torch.as_tensor([edge_src, edge_dst], dtype=torch.long)
    return node_index, edge_index


def _discover_reachable_targets(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    answer_node_ids: torch.Tensor,
    num_nodes: int,
    path_mode: str,
):
    is_anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if anchor_node_ids.numel() > 0:
        is_anchor_mask[anchor_node_ids] = True

    target_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if answer_node_ids.numel() > 0:
        target_mask[answer_node_ids] = True

    return compute_shortest_path_teacher_targets(
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
        target_mask=target_mask,
        num_nodes=num_nodes,
        path_mode=path_mode,
    )


def _dedup_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _write_sample_filter(
    path: Path,
    *,
    dataset: str,
    sample_ids: Sequence[str],
    prepared_sample_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "filter_kind": "answer_reachable_subset",
                "prepared_sample_count": int(prepared_sample_count),
                "selected_sample_count": int(len(sample_ids)),
                "sample_ids": sorted(str(sample_id) for sample_id in sample_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )