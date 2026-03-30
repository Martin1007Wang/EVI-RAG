from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.data.schema.constants import _PATH_MODE_UNDIRECTED
from src.data.schema.types import Sample, SampleFilterOutcome, SplitFilter
from src.data.utils.connectivity import has_connectivity


def _resolve_split_filter(
    split: str,
    train_filter: SplitFilter,
    eval_filter: SplitFilter,
    override_filters: Dict[str, SplitFilter],
) -> SplitFilter:
    override = override_filters.get(split)
    if override is not None:
        return override
    return train_filter if split == "train" else eval_filter


def _partition_graph_edges(
    graph: Sequence[Tuple[str, str, str]],
    *,
    remove_self_loops: bool,
) -> List[Tuple[str, str, str]]:
    kept_edges: List[Tuple[str, str, str]] = []
    for head, rel, tail in graph:
        if remove_self_loops and head == tail:
            continue
        kept_edges.append((head, rel, tail))
    return kept_edges


def _should_keep_sample(
    sample: Sample,
    split_filter: SplitFilter,
    connectivity_cache: Dict[Tuple[str, str, str], bool],
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
    remove_self_loops: bool,
    kept_edges: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> SampleFilterOutcome:
    if kept_edges is None:
        kept_edges = _partition_graph_edges(
            sample.graph,
            remove_self_loops=remove_self_loops,
        )
    node_strings = {h for h, _, t in kept_edges} | {t for _, _, t in kept_edges}
    has_question_entity = any(ent in node_strings for ent in sample.question_entities)
    has_answer = any(ent in node_strings for ent in sample.answer_entities)

    cache_key = (sample.dataset, sample.split, sample.question_id)
    has_path = connectivity_cache.get(cache_key)
    if has_path is None:
        if split_filter.skip_no_path:
            has_path = has_connectivity(
                kept_edges,
                sample.question_entities,
                sample.answer_entities,
                path_mode=path_mode,
            )
        else:
            has_path = True
        connectivity_cache[cache_key] = has_path

    if split_filter.skip_no_question_entity and not has_question_entity:
        return SampleFilterOutcome(False, has_question_entity, has_answer, has_path)
    if split_filter.skip_no_ans and not has_answer:
        return SampleFilterOutcome(False, has_question_entity, has_answer, has_path)
    if split_filter.skip_no_path and not has_path:
        return SampleFilterOutcome(False, has_question_entity, has_answer, has_path)
    return SampleFilterOutcome(True, has_question_entity, has_answer, has_path)
