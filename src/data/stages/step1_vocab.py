from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.data.relation_cleaning_rules import (
    RELATION_ACTION_DROP,
    RELATION_ACTION_KEEP,
    RELATION_ACTION_TYPE,
    RelationCleaningRules,
    relation_action,
)
from src.data.schema.constants import _PATH_MODE_UNDIRECTED
from src.data.schema.types import Sample, SampleFilterOutcome, SplitFilter
from src.data.utils.connectivity import _validate_path_mode, has_connectivity


def _resolve_split_filter(
    split: str, train_filter: SplitFilter, eval_filter: SplitFilter, override_filters: Dict[str, SplitFilter]
) -> SplitFilter:
    override = override_filters.get(split)
    if override is not None:
        return override
    return train_filter if split == "train" else eval_filter


def _partition_graph_edges(
    graph: Sequence[Tuple[str, str, str]],
    rules: RelationCleaningRules,
    *,
    remove_self_loops: bool,
    relation_cleaning_enabled: bool,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    kept_edges: List[Tuple[str, str, str]] = []
    type_edges: List[Tuple[str, str, str]] = []
    for h, r, t in graph:
        if remove_self_loops and h == t:
            continue
        action = relation_action(r, rules, enabled=relation_cleaning_enabled)
        if action == RELATION_ACTION_KEEP:
            kept_edges.append((h, r, t))
        elif action == RELATION_ACTION_TYPE:
            type_edges.append((h, r, t))
    return kept_edges, type_edges


def _should_keep_sample(
    sample: Sample,
    split_filter: SplitFilter,
    connectivity_cache: Dict[Tuple[str, str, str], bool],
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
    remove_self_loops: bool,
    relation_cleaning_enabled: bool,
    relation_cleaning_rules: RelationCleaningRules,
    kept_edges: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> SampleFilterOutcome:
    if kept_edges is None:
        kept_edges, _ = _partition_graph_edges(
            sample.graph,
            relation_cleaning_rules,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
        )
    node_strings = {h for h, _, t in kept_edges} | {t for _, _, t in kept_edges}
    has_topic = any(ent in node_strings for ent in sample.q_entity)
    has_answer = any(ent in node_strings for ent in sample.a_entity)

    cache_key = (sample.dataset, sample.split, sample.question_id)
    has_path = connectivity_cache.get(cache_key)
    if has_path is None:
        if split_filter.skip_no_path:
            has_path = has_connectivity(kept_edges, sample.q_entity, sample.a_entity, path_mode=path_mode)
        else:
            has_path = True
        connectivity_cache[cache_key] = has_path

    if split_filter.skip_no_topic and not has_topic:
        return SampleFilterOutcome(False, has_topic, has_answer, has_path)
    if split_filter.skip_no_ans and not has_answer:
        return SampleFilterOutcome(False, has_topic, has_answer, has_path)
    if split_filter.skip_no_path and not has_path:
        return SampleFilterOutcome(False, has_topic, has_answer, has_path)
    return SampleFilterOutcome(True, has_topic, has_answer, has_path)
