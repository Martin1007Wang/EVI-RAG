from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.data.relation_cleaning_rules import (
    RELATION_ACTION_DROP,
    RELATION_ACTION_KEEP,
    RELATION_ACTION_TYPE,
    RelationCleaningRules,
    relation_action,
)
from src.data.schema.constants import (
    _PATH_MODE_UNDIRECTED,
    _TIME_RELATION_MODE_DROP,
    _TIME_RELATION_MODE_KEEP,
    _TIME_RELATION_MODE_QUESTION,
)
from src.data.schema.types import Sample, SampleFilterOutcome, SplitFilter, TimeRelationConfig
from src.data.utils.connectivity import _validate_path_mode, has_connectivity


def _resolve_split_filter(
    split: str, train_filter: SplitFilter, eval_filter: SplitFilter, override_filters: Dict[str, SplitFilter]
) -> SplitFilter:
    override = override_filters.get(split)
    if override is not None:
        return override
    return train_filter if split == "train" else eval_filter


def _resolve_time_gate(time_relation_cfg: Optional[TimeRelationConfig], question_text: Optional[str]) -> Optional[bool]:
    if time_relation_cfg is None:
        return None
    mode = time_relation_cfg.mode
    if mode == _TIME_RELATION_MODE_KEEP:
        return None
    if mode == _TIME_RELATION_MODE_DROP:
        return False
    if mode == _TIME_RELATION_MODE_QUESTION:
        if question_text is None:
            raise ValueError("time_relation_mode=question_gated requires question_text.")
        return time_relation_cfg.is_time_question(question_text)
    raise ValueError(f"Unsupported time_relation_mode: {mode!r}.")


def _edge_nodes(edges: Sequence[Tuple[str, str, str]]) -> set[str]:
    nodes: set[str] = set()
    for head, _, tail in edges:
        nodes.add(head)
        nodes.add(tail)
    return nodes


def _relax_anchor_edges(
    graph: Sequence[Tuple[str, str, str]],
    kept_edges: List[Tuple[str, str, str]],
    *,
    anchor_entities: Optional[Sequence[str]],
    remove_self_loops: bool,
) -> List[Tuple[str, str, str]]:
    if not anchor_entities:
        return kept_edges
    anchor_set = set(anchor_entities)
    if not anchor_set:
        return kept_edges
    if _edge_nodes(kept_edges) & anchor_set:
        return kept_edges
    relaxed_edges: List[Tuple[str, str, str]] = []
    for head, rel, tail in graph:
        if remove_self_loops and head == tail:
            continue
        if head in anchor_set or tail in anchor_set:
            # Anchor retention bypasses relation/time filtering to keep the start node.
            relaxed_edges.append((head, rel, tail))
    if not relaxed_edges:
        return kept_edges
    return kept_edges + relaxed_edges


def _partition_graph_edges(
    graph: Sequence[Tuple[str, str, str]],
    rules: RelationCleaningRules,
    *,
    remove_self_loops: bool,
    relation_cleaning_enabled: bool,
    time_relation_cfg: Optional[TimeRelationConfig] = None,
    question_text: Optional[str] = None,
    anchor_entities: Optional[Sequence[str]] = None,
    keep_anchor_edges: bool = False,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    kept_edges: List[Tuple[str, str, str]] = []
    type_edges: List[Tuple[str, str, str]] = []
    time_gate = _resolve_time_gate(time_relation_cfg, question_text)
    for head, rel, tail in graph:
        if remove_self_loops and head == tail:
            continue
        if (
            time_gate is not None
            and time_relation_cfg is not None
            and time_relation_cfg.is_time_relation(rel)
            and not time_gate
        ):
            continue
        action = relation_action(rel, rules, enabled=relation_cleaning_enabled)
        if action == RELATION_ACTION_KEEP:
            kept_edges.append((head, rel, tail))
        elif action == RELATION_ACTION_TYPE:
            type_edges.append((head, rel, tail))
    if keep_anchor_edges:
        kept_edges = _relax_anchor_edges(
            graph,
            kept_edges,
            anchor_entities=anchor_entities,
            remove_self_loops=remove_self_loops,
        )
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
    time_relation_cfg: Optional[TimeRelationConfig] = None,
    anchor_entities: Optional[Sequence[str]] = None,
    keep_anchor_edges: bool = False,
) -> SampleFilterOutcome:
    if kept_edges is None:
        kept_edges, _ = _partition_graph_edges(
            sample.graph,
            relation_cleaning_rules,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
            time_relation_cfg=time_relation_cfg,
            question_text=sample.question,
            anchor_entities=anchor_entities,
            keep_anchor_edges=keep_anchor_edges,
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
