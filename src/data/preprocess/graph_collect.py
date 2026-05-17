from __future__ import annotations
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import torch
from src.graph.ops import build_local_graph
from src.graph.paths import compute_path_labels
from .samples import PreparedSample, RawSample, SplitFilter
from .vocab import EntityVocab, RelationVocab

log = logging.getLogger(__name__)
Edge = tuple[str, str, str]


@dataclass
class GraphCollectStats:
    kept: int = 0
    empty_graph: int = 0
    no_anchor_in_graph: int = 0
    no_answer_in_graph: int = 0
    no_reachable_answer: int = 0
    unknown_split: int = 0

    def drops(self) -> dict[str, int]:
        return {
            "empty_graph": self.empty_graph,
            "no_anchor_in_graph": self.no_anchor_in_graph,
            "no_answer_in_graph": self.no_answer_in_graph,
            "no_reachable_answer": self.no_reachable_answer,
            "unknown_split": self.unknown_split,
        }


def collect_and_filter_graphs(
    sample_iter: Iterable[RawSample],
    *,
    split_filters: dict[str, SplitFilter],
    dedup_edges: bool = True,
    remove_self_loops: bool = True,
    validate_alignment: bool = True,
) -> tuple[list[PreparedSample], EntityVocab, RelationVocab]:
    entity_vocab = EntityVocab()
    relation_vocab = RelationVocab()
    prepared_samples: list[PreparedSample] = []
    drops = {
        "empty_graph": 0,
        "no_anchor_in_graph": 0,
        "no_answer_in_graph": 0,
        "no_reachable_answer": 0,
        "unknown_split": 0,
    }
    for sample in sample_iter:
        stats = GraphCollectStats()
        prepared = prepare_sample(
            sample=sample,
            split_filters=split_filters,
            entity_vocab=entity_vocab,
            relation_vocab=relation_vocab,
            dedup_edges=dedup_edges,
            remove_self_loops=remove_self_loops,
            validate_alignment=validate_alignment,
            update_vocab=True,
            stats=stats,
        )
        for key, value in stats.drops().items():
            drops[key] += value
        if prepared is not None:
            prepared_samples.append(prepared)
    if not prepared_samples:
        raise RuntimeError("No valid samples after graph collection.")
    log.info(
        "Collected %d samples (drop: %s)",
        len(prepared_samples),
        drops,
    )
    return prepared_samples, entity_vocab, relation_vocab


def prepare_sample(
    *,
    sample: RawSample,
    split_filters: dict[str, SplitFilter],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    dedup_edges: bool = True,
    remove_self_loops: bool = True,
    validate_alignment: bool = True,
    update_vocab: bool = True,
    stats: GraphCollectStats | None = None,
) -> PreparedSample | None:
    stats = stats or GraphCollectStats()
    split_filter = split_filters.get(sample.split)
    if split_filter is None:
        stats.unknown_split += 1
        return None
    graph_edges = _clean_edges(
        sample.graph,
        remove_self_loops=remove_self_loops,
        dedup_edges=dedup_edges,
    )
    if not graph_edges:
        stats.empty_graph += 1
        return None
    node_index, edge_index = build_local_graph(graph_edges)
    edge_index = edge_index.long().contiguous()
    num_nodes = len(node_index)
    num_edges = int(edge_index.size(1))
    if num_nodes == 0 or num_edges == 0:
        stats.empty_graph += 1
        return None
    if validate_alignment:
        _validate_edge_alignment(graph_edges, node_index, edge_index)
    question_entities = _filter_graph_entities(
        sample.question_entities,
        node_index,
    )
    if not question_entities:
        stats.no_anchor_in_graph += 1
        return None
    answer_entities = _filter_graph_entities(
        sample.answer_entities,
        node_index,
    )
    if split_filter.require_answer_in_graph and not answer_entities:
        stats.no_answer_in_graph += 1
        return None
    anchor_node_ids = torch.tensor(
        [node_index[entity] for entity in question_entities],
        dtype=torch.long,
    )
    target_node_ids = torch.tensor(
        [node_index[entity] for entity in answer_entities],
        dtype=torch.long,
    )
    path_labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        num_nodes=num_nodes,
    )
    if (
        split_filter.require_reachable_answer
        and path_labels.reachable_target_node_ids.numel() == 0
    ):
        stats.no_reachable_answer += 1
        return None
    node_entity_catalog_ids, edge_relation_catalog_ids = _build_graph_catalog_ids(
        graph_edges=graph_edges,
        node_index=node_index,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        update_vocab=update_vocab,
    )
    stats.kept += 1
    return PreparedSample(
        dataset=sample.dataset,
        split=sample.split,
        question_id=sample.question_id,
        question=sample.question,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=num_edges,
        question_entities=question_entities,
        answer_entities=answer_entities,
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        reachable_target_node_ids=path_labels.reachable_target_node_ids,
        node_entity_catalog_ids=node_entity_catalog_ids,
        edge_relation_catalog_ids=edge_relation_catalog_ids,
        anchor_node_forward_distances_flat=path_labels.anchor_node_forward_distances_flat,
        anchor_node_backward_distances_flat=path_labels.anchor_node_backward_distances_flat,
        node_target_distance=path_labels.node_target_distance,
        node_target_distances_flat=path_labels.node_target_distances_flat,
        node_target_shortest_path_count_flat=path_labels.node_target_shortest_path_count_flat,
        node_target_shortest_path_edge_mask_flat=path_labels.node_target_shortest_path_edge_mask_flat,
        node_target_shortest_path_edge_count_flat=path_labels.node_target_shortest_path_edge_count_flat,
    )


def _clean_edges(
    graph: Sequence[Edge],
    *,
    remove_self_loops: bool,
    dedup_edges: bool,
) -> list[Edge]:
    """Clean source triples while preserving relation-distinct parallel edges."""
    edges: list[Edge] = []
    seen: set[Edge] = set()
    for head, relation, tail in graph:
        edge = (head, relation, tail)
        if remove_self_loops and head == tail:
            continue
        if dedup_edges and edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges


def _filter_graph_entities(
    entities: Sequence[str],
    node_index: dict[str, int],
) -> tuple[str, ...]:
    seen: set[str] = set()
    kept: list[str] = []
    for entity in entities:
        if entity in seen:
            continue
        if entity not in node_index:
            continue
        seen.add(entity)
        kept.append(entity)
    return tuple(kept)


def _build_graph_catalog_ids(
    *,
    graph_edges: Sequence[Edge],
    node_index: dict[str, int],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    update_vocab: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_entity_catalog_ids = torch.empty(len(node_index), dtype=torch.long)
    for entity, local_node_id in node_index.items():
        node_entity_catalog_ids[local_node_id] = (
            entity_vocab.add(entity) if update_vocab else entity_vocab.id(entity)
        )
    edge_relation_catalog_ids = torch.tensor(
        [
            relation_vocab.add(relation) if update_vocab else relation_vocab.id(relation)
            for _, relation, _ in graph_edges
        ],
        dtype=torch.long,
    )
    return (
        node_entity_catalog_ids.contiguous(),
        edge_relation_catalog_ids.contiguous(),
    )


def _validate_edge_alignment(
    graph_edges: Sequence[Edge],
    node_index: dict[str, int],
    edge_index: torch.Tensor,
) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )
    if edge_index.size(1) != len(graph_edges):
        raise ValueError(
            f"edge_index has {edge_index.size(1)} edges, "
            f"but graph_edges has {len(graph_edges)}."
        )
    for edge_id, (head, _, tail) in enumerate(graph_edges):
        expected = (node_index[head], node_index[tail])
        actual = (
            int(edge_index[0, edge_id].item()),
            int(edge_index[1, edge_id].item()),
        )
        if actual != expected:
            raise ValueError(
                "edge_index order is not aligned with graph_edges at "
                f"edge_id={edge_id}: expected {expected}, got {actual}."
            )
