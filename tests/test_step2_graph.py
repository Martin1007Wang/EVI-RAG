from __future__ import annotations

from src.data.preprocess.stages.step2_graph import (
    _all_anchors_reachable_targets_by_index,
    build_graph,
)
from src.data.schema.constants import _PATH_MODE_QA_DIRECTED
from src.data.schema.types import EntityLookup, RelationLookup, Sample


def _sample_with_duplicate_edges() -> Sample:
    return Sample(
        dataset="unit",
        split="train",
        question_id="q1",
        kb="freebase",
        question="who?",
        graph=[
            ("m.topic", "people.person.nationality", "m.country"),
            ("m.topic", "people.person.nationality", "m.country"),
        ],
        question_entities=["m.topic"],
        answer_entities=["m.country"],
        answer_texts=["country"],
    )


def test_build_graph_preserves_duplicate_edges_when_dedup_disabled() -> None:
    sample = _sample_with_duplicate_edges()

    graph = build_graph(
        sample,
        EntityLookup(
            entity_to_struct={"m.topic": 0, "m.country": 1},
            text_kg_id_to_embed_id={},
        ),
        RelationLookup(rel_to_id={"people.person.nationality": 0}),
        "unit/train/q1",
        dedup_edges=False,
        validate_graph_edges=True,
        remove_self_loops=True,
    )

    assert graph is not None
    assert graph.edge_src == [0, 0]
    assert graph.edge_dst == [1, 1]
    assert graph.edge_relation_ids == [0, 0]


def test_build_graph_dedups_duplicate_edges_when_enabled() -> None:
    sample = _sample_with_duplicate_edges()

    graph = build_graph(
        sample,
        EntityLookup(
            entity_to_struct={"m.topic": 0, "m.country": 1},
            text_kg_id_to_embed_id={},
        ),
        RelationLookup(rel_to_id={"people.person.nationality": 0}),
        "unit/train/q1",
        dedup_edges=True,
        validate_graph_edges=True,
        remove_self_loops=True,
    )

    assert graph is not None
    assert graph.edge_src == [0]
    assert graph.edge_dst == [1]
    assert graph.edge_relation_ids == [0]


def test_all_anchors_reachable_targets_by_index_returns_intersection() -> None:
    legal_targets = _all_anchors_reachable_targets_by_index(
        num_nodes=4,
        edge_src=[0, 1, 0],
        edge_dst=[2, 2, 3],
        anchor_local_indices=[0, 1],
        target_local_indices=[2, 3],
        path_mode=_PATH_MODE_QA_DIRECTED,
    )

    assert legal_targets == [2]


def test_build_graph_target_pruning_keeps_only_all_anchor_reachable_region() -> None:
    sample = Sample(
        dataset="unit",
        split="train",
        question_id="multi-anchor-keep",
        kb="freebase",
        question="who fits both anchors?",
        graph=[
            ("m.anchor_1", "rel.common", "m.common"),
            ("m.anchor_2", "rel.common", "m.common"),
            ("m.anchor_1", "rel.pseudo", "m.pseudo"),
        ],
        question_entities=["m.anchor_1", "m.anchor_2"],
        answer_entities=["m.common", "m.pseudo"],
        answer_texts=["common", "pseudo"],
    )

    graph = build_graph(
        sample,
        EntityLookup(
            entity_to_struct={
                "m.anchor_1": 0,
                "m.anchor_2": 1,
                "m.common": 2,
                "m.pseudo": 3,
            },
            text_kg_id_to_embed_id={},
        ),
        RelationLookup(rel_to_id={"rel.common": 0, "rel.pseudo": 1}),
        "unit/train/multi-anchor-keep",
        path_mode=_PATH_MODE_QA_DIRECTED,
        target_reachable_pruning=True,
    )

    assert graph is not None
    kept_edges = {
        (
            graph.node_labels[src],
            graph.edge_relation_ids[idx],
            graph.node_labels[dst],
        )
        for idx, (src, dst) in enumerate(zip(graph.edge_src, graph.edge_dst))
    }
    assert kept_edges == {
        ("m.anchor_1", 0, "m.common"),
        ("m.anchor_2", 0, "m.common"),
    }


def test_build_graph_target_pruning_drops_answers_missing_anchor_intersection() -> None:
    sample = Sample(
        dataset="unit",
        split="train",
        question_id="multi-anchor-drop",
        kb="freebase",
        question="who fits both anchors?",
        graph=[
            ("m.anchor_1", "rel.left", "m.left_only"),
            ("m.anchor_2", "rel.right", "m.right_only"),
        ],
        question_entities=["m.anchor_1", "m.anchor_2"],
        answer_entities=["m.left_only", "m.right_only"],
        answer_texts=["left", "right"],
    )

    graph = build_graph(
        sample,
        EntityLookup(
            entity_to_struct={
                "m.anchor_1": 0,
                "m.anchor_2": 1,
                "m.left_only": 2,
                "m.right_only": 3,
            },
            text_kg_id_to_embed_id={},
        ),
        RelationLookup(rel_to_id={"rel.left": 0, "rel.right": 1}),
        "unit/train/multi-anchor-drop",
        path_mode=_PATH_MODE_QA_DIRECTED,
        target_reachable_pruning=True,
    )

    assert graph is None
