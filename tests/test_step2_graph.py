from __future__ import annotations

from src.data.preprocess.stages.step2_graph import build_graph
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
        q_entity=["m.topic"],
        a_entity=["m.country"],
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
