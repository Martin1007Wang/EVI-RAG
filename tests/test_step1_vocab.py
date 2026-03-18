from __future__ import annotations

from src.data.preprocess.stages.step1_vocab import (
    _partition_graph_edges,
    _should_keep_sample,
)
from src.data.schema.types import Sample, SplitFilter


def test_partition_graph_edges_keeps_relations_that_used_to_be_cleaned() -> None:
    graph = [
        ("m.topic", "type.object.type", "people.person"),
        ("m.topic", "people.person.nationality", "m.country"),
        ("m.topic", "type.object.type", "m.topic"),
    ]

    kept_edges = _partition_graph_edges(graph, remove_self_loops=True)

    assert kept_edges == [
        ("m.topic", "type.object.type", "people.person"),
        ("m.topic", "people.person.nationality", "m.country"),
    ]


def test_should_keep_sample_uses_full_non_self_loop_graph() -> None:
    sample = Sample(
        dataset="unit",
        split="train",
        question_id="q1",
        kb="freebase",
        question="what nationality is the person?",
        graph=[("m.topic", "people.person.nationality", "m.country")],
        q_entity=["m.topic"],
        a_entity=["m.country"],
        answer_texts=["country"],
    )

    outcome = _should_keep_sample(
        sample,
        SplitFilter(skip_no_topic=True, skip_no_ans=True, skip_no_path=True),
        {},
        remove_self_loops=True,
    )

    assert outcome.keep is True
    assert outcome.has_topic is True
    assert outcome.has_answer is True
    assert outcome.has_path is True
