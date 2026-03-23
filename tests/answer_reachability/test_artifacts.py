from __future__ import annotations

from dataclasses import asdict
import json
import pytest

from src.metrics.answer_metrics import (
    AnswerPosteriorRecord,
    AnswerSupportRecord,
    EdgeRecord,
    SupportWindowArtifactWriter,
    SupportWindowLabelRecord,
    SupportWindowResult,
    TrajectoryRecord,
    UNSPECIFIED_INFERENCE_MODE,
    UNSPECIFIED_MASS_REFERENCE,
)


def test_artifact_writer_splits_prompt_and_label_records(tmp_path) -> None:
    result = SupportWindowResult(
        sample_id="s1",
        dataset_scope="full",
        mass_threshold=0.9,
        window_size=1,
        covered_mass=0.6,
        residual_mass=0.4,
        gold_answer_mass=0.6,
        covered_gold_answer_mass=0.6,
        missed_gold_answer_mass=0.0,
        unique_answer_count=1,
        unique_path_count=1,
        gold_answer_entity_ids=[2],
        start_entity_ids=[1],
        trajectories=[
            TrajectoryRecord(
                sample_id="s1",
                path_rank=1,
                log_prob=-0.5108256,
                prob=0.6,
                cumulative_mass=0.6,
                terminal_entity_id=2,
                is_gold=True,
                edges=[
                    EdgeRecord(
                        edge_id=0,
                        src_entity_id=1,
                        relation_id=7,
                        dst_entity_id=2,
                    )
                ],
                start_entity_id=1,
                answer_rank=1,
                support_rank=1,
                conditional_prob=1.0,
                conditional_cumulative_mass=1.0,
            )
        ],
        inference_mode="monte_carlo",
        answer_mass_threshold=0.9,
        support_mass_threshold=0.9,
        probe_count=4,
        emit_path_count=1,
        remaining_mass_upper=0.4,
        stop_reason="support_mass_reached",
        coverage_certified=False,
        answer_mass_reference="monte_carlo",
        support_mass_reference="monte_carlo",
        selected_answer_ids=[2],
        answer_posterior=[
            AnswerPosteriorRecord(
                answer_entity_id=2,
                prob=0.6,
                cumulative_mass=0.6,
                is_gold=True,
                is_selected=True,
                support_mass=0.6,
                support_conditioned_mass=1.0,
                support_path_count=1,
            )
        ],
        answer_support=[
            AnswerSupportRecord(
                answer_entity_id=2,
                answer_rank=1,
                prob=0.6,
                cumulative_mass=0.6,
                is_gold=True,
                is_selected=True,
                support_mass=0.6,
                support_conditioned_mass=1.0,
                support_path_count=1,
                trajectories=[
                    TrajectoryRecord(
                        sample_id="s1",
                        path_rank=1,
                        log_prob=-0.5108256,
                        prob=0.6,
                        cumulative_mass=0.6,
                        terminal_entity_id=2,
                        is_gold=True,
                        edges=[
                            EdgeRecord(
                                edge_id=0,
                                src_entity_id=1,
                                relation_id=7,
                                dst_entity_id=2,
                            )
                        ],
                        start_entity_id=1,
                        answer_rank=1,
                        support_rank=1,
                        conditional_prob=1.0,
                        conditional_cumulative_mass=1.0,
                    )
                ],
            )
        ],
    )
    label = SupportWindowLabelRecord(
        sample_id="s1",
        question="Where was X born?",
        start_entity_ids=[1],
        answer_entity_ids=[2],
        a_entity_in_graph=True,
    )

    writer = SupportWindowArtifactWriter(output_dir=tmp_path, split="test")
    paths = writer.write(results=[result], labels=[label])

    prompt_record = json.loads(paths["prompt_path"].read_text(encoding="utf-8").strip())
    label_record = json.loads(paths["labels_path"].read_text(encoding="utf-8").strip())

    assert prompt_record["sample_id"] == "s1"
    assert prompt_record["question"] == "Where was X born?"
    assert prompt_record["dataset_scope"] == "full"
    assert prompt_record["mass_threshold"] == 0.9
    assert prompt_record["inference_mode"] == "monte_carlo"
    assert prompt_record["answer_mass_threshold"] == 0.9
    assert prompt_record["support_mass_threshold"] == 0.9
    assert prompt_record["probe_count"] == 4
    assert prompt_record["emit_path_count"] == 1
    assert prompt_record["remaining_mass_upper"] == 0.4
    assert prompt_record["covered_mass_ci_low"] is None
    assert prompt_record["covered_mass_ci_high"] is None
    assert prompt_record["gold_answer_mass"] == 0.6
    assert prompt_record["gold_answer_mass_ci_low"] is None
    assert prompt_record["gold_answer_mass_ci_high"] is None
    assert prompt_record["ci_confidence_level"] is None
    assert prompt_record["coverage_certified"] is False
    assert prompt_record["answer_mass_reference"] == "monte_carlo"
    assert prompt_record["support_mass_reference"] == "monte_carlo"
    assert prompt_record["selected_answer_ids"] == [2]
    assert prompt_record["residual_mass"] == 0.4
    assert prompt_record["answer_posterior"] == [
        {
            "answer_entity_id": 2,
            "prob": 0.6,
            "prob_ci_low": 0.0,
            "prob_ci_high": 0.0,
            "cumulative_mass": 0.6,
            "is_gold": True,
            "is_selected": True,
            "support_mass": 0.6,
            "support_conditioned_mass": 1.0,
            "support_path_count": 1,
        }
    ]
    assert prompt_record["answer_support"] == [
        {
            "answer_entity_id": 2,
            "answer_rank": 1,
            "prob": 0.6,
            "prob_ci_low": 0.0,
            "prob_ci_high": 0.0,
            "cumulative_mass": 0.6,
            "is_gold": True,
            "is_selected": True,
            "support_mass": 0.6,
            "support_conditioned_mass": 1.0,
            "support_path_count": 1,
            "trajectories": [
                {
                    "path_rank": 1,
                    "log_prob": -0.5108256,
                    "prob": 0.6,
                    "cumulative_mass": 0.6,
                    "terminal_entity_id": 2,
                    "start_entity_id": 1,
                    "answer_rank": 1,
                    "support_rank": 1,
                    "conditional_prob": 1.0,
                    "conditional_cumulative_mass": 1.0,
                    "edges": [
                        {
                            "edge_id": 0,
                            "src_entity_id": 1,
                            "relation_id": 7,
                            "dst_entity_id": 2,
                        }
                    ],
                    "trajectory_text": "1 --7--> 2",
                }
            ],
        }
    ]
    assert prompt_record["trajectories"] == [
        {
            "path_rank": 1,
            "log_prob": -0.5108256,
            "prob": 0.6,
            "cumulative_mass": 0.6,
            "terminal_entity_id": 2,
            "start_entity_id": 1,
            "answer_rank": 1,
            "support_rank": 1,
            "conditional_prob": 1.0,
            "conditional_cumulative_mass": 1.0,
            "edges": [
                {
                    "edge_id": 0,
                    "src_entity_id": 1,
                    "relation_id": 7,
                    "dst_entity_id": 2,
                }
            ],
            "trajectory_text": "1 --7--> 2",
        }
    ]
    assert label_record == {
        "sample_id": "s1",
        "question": "Where was X born?",
        "start_entity_ids": [1],
        "answer_entity_ids": [2],
        "answer_texts": [],
        "a_entity_in_graph": True,
    }


def test_artifact_writer_rejects_existing_outputs_when_overwrite_false(
    tmp_path,
) -> None:
    result = SupportWindowResult(
        sample_id="s1",
        dataset_scope="full",
        mass_threshold=0.9,
        window_size=0,
        covered_mass=0.0,
        residual_mass=1.0,
        gold_answer_mass=0.0,
        covered_gold_answer_mass=0.0,
        missed_gold_answer_mass=0.0,
        unique_answer_count=0,
        unique_path_count=0,
        gold_answer_entity_ids=[],
        start_entity_ids=[],
        trajectories=[],
        answer_posterior=[],
    )
    label = SupportWindowLabelRecord(
        sample_id="s1",
        question="Where was X born?",
        start_entity_ids=[],
        answer_entity_ids=[],
        a_entity_in_graph=False,
    )
    (tmp_path / "test.jsonl").write_text("existing\n", encoding="utf-8")

    writer = SupportWindowArtifactWriter(
        output_dir=tmp_path,
        split="test",
        overwrite=False,
    )

    with pytest.raises(FileExistsError, match="Artifact already exists"):
        writer.write(results=[result], labels=[label])


def test_artifact_writer_uses_backend_neutral_metadata_defaults(tmp_path) -> None:
    result = SupportWindowResult(
        sample_id="s1",
        dataset_scope="full",
        mass_threshold=0.9,
        window_size=0,
        covered_mass=0.0,
        residual_mass=1.0,
        gold_answer_mass=0.0,
        covered_gold_answer_mass=0.0,
        missed_gold_answer_mass=0.0,
        unique_answer_count=0,
        unique_path_count=0,
        gold_answer_entity_ids=[],
        start_entity_ids=[],
        trajectories=[],
        answer_posterior=[],
    )
    writer = SupportWindowArtifactWriter(output_dir=tmp_path, split="test")

    prompt_record = writer._build_prompt_record(result, label=None)

    assert prompt_record["inference_mode"] == UNSPECIFIED_INFERENCE_MODE
    assert prompt_record["answer_mass_reference"] == UNSPECIFIED_MASS_REFERENCE
    assert prompt_record["support_mass_reference"] == UNSPECIFIED_MASS_REFERENCE


def test_artifact_writer_streams_from_jsonl_inputs(tmp_path) -> None:
    result = SupportWindowResult(
        sample_id="s1",
        dataset_scope="full",
        mass_threshold=0.9,
        window_size=1,
        covered_mass=0.6,
        residual_mass=0.4,
        gold_answer_mass=0.6,
        covered_gold_answer_mass=0.6,
        missed_gold_answer_mass=0.0,
        unique_answer_count=1,
        unique_path_count=1,
        gold_answer_entity_ids=[2],
        start_entity_ids=[1],
        trajectories=[],
        answer_posterior=[],
    )
    label = SupportWindowLabelRecord(
        sample_id="s1",
        question="Where was X born?",
        start_entity_ids=[1],
        answer_entity_ids=[2],
        a_entity_in_graph=True,
    )
    raw_results_path = tmp_path / "raw.results.jsonl"
    raw_labels_path = tmp_path / "raw.labels.jsonl"
    raw_results_path.write_text(json.dumps(asdict(result)) + "\n", encoding="utf-8")
    raw_labels_path.write_text(json.dumps(asdict(label)) + "\n", encoding="utf-8")

    writer = SupportWindowArtifactWriter(output_dir=tmp_path / "out", split="test")
    paths = writer.write_from_jsonl(
        results_path=raw_results_path,
        labels_path=raw_labels_path,
    )

    prompt_record = json.loads(paths["prompt_path"].read_text(encoding="utf-8").strip())
    label_record = json.loads(paths["labels_path"].read_text(encoding="utf-8").strip())
    assert prompt_record["sample_id"] == "s1"
    assert prompt_record["question"] == "Where was X born?"
    assert label_record["answer_entity_ids"] == [2]
