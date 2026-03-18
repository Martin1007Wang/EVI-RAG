from __future__ import annotations

import json

import pytest

from src.llm.metrics import (
    _subgraphrag_get_pred_lines,
    _subgraphrag_no_answer,
    compute_llm_metrics,
    write_llm_metrics_artifacts,
)


def test_no_answer_false_when_valid_answer_exists() -> None:
    prediction = "ans: not available\nans: Paris"
    pred_lines = _subgraphrag_get_pred_lines(prediction)
    assert pred_lines == ["ans: Paris"]
    assert _subgraphrag_no_answer(prediction, pred_lines) is False


def test_no_answer_true_when_only_no_answer_markers() -> None:
    prediction = "ans: not available\nans: no information available"
    pred_lines = _subgraphrag_get_pred_lines(prediction)
    assert pred_lines == []
    assert _subgraphrag_no_answer(prediction, pred_lines) is True


def test_sub_hal_score_is_computed_from_sub_scope_samples(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "answer_texts": ["Paris"],
        "a_entity_in_graph": True,
        "trajectories": [
            {
                "edges": [
                    {"src_entity_id": 1, "dst_entity_id": 2},
                ]
            }
        ],
    }
    output_record = {"sample_id": "s1", "answer": "ans: Paris"}
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    metrics = compute_llm_metrics(
        input_path=input_path,
        output_path=output_path,
        split="test",
        provider="unit",
        top_k=1,
        answer_key="answer",
        answer_separator=" | ",
    )

    assert metrics["llm/subgraphrag/sub/total_cnt"] == 1
    assert metrics["llm/subgraphrag/sub/hal_score"] == pytest.approx(100.0)
    assert metrics["llm/subgraphrag/sub/stats/total_samples"] == 1.0
    assert metrics["llm/input/trajectory_count_mean"] == pytest.approx(1.0)
    assert metrics["llm/input/trajectory_count_min"] == 1
    assert metrics["llm/input/trajectory_count_max"] == 1


def test_metrics_use_structured_answer_field_not_raw_response(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "answer_texts": ["Paris"],
        "a_entity_in_graph": True,
        "trajectories": [],
    }
    output_record = {
        "sample_id": "s1",
        "answer": "Paris",
        "raw_response": "ans: London",
    }
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    metrics = compute_llm_metrics(
        input_path=input_path,
        output_path=output_path,
        split="test",
        provider="unit",
        top_k=1,
        answer_key="answer",
        answer_separator=" | ",
    )

    assert metrics["llm/subgraphrag/full/hit@1"] == pytest.approx(100.0)
    assert metrics["llm/subgraphrag/full/macro_f1"] == pytest.approx(100.0)


def test_metrics_can_read_gold_labels_from_sidecar(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    labels_path = tmp_path / "input.labels.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "trajectories": [],
    }
    labels_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "answer_texts": ["Paris"],
        "a_entity_in_graph": True,
    }
    output_record = {"sample_id": "s1", "answer": "Paris"}
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(labels_record) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    metrics = compute_llm_metrics(
        input_path=input_path,
        input_labels_path=labels_path,
        output_path=output_path,
        split="test",
        provider="unit",
        top_k=1,
        answer_key="answer",
        answer_separator=" | ",
    )

    assert metrics["llm/subgraphrag/full/hit@1"] == pytest.approx(100.0)
    assert metrics["llm/subgraphrag/sub/total_cnt"] == 1


def test_metrics_fail_fast_when_predicted_sample_has_no_gold_labels(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "trajectories": [],
    }
    output_record = {"sample_id": "s1", "answer": "Paris"}
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing or invalid gold labels"):
        compute_llm_metrics(
            input_path=input_path,
            output_path=output_path,
            split="test",
            provider="unit",
            top_k=1,
            answer_key="answer",
            answer_separator=" | ",
        )


def test_metrics_fail_fast_on_duplicate_label_sample_id(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    labels_path = tmp_path / "input.labels.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "trajectories": [],
    }
    labels_records = [
        {"sample_id": "s1", "answer_texts": ["Paris"], "a_entity_in_graph": True},
        {"sample_id": "s1", "answer_texts": ["London"], "a_entity_in_graph": True},
    ]
    output_record = {"sample_id": "s1", "answer": "Paris"}
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    labels_path.write_text(
        "\n".join(json.dumps(r) for r in labels_records) + "\n", encoding="utf-8"
    )
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate sample_id in label JSONL"):
        compute_llm_metrics(
            input_path=input_path,
            input_labels_path=labels_path,
            output_path=output_path,
            split="test",
            provider="unit",
            top_k=1,
            answer_key="answer",
            answer_separator=" | ",
        )


def test_metrics_artifacts_keep_json_and_jsonl_together(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    output_dir = tmp_path / "metrics"
    metrics_log_dir = tmp_path / "logs"
    input_record = {
        "sample_id": "s1",
        "question": "Where was X born?",
        "answer_texts": ["Paris"],
        "a_entity_in_graph": True,
        "trajectories": [],
    }
    output_record = {"sample_id": "s1", "answer": "Paris"}
    input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")

    metrics_path, metrics = write_llm_metrics_artifacts(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
        split="test",
        provider="unit",
        top_k=1,
        answer_key="answer",
        answer_separator=" | ",
        metrics_log_dir=metrics_log_dir,
        metrics_jsonl_name="llm_eval.jsonl",
        dataset_name="webqsp",
        dataset_scope="full",
    )

    assert metrics_path.exists()
    assert metrics["llm/subgraphrag/full/hit@1"] == pytest.approx(100.0)

    log_path = metrics_log_dir / "llm_eval.jsonl"
    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["stage"] == "llm"
    assert records[0]["metadata"] == {
        "dataset_name": "webqsp",
        "dataset_scope": "full",
        "split": "test",
        "provider": "unit",
        "top_k": 1,
        "input_path": str(input_path),
        "input_labels_path": "",
        "output_path": str(output_path),
    }
    assert records[0]["metrics"]["llm/subgraphrag/full/hit@1"] == pytest.approx(100.0)
