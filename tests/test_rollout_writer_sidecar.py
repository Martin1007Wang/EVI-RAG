from __future__ import annotations

from src.callbacks.dual_flow_rollout_artifact_writer import DualFlowRolloutArtifactWriter


def test_rollout_writer_splits_prompt_and_label_records(tmp_path) -> None:
    writer = DualFlowRolloutArtifactWriter(output_dir=tmp_path)
    records = [
        {
            "sample_id": "s1",
            "question_text": "Where was X born?",
            "rollouts": [{"rollout_index": 0, "score": 1.0, "edges": []}],
            "start_entity_ids": [1],
            "answer_entity_ids": [2],
            "a_entity_in_graph": True,
            "answer_texts": ["Paris"],
            "answer_text": "Paris",
        }
    ]

    prompt_records, label_records = writer._split_records(records)

    assert prompt_records == [
        {
            "sample_id": "s1",
            "question": "Where was X born?",
            "rollouts": [{"rollout_index": 0, "score": 1.0, "edges": []}],
        }
    ]
    assert label_records == [
        {
            "sample_id": "s1",
            "question": "Where was X born?",
            "question_text": "Where was X born?",
            "start_entity_ids": [1],
            "answer_entity_ids": [2],
            "answer_texts": ["Paris"],
            "answer_text": "Paris",
            "a_entity_in_graph": True,
        }
    ]

