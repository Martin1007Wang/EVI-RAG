from pathlib import Path

import pandas as pd
import pytest
import torch

pytest.importorskip("lightning")

from src.data.reasoner_triplet_datamodule import ReasonerTripletDataModule


def test_reasoner_triplet_datamodule_windows(tmp_path: Path) -> None:
    score_dict_path = tmp_path / "test_g_agent.pt"
    questions_path = tmp_path / "questions.parquet"
    missing_vocab = tmp_path / "missing_vocab.parquet"

    record = {
        "sample_id": "q1",
        "question": "mock question 1",
        "node_entity_ids": [42, 3],
        "edge_head_locals": [0, 1],
        "edge_tail_locals": [1, 0],
        "edge_relations": [0, 0],
        "edge_scores": [0.9, 0.8],
        "edge_labels": [1.0, 0.0],
        "top_edge_mask": [True, True],
        "answer_entity_ids": [42],
    }
    torch.save({"samples": [record]}, score_dict_path)

    df = pd.DataFrame(
        [
            {
                "question_uid": "q1",
                "question": "mock question 1",
                "answer_texts": ["Ans42"],
            }
        ]
    )
    df.to_parquet(questions_path, index=False)

    dm = ReasonerTripletDataModule(
        dataset="mock",
        split="test",
        score_dict_path=str(score_dict_path),
        questions_path=str(questions_path),
        entity_vocab_path=str(missing_vocab),
        relation_vocab_path=str(missing_vocab),
        triplet_limits=[1, 5],
        token_budget=3,
        token_budget_encoding=None,
        num_workers=0,
        prompt_tag="triplet",
    )
    dm.setup()
    assert len(dm.data) == 2
    ks = sorted(int(s["window_k"]) for s in dm.data)
    assert ks == [1, 5]
    for sample in dm.data:
        k = int(sample["window_k"])
        assert len(sample["triplets"]) <= k
        assert "retrieved_edge_ids" in sample and len(sample["retrieved_edge_ids"]) == len(sample["triplets"])
        assert "gt_path_edge_local_ids" in sample
        assert sample.get("prompt_token_count") is not None
        assert sample.get("evidence_token_count") is not None

    by_k = {int(s["window_k"]): s for s in dm.data}
    assert by_k[1]["hit_set"] is True
    assert by_k[1]["hit_vis"] is True
    assert by_k[1]["evidence_truncated"] is False
    assert by_k[1]["token_budget"] == 3
    assert len(by_k[1]["visible_edge_ids"]) == 1

    assert by_k[5]["hit_set"] is False
    assert by_k[5]["hit_vis"] is True
    assert by_k[5]["evidence_truncated"] is True
    assert by_k[5]["token_budget"] == 3
    assert len(by_k[5]["retrieved_edge_ids"]) == 2
    assert len(by_k[5]["visible_edge_ids"]) == 1
