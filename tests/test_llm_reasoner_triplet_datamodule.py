import pickle
from pathlib import Path

import lmdb
import pandas as pd
import pytest
import torch

pytest.importorskip("lightning")

from src.data.llm_reasoner_triplet_datamodule import LLMReasonerTripletDataModule


def _write_lmdb(path: Path, entries: dict[str, dict]) -> None:
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            for key, value in entries.items():
                txn.put(key.encode("utf-8"), pickle.dumps(value))
    finally:
        env.close()


def test_llm_reasoner_triplet_datamodule_windows(tmp_path: Path) -> None:
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

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True)
    _write_lmdb(
        embeddings_dir / "test.lmdb",
        {
            "q1": {
                # GT triple is the *second* edge in the g_agent record, so K=1 misses, K=5 hits.
                "gt_paths_triples": [[(3, 0, 42)]],
            }
        },
    )

    dm = LLMReasonerTripletDataModule(
        dataset="mock",
        split="test",
        score_dict_path=str(score_dict_path),
        questions_path=str(questions_path),
        entity_vocab_path=str(missing_vocab),
        relation_vocab_path=str(missing_vocab),
        triplet_limits=[1, 5],
        retrieval_lmdb_dir=str(embeddings_dir),
        token_budget=3,
        token_budget_encoding=None,
        num_workers=0,
        prompt_tag="triplets",
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
    assert by_k[1]["hit_set"] is False
    assert by_k[1]["hit_vis"] is False
    assert by_k[1]["evidence_truncated"] is False
    assert by_k[1]["token_budget"] == 3
    assert len(by_k[1]["visible_edge_ids"]) == 1

    assert by_k[5]["hit_set"] is True
    assert by_k[5]["hit_vis"] is False
    assert by_k[5]["evidence_truncated"] is True
    assert by_k[5]["token_budget"] == 3
    assert len(by_k[5]["retrieved_edge_ids"]) == 2
    assert len(by_k[5]["visible_edge_ids"]) == 1
