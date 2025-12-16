import json
from pathlib import Path

import torch

from src.data.llm_reasoner_truth_datamodule import LLMReasonerTruthDataModule
from src.models.llm_reasoner_truth_module import LLMReasonerTruthModule


def test_llm_reasoner_truth_oracle_metrics(tmp_path: Path) -> None:
    fixture_path = Path(__file__).resolve().parent / "data" / "retriever_eval_tiny.json"
    payload = json.loads(fixture_path.read_text())

    cache_path = tmp_path / "test_retriever_eval.pt"
    torch.save(payload, cache_path)

    dm = LLMReasonerTruthDataModule(
        dataset="mock",
        split="test",
        retriever_eval_path=str(cache_path),
        k_values=[1, 5],
        num_workers=0,
        prompt_tag="truth",
    )
    dm.setup()

    model = LLMReasonerTruthModule(
        dataset="mock",
        split="test",
        output_dir=str(tmp_path),
        k_values=[1, 5],
        prompt_tag="truth",
    )
    rows = []
    for sample in dm.data:
        rows.extend(model.predict_step([sample], batch_idx=0))
    metrics = model._aggregate_metrics(rows, k_values=model.k_values)

    assert metrics["oracle/total"] == 2.0
    assert metrics["oracle/answer_hit@1"] == 0.5
    assert metrics["oracle/answer_hit@5"] == 1.0
    assert metrics["oracle/answer_recall@1"] == 0.25
    assert metrics["oracle/answer_recall@5"] == 0.75
