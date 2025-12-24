import json
from pathlib import Path

import torch

from src.data.reasoner_oracle_datamodule import ReasonerOracleDataModule
from src.models.reasoner_module import ReasonerModule


def test_reasoner_oracle_metrics(tmp_path: Path) -> None:
    fixture_path = Path(__file__).resolve().parent / "data" / "eval_retriever_tiny.json"
    payload = json.loads(fixture_path.read_text())

    cache_path = tmp_path / "test_eval_retriever.pt"
    torch.save(payload, cache_path)
    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "artifact": "eval_retriever",
                "schema_version": 1,
                "file": cache_path.name,
                "created_at": "2025-01-01T00:00:00Z",
                "producer": "retriever_module",
            }
        ),
        encoding="utf-8",
    )

    dm = ReasonerOracleDataModule(
        dataset="mock",
        split="test",
        eval_retriever_path=str(cache_path),
        k_values=[1, 5],
        num_workers=0,
        prompt_tag="oracle",
    )
    dm.setup()

    model = ReasonerModule(
        mode="oracle",
        dataset="mock",
        split="test",
        output_dir=str(tmp_path),
        k_values=[1, 5],
        prompt_tag="oracle",
    )
    rows = []
    for sample in dm.data:
        rows.extend(model.predict_step([sample], batch_idx=0))
    metrics = model._aggregate_oracle_metrics(rows, k_values=model.k_values)

    assert metrics["oracle/total"] == 2.0
    assert metrics["oracle/answer_hit@1"] == 0.5
    assert metrics["oracle/answer_hit@5"] == 1.0
    assert metrics["oracle/answer_recall@1"] == 0.25
    assert metrics["oracle/answer_recall@5"] == 0.75
