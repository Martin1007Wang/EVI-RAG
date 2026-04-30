from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.eval.llm.artifacts import write_llm_metrics_artifacts


def test_write_llm_metrics_artifacts_writes_json_and_jsonl(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "artifacts"
    metrics_log_dir = tmp_path / "logs"

    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "s1",
                        "question": "Where is the Eiffel Tower?",
                        "answer_texts": ["Paris"],
                        "gold_answer_in_graph": True,
                        "trajectories": [{"edges": [{"src_text": "Eiffel Tower", "dst_text": "Paris"}]}],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps({"sample_id": "s1", "answer": "ans: Paris"}) + "\n",
        encoding="utf-8",
    )

    metrics_path, metrics = write_llm_metrics_artifacts(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
        split="dev",
        provider="mock",
        top_k=1,
        answer_key="answer",
        answer_separator=" | ",
        metrics_log_dir=metrics_log_dir,
        dataset_name="demo",
        dataset_scope="tiny",
    )

    assert metrics_path.exists()
    assert metrics["llm/subgraphrag/full/hit@1"] == 100.0
    assert metrics["llm/subgraphrag/sub/hit@1"] == 100.0

    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted["llm/subgraphrag/full/hit@1"] == 100.0

    jsonl_payload = json.loads((metrics_log_dir / "llm.jsonl").read_text(encoding="utf-8").strip())
    assert jsonl_payload["stage"] == "llm"
    assert jsonl_payload["metadata"]["dataset_name"] == "demo"
    assert jsonl_payload["metrics"]["llm/subgraphrag/full/hit@1"] == 100.0
