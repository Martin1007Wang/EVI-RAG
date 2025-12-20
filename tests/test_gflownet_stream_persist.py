from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hydra")

from src.models.gflownet_module import GFlowNetModule


def test_stream_rollouts_writes_jsonl(tmp_path: Path) -> None:
    # Construct a minimal module stub (bypass heavy __init__) to exercise stream writer.
    module = GFlowNetModule.__new__(GFlowNetModule)
    module.eval_persist_cfg = {"output_dir": str(tmp_path), "textualize": False}
    module._stream_processor = None
    module._stream_output_dir = None
    records = [
        {
            "sample_id": "s1",
            "question": "q1",
            "answer_entity_ids": [1],
            "rollouts": [
                {
                    "rollout_index": 0,
                    "success": True,
                    "log_pf": 0.0,
                    "log_reward": 0.0,
                    "edges": [
                        {"head_entity_id": 10, "relation_id": 0, "tail_entity_id": 20, "edge_score": 0.1, "edge_label": 1.0}
                    ],
                }
            ],
        },
        {
            "sample_id": "s2",
            "question": "q2",
            "answer_entity_ids": [2],
            "rollouts": [],
        },
    ]

    module._stream_rollouts(records, split="predict")
    out = tmp_path / "predict_gflownet_eval.jsonl"
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(records)
