from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("hydra")

from src.callbacks.gflownet_rollout_writer import GFlowNetRolloutWriter


def test_stream_rollouts_writes_jsonl(tmp_path: Path) -> None:
    writer = GFlowNetRolloutWriter(output_dir=tmp_path, split="predict", textualize=False)
    records = [
        {
            "sample_id": "s1",
            "question": "q1",
            "rollouts": [
                {
                    "rollout_index": 0,
                    "log_pf": 0.0,
                    "log_reward": 0.0,
                    "edges": [
                        {"head_entity_id": 10, "relation_id": 0, "tail_entity_id": 20}
                    ],
                }
            ],
        },
        {
            "sample_id": "s2",
            "question": "q2",
            "rollouts": [],
        },
    ]

    trainer = SimpleNamespace(global_rank=0)
    writer.on_predict_start(trainer, None)
    writer.write_on_batch_end(trainer, None, records, None, None, 0, 0)
    out = tmp_path / "predict.jsonl"
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(records)
    manifest = tmp_path / "predict.manifest.json"
    assert manifest.exists()
