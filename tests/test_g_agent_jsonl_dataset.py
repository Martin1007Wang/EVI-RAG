from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.g_agent_dataset import GAgentPyGDataset, _builder_sample_to_record, GAgentSample


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_g_agent_jsonl_dataset_stream_read(tmp_path: Path) -> None:
    # Build a minimal GAgentSample then export to jsonl (streaming-friendly).
    sample = GAgentSample(
        sample_id="s1",
        question="q",
        question_emb=torch.tensor([1.0, 2.0]),
        node_entity_ids=torch.tensor([1, 2]),
        node_embedding_ids=torch.tensor([1, 2]),
        edge_head_locals=torch.tensor([0]),
        edge_tail_locals=torch.tensor([1]),
        edge_relations=torch.tensor([0]),
        edge_scores=torch.tensor([0.5]),
        edge_labels=torch.tensor([1.0]),
        top_edge_mask=torch.tensor([True]),
        start_entity_ids=torch.tensor([1]),
        answer_entity_ids=torch.tensor([2]),
        gt_path_edge_local_ids=torch.tensor([0]),
        gt_path_node_local_ids=torch.tensor([0, 1]),
        start_node_locals=torch.tensor([0]),
        answer_node_locals=torch.tensor([1]),
        gt_path_exists=True,
        is_answer_reachable=True,
    )
    record = _builder_sample_to_record(sample)
    jsonl_path = tmp_path / "cache.jsonl"
    _write_jsonl(jsonl_path, record)

    ds = GAgentPyGDataset(jsonl_path, drop_unreachable=False, prefer_jsonl=True, convert_pt_to_jsonl=False)
    assert len(ds) == 1
    data: Data = ds[0]
    assert data.edge_index.shape == (2, 1)
    assert torch.equal(data.answer_entity_ids, torch.tensor([2]))
    assert torch.equal(data.start_node_locals, torch.tensor([0]))
