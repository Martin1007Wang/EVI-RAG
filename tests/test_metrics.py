import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

_metrics_path = Path(__file__).resolve().parent.parent / "src" / "utils" / "metrics.py"
_metrics_spec = importlib.util.spec_from_file_location("metrics", _metrics_path)
if _metrics_spec is None or _metrics_spec.loader is None:
    raise RuntimeError(f"Failed to load metrics module from {_metrics_path}")
metrics = importlib.util.module_from_spec(_metrics_spec)
sys.modules["metrics"] = metrics
_metrics_spec.loader.exec_module(metrics)  # type: ignore[misc]
extract_answer_entity_ids = metrics.extract_answer_entity_ids


def test_extract_answer_entity_ids_uses_slice_dict() -> None:
    data_list = [
        Data(
            num_nodes=3,
            node_global_ids=torch.tensor([10, 11, 12], dtype=torch.long),
            a_local_indices=torch.tensor([1], dtype=torch.long),
        ),
        Data(
            num_nodes=2,
            node_global_ids=torch.tensor([20, 21], dtype=torch.long),
            a_local_indices=torch.tensor([0, 1], dtype=torch.long),
        ),
    ]

    batch = Batch.from_data_list(data_list)

    ids0 = extract_answer_entity_ids(batch, sample_idx=0, node_ptr=batch.ptr, node_ids=batch.node_global_ids)
    ids1 = extract_answer_entity_ids(batch, sample_idx=1, node_ptr=batch.ptr, node_ids=batch.node_global_ids)

    assert torch.equal(ids0, torch.tensor([11], dtype=torch.long))
    assert torch.equal(ids1, torch.tensor([20, 21], dtype=torch.long))
