from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mask_debug import collect_mask_debug_summaries


def _build_batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_graphs=2,
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 3, 4], [1, 2, 4, 5]],
            dtype=torch.long,
        ),
        is_anchor_mask=torch.tensor([True, False, False, True, False, False]),
        is_target_mask=torch.tensor([False, False, True, False, False, True]),
        positive_edge_mask=torch.tensor([False, True, True, False]),
    )


def test_collect_mask_debug_summaries_reports_positive_edge_target_alignment() -> None:
    batch = _build_batch()

    summaries = collect_mask_debug_summaries(batch)

    assert len(summaries) == 2

    assert summaries[0]["graph_idx"] == 0
    assert summaries[0]["anchor_nodes"] == 1
    assert summaries[0]["target_nodes"] == 1
    assert summaries[0]["positive_edges"] == 1
    assert summaries[0]["positive_edge_target_dst_ratio"] == 1.0
    assert summaries[0]["positive_edge_target_dst_hits"] == 1
    assert summaries[0]["non_target_positive_dst_local_ids_sample"] == ()

    assert summaries[1]["graph_idx"] == 1
    assert summaries[1]["anchor_nodes"] == 1
    assert summaries[1]["target_nodes"] == 1
    assert summaries[1]["positive_edges"] == 1
    assert summaries[1]["positive_edge_target_dst_ratio"] == 0.0
    assert summaries[1]["positive_edge_target_dst_hits"] == 0
    assert summaries[1]["non_target_positive_dst_local_ids_sample"] == (1,)
