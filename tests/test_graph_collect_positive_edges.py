from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_steps.graph_collect import (
    _build_positive_edge_mask,
    _prepare_graph_edges,
)


def test_build_positive_edge_mask_keeps_all_shortest_path_edges() -> None:
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    positive_edge_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    positive_edge_mask = _build_positive_edge_mask(
        edge_index=edge_index,
        positive_edge_ids=positive_edge_ids,
    )

    assert positive_edge_mask.dtype == torch.bool
    assert positive_edge_mask.tolist() == [False, True, True, True]


def test_prepare_graph_edges_keeps_only_cleaned_input_edges() -> None:
    edges = _prepare_graph_edges(
        [("a", "r", "b"), ("b", "inverse::r", "a"), ("c", "self", "c")],
        remove_self_loops=True,
        dedup_edges=True,
    )

    assert edges == [("a", "r", "b"), ("b", "inverse::r", "a")]
