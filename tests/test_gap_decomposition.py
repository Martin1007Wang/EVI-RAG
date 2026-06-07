from __future__ import annotations

import math

import torch

from scripts.diagnose_weaver_gap_decomposition import (
    best_edge_masks_by_budget,
    edge_ids_from_mask,
    greedy_edge_masks_by_budget,
    pearson,
    proxy_reward_from_recall,
)
from scripts.analyze_budget_recall_oracle import PathCandidate


def test_edge_ids_from_mask_roundtrip() -> None:
    mask = (1 << 0) | (1 << 3) | (1 << 9)
    assert edge_ids_from_mask(mask) == [0, 3, 9]


def test_best_edge_masks_by_budget_prefers_more_coverage_then_fewer_edges() -> None:
    candidates = [
        PathCandidate(edge_ids=frozenset({0}), edge_bits=(1 << 0), covered_target_bits=(1 << 0), target_pos=0),
        PathCandidate(edge_ids=frozenset({1}), edge_bits=(1 << 1), covered_target_bits=(1 << 1), target_pos=1),
        PathCandidate(edge_ids=frozenset({2, 3}), edge_bits=(1 << 2) | (1 << 3), covered_target_bits=(1 << 0) | (1 << 1), target_pos=0),
    ]
    best = best_edge_masks_by_budget(
        candidates=candidates,
        initial_target_bits=0,
        target_count=2,
        budgets=[0, 1, 2],
    )
    assert best[0] == 0
    assert edge_ids_from_mask(best[1]) in ([0], [1])
    assert edge_ids_from_mask(best[2]) == [2, 3]


def test_greedy_edge_masks_by_budget_is_budget_respecting() -> None:
    candidates = [
        PathCandidate(edge_ids=frozenset({0, 1}), edge_bits=(1 << 0) | (1 << 1), covered_target_bits=(1 << 0), target_pos=0),
        PathCandidate(edge_ids=frozenset({2}), edge_bits=(1 << 2), covered_target_bits=(1 << 1), target_pos=1),
    ]
    best = greedy_edge_masks_by_budget(
        candidates=candidates,
        initial_target_bits=0,
        target_count=2,
        budgets=[1, 2],
    )
    assert len(edge_ids_from_mask(best[1])) == 1
    assert len(edge_ids_from_mask(best[2])) <= 2


def test_proxy_reward_matches_expected_formula() -> None:
    value = proxy_reward_from_recall(recall=0.5, edge_count=4.0, budget=8)
    assert math.isclose(value, 0.5 - 0.1 * 4.0 / 8.0, rel_tol=0.0, abs_tol=1.0e-8)
    zero_hit = proxy_reward_from_recall(recall=0.0, edge_count=3.0, budget=8)
    assert zero_hit < 0.0


def test_pearson_returns_one_for_affine_relation() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([3.0, 5.0, 7.0, 9.0])
    assert math.isclose(pearson(x, y), 1.0, rel_tol=0.0, abs_tol=1.0e-6)
