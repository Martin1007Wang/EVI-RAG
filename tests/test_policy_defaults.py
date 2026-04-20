from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.policy import Policy


def test_policy_defaults_to_directed_mode() -> None:
    policy = Policy(
        backbone_cfg={"embedding_dim": 8, "hidden_dim": 8},
        hidden_dim=8,
    )

    assert policy.undirected is False


def test_directed_policy_keeps_oracle_edge_candidates() -> None:
    policy = Policy(
        backbone_cfg={"embedding_dim": 8, "hidden_dim": 8},
        hidden_dim=8,
    )

    candidate_mask = policy._build_candidate_mask(
        edge_src=torch.tensor([2, 1], dtype=torch.long),
        edge_dst=torch.tensor([1, 0], dtype=torch.long),
        active_nodes=torch.tensor([False, False, True]),
        active_edges=torch.tensor([False, False]),
    )

    assert candidate_mask.tolist() == [True, False]
