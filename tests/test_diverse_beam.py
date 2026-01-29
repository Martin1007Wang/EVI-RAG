import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import torch

from src.models.dual_flow_module import DualFlowModule


def _module():
    return DualFlowModule.__new__(DualFlowModule)


def test_diverse_select_positions_hard_tail():
    module = _module()
    scores = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
    keys = torch.tensor([[1, 1, 2, 3]])
    counts = torch.tensor([4])
    sel_pos = module._diverse_select_positions(
        scores=scores,
        keys=keys,
        counts=counts,
        beam_size=3,
        groups=3,
        penalty="hard",
        penalty_lambda=1.0,
        neg_inf=float("-inf"),
    )
    assert sel_pos.shape == (1, 3)
    assert sel_pos[0].tolist() == [0, 2, 3]


def test_select_beam_positions_respects_counts():
    module = _module()
    scores = torch.tensor([[5.0, 1.0]])
    keys = torch.tensor([[0, 1]])
    counts = torch.tensor([1])
    sel_pos, sel_scores = module._select_beam_positions(
        scores=scores,
        keys=keys,
        counts=counts,
        beam_size=3,
        diverse_cfg={"enabled": False, "groups": 1, "penalty": "hard", "lambda": 1.0},
        neg_inf=float("-inf"),
    )
    assert sel_pos.shape == (1, 3)
    assert sel_scores.shape == (1, 3)
    assert sel_pos[0, 0].item() == 0
    assert sel_pos[0, 1].item() == -1
    assert sel_pos[0, 2].item() == -1
