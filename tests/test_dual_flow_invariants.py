from __future__ import annotations

import torch

from src.models.dual_flow_module import DualFlowModule


def test_reverse_actions_by_length() -> None:
    actions = torch.tensor([[5, 6, -1], [7, 8, 9]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)
    reversed_actions = DualFlowModule._reverse_actions_by_length(actions=actions, lengths=lengths)
    assert reversed_actions.tolist() == [[6, 5, -1], [9, 8, 7]]
