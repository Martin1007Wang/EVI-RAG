from __future__ import annotations

import pytest
import torch

from src.models.dual_flow_module import DualFlowModule
from src.utils.graph import build_edge_inverse_map


def test_edge_inverse_map_and_action_mapping() -> None:
    # Two-node graph with inverse relations.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_relations = torch.tensor([0, 1], dtype=torch.long)
    inverse_map = torch.tensor([1, 0], dtype=torch.long)
    edge_inverse_map = build_edge_inverse_map(
        edge_index=edge_index,
        edge_relations=edge_relations,
        num_nodes_total=2,
        inverse_map=inverse_map,
        num_relations=2,
    )
    assert edge_inverse_map.tolist() == [1, 0]

    actions = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mapped = DualFlowModule._map_inverse_actions(actions=actions, edge_inverse_map=edge_inverse_map)
    assert mapped.tolist() == [[1, 0], [0, 1]]

    bad_inverse_map = torch.tensor([-1, -1], dtype=torch.long)
    with pytest.raises(AssertionError):
        DualFlowModule._map_inverse_actions(actions=actions, edge_inverse_map=bad_inverse_map)


def test_reverse_actions_by_length() -> None:
    actions = torch.tensor([[5, 6, -1], [7, 8, 9]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)
    reversed_actions = DualFlowModule._reverse_actions_by_length(actions=actions, lengths=lengths)
    assert reversed_actions.tolist() == [[6, 5, -1], [9, 8, 7]]
