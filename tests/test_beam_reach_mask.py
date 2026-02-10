from __future__ import annotations

import types

import torch

from src.models.dual_flow_module import DualFlowModule


def _tiny_dual_flow_module() -> DualFlowModule:
    return DualFlowModule(
        hidden_dim=8,
        emb_dim=8,
        max_steps=2,
        training_cfg={
            "db_cfg": {"sampling_temperature_start": 1.0, "sampling_temperature_end": 1.0},
        },
        evaluation_cfg={},
        runtime_cfg={},
    )


def test_compute_beam_reach_mask_accepts_3d_paths() -> None:
    module = _tiny_dual_flow_module()
    prepared = types.SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    beam_paths = torch.tensor([[[0, 1], [0, -1]]], dtype=torch.long)  # [graphs=1, beams=2, steps=2]
    beam_lengths = torch.tensor([[2, 1]], dtype=torch.long)
    beam_nodes = torch.tensor([[2, 1]], dtype=torch.long)
    node_is_target = torch.tensor([False, False, True], dtype=torch.bool)

    reach = module._compute_beam_reach_mask(
        prepared_fwd=prepared,
        beam_paths=beam_paths,
        beam_lengths=beam_lengths,
        beam_nodes=beam_nodes,
        node_is_target=node_is_target,
    )

    assert reach.tolist() == [[True, False]]

