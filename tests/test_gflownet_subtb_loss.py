from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

pytest.importorskip("hydra")

from src.models.gflownet_module import GFlowNetModule


def test_subtb_loss_zero_for_consistent_single_step_trajectory() -> None:
    # One graph, one action (stop at step0), terminal logF(s1)=logR=0.
    # TB constraint: logF(s0) + logP(stop|s0) = logF(s1).
    log_p_stop = torch.tensor([[-1.6094379]], dtype=torch.float32)  # log(0.2)
    log_flow_states = torch.tensor([[1.6094379, 0.0]], dtype=torch.float32)  # -log(0.2), logR
    log_pb_steps = torch.zeros_like(log_p_stop)
    edge_lengths = torch.tensor([0], dtype=torch.long)

    loss = GFlowNetModule._compute_subtb_loss(
        None,
        log_flow_states=log_flow_states,
        log_pf_steps=log_p_stop,
        log_pb_steps=log_pb_steps,
        edge_lengths=edge_lengths,
    )
    assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_mask_subtb_loss_by_path_exists_uses_only_valid_graphs() -> None:
    subtb_per_graph = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    path_exists = torch.tensor([True, False, True, False], dtype=torch.bool)

    masked = GFlowNetModule._mask_subtb_loss_by_path_exists(subtb_per_graph=subtb_per_graph, path_exists=path_exists)
    assert masked.shape == torch.Size([])
    assert_close(masked, torch.tensor(2.0), atol=1e-6, rtol=0.0)


def test_mask_subtb_loss_by_path_exists_all_missing_returns_zero_scalar() -> None:
    subtb_per_graph = torch.tensor([1.0, 2.0], dtype=torch.float32)
    path_exists = torch.tensor([False, False], dtype=torch.bool)

    masked = GFlowNetModule._mask_subtb_loss_by_path_exists(subtb_per_graph=subtb_per_graph, path_exists=path_exists)
    assert masked.shape == torch.Size([])
    assert_close(masked, torch.tensor(0.0), atol=1e-6, rtol=0.0)
