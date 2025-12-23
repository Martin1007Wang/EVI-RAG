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

