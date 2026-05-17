from __future__ import annotations

import torch

from src.weaver.loss import ProbabilityDBLoss


def test_probability_db_loss_emits_logging_metric() -> None:
    loss_fn = ProbabilityDBLoss()

    output = loss_fn(
        parent_log_reward=torch.tensor([0.2, 0.4], dtype=torch.float32),
        child_log_reward=torch.tensor([0.3, 0.1], dtype=torch.float32),
        log_backward_prob=torch.tensor([0.0, -0.2], dtype=torch.float32),
        parent_stop_log_prob=torch.tensor([-0.2, -0.1], dtype=torch.float32),
        parent_continue_log_prob=torch.tensor([-1.7, -1.4], dtype=torch.float32),
        parent_edge_log_prob=torch.tensor([-0.6, -0.8], dtype=torch.float32, requires_grad=True),
        child_stop_log_prob=torch.tensor([-0.3, -0.4], dtype=torch.float32),
    )

    assert set(output.metrics) == {"db/residual_abs_mean"}
    assert output.loss.requires_grad
