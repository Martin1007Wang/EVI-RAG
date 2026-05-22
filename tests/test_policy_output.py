from __future__ import annotations

import torch

from src.weaver.policy.output import PolicyOutput, STOP_EDGE_ID
from src.weaver.state import Frontier


def test_empty_frontier_row_is_forced_stop_in_policy_semantics() -> None:
    out = PolicyOutput(
        stop_logit=torch.tensor([-3.0, 0.7], dtype=torch.float32),
        log_flow=torch.zeros(2, dtype=torch.float32),
        edge_logit=torch.tensor([1.5], dtype=torch.float32),
        frontier=Frontier(
            row_ids=torch.tensor([1], dtype=torch.long),
            edge_ids=torch.tensor([4], dtype=torch.long),
        ),
        num_rows=2,
        num_edges=8,
    )

    stop_log_prob = out.stop_log_prob
    continue_log_prob = out.continue_log_prob

    assert stop_log_prob[0].item() == 0.0
    assert continue_log_prob[0].item() == float("-inf")
    assert out.gather_log_prob(
        row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([STOP_EDGE_ID], dtype=torch.long),
    ).item() == 0.0

    sampled = out.sample(rows=torch.tensor([0, 1], dtype=torch.long))
    assert sampled[0].item() == STOP_EDGE_ID
    assert sampled[1].item() in {STOP_EDGE_ID, 4}
