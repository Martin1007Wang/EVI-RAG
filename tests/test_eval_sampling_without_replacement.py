from __future__ import annotations

import torch

from src.models.rollout import ActionSampler


def test_eval_sampling_without_replacement_diversifies_same_state_group() -> None:
    sampler = ActionSampler()
    neg_inf = torch.finfo(torch.float32).min
    policy_output = {
        "edge_logits": torch.tensor(
            [8.0, 7.0, 6.0, 8.0, 7.0, 6.0, 8.0, 7.0, 6.0],
            dtype=torch.float32,
        ),
        "out_degrees": torch.tensor([[3, 3, 3]], dtype=torch.long),
        "stop_logits": torch.tensor([[neg_inf, neg_inf, neg_inf]], dtype=torch.float32),
        "edge_ids": torch.tensor(
            [10, 11, 12, 20, 21, 22, 30, 31, 32], dtype=torch.long
        ),
        "target_nodes": torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=torch.long),
    }

    torch.manual_seed(0)
    out = sampler(
        policy_output,
        is_training=False,
        deterministic=True,
        sampling_mode="gumbel",
        sampling_temperature=1.0,
        eval_sampling_temperature=0.5,
        eval_sample_without_replacement=True,
        agent_graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        source_nodes=torch.tensor([5, 5, 5], dtype=torch.long),
        active_mask=torch.tensor([True, True, True]),
        num_nodes_total=16,
    )
    chosen = out["chosen_edge_ids"]
    assert chosen.unique().numel() == 3
    assert not bool(out["is_stop"].any().item())
