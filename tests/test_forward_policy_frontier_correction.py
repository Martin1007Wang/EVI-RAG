import torch

from src.weaver.policy.forward import ForwardPolicy
from src.weaver.state import ActionSpace


def test_frontier_size_correction_beta_one_uses_log_mean_exp() -> None:
    action_space = ActionSpace(
        num_states=2,
        expand_state_ids=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        expand_edge_ids=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        expand_ptr=torch.tensor([0, 2, 5], dtype=torch.long),
    )
    raw = torch.full((5,), 2.0)

    edge_flow = ForwardPolicy.size_normalized_edge_flow(
        edge_raw_score=raw,
        action_space=action_space,
        frontier_size_correction=1.0,
    )

    assert torch.allclose(edge_flow[:2], torch.full((2,), 2.0 - torch.log(torch.tensor(2.0))))
    assert torch.allclose(edge_flow[2:], torch.full((3,), 2.0 - torch.log(torch.tensor(3.0))))


def test_frontier_size_correction_beta_zero_keeps_raw_edge_scores() -> None:
    action_space = ActionSpace(
        num_states=2,
        expand_state_ids=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        expand_edge_ids=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        expand_ptr=torch.tensor([0, 2, 5], dtype=torch.long),
    )
    raw = torch.tensor([1.0, 2.0, -1.0, 0.0, 3.0])

    edge_flow = ForwardPolicy.size_normalized_edge_flow(
        edge_raw_score=raw,
        action_space=action_space,
        frontier_size_correction=0.0,
    )

    assert torch.equal(edge_flow, raw)
