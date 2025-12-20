from __future__ import annotations

import torch

from src.models.components.gflownet_state_encoder import TrajectoryStateEncoder


def test_state_encoder_is_order_sensitive() -> None:
    torch.manual_seed(0)
    encoder = TrajectoryStateEncoder(hidden_dim=4, dropout=0.0, use_start_state=False)
    edge_tokens = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    actions_seq = torch.tensor([[0, 1, 3], [1, 0, 3]], dtype=torch.long)
    stop_indices = torch.tensor([3, 3], dtype=torch.long)
    question_tokens = torch.zeros((2, 4), dtype=torch.float32)

    state_emb = encoder(
        actions_seq=actions_seq,
        edge_tokens=edge_tokens,
        stop_indices=stop_indices,
        question_tokens=question_tokens,
    )

    assert state_emb.shape == (2, 3, 4)
    assert not torch.allclose(state_emb[0], state_emb[1])
