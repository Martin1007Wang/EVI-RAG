from __future__ import annotations

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import State
from src.weaver.utility import EvidenceUtilityReward

from tests.test_replay_alignment import replay_batch


def test_evidence_reward_uses_shortest_path_support_prior() -> None:
    batch = replay_batch()
    context = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=context)
    reward = EvidenceUtilityReward(
        answer_weight=1.0,
        support_weight=1.0,
        connect_weight=0.0,
        edge_cost=0.0,
        unsupported_edge_cost=1.0,
        fail_cost=0.0,
    )

    state = State.from_selected_edges(
        graph=context,
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        selected_edge_mask=torch.tensor(
            [
                [True, False, True, False, False, False, False, False],
                [False, False, False, True, False, False, False, False],
            ],
            dtype=torch.bool,
        ),
        expand_budget=3,
    )

    out = reward(state=state, graph_context=context, target_context=target)

    assert out.metrics["reward_support_coverage_mean"].item() > 0.0
    assert out.log_reward[0].item() > out.log_reward[1].item()
