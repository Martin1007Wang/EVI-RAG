from __future__ import annotations

import torch

from src.graph.oracle_replay import build_replay_bank


def test_build_replay_bank_outputs_frontier_legal_trajectories() -> None:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 2, 2, 3],
        ],
        dtype=torch.long,
    )
    replay = build_replay_bank(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2, 3], dtype=torch.long),
        num_nodes=4,
        sample_id="unit/test/0",
        max_edges=3,
        round_variants=2,
        trajectories_per_graph=3,
        beam_width=8,
        path_variants_per_pair=2,
        max_expansions_per_state=8,
        seed=7,
    )

    for variant in range(int(replay.edge_ids.size(0))):
        for slot in range(int(replay.edge_ids.size(1))):
            edge_count = int(replay.edge_count[variant, slot].item())
            if edge_count < 0:
                continue
            active = {0}
            for edge_id in replay.edge_ids[variant, slot, :edge_count].tolist():
                src = int(edge_index[0, edge_id].item())
                dst = int(edge_index[1, edge_id].item())
                assert src in active
                active.add(dst)
