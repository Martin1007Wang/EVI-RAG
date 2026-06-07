from __future__ import annotations

import torch

from src.data.collate import RetrievalCollator
from src.data.schema import ReplayBankSample, RetrievalData
from src.weaver.feature import FeatureEncoder


def test_feature_encoder_emits_static_frontier_prune_scores_from_raw_question_relation_dot_product() -> None:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        text_row_by_entity_id=torch.tensor([0, 1], dtype=torch.long),
        entity_relation_neighborhood_semantic_table=torch.empty((0, 2), dtype=torch.float32),
        relation_neighborhood_row_by_entity_id=torch.tensor([-1, -1], dtype=torch.long),
        relation_semantic_table=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        sem_dim=2,
        hidden_dim=2,
    )
    batch = RetrievalCollator()(
        [
            RetrievalData(
                sample_id="toy/0",
                edge_index=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
                edge_relation_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
                num_nodes=2,
                num_edges=2,
                question_emb=torch.tensor([0.6, 0.8], dtype=torch.float32),
                anchor_node_ids=torch.tensor([0], dtype=torch.long),
                target_node_ids=torch.tensor([1], dtype=torch.long),
                reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
                node_target_distance=torch.tensor([1, 0], dtype=torch.long),
                edge_on_shortest_path=torch.tensor([True, False], dtype=torch.bool),
                reachable_target_max_distance=torch.tensor(1, dtype=torch.long),
                replay_bank=ReplayBankSample(
                    edge_ids_local=torch.empty((0, 0), dtype=torch.long),
                    edge_count=torch.empty((0,), dtype=torch.long),
                    priority=torch.empty((0,), dtype=torch.float32),
                ),
            )
        ]
    )

    features = encoder(batch)

    expected = torch.tensor([0.6, 0.8], dtype=torch.float32)
    assert torch.allclose(features.frontier_prune_score, expected)
