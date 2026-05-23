from __future__ import annotations

import torch
import torch.nn.functional as F

from src.weaver.nn.feature_encoder import FeatureEncoder


def test_feature_encoder_l2_normalizes_before_projection() -> None:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor([[1.0, 0.0]]),
        text_row_by_entity_id=torch.tensor([0]),
        relation_semantic_table=torch.tensor([[0.0, 1.0]]),
        model_dim=2,
    )
    with torch.no_grad():
        encoder.project_query_to_model.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )

    semantic = torch.tensor([[3.0, 4.0]])

    actual = encoder.to_model_space(
        semantic,
        projector=encoder.project_query_to_model,
    )
    expected = F.linear(
        F.normalize(semantic, p=2, dim=-1),
        encoder.project_query_to_model.weight,
    )

    assert torch.allclose(actual, expected)


def test_feature_encoder_uses_role_specific_projections() -> None:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        text_row_by_entity_id=torch.tensor([0], dtype=torch.long),
        relation_semantic_table=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        model_dim=2,
    )
    with torch.no_grad():
        encoder.project_query_to_model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        encoder.project_node_to_model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32))
        encoder.project_relation_to_model.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32))

    semantic = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    normalized = F.normalize(semantic, p=2, dim=-1)

    assert torch.allclose(
        encoder.project_query_semantic(semantic),
        F.linear(normalized, encoder.project_query_to_model.weight),
    )
    assert torch.allclose(
        encoder.project_node_semantic(semantic),
        F.linear(normalized, encoder.project_node_to_model.weight),
    )
    assert torch.allclose(
        encoder.project_relation_semantic(semantic),
        F.linear(normalized, encoder.project_relation_to_model.weight),
    )
