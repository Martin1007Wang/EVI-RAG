from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.data.artifacts import parse_materialization_artifact
from src.data.preprocess.catalog import CatalogBuilder, EntityTextPolicy
from src.data.preprocess.relation_neighborhood import (
    build_relation_neighborhood_semantic_table,
)
from src.training.config import validate_model_resources


def test_relation_neighborhood_uses_all_retained_incident_relation_sets_for_non_text_entities() -> None:
    builder = CatalogBuilder()
    text_entity = builder.add_entity("named entity")
    mid_a = builder.add_entity("m.a")
    mid_b = builder.add_entity("m.b")
    mid_validation_only = builder.add_entity("m.validation")
    rel_a = builder.add_relation("rel.a")
    rel_b = builder.add_relation("rel.b")
    rel_validation = builder.add_relation("rel.validation")
    catalog = builder.build(text_policy=EntityTextPolicy(non_text_prefixes=("m.",)))
    relation_semantic_table = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    train = _sample(
        split="train",
        node_entity_catalog_ids=[text_entity, mid_a, mid_b],
        edges=[(0, 1), (1, 2), (0, 1)],
        relation_ids=[rel_a, rel_b, rel_a],
    )
    validation = _sample(
        split="validation",
        node_entity_catalog_ids=[mid_b, mid_validation_only],
        edges=[(0, 1)],
        relation_ids=[rel_validation],
    )

    table, row_by_entity_id = build_relation_neighborhood_semantic_table(
        prepared_samples=[train, validation],
        catalog=catalog,
        relation_semantic_table=relation_semantic_table,
    )

    assert row_by_entity_id.tolist() == [-1, 0, 1, 2]
    assert torch.allclose(
        table,
        torch.tensor(
            [
                [2**-0.5, 2**-0.5],
                [-2**-0.5, 2**-0.5],
                [-1.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(table.norm(dim=-1), torch.ones(3))


def test_relation_neighborhood_rejects_zero_sum_rows() -> None:
    builder = CatalogBuilder()
    mid = builder.add_entity("m.zero")
    rel_a = builder.add_relation("rel.a")
    rel_b = builder.add_relation("rel.b")
    catalog = builder.build(text_policy=EntityTextPolicy(non_text_prefixes=("m.",)))

    with pytest.raises(ValueError, match="sum to zero"):
        build_relation_neighborhood_semantic_table(
            prepared_samples=[
                _sample(
                    split="train",
                    node_entity_catalog_ids=[mid, mid],
                    edges=[(0, 1), (1, 0)],
                    relation_ids=[rel_a, rel_b],
                )
            ],
            catalog=catalog,
            relation_semantic_table=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        )


def test_catalog_round_trips_relation_neighborhood_row_map(tmp_path) -> None:
    builder = CatalogBuilder()
    builder.add_entity("m.a")
    builder.add_relation("rel.a")
    catalog = builder.build(text_policy=EntityTextPolicy(non_text_prefixes=("m.",)))
    catalog.relation_neighborhood_row_by_entity_id[0] = 0
    path = tmp_path / "catalog.pt"

    catalog.save(path)
    loaded = catalog.load(path)

    assert loaded.relation_neighborhood_row_by_entity_id.tolist() == [0]


def test_model_resources_reject_relation_neighborhood_row_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside its semantic table"):
        validate_model_resources(
            entity_text_semantic_table=torch.tensor([[1.0, 0.0]]),
            text_row_by_entity_id=torch.tensor([0, -1]),
            entity_relation_neighborhood_semantic_table=torch.tensor([[0.0, 1.0]]),
            relation_neighborhood_row_by_entity_id=torch.tensor([-1, 1]),
            relation_semantic_table=torch.tensor([[1.0, 0.0]]),
        )


def test_model_resources_reject_non_normalized_relation_neighborhood_rows() -> None:
    with pytest.raises(ValueError, match="rows must be L2-normalized"):
        validate_model_resources(
            entity_text_semantic_table=torch.tensor([[1.0, 0.0]]),
            text_row_by_entity_id=torch.tensor([0, -1]),
            entity_relation_neighborhood_semantic_table=torch.tensor([[0.0, 2.0]]),
            relation_neighborhood_row_by_entity_id=torch.tensor([-1, 0]),
            relation_semantic_table=torch.tensor([[1.0, 0.0]]),
        )


def test_model_resources_require_every_entity_to_have_a_feature() -> None:
    with pytest.raises(ValueError, match="Every entity must have"):
        validate_model_resources(
            entity_text_semantic_table=torch.tensor([[1.0, 0.0]]),
            text_row_by_entity_id=torch.tensor([0, -1]),
            entity_relation_neighborhood_semantic_table=torch.empty((0, 2)),
            relation_neighborhood_row_by_entity_id=torch.tensor([-1, -1]),
            relation_semantic_table=torch.tensor([[1.0, 0.0]]),
        )


def test_legacy_manifest_requires_relation_neighborhood_rebuild(tmp_path) -> None:
    with pytest.raises(ValueError, match="Re-run preprocessing"):
        parse_materialization_artifact(
            {
                "generation_id": "legacy",
                "materialization_dir": ".",
                "catalogs": {"catalog": "catalog.pt"},
                "embeddings": {
                    "entity_text_semantic_table": {
                        "path": "entity.f32",
                        "dtype": "float32",
                        "shape": [1, 2],
                    },
                    "relation_semantic_table": {
                        "path": "relation.f32",
                        "dtype": "float32",
                        "shape": [1, 2],
                    },
                },
                "splits": {},
            },
            manifest_path=tmp_path / "materialization_manifest.json",
        )


def _sample(
    *,
    split: str,
    node_entity_catalog_ids: list[int],
    edges: list[tuple[int, int]],
    relation_ids: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        split=split,
        node_entity_catalog_ids=torch.tensor(node_entity_catalog_ids),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_relation_catalog_ids=torch.tensor(relation_ids),
    )
