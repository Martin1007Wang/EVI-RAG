from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.catalog_lookup import CatalogLookup
from src.data.dataset import RetrievalDataset
from src.data.schema import SampleFields, StorageSchema


def _sample_dict_with_catalog_ids(*, use_legacy_names: bool) -> dict[str, torch.Tensor]:
    node_key = (
        SampleFields.LEGACY_NODE_ENTITY_IDS
        if use_legacy_names
        else SampleFields.NODE_ENTITY_CATALOG_IDS
    )
    edge_key = (
        SampleFields.LEGACY_EDGE_RELATION_IDS
        if use_legacy_names
        else SampleFields.EDGE_RELATION_CATALOG_IDS
    )
    return {
        SampleFields.EDGE_INDEX: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        node_key: torch.tensor([11, 12], dtype=torch.long),
        edge_key: torch.tensor([21, 22], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(2, dtype=torch.long),
        SampleFields.QUESTION_EMB: torch.tensor([0.1, 0.2], dtype=torch.float32),
        SampleFields.IS_ANCHOR_MASK: torch.tensor([True, False], dtype=torch.bool),
        SampleFields.TRAIN_TARGET_MASK: torch.tensor([False, True], dtype=torch.bool),
        SampleFields.ANCHOR_SIGNED_DISTANCE: torch.tensor([0, 1], dtype=torch.long),
        SampleFields.TRAIN_TARGET_NODE_IDS: torch.tensor([1], dtype=torch.long),
        SampleFields.TARGET_NODE_DISTANCE_FLAT: torch.tensor([1, 0], dtype=torch.long),
        SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor([1, 1], dtype=torch.long),
        SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT: torch.tensor(
            [True, False], dtype=torch.bool
        ),
    }


def test_storage_schema_accepts_legacy_catalog_id_field_names() -> None:
    StorageSchema.validate(_sample_dict_with_catalog_ids(use_legacy_names=True))


def test_retrieval_dataset_build_sample_reads_legacy_catalog_id_field_names() -> None:
    dataset = RetrievalDataset(sample_ids=[], lmdb_paths=[])

    sample = dataset._build_sample(
        _sample_dict_with_catalog_ids(use_legacy_names=True),
        sample_id="sample-1",
    )

    assert torch.equal(
        sample.node_entity_catalog_ids,
        torch.tensor([11, 12], dtype=torch.long),
    )
    assert torch.equal(
        sample.edge_relation_catalog_ids,
        torch.tensor([21, 22], dtype=torch.long),
    )


def test_catalog_lookup_accepts_canonical_and_legacy_graph_attrs() -> None:
    lookup = CatalogLookup(entity_labels=["e0", "e1", "e2"], relation_labels=["r0", "r1"])

    canonical_graph = SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([1, 2], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    legacy_graph = SimpleNamespace(
        node_entity_ids=torch.tensor([2, 1], dtype=torch.long),
        edge_relation_ids=torch.tensor([1], dtype=torch.long),
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert lookup.local_node_global_id(canonical_graph, 0) == 1
    assert lookup.local_edge_global_id(canonical_graph, 0) == 0
    assert lookup.local_node_global_id(legacy_graph, 0) == 2
    assert lookup.local_edge_global_id(legacy_graph, 0) == 1
