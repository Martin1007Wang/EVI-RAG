from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.datamodule import _extract_entity_embedding_map
from src.data.preprocess.materialize import _entity_text_embedding_ids_to_map


def test_extract_entity_embedding_map_accepts_materialized_entity_text_embedding_ids() -> None:
    entity_metadata = {
        "entity_text_embedding_ids": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "non_text_entity_mask": torch.tensor([True, False, False, True], dtype=torch.bool),
    }

    entity_embedding_map = _extract_entity_embedding_map(
        artifact=entity_metadata,
        name="entity_metadata",
    )

    assert torch.equal(
        entity_embedding_map,
        torch.tensor([-1, 0, 1, -1], dtype=torch.long),
    )


def test_entity_text_embedding_ids_to_map_uses_minus_one_non_text_sentinel() -> None:
    entity_text_embedding_ids = torch.tensor([0, 3, 1, 0, 2], dtype=torch.long)

    entity_embedding_map = _entity_text_embedding_ids_to_map(entity_text_embedding_ids)

    assert torch.equal(
        entity_embedding_map,
        torch.tensor([-1, 2, 0, -1, 1], dtype=torch.long),
    )
