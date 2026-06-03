from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .catalog import Catalog
from .samples import PreparedSample


def build_relation_neighborhood_semantic_table(
    *,
    prepared_samples: Sequence[PreparedSample],
    catalog: Catalog,
    relation_semantic_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build compact pseudo-features for non-text entities from incident relations.

    Relations are deduplicated per entity across all retained graphs. Incoming
    and outgoing relations intentionally share one set in this first version.
    """

    if relation_semantic_table.ndim != 2:
        raise ValueError("relation_semantic_table must be 2D.")
    if int(relation_semantic_table.size(0)) != catalog.num_relations:
        raise ValueError(
            "relation_semantic_table rows must equal catalog.num_relations."
        )

    relation_ids_by_entity_id: dict[int, set[int]] = {}
    for sample in prepared_samples:
        src_entity_ids = sample.node_entity_catalog_ids.index_select(
            0,
            sample.edge_index[0].long(),
        )
        dst_entity_ids = sample.node_entity_catalog_ids.index_select(
            0,
            sample.edge_index[1].long(),
        )
        for entity_id, relation_id in zip(
            torch.cat((src_entity_ids, dst_entity_ids)).tolist(),
            torch.cat(
                (
                    sample.edge_relation_catalog_ids,
                    sample.edge_relation_catalog_ids,
                )
            ).tolist(),
            strict=True,
        ):
            if int(catalog.entity_text_row_by_entity_id[int(entity_id)].item()) >= 0:
                continue
            relation_ids_by_entity_id.setdefault(int(entity_id), set()).add(
                int(relation_id)
            )

    row_by_entity_id = torch.full((catalog.num_entities,), -1, dtype=torch.long)
    rows: list[torch.Tensor] = []
    for entity_id in sorted(relation_ids_by_entity_id):
        relation_ids = sorted(relation_ids_by_entity_id[entity_id])
        semantic_h = relation_semantic_table.index_select(
            0,
            torch.tensor(relation_ids, dtype=torch.long),
        ).sum(dim=0)
        if float(torch.linalg.vector_norm(semantic_h).item()) == 0.0:
            raise ValueError(
                "Cannot build a relation-neighborhood feature because incident "
                f"relation embeddings sum to zero for catalog entity id {entity_id}."
            )
        row_by_entity_id[entity_id] = len(rows)
        rows.append(F.normalize(semantic_h, p=2, dim=0))

    missing_entity_ids = [
        entity_id
        for entity_id in range(catalog.num_entities)
        if int(catalog.entity_text_row_by_entity_id[entity_id].item()) < 0
        and int(row_by_entity_id[entity_id].item()) < 0
    ]
    if missing_entity_ids:
        raise ValueError(
            "Every non-text entity must occur in at least one retained graph edge; "
            f"missing catalog entity ids: {missing_entity_ids}."
        )

    if not rows:
        return relation_semantic_table.new_empty((0, relation_semantic_table.size(1))), row_by_entity_id
    return torch.stack(rows).to(dtype=torch.float32, device="cpu").contiguous(), row_by_entity_id


__all__ = ["build_relation_neighborhood_semantic_table"]
