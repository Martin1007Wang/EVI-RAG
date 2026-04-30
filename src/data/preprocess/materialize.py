from __future__ import annotations

import logging
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

import lmdb
import torch

from src.data.schema.fields import SampleFields, StorageSchema
from src.utils.lmdb_utils import serialize_sample

from .samples import PreparedSample
from .vocab import EntityCatalog, RelationCatalog

log = logging.getLogger(__name__)


def materialize_preprocessed_data(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_text_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    question_embeddings: torch.Tensor,
    lmdb_dir: Path,
    metadata_dir: Path,
    embeddings_dir: Path,
    entity_metadata_path: Path,
    entity_catalog_path: Path,
    relation_catalog_path: Path,
    overwrite_lmdb: bool = True,
    map_size_gb: float = 128,
    commit_frequency: int = 1000,
    schema_version: int = 2,
) -> None:
    """
    Materialize prepared retrieval samples.

    Outputs:
    - LMDB samples:
        lmdb_dir/{split}.lmdb

    - Split indices:
        metadata_dir/{split}.index.pt

    - Catalogs:
        entity_catalog_path
        relation_catalog_path
        entity_metadata_path

    - Embeddings:
        embeddings_dir/entity_text_embeddings.pt
        embeddings_dir/entity_text_embedding_ids.pt
        embeddings_dir/non_text_entity_mask.pt
        embeddings_dir/relation_embeddings.pt
        embeddings_dir/relation_text_labels.pt

    This function does not:
    - filter samples
    - recompute path labels
    - create train target masks
    - create anchor masks
    - support legacy on-disk fields
    - support LMDB sharding
    """
    _validate_materialize_inputs(
        prepared_samples=prepared_samples,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        entity_text_embeddings=entity_text_embeddings,
        relation_embeddings=relation_embeddings,
        question_embeddings=question_embeddings,
        commit_frequency=commit_frequency,
    )

    lmdb_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    _save_catalog_artifacts(
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        entity_metadata_path=entity_metadata_path,
        entity_catalog_path=entity_catalog_path,
        relation_catalog_path=relation_catalog_path,
    )

    _save_embedding_artifacts(
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        entity_text_embeddings=entity_text_embeddings,
        relation_embeddings=relation_embeddings,
        embeddings_dir=embeddings_dir,
    )

    _write_lmdb_samples(
        prepared_samples=prepared_samples,
        question_embeddings=question_embeddings,
        lmdb_dir=lmdb_dir,
        overwrite_lmdb=overwrite_lmdb,
        map_size_gb=map_size_gb,
        commit_frequency=commit_frequency,
    )

    _save_split_indices(
        prepared_samples=prepared_samples,
        metadata_dir=metadata_dir,
        schema_version=schema_version,
    )

    log.info(
        "Materialization complete: samples=%d, splits=%s",
        len(prepared_samples),
        sorted({sample.split for sample in prepared_samples}),
    )


def _validate_materialize_inputs(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_text_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    question_embeddings: torch.Tensor,
    commit_frequency: int,
) -> None:
    if not prepared_samples:
        raise ValueError("prepared_samples must be non-empty")

    if question_embeddings.ndim != 2:
        raise ValueError("question_embeddings must be 2D")

    if int(question_embeddings.size(0)) != len(prepared_samples):
        raise ValueError(
            "question_embeddings size mismatch: "
            f"got {int(question_embeddings.size(0))}, "
            f"expected {len(prepared_samples)}"
        )

    if entity_text_embeddings.ndim != 2:
        raise ValueError("entity_text_embeddings must be 2D")

    if int(entity_text_embeddings.size(0)) != entity_catalog.num_text_entities:
        raise ValueError(
            "entity_text_embeddings size mismatch: "
            f"got {int(entity_text_embeddings.size(0))}, "
            f"expected {entity_catalog.num_text_entities}"
        )

    if relation_embeddings.ndim != 2:
        raise ValueError("relation_embeddings must be 2D")

    if int(relation_embeddings.size(0)) != len(relation_catalog.relation_labels):
        raise ValueError(
            "relation_embeddings size mismatch: "
            f"got {int(relation_embeddings.size(0))}, "
            f"expected {len(relation_catalog.relation_labels)}"
        )

    if commit_frequency <= 0:
        raise ValueError("commit_frequency must be positive")


def _save_catalog_artifacts(
    *,
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_metadata_path: Path,
    entity_catalog_path: Path,
    relation_catalog_path: Path,
) -> None:
    entity_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    relation_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    entity_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    entity_catalog.save(entity_catalog_path)
    relation_catalog.save(relation_catalog_path)

    torch.save(
        {
            "entity_text_embedding_ids": (
                entity_catalog.entity_text_embedding_ids.long().contiguous()
            ),
            "entity_embedding_map": _entity_text_embedding_ids_to_map(
                entity_catalog.entity_text_embedding_ids
            ),
            "non_text_entity_mask": (
                entity_catalog.non_text_entity_mask.bool().contiguous()
            ),
        },
        entity_metadata_path,
    )


def _save_embedding_artifacts(
    *,
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_text_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    embeddings_dir: Path,
) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        entity_text_embeddings.float().contiguous(),
        embeddings_dir / "entity_text_embeddings.pt",
    )
    torch.save(
        entity_catalog.entity_text_embedding_ids.long().contiguous(),
        embeddings_dir / "entity_text_embedding_ids.pt",
    )
    torch.save(
        entity_catalog.non_text_entity_mask.bool().contiguous(),
        embeddings_dir / "non_text_entity_mask.pt",
    )
    torch.save(
        relation_embeddings.float().contiguous(),
        embeddings_dir / "relation_embeddings.pt",
    )
    torch.save(
        list(relation_catalog.relation_text_labels),
        embeddings_dir / "relation_text_labels.pt",
    )


def _entity_text_embedding_ids_to_map(
    entity_text_embedding_ids: torch.Tensor,
) -> torch.Tensor:
    if entity_text_embedding_ids.ndim != 1:
        raise ValueError(
            "entity_text_embedding_ids must be 1D, "
            f"got shape={tuple(entity_text_embedding_ids.shape)}."
        )

    return entity_text_embedding_ids.long().contiguous() - 1


def _write_lmdb_samples(
    *,
    prepared_samples: Sequence[PreparedSample],
    question_embeddings: torch.Tensor,
    lmdb_dir: Path,
    overwrite_lmdb: bool,
    map_size_gb: float,
    commit_frequency: int,
) -> None:
    map_size_bytes = int(map_size_gb * (1024**3))
    if map_size_bytes <= 0:
        raise ValueError(f"map_size_gb must be positive, got {map_size_gb}")

    splits = sorted({sample.split for sample in prepared_samples})

    with ExitStack() as stack:
        envs: dict[str, lmdb.Environment] = {}
        txns: dict[str, lmdb.Transaction] = {}
        uncommitted: dict[str, int] = {}

        for split in splits:
            path = _lmdb_path(lmdb_dir=lmdb_dir, split=split)
            _reset_output_path(path, overwrite=overwrite_lmdb)

            env = stack.enter_context(
                lmdb.open(
                    str(path),
                    map_size=map_size_bytes,
                    subdir=True,
                    lock=True,
                    create=True,
                    max_dbs=1,
                )
            )

            envs[split] = env
            txns[split] = env.begin(write=True)
            uncommitted[split] = 0

        for idx, sample in enumerate(prepared_samples):
            sample_dict = _sample_to_lmdb_record(
                sample=sample,
                question_embedding=question_embeddings[idx],
            )
            StorageSchema.validate(sample_dict)

            split = sample.split
            txns[split].put(
                sample.question_id.encode("utf-8"),
                serialize_sample(sample_dict),
            )

            uncommitted[split] += 1
            if uncommitted[split] >= commit_frequency:
                txns[split].commit()
                txns[split] = envs[split].begin(write=True)
                uncommitted[split] = 0

        for txn in txns.values():
            txn.commit()


def _sample_to_lmdb_record(
    *,
    sample: PreparedSample,
    question_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_prepared_sample_shapes(sample)

    return {
        SampleFields.EDGE_INDEX: sample.edge_index.long().contiguous(),
        SampleFields.NODE_ENTITY_CATALOG_IDS: (
            sample.node_entity_catalog_ids.long().contiguous()
        ),
        SampleFields.EDGE_RELATION_CATALOG_IDS: (
            sample.edge_relation_catalog_ids.long().contiguous()
        ),
        SampleFields.NUM_NODES: torch.as_tensor(sample.num_nodes, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.as_tensor(sample.num_edges, dtype=torch.long),
        SampleFields.QUESTION_EMB: question_embedding.float().contiguous(),
        SampleFields.ANCHOR_NODE_IDS: sample.anchor_node_ids.long().contiguous(),
        SampleFields.TARGET_NODE_IDS: sample.target_node_ids.long().contiguous(),
        SampleFields.REACHABLE_TARGET_NODE_IDS: (
            sample.reachable_target_node_ids.long().contiguous()
        ),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: (
            sample.anchor_node_forward_distances_flat.long().contiguous()
        ),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: (
            sample.anchor_node_backward_distances_flat.long().contiguous()
        ),
        SampleFields.NODE_TARGET_DISTANCE: (
            sample.node_target_distance.long().contiguous()
        ),
        SampleFields.TARGET_NODE_DISTANCE_FLAT: (
            sample.target_node_distances_flat.long().contiguous()
        ),
        SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT: (
            sample.target_shortest_path_count_flat.float().contiguous()
        ),
        SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT: (
            sample.target_shortest_path_edge_mask_flat.bool().contiguous()
        ),
    }


def _validate_prepared_sample_shapes(sample: PreparedSample) -> None:
    num_nodes = int(sample.num_nodes)
    num_edges = int(sample.num_edges)
    num_reachable_targets = int(sample.reachable_target_node_ids.numel())

    if num_nodes <= 0:
        raise ValueError(f"{sample.question_id}: num_nodes must be positive")

    if num_edges <= 0:
        raise ValueError(f"{sample.question_id}: num_edges must be positive")

    if sample.edge_index.shape != (2, num_edges):
        raise ValueError(
            f"{sample.question_id}: edge_index shape mismatch, "
            f"got {tuple(sample.edge_index.shape)}, expected {(2, num_edges)}"
        )

    _require_numel(
        sample_id=sample.question_id,
        name="node_entity_catalog_ids",
        tensor=sample.node_entity_catalog_ids,
        expected=num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="edge_relation_catalog_ids",
        tensor=sample.edge_relation_catalog_ids,
        expected=num_edges,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="anchor_node_forward_distances_flat",
        tensor=sample.anchor_node_forward_distances_flat,
        expected=num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="anchor_node_backward_distances_flat",
        tensor=sample.anchor_node_backward_distances_flat,
        expected=num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="node_target_distance",
        tensor=sample.node_target_distance,
        expected=num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="target_node_distances_flat",
        tensor=sample.target_node_distances_flat,
        expected=num_reachable_targets * num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="target_shortest_path_count_flat",
        tensor=sample.target_shortest_path_count_flat,
        expected=num_reachable_targets * num_nodes,
    )
    _require_numel(
        sample_id=sample.question_id,
        name="target_shortest_path_edge_mask_flat",
        tensor=sample.target_shortest_path_edge_mask_flat,
        expected=num_reachable_targets * num_edges,
    )


def _save_split_indices(
    *,
    prepared_samples: Sequence[PreparedSample],
    metadata_dir: Path,
    schema_version: int,
) -> None:
    by_split: dict[str, list[str]] = {}

    for sample in prepared_samples:
        by_split.setdefault(sample.split, []).append(sample.question_id)

    for split, sample_ids in by_split.items():
        torch.save(
            {
                "schema_version": int(schema_version),
                "sample_ids": list(sample_ids),
            },
            metadata_dir / f"{split}.index.pt",
        )


def _require_numel(
    *,
    sample_id: str,
    name: str,
    tensor: torch.Tensor,
    expected: int,
) -> None:
    actual = int(tensor.numel())
    if actual != expected:
        raise ValueError(
            f"{sample_id}: {name} length mismatch, "
            f"got {actual}, expected {expected}"
        )


def _lmdb_path(
    *,
    lmdb_dir: Path,
    split: str,
) -> Path:
    return lmdb_dir / f"{split}.lmdb"


def _reset_output_path(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return

    if not overwrite:
        raise FileExistsError(f"{path} exists")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
