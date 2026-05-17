from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import lmdb
import torch

from src.data.artifacts import manifest_path
from src.data.schema.fields import SampleFields
from src.data.split_index import SplitIndexWriter
from src.data.tensor_table import (
    TensorTable,
    TensorTableWriter,
    manifest_entry,
    write_table,
)
from src.utils.lmdb_utils import serialize_sample

from .samples import PreparedSample
from .vocab import EntityCatalog, RelationCatalog

log = logging.getLogger(__name__)

ENTITY_METADATA_NAME = "entity_metadata.pt"
ENTITY_CATALOG_NAME = "entity_catalog.pt"
RELATION_CATALOG_NAME = "relation_catalog.pt"


@dataclass(frozen=True)
class SplitPlan:
    num_samples: int


@dataclass(frozen=True)
class MaterializationPlan:
    split_plans: Mapping[str, SplitPlan]
    question_embedding_dim: int


class StreamingMaterializer:
    """
    Writes one complete, manifest-addressed materialized dataset generation.

    Contract:
    - metadata_dir is the only artifact root.
    - data are written to metadata_dir/.materialize_tmp/{generation_id} first.
    - successful finish publishes them to metadata_dir/.materializations/{generation_id}.
    - metadata_dir/materialization_manifest.json is atomically replaced last.
    - training code should enter only through the active manifest.
    """

    def __init__(
        self,
        *,
        plan: MaterializationPlan,
        entity_catalog: EntityCatalog,
        relation_catalog: RelationCatalog,
        entity_text_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        metadata_dir: Path,
        overwrite: bool = True,
        map_size_gb: float = 128,
        commit_frequency: int = 1000,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        _validate_inputs(
            plan=plan,
            entity_catalog=entity_catalog,
            relation_catalog=relation_catalog,
            entity_text_embeddings=entity_text_embeddings,
            relation_embeddings=relation_embeddings,
            commit_frequency=commit_frequency,
        )

        self.plan = plan
        self.entity_catalog = entity_catalog
        self.relation_catalog = relation_catalog
        self.entity_text_embeddings = entity_text_embeddings
        self.relation_embeddings = relation_embeddings

        self.metadata_dir = Path(metadata_dir)
        self.entity_metadata_name = ENTITY_METADATA_NAME
        self.entity_catalog_name = ENTITY_CATALOG_NAME
        self.relation_catalog_name = RELATION_CATALOG_NAME

        self.overwrite = bool(overwrite)
        self.map_size_bytes = int(float(map_size_gb) * (1024**3))
        if self.map_size_bytes <= 0:
            raise ValueError(f"map_size_gb must be positive, got {map_size_gb}")
        self.commit_frequency = int(commit_frequency)
        self.provenance = None if provenance is None else dict(provenance)

        self.generation_id = uuid.uuid4().hex
        self.tmp_generation_dir = self.metadata_dir / ".materialize_tmp" / self.generation_id
        self.generation_dir = self.metadata_dir / ".materializations" / self.generation_id
        self.tmp_lmdb_dir = self.tmp_generation_dir / "lmdb"
        self.tmp_metadata_dir = self.tmp_generation_dir / "metadata"
        self.tmp_embeddings_dir = self.tmp_generation_dir / "embeddings"

        self._stack: ExitStack | None = None
        self._envs: dict[str, lmdb.Environment] = {}
        self._txns: dict[str, lmdb.Transaction] = {}
        self._uncommitted: dict[str, int] = {}
        self._index_writers: dict[str, SplitIndexWriter] = {}
        self._question_writers: dict[str, TensorTableWriter] = {}
        self._rows_written: dict[str, int] = {}
        self._entity_text_table: TensorTable | None = None
        self._relation_table: TensorTable | None = None
        self._closed = False

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(self._rows_written)

    def __enter__(self) -> StreamingMaterializer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.abort()
            return
        self.finish()

    def open(self) -> None:
        if self._stack is not None:
            raise RuntimeError("StreamingMaterializer is already open")

        _preflight_generation(
            metadata_dir=self.metadata_dir,
            generation_dir=self.generation_dir,
            overwrite=self.overwrite,
        )

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.tmp_generation_dir.exists():
            shutil.rmtree(self.tmp_generation_dir)
        self.tmp_lmdb_dir.mkdir(parents=True)
        self.tmp_metadata_dir.mkdir(parents=True)
        self.tmp_embeddings_dir.mkdir(parents=True)

        stack = ExitStack()
        self._stack = stack
        try:
            self._write_global_artifacts()
            self._open_split_writers(stack)
        except Exception:
            self.abort()
            raise

    def write_chunk(
        self,
        *,
        prepared_samples: list[PreparedSample],
        question_embeddings: torch.Tensor,
    ) -> None:
        if self._stack is None:
            raise RuntimeError("StreamingMaterializer must be opened before writing")
        _validate_question_embeddings(
            embeddings=question_embeddings,
            num_samples=len(prepared_samples),
            dim=self.plan.question_embedding_dim,
        )

        rows_by_split: dict[str, list[tuple[PreparedSample, torch.Tensor]]] = {}
        for idx, sample in enumerate(prepared_samples):
            if sample.split not in self.plan.split_plans:
                raise ValueError(f"Unexpected split during materialization: {sample.split}")
            rows_by_split.setdefault(sample.split, []).append((sample, question_embeddings[idx]))

        for split, rows in rows_by_split.items():
            start_row = self._rows_written[split]
            embeddings = torch.stack([embedding for _, embedding in rows], dim=0)
            self._question_writers[split].write(start_row, embeddings)

            for offset, (sample, _embedding) in enumerate(rows):
                self._write_sample(
                    split=split,
                    row_idx=start_row + offset,
                    sample=sample,
                )

            self._rows_written[split] += len(rows)

    def finish(self) -> None:
        if self._closed:
            return
        if self._stack is None:
            raise RuntimeError("StreamingMaterializer was not opened")

        try:
            self._check_expected_split_counts()
            self._commit_lmdb_transactions()
            self._close_open_resources()

            manifest = _build_manifest(
                plan=self.plan,
                generation_id=self.generation_id,
                generation_dir=self.generation_dir,
                metadata_dir=self.metadata_dir,
                entity_metadata_path=(self.generation_dir / "metadata" / self.entity_metadata_name),
                entity_catalog_path=(self.generation_dir / "metadata" / self.entity_catalog_name),
                relation_catalog_path=(self.generation_dir / "metadata" / self.relation_catalog_name),
                entity_text_table=self._published_table(_require_table(self._entity_text_table)),
                relation_table=self._published_table(_require_table(self._relation_table)),
                question_tables={split: self._published_table(writer.table) for split, writer in self._question_writers.items()},
                provenance=self.provenance,
            )
            _publish_generation(
                tmp_generation_dir=self.tmp_generation_dir,
                generation_dir=self.generation_dir,
                metadata_dir=self.metadata_dir,
                manifest=manifest,
            )
        except Exception:
            self.abort()
            raise

        self._closed = True
        log.info(
            "Materialization complete: generation=%s, samples=%d, splits=%s",
            self.generation_id,
            sum(split.num_samples for split in self.plan.split_plans.values()),
            sorted(self.plan.split_plans),
        )

    def abort(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        if self.tmp_generation_dir.exists():
            shutil.rmtree(self.tmp_generation_dir)
        self._closed = True

    def _write_global_artifacts(self) -> None:
        _save_catalog_artifacts(
            entity_catalog=self.entity_catalog,
            relation_catalog=self.relation_catalog,
            entity_metadata_path=self.tmp_metadata_dir / self.entity_metadata_name,
            entity_catalog_path=self.tmp_metadata_dir / self.entity_catalog_name,
            relation_catalog_path=self.tmp_metadata_dir / self.relation_catalog_name,
        )

        self._entity_text_table = write_table(
            self.tmp_embeddings_dir / "entity_text_embeddings.f32",
            self.entity_text_embeddings,
        )
        self._relation_table = write_table(
            self.tmp_embeddings_dir / "relation_embeddings.f32",
            self.relation_embeddings,
        )

        entity_embedding_map = entity_text_row_ids(
            self.entity_catalog.entity_text_embedding_ids
        )
        torch.save(
            entity_embedding_map,
            self.tmp_embeddings_dir / "entity_embedding_map.pt",
        )
        torch.save(
            self.entity_catalog.non_text_entity_mask.bool().contiguous(),
            self.tmp_embeddings_dir / "non_text_entity_mask.pt",
        )
        torch.save(
            list(self.relation_catalog.relation_text_labels),
            self.tmp_embeddings_dir / "relation_text_labels.pt",
        )

    def _open_split_writers(self, stack: ExitStack) -> None:
        index_map_size = _index_map_size(self.plan)
        for split, split_plan in self.plan.split_plans.items():
            lmdb_path = self.tmp_lmdb_dir / f"{split}.lmdb"
            env = stack.enter_context(
                lmdb.open(
                    str(lmdb_path),
                    map_size=self.map_size_bytes,
                    subdir=True,
                    lock=True,
                    create=True,
                    max_dbs=1,
                    sync=True,
                    writemap=False,
                )
            )
            self._envs[split] = env
            self._txns[split] = env.begin(write=True)
            self._uncommitted[split] = 0
            self._rows_written[split] = 0

            self._index_writers[split] = stack.enter_context(
                SplitIndexWriter(
                    self.tmp_metadata_dir / f"{split}.index.lmdb",
                    map_size=index_map_size,
                    commit_frequency=self.commit_frequency,
                )
            )
            self._question_writers[split] = stack.enter_context(
                TensorTableWriter(
                    self.tmp_embeddings_dir / f"{split}.questions.f32",
                    rows=split_plan.num_samples,
                    dim=self.plan.question_embedding_dim,
                )
            )

    def _write_sample(self, *, split: str, row_idx: int, sample: PreparedSample) -> None:
        record = storage_record_from_sample(sample)

        storage_key = sample_storage_key(sample)
        inserted = self._txns[split].put(
            storage_key.encode("utf-8"),
            serialize_sample(record),
            overwrite=False,
        )
        if not inserted:
            raise ValueError(f"duplicate LMDB sample storage key: {storage_key}")

        self._index_writers[split].put(row_idx, storage_key)
        self._uncommitted[split] += 1
        if self._uncommitted[split] >= self.commit_frequency:
            self._txns[split].commit()
            self._txns[split] = self._envs[split].begin(write=True)
            self._uncommitted[split] = 0

    def _check_expected_split_counts(self) -> None:
        for split, split_plan in self.plan.split_plans.items():
            written = self._rows_written.get(split, 0)
            if written != split_plan.num_samples:
                raise ValueError(f"Split {split!r} wrote {written} samples, " f"expected {split_plan.num_samples}")

    def _commit_lmdb_transactions(self) -> None:
        for txn in self._txns.values():
            txn.commit()
        self._txns.clear()
        for env in self._envs.values():
            env.sync(True)

    def _close_open_resources(self) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None

    def _published_table(self, table: TensorTable) -> TensorTable:
        try:
            relative_path = table.path.relative_to(self.tmp_generation_dir)
        except ValueError:
            return table
        return TensorTable(
            path=self.generation_dir / relative_path,
            shape=table.shape,
        )


def _validate_inputs(
    *,
    plan: MaterializationPlan,
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_text_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    commit_frequency: int,
) -> None:
    if not plan.split_plans:
        raise ValueError("materialization plan must contain at least one split")
    for split, split_plan in plan.split_plans.items():
        if not str(split).strip():
            raise ValueError("split names must be non-empty")
        if split_plan.num_samples < 0:
            raise ValueError(f"split {split!r} num_samples must be nonnegative, " f"got {split_plan.num_samples}")
    if sum(split.num_samples for split in plan.split_plans.values()) <= 0:
        raise ValueError("materialization plan must contain at least one sample")
    if plan.question_embedding_dim <= 0:
        raise ValueError("question_embedding_dim must be positive")
    if entity_text_embeddings.ndim != 2:
        raise ValueError("entity_text_embeddings must be 2D")
    if int(entity_text_embeddings.size(0)) != entity_catalog.num_text_entities:
        raise ValueError(
            "entity_text_embeddings size mismatch: " f"got {int(entity_text_embeddings.size(0))}, " f"expected {entity_catalog.num_text_entities}"
        )
    if relation_embeddings.ndim != 2:
        raise ValueError("relation_embeddings must be 2D")
    if int(relation_embeddings.size(0)) != len(relation_catalog.relation_labels):
        raise ValueError(
            "relation_embeddings size mismatch: " f"got {int(relation_embeddings.size(0))}, " f"expected {len(relation_catalog.relation_labels)}"
        )
    if int(entity_text_embeddings.size(1)) != plan.question_embedding_dim:
        raise ValueError(
            "entity_text_embeddings dim mismatch: " f"got {int(entity_text_embeddings.size(1))}, " f"expected {plan.question_embedding_dim}"
        )
    if int(relation_embeddings.size(1)) != plan.question_embedding_dim:
        raise ValueError("relation_embeddings dim mismatch: " f"got {int(relation_embeddings.size(1))}, " f"expected {plan.question_embedding_dim}")
    if commit_frequency <= 0:
        raise ValueError("commit_frequency must be positive")


def _validate_question_embeddings(
    *,
    embeddings: torch.Tensor,
    num_samples: int,
    dim: int,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError("question_embeddings must be 2D")
    if int(embeddings.size(0)) != num_samples:
        raise ValueError("question_embeddings row mismatch: " f"got {int(embeddings.size(0))}, expected {num_samples}")
    if int(embeddings.size(1)) != dim:
        raise ValueError("question_embeddings dim mismatch: " f"got {int(embeddings.size(1))}, expected {dim}")


def sample_storage_key(sample: PreparedSample) -> str:
    return f"{sample.dataset}/{sample.split}/{sample.question_id}"


def _save_catalog_artifacts(
    *,
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    entity_metadata_path: Path,
    entity_catalog_path: Path,
    relation_catalog_path: Path,
) -> None:
    entity_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    entity_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    relation_catalog_path.parent.mkdir(parents=True, exist_ok=True)

    entity_catalog.save(entity_catalog_path)
    relation_catalog.save(relation_catalog_path)

    entity_embedding_map = entity_text_row_ids(entity_catalog.entity_text_embedding_ids)
    torch.save(
        {
            "entity_text_row_ids": entity_embedding_map,
            "entity_embedding_map": entity_embedding_map,
            "non_text_entity_mask": (entity_catalog.non_text_entity_mask.bool().contiguous()),
        },
        entity_metadata_path,
    )


def entity_text_row_ids(entity_text_embedding_ids: torch.Tensor) -> torch.Tensor:
    """
    Converts catalog entity text ids to tensor-table row ids.

    Convention inherited from EntityCatalog:
    - 0 means the entity has no text embedding.
    - positive ids are 1-based text embedding ids.

    Returned convention:
    - -1 means no text embedding.
    - nonnegative values are 0-based rows in entity_text_embeddings.
    """
    if entity_text_embedding_ids.ndim != 1:
        raise ValueError("entity_text_embedding_ids must be 1D, " f"got shape={tuple(entity_text_embedding_ids.shape)}.")
    return entity_text_embedding_ids.long().contiguous() - 1


def _preflight_generation(
    *,
    metadata_dir: Path,
    generation_dir: Path,
    overwrite: bool,
) -> None:
    if generation_dir.exists():
        raise FileExistsError(f"materialization generation already exists: {generation_dir}")
    if overwrite:
        return
    active_manifest_path = manifest_path(metadata_dir)
    if active_manifest_path.exists():
        raise FileExistsError(f"materialization manifest already exists: {active_manifest_path}")


def _build_manifest(
    *,
    plan: MaterializationPlan,
    generation_id: str,
    generation_dir: Path,
    metadata_dir: Path,
    entity_metadata_path: Path,
    entity_catalog_path: Path,
    relation_catalog_path: Path,
    entity_text_table: TensorTable,
    relation_table: TensorTable,
    question_tables: Mapping[str, TensorTable],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for split, split_plan in sorted(plan.split_plans.items()):
        question_table = question_tables[split]
        splits[split] = {
            "lmdb_path": _relative_to_metadata(
                generation_dir / "lmdb" / f"{split}.lmdb",
                metadata_dir=metadata_dir,
            ),
            "index_path": _relative_to_metadata(
                generation_dir / "metadata" / f"{split}.index.lmdb",
                metadata_dir=metadata_dir,
            ),
            "num_samples": int(split_plan.num_samples),
            "question_embeddings": _tensor_manifest_entry(
                question_table,
                metadata_dir=metadata_dir,
            ),
        }

    manifest: dict[str, Any] = {
        "generation_id": generation_id,
        "splits": splits,
        "catalogs": {
            "entity_metadata": _relative_to_metadata(
                entity_metadata_path,
                metadata_dir=metadata_dir,
            ),
            "entity_catalog": _relative_to_metadata(
                entity_catalog_path,
                metadata_dir=metadata_dir,
            ),
            "relation_catalog": _relative_to_metadata(
                relation_catalog_path,
                metadata_dir=metadata_dir,
            ),
        },
        "embeddings": {
            "entity_text_embeddings": _tensor_manifest_entry(
                entity_text_table,
                metadata_dir=metadata_dir,
            ),
            "entity_embedding_map": _relative_to_metadata(
                generation_dir / "embeddings" / "entity_embedding_map.pt",
                metadata_dir=metadata_dir,
            ),
            "non_text_entity_mask": _relative_to_metadata(
                generation_dir / "embeddings" / "non_text_entity_mask.pt",
                metadata_dir=metadata_dir,
            ),
            "relation_embeddings": _tensor_manifest_entry(
                relation_table,
                metadata_dir=metadata_dir,
            ),
            "relation_text_labels": _relative_to_metadata(
                generation_dir / "embeddings" / "relation_text_labels.pt",
                metadata_dir=metadata_dir,
            ),
        },
    }
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    return manifest


def _tensor_manifest_entry(
    table: TensorTable,
    *,
    metadata_dir: Path,
) -> dict[str, Any]:
    return manifest_entry(
        path=_relative_to_metadata(table.path, metadata_dir=metadata_dir),
        shape=table.shape,
    )


def _relative_to_metadata(path: Path, *, metadata_dir: Path) -> str:
    try:
        return path.relative_to(metadata_dir).as_posix()
    except ValueError:
        return str(path)


def _publish_generation(
    *,
    tmp_generation_dir: Path,
    generation_dir: Path,
    metadata_dir: Path,
    manifest: dict[str, Any],
) -> None:
    generation_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_generation_dir, generation_dir)

    tmp_manifest_path = _tmp_manifest_path(metadata_dir=metadata_dir, manifest=manifest)
    try:
        _verify_manifest_artifacts(manifest=manifest, metadata_dir=metadata_dir)
        with tmp_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_manifest_path, manifest_path(metadata_dir))
        _fsync_dir(metadata_dir)
    except Exception:
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink()
        raise


def _tmp_manifest_path(*, metadata_dir: Path, manifest: Mapping[str, Any]) -> Path:
    return metadata_dir / f".{manifest['generation_id']}.manifest.tmp"


def _verify_manifest_artifacts(
    *,
    manifest: Mapping[str, Any],
    metadata_dir: Path,
) -> None:
    paths: list[Path] = []

    for split_entry in manifest["splits"].values():
        paths.append(_manifest_path(metadata_dir, split_entry["lmdb_path"]))
        paths.append(_manifest_path(metadata_dir, split_entry["index_path"]))
        question_entry = split_entry["question_embeddings"]
        paths.append(_manifest_path(metadata_dir, question_entry["path"]))

    for entry in manifest["catalogs"].values():
        paths.append(_manifest_path(metadata_dir, entry))

    for entry in manifest["embeddings"].values():
        if isinstance(entry, Mapping):
            paths.append(_manifest_path(metadata_dir, entry["path"]))
        else:
            paths.append(_manifest_path(metadata_dir, entry))

    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"materialization generation is incomplete: {formatted}")


def _manifest_path(metadata_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else metadata_dir / path


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def storage_record_from_sample(sample: PreparedSample) -> dict[str, torch.Tensor]:
    edge_count_indices, edge_count_values = _sparse_nonzero_counts(
        sample.node_target_shortest_path_edge_count_flat,
    )

    return {
        SampleFields.EDGE_INDEX: sample.edge_index.long().contiguous(),
        SampleFields.NODE_ENTITY_CATALOG_IDS: (sample.node_entity_catalog_ids.long().contiguous()),
        SampleFields.EDGE_RELATION_CATALOG_IDS: (sample.edge_relation_catalog_ids.long().contiguous()),
        SampleFields.NUM_NODES: torch.as_tensor(sample.num_nodes, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.as_tensor(sample.num_edges, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: sample.anchor_node_ids.long().contiguous(),
        SampleFields.TARGET_NODE_IDS: sample.target_node_ids.long().contiguous(),
        SampleFields.REACHABLE_TARGET_NODE_IDS: (sample.reachable_target_node_ids.long().contiguous()),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: (sample.anchor_node_forward_distances_flat.long().contiguous()),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: (sample.anchor_node_backward_distances_flat.long().contiguous()),
        SampleFields.NODE_TARGET_DISTANCE: sample.node_target_distance.long().contiguous(),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: (sample.node_target_distances_flat.long().contiguous()),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: (
            sample.node_target_shortest_path_count_flat.float().contiguous()
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: (edge_count_indices.long().contiguous()),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: (edge_count_values.float().contiguous()),
    }


def _sparse_nonzero_counts(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dense = counts.to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    indices = dense.nonzero(as_tuple=False).view(-1).to(dtype=torch.long)
    values = dense.index_select(0, indices).to(dtype=torch.float32).contiguous()
    return indices.contiguous(), values

def _index_map_size(plan: MaterializationPlan) -> int:
    total = sum(split.num_samples for split in plan.split_plans.values())
    return max(1024 * 1024, total * 256)


def _require_table(table: TensorTable | None) -> TensorTable:
    if table is None:
        raise RuntimeError("global tensor table is not initialized")
    return table


__all__ = [
    "MaterializationPlan",
    "SplitPlan",
    "StreamingMaterializer",
    "entity_text_row_ids",
    "sample_storage_key",
    "storage_record_from_sample",
]
