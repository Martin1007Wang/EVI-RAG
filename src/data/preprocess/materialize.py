from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import lmdb
import torch

from src.data.artifacts import manifest_path
from src.data.keys import row_key
from src.data.tensor_table import (
    TensorTable,
    TensorTableWriter,
    manifest_entry,
    write_table,
)
from src.utils.lmdb_utils import serialize_sample

from .catalog import Catalog
from .samples import PreparedSample

log = logging.getLogger(__name__)

CATALOG_NAME = "catalog.pt"
QUESTION_TEXTS_NAME = "question_texts.json"


class Materializer:
    """
    Stream prepared samples into one manifest-addressed materialization.

    The only non-negotiable safety property is atomic publish:
    write into a temporary generation directory first, then publish the manifest last.
    """

    def __init__(
        self,
        *,
        split_counts: Mapping[str, int],
        question_dim: int,
        catalog: Catalog,
        entity_text_semantic_table: torch.Tensor,
        entity_relation_neighborhood_semantic_table: torch.Tensor,
        relation_semantic_table: torch.Tensor,
        metadata_dir: Path,
        overwrite: bool = True,
        map_size_gb: float = 64,
        commit_frequency: int = 1000,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.split_expected = _clean_split_counts(split_counts)
        self.question_dim = int(question_dim)
        self.catalog = catalog
        self.entity_text_semantic_table = entity_text_semantic_table
        self.entity_relation_neighborhood_semantic_table = entity_relation_neighborhood_semantic_table
        self.relation_semantic_table = relation_semantic_table
        self.metadata_dir = Path(metadata_dir)
        self.overwrite = bool(overwrite)
        self.map_size_bytes = int(float(map_size_gb) * 1024**3)
        self.commit_frequency = int(commit_frequency)
        self.provenance = None if provenance is None else dict(provenance)

        _validate_inputs(
            split_counts=self.split_expected,
            question_dim=self.question_dim,
            catalog=self.catalog,
            entity_text_semantic_table=self.entity_text_semantic_table,
            entity_relation_neighborhood_semantic_table=self.entity_relation_neighborhood_semantic_table,
            relation_semantic_table=self.relation_semantic_table,
            map_size_bytes=self.map_size_bytes,
            commit_frequency=self.commit_frequency,
        )

        self.generation_id = uuid.uuid4().hex
        self.tmp_dir = self.metadata_dir / ".materialize_tmp" / self.generation_id
        self.generation_dir = (
            self.metadata_dir / ".materializations" / self.generation_id
        )
        self.tmp_lmdb_dir = self.tmp_dir / "lmdb"
        self.tmp_metadata_dir = self.tmp_dir / "metadata"
        self.tmp_embeddings_dir = self.tmp_dir / "embeddings"

        self._stack: ExitStack | None = None
        self._envs: dict[str, lmdb.Environment] = {}
        self._txns: dict[str, lmdb.Transaction] = {}
        self._uncommitted: dict[str, int] = {}
        self._question_writers: dict[str, TensorTableWriter] = {}
        self._rows_written: dict[str, int] = {split: 0 for split in self.split_expected}
        self._question_text_by_sample_id: dict[str, str] = {}
        self._entity_text_table: TensorTable | None = None
        self._entity_relation_neighborhood_table: TensorTable | None = None
        self._relation_table: TensorTable | None = None
        self._closed = False

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(self._rows_written)

    def __enter__(self) -> Materializer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.abort()
            return
        self.finish()

    def open(self) -> None:
        if self._stack is not None:
            raise RuntimeError("Materializer is already open.")

        _preflight(metadata_dir=self.metadata_dir, overwrite=self.overwrite)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_lmdb_dir.mkdir(parents=True)
        self.tmp_metadata_dir.mkdir(parents=True)
        self.tmp_embeddings_dir.mkdir(parents=True)

        self._stack = ExitStack()
        try:
            self._write_global_artifacts()
            self._open_split_writers()
        except Exception:
            self.abort()
            raise

    def write_chunk(
        self,
        *,
        prepared_samples: list[PreparedSample],
        question_embeddings: torch.Tensor,
    ) -> None:
        if not prepared_samples:
            return
        if self._stack is None:
            raise RuntimeError("Materializer must be opened before writing.")
        _validate_question_embeddings(
            embeddings=question_embeddings,
            num_samples=len(prepared_samples),
            dim=self.question_dim,
        )

        by_split: dict[str, list[int]] = {}
        for row, sample in enumerate(prepared_samples):
            if sample.split not in self.split_expected:
                raise ValueError(
                    f"Unexpected split during materialization: {sample.split!r}."
                )
            by_split.setdefault(sample.split, []).append(row)

        for split, rows in by_split.items():
            row_tensor = torch.as_tensor(rows, dtype=torch.long)
            start = self._question_writers[split].append(
                question_embeddings.index_select(0, row_tensor).contiguous()
            )
            if start != self._rows_written[split]:
                raise RuntimeError(
                    f"question table cursor mismatch for split {split!r}: "
                    f"writer_start={start}, rows_written={self._rows_written[split]}"
                )

            for offset, source_row in enumerate(rows):
                sample = prepared_samples[source_row]
                self._write_sample(split=split, row_idx=start + offset, sample=sample)
                self._question_text_by_sample_id[sample.storage_key()] = str(
                    sample.question
                )

            self._rows_written[split] += len(rows)

    def finish(self) -> None:
        if self._closed:
            return
        if self._stack is None:
            raise RuntimeError("Materializer was not opened.")

        try:
            self._check_split_counts()
            self._commit_lmdb()
            self._close_resources()
            _write_json(
                self.tmp_metadata_dir / QUESTION_TEXTS_NAME,
                dict(sorted(self._question_text_by_sample_id.items())),
            )
            manifest = self._manifest()
            _publish(
                tmp_dir=self.tmp_dir,
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
            sum(self.split_expected.values()),
            sorted(self.split_expected),
        )

    def abort(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self._closed = True

    def _write_global_artifacts(self) -> None:
        self.catalog.save(self.tmp_metadata_dir / CATALOG_NAME)
        self._entity_text_table = write_table(
            self.tmp_embeddings_dir / "entity_text_semantic_table.f32",
            self.entity_text_semantic_table,
        )
        self._entity_relation_neighborhood_table = write_table(
            self.tmp_embeddings_dir / "entity_relation_neighborhood_semantic_table.f32",
            self.entity_relation_neighborhood_semantic_table,
        )
        self._relation_table = write_table(
            self.tmp_embeddings_dir / "relation_semantic_table.f32",
            self.relation_semantic_table,
        )

    def _open_split_writers(self) -> None:
        if self._stack is None:
            raise RuntimeError("Materializer is not open.")

        for split, count in self.split_expected.items():
            env = self._stack.enter_context(
                lmdb.open(
                    str(self.tmp_lmdb_dir / f"{split}.lmdb"),
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
            self._question_writers[split] = self._stack.enter_context(
                TensorTableWriter(
                    self.tmp_embeddings_dir / f"{split}.questions.f32",
                    rows=count,
                    dim=self.question_dim,
                )
            )

    def _write_sample(
        self, *, split: str, row_idx: int, sample: PreparedSample
    ) -> None:
        key = row_key(row_idx)
        inserted = self._txns[split].put(
            key,
            serialize_sample(sample.to_storage_record()),
            overwrite=False,
        )
        if not inserted:
            raise ValueError(f"Duplicate LMDB row key: split={split!r}, row_idx={row_idx}")

        self._uncommitted[split] += 1
        if self._uncommitted[split] >= self.commit_frequency:
            self._txns[split].commit()
            self._txns[split] = self._envs[split].begin(write=True)
            self._uncommitted[split] = 0

    def _check_split_counts(self) -> None:
        for split, expected in self.split_expected.items():
            actual = self._rows_written[split]
            if actual != expected:
                raise ValueError(
                    f"Split {split!r} wrote {actual} samples, expected {expected}."
                )

    def _commit_lmdb(self) -> None:
        for txn in self._txns.values():
            txn.commit()
        self._txns.clear()
        for env in self._envs.values():
            env.sync(True)

    def _close_resources(self) -> None:
        if self._stack is None:
            return
        self._stack.close()
        self._stack = None

    def _manifest(self) -> dict[str, Any]:
        entity_table = _published_table(
            _require_table(self._entity_text_table), self.tmp_dir, self.generation_dir
        )
        entity_relation_neighborhood_table = _published_table(
            _require_table(self._entity_relation_neighborhood_table),
            self.tmp_dir,
            self.generation_dir,
        )
        relation_table = _published_table(
            _require_table(self._relation_table), self.tmp_dir, self.generation_dir
        )
        question_tables = {
            split: _published_table(writer.table, self.tmp_dir, self.generation_dir)
            for split, writer in self._question_writers.items()
        }

        splits = {
            split: {
                "lmdb_path": f"lmdb/{split}.lmdb",
                "num_samples": int(count),
                "question_embeddings": _tensor_entry(
                    question_tables[split], self.generation_dir
                ),
            }
            for split, count in sorted(self.split_expected.items())
        }

        manifest: dict[str, Any] = {
            "generation_id": self.generation_id,
            "materialization_dir": self.generation_dir.relative_to(
                self.metadata_dir
            ).as_posix(),
            "splits": splits,
            "catalogs": {"catalog": f"metadata/{CATALOG_NAME}"},
            "embeddings": {
                "entity_text_semantic_table": _tensor_entry(
                    entity_table, self.generation_dir
                ),
                "entity_relation_neighborhood_semantic_table": _tensor_entry(
                    entity_relation_neighborhood_table, self.generation_dir
                ),
                "relation_semantic_table": _tensor_entry(
                    relation_table, self.generation_dir
                ),
            },
            "debug": {"question_texts": f"metadata/{QUESTION_TEXTS_NAME}"},
        }
        if self.provenance is not None:
            manifest["provenance"] = dict(self.provenance)
        return manifest

def _clean_split_counts(split_counts: Mapping[str, int]) -> dict[str, int]:
    out = {
        str(split): int(count)
        for split, count in split_counts.items()
        if int(count) > 0
    }
    if not out:
        raise ValueError("split_counts must contain at least one positive split.")
    for split in out:
        if not split.strip():
            raise ValueError("split name must be non-empty.")
    return out


def _validate_inputs(
    *,
    split_counts: Mapping[str, int],
    question_dim: int,
    catalog: Catalog,
    entity_text_semantic_table: torch.Tensor,
    entity_relation_neighborhood_semantic_table: torch.Tensor,
    relation_semantic_table: torch.Tensor,
    map_size_bytes: int,
    commit_frequency: int,
) -> None:
    if question_dim <= 0:
        raise ValueError("question_dim must be positive.")
    if map_size_bytes <= 0:
        raise ValueError("map_size_gb must be positive.")
    if commit_frequency <= 0:
        raise ValueError("commit_frequency must be positive.")
    if sum(split_counts.values()) <= 0:
        raise ValueError("split_counts must contain at least one sample.")

    catalog.validate_embeddings(
        entity_text_semantic_table=entity_text_semantic_table,
        entity_relation_neighborhood_semantic_table=entity_relation_neighborhood_semantic_table,
        relation_semantic_table=relation_semantic_table,
    )
    if int(entity_text_semantic_table.size(1)) != question_dim:
        raise ValueError(
            "entity_text_semantic_table dim mismatch: "
            f"got {int(entity_text_semantic_table.size(1))}, expected {question_dim}."
        )
    if int(relation_semantic_table.size(1)) != question_dim:
        raise ValueError(
            "relation_semantic_table dim mismatch: "
            f"got {int(relation_semantic_table.size(1))}, expected {question_dim}."
        )
    if int(entity_relation_neighborhood_semantic_table.size(1)) != question_dim:
        raise ValueError(
            "entity_relation_neighborhood_semantic_table dim mismatch: "
            f"got {int(entity_relation_neighborhood_semantic_table.size(1))}, expected {question_dim}."
        )


def _validate_question_embeddings(
    *, embeddings: torch.Tensor, num_samples: int, dim: int
) -> None:
    if embeddings.ndim != 2:
        raise ValueError("question_embeddings must be 2D.")
    if int(embeddings.size(0)) != num_samples:
        raise ValueError(
            f"question_embeddings row mismatch: got {int(embeddings.size(0))}, expected {num_samples}."
        )
    if int(embeddings.size(1)) != dim:
        raise ValueError(
            f"question_embeddings dim mismatch: got {int(embeddings.size(1))}, expected {dim}."
        )


def _preflight(*, metadata_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    active_manifest = manifest_path(metadata_dir)
    if active_manifest.exists():
        raise FileExistsError(
            f"Materialization manifest already exists: {active_manifest}"
        )


def _publish(
    *,
    tmp_dir: Path,
    generation_dir: Path,
    metadata_dir: Path,
    manifest: dict[str, Any],
) -> None:
    generation_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_dir, generation_dir)
    _verify_manifest(manifest=manifest, metadata_dir=metadata_dir)

    tmp_manifest = metadata_dir / f".{manifest['generation_id']}.manifest.tmp"
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_manifest, manifest_path(metadata_dir))
    _fsync_dir(metadata_dir)


def _verify_manifest(*, manifest: Mapping[str, Any], metadata_dir: Path) -> None:
    root = metadata_dir / str(manifest["materialization_dir"])
    paths: list[Path] = []

    for split in manifest["splits"].values():
        paths.append(root / split["lmdb_path"])
        paths.append(root / split["question_embeddings"]["path"])
    for path in manifest["catalogs"].values():
        paths.append(root / path)
    for item in manifest["embeddings"].values():
        paths.append(root / (item["path"] if isinstance(item, Mapping) else item))
    for item in manifest.get("debug", {}).values():
        paths.append(root / item)

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete materialization: " + ", ".join(missing))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _tensor_entry(table: TensorTable, root_dir: Path) -> dict[str, Any]:
    return manifest_entry(
        path=table.path.relative_to(root_dir).as_posix(),
        shape=table.shape,
    )


def _published_table(
    table: TensorTable, tmp_dir: Path, generation_dir: Path
) -> TensorTable:
    return TensorTable(
        path=generation_dir / table.path.relative_to(tmp_dir), shape=table.shape
    )

def _require_table(table: TensorTable | None) -> TensorTable:
    if table is None:
        raise RuntimeError("Tensor table is not initialized.")
    return table


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "Materializer",
]
