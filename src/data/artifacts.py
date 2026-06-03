from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensor_table import TensorTable
from .tensor_table import from_manifest as tensor_table_from_manifest
from .tensor_table import manifest_entry as tensor_table_manifest_entry

MANIFEST_NAME = "materialization_manifest.json"


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    lmdb: Path
    question_embeddings: TensorTable
    num_samples: int


@dataclass(frozen=True, slots=True)
class MaterializationArtifact:
    generation_id: str
    manifest_path: Path
    materialization_dir: Path
    catalog: Path
    entity_text_semantic_table: TensorTable
    entity_relation_neighborhood_semantic_table: TensorTable
    relation_semantic_table: TensorTable
    question_texts: Path | None
    splits: dict[str, SplitArtifacts]
    provenance: Mapping[str, Any] | None = None

    def require_split(self, split: str) -> SplitArtifacts:
        entry = self.splits.get(split)
        if entry is None:
            available = ", ".join(sorted(self.splits)) or "<none>"
            raise FileNotFoundError(
                f"Split {split!r} is not present in materialization artifact "
                f"{self.manifest_path}. Available splits: {available}."
            )
        return entry


def manifest_path(metadata_dir: str | Path) -> Path:
    return Path(metadata_dir) / MANIFEST_NAME


def load_manifest(metadata_dir: str | Path) -> dict[str, Any] | None:
    return _load_manifest_file(manifest_path(metadata_dir))


def load_materialization_artifact(
    metadata_dir: str | Path,
) -> MaterializationArtifact | None:
    return load_materialization_artifact_from_path(manifest_path(metadata_dir))


def load_materialization_artifact_from_path(
    path: str | Path,
) -> MaterializationArtifact | None:
    manifest_file = Path(path)
    payload = _load_manifest_file(manifest_file)
    if payload is None:
        return None
    return parse_materialization_artifact(payload, manifest_path=manifest_file)


def parse_materialization_artifact(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> MaterializationArtifact:
    path = Path(manifest_path)
    manifest_dir = path.parent
    generation_id = string(payload, "generation_id", path)
    root = resolve_path(
        metadata_dir=manifest_dir,
        value=string(payload, "materialization_dir", path),
    )

    catalogs_payload = mapping(payload, "catalogs", path)
    embeddings_payload = mapping(payload, "embeddings", path)
    if "entity_relation_neighborhood_semantic_table" not in embeddings_payload:
        raise ValueError(
            "Materialization is missing entity_relation_neighborhood_semantic_table. "
            "Re-run preprocessing to rebuild MID relation-neighborhood features."
        )
    debug_payload = mapping(payload, "debug", path) if "debug" in payload else {}
    splits_payload = mapping(payload, "splits", path)

    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError(
            f"Artifact manifest field 'provenance' must be a mapping: {path}"
        )

    splits: dict[str, SplitArtifacts] = {}
    for raw_name, entry in splits_payload.items():
        split_name = str(raw_name).strip()
        if not split_name:
            raise ValueError(f"Artifact manifest split name must be non-empty: {path}")
        split_entry = mapping_value(entry, f"splits.{split_name}", path)
        splits[split_name] = SplitArtifacts(
            lmdb=_resolve_artifact_path(
                root=root,
                value=Path(string(split_entry, "lmdb_path", path)),
            ),
            question_embeddings=tensor_table_from_manifest(
                entry=mapping(split_entry, "question_embeddings", path),
                metadata_dir=root,
            ),
            num_samples=integer(split_entry, "num_samples", path),
        )

    question_texts: Path | None = None
    if "question_texts" in debug_payload:
        question_texts = _resolve_artifact_path(
            root=root,
            value=Path(string(debug_payload, "question_texts", path)),
        )

    return MaterializationArtifact(
        generation_id=generation_id,
        manifest_path=path,
        materialization_dir=root,
        catalog=_resolve_artifact_path(
            root=root,
            value=Path(string(catalogs_payload, "catalog", path)),
        ),
        entity_text_semantic_table=tensor_table_from_manifest(
            entry=mapping(embeddings_payload, "entity_text_semantic_table", path),
            metadata_dir=root,
        ),
        entity_relation_neighborhood_semantic_table=tensor_table_from_manifest(
            entry=mapping(
                embeddings_payload,
                "entity_relation_neighborhood_semantic_table",
                path,
            ),
            metadata_dir=root,
        ),
        relation_semantic_table=tensor_table_from_manifest(
            entry=mapping(embeddings_payload, "relation_semantic_table", path),
            metadata_dir=root,
        ),
        question_texts=question_texts,
        splits=splits,
        provenance=provenance,
    )


def canonicalize_materialization_manifest_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = parse_materialization_artifact(payload, manifest_path=manifest_path)
    canonical: dict[str, Any] = {
        "generation_id": manifest.generation_id,
        "materialization_dir": _path_for_manifest(
            root=manifest.manifest_path.parent,
            value=manifest.materialization_dir,
        ),
        "catalogs": {
            "catalog": _path_for_manifest(
                root=manifest.materialization_dir,
                value=manifest.catalog,
            )
        },
        "embeddings": {
            "entity_text_semantic_table": _tensor_table_entry_for_manifest(
                root=manifest.materialization_dir,
                table=manifest.entity_text_semantic_table,
            ),
            "entity_relation_neighborhood_semantic_table": _tensor_table_entry_for_manifest(
                root=manifest.materialization_dir,
                table=manifest.entity_relation_neighborhood_semantic_table,
            ),
            "relation_semantic_table": _tensor_table_entry_for_manifest(
                root=manifest.materialization_dir,
                table=manifest.relation_semantic_table,
            ),
        },
        "debug": {},
        "splits": {},
    }
    if manifest.provenance is not None:
        canonical["provenance"] = dict(manifest.provenance)
    if manifest.question_texts is not None:
        canonical["debug"]["question_texts"] = _path_for_manifest(
            root=manifest.materialization_dir,
            value=manifest.question_texts,
        )

    splits: dict[str, Any] = {}
    for split_name, split in manifest.splits.items():
        splits[split_name] = {
            "lmdb_path": _path_for_manifest(
                root=manifest.materialization_dir,
                value=split.lmdb,
            ),
            "num_samples": split.num_samples,
            "question_embeddings": _tensor_table_entry_for_manifest(
                root=manifest.materialization_dir,
                table=split.question_embeddings,
            ),
        }
    canonical["splits"] = splits
    return canonical


def resolve_path(*, metadata_dir: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(metadata_dir) / path


def mapping(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Artifact manifest field {key!r} must be a mapping: {path}")
    return value


def mapping_value(value: Any, key: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Artifact manifest field {key!r} must be a mapping: {path}")
    return value


def string(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Artifact manifest field {key!r} must be a non-empty string: {path}"
        )
    return value


def integer(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"Artifact manifest field {key!r} must be an integer: {path}")
    return int(value)


def split_artifacts(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationArtifact,
    split: str,
) -> SplitArtifacts:
    return _materialization_artifact(
        metadata_dir=metadata_dir,
        manifest=manifest,
    ).require_split(split)


def _load_manifest_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid artifact manifest JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"Artifact manifest must be a mapping: {path}")
    return dict(payload)


def _materialization_artifact(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationArtifact,
) -> MaterializationArtifact:
    if isinstance(manifest, MaterializationArtifact):
        return manifest
    if isinstance(manifest, Mapping):
        return parse_materialization_artifact(
            manifest,
            manifest_path=manifest_path(metadata_dir),
        )
    raise TypeError(f"Unsupported manifest type: {type(manifest)!r}")


def _resolve_artifact_path(*, root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def _tensor_table_entry_for_manifest(*, root: Path, table: TensorTable) -> dict[str, Any]:
    return tensor_table_manifest_entry(
        path=_path_for_manifest(root=root, value=table.path),
        shape=table.shape,
    )


def _path_for_manifest(*, root: Path, value: Path) -> str:
    if not value.is_absolute():
        return value.as_posix()
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return str(value)


__all__ = [
    "MANIFEST_NAME",
    "MaterializationArtifact",
    "SplitArtifacts",
    "canonicalize_materialization_manifest_payload",
    "integer",
    "load_manifest",
    "load_materialization_artifact",
    "load_materialization_artifact_from_path",
    "manifest_path",
    "mapping",
    "parse_materialization_artifact",
    "resolve_path",
    "split_artifacts",
    "string",
]


MaterializationManifest = MaterializationArtifact
ResolvedMaterialization = MaterializationArtifact


def load_materialization_manifest(
    metadata_dir: str | Path,
) -> MaterializationArtifact | None:
    return load_materialization_artifact(metadata_dir)


def load_materialization_manifest_from_path(
    path: str | Path,
) -> MaterializationArtifact | None:
    return load_materialization_artifact_from_path(path)


def parse_materialization_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> MaterializationArtifact:
    return parse_materialization_artifact(payload, manifest_path=manifest_path)
