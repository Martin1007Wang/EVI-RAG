from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensor_table import TensorTable


MANIFEST_NAME = "materialization_manifest.json"


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    lmdb: Path
    index: Path
    question_embeddings: TensorTable
    num_samples: int


@dataclass(frozen=True, slots=True)
class TensorTableSpec:
    path: Path
    shape: tuple[int, int]
    dtype: str = "float32"

    def resolve(self, root: Path) -> TensorTable:
        return TensorTable(
            path=_resolve_artifact_path(root=root, value=self.path),
            shape=self.shape,
        )


@dataclass(frozen=True, slots=True)
class CatalogPaths:
    entity_catalog: Path
    entity_metadata: Path
    relation_catalog: Path


@dataclass(frozen=True, slots=True)
class EmbeddingPaths:
    entity_embedding_map: Path
    entity_text_embeddings: TensorTableSpec
    non_text_entity_mask: Path
    relation_embeddings: TensorTableSpec
    relation_text_labels: Path


@dataclass(frozen=True, slots=True)
class SplitPaths:
    lmdb: Path
    index: Path
    question_embeddings: TensorTableSpec
    num_samples: int

    def resolve(self, root: Path) -> SplitArtifacts:
        return SplitArtifacts(
            lmdb=_resolve_artifact_path(root=root, value=self.lmdb),
            index=_resolve_artifact_path(root=root, value=self.index),
            question_embeddings=self.question_embeddings.resolve(root),
            num_samples=self.num_samples,
        )


@dataclass(frozen=True, slots=True)
class ResolvedMaterialization:
    generation_id: str
    manifest_path: Path
    materialization_dir: Path
    entity_catalog: Path
    entity_metadata: Path
    relation_catalog: Path
    entity_embedding_map: Path
    entity_text_embeddings: TensorTable
    non_text_entity_mask: Path
    relation_embeddings: TensorTable
    relation_text_labels: Path
    splits: dict[str, SplitArtifacts]

    def require_split(self, split: str) -> SplitArtifacts:
        entry = self.splits.get(split)
        if entry is None:
            available = ", ".join(sorted(self.splits)) or "<none>"
            raise FileNotFoundError(
                f"Split {split!r} is not present in materialization manifest "
                f"{self.manifest_path}. Available splits: {available}."
            )
        return entry


@dataclass(frozen=True, slots=True)
class MaterializationManifest:
    generation_id: str
    manifest_path: Path
    materialization_dir: Path
    catalogs: CatalogPaths
    embeddings: EmbeddingPaths
    splits: dict[str, SplitPaths]
    provenance: Mapping[str, Any] | None = None

    def resolve(self) -> ResolvedMaterialization:
        root = _resolve_artifact_path(
            root=self.manifest_path.parent,
            value=self.materialization_dir,
        )
        return ResolvedMaterialization(
            generation_id=self.generation_id,
            manifest_path=self.manifest_path,
            materialization_dir=root,
            entity_catalog=_resolve_artifact_path(root=root, value=self.catalogs.entity_catalog),
            entity_metadata=_resolve_artifact_path(root=root, value=self.catalogs.entity_metadata),
            relation_catalog=_resolve_artifact_path(root=root, value=self.catalogs.relation_catalog),
            entity_embedding_map=_resolve_artifact_path(
                root=root,
                value=self.embeddings.entity_embedding_map,
            ),
            entity_text_embeddings=self.embeddings.entity_text_embeddings.resolve(root),
            non_text_entity_mask=_resolve_artifact_path(
                root=root,
                value=self.embeddings.non_text_entity_mask,
            ),
            relation_embeddings=self.embeddings.relation_embeddings.resolve(root),
            relation_text_labels=_resolve_artifact_path(
                root=root,
                value=self.embeddings.relation_text_labels,
            ),
            splits={
                name: split.resolve(root)
                for name, split in self.splits.items()
            },
        )


def manifest_path(metadata_dir: str | Path) -> Path:
    return Path(metadata_dir) / MANIFEST_NAME


def load_manifest(metadata_dir: str | Path) -> dict[str, Any] | None:
    path = manifest_path(metadata_dir)
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


def load_materialization_manifest(
    metadata_dir: str | Path,
) -> MaterializationManifest | None:
    return load_materialization_manifest_from_path(manifest_path(metadata_dir))


def load_materialization_manifest_from_path(
    path: str | Path,
) -> MaterializationManifest | None:
    manifest_file = Path(path)
    if not manifest_file.exists():
        return None

    payload = load_manifest(manifest_file.parent)
    if payload is None:
        return None
    return parse_materialization_manifest(payload, manifest_path=manifest_file)


def parse_materialization_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> MaterializationManifest:
    path = Path(manifest_path)
    manifest_dir = path.parent

    generation_id = string(payload, "generation_id", path)
    materialization_dir, legacy_prefix = _materialization_dir(
        payload=payload,
        manifest_dir=manifest_dir,
        generation_id=generation_id,
        path=path,
    )

    catalogs_payload = mapping(payload, "catalogs", path)
    embeddings_payload = mapping(payload, "embeddings", path)
    splits_payload = mapping(payload, "splits", path)

    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError(f"Artifact manifest field 'provenance' must be a mapping: {path}")

    splits: dict[str, SplitPaths] = {}
    for raw_name, entry in splits_payload.items():
        split_name = str(raw_name).strip()
        if not split_name:
            raise ValueError(f"Artifact manifest split name must be non-empty: {path}")
        if not isinstance(entry, Mapping):
            raise TypeError(f"Artifact manifest split {split_name!r} must be a mapping: {path}")

        splits[split_name] = SplitPaths(
            lmdb=_manifest_relative_path(
                value=string(entry, "lmdb_path", path),
                legacy_prefix=legacy_prefix,
            ),
            index=_manifest_relative_path(
                value=string(entry, "index_path", path),
                legacy_prefix=legacy_prefix,
            ),
            question_embeddings=_tensor_table_spec(
                entry=mapping(entry, "question_embeddings", path),
                path=path,
                legacy_prefix=legacy_prefix,
            ),
            num_samples=integer(entry, "num_samples", path),
        )

    return MaterializationManifest(
        generation_id=generation_id,
        manifest_path=path,
        materialization_dir=materialization_dir,
        catalogs=CatalogPaths(
            entity_catalog=_manifest_relative_path(
                value=string(catalogs_payload, "entity_catalog", path),
                legacy_prefix=legacy_prefix,
            ),
            entity_metadata=_manifest_relative_path(
                value=string(catalogs_payload, "entity_metadata", path),
                legacy_prefix=legacy_prefix,
            ),
            relation_catalog=_manifest_relative_path(
                value=string(catalogs_payload, "relation_catalog", path),
                legacy_prefix=legacy_prefix,
            ),
        ),
        embeddings=EmbeddingPaths(
            entity_embedding_map=_manifest_relative_path(
                value=string(embeddings_payload, "entity_embedding_map", path),
                legacy_prefix=legacy_prefix,
            ),
            entity_text_embeddings=_tensor_table_spec(
                entry=mapping(embeddings_payload, "entity_text_embeddings", path),
                path=path,
                legacy_prefix=legacy_prefix,
            ),
            non_text_entity_mask=_manifest_relative_path(
                value=string(embeddings_payload, "non_text_entity_mask", path),
                legacy_prefix=legacy_prefix,
            ),
            relation_embeddings=_tensor_table_spec(
                entry=mapping(embeddings_payload, "relation_embeddings", path),
                path=path,
                legacy_prefix=legacy_prefix,
            ),
            relation_text_labels=_manifest_relative_path(
                value=string(embeddings_payload, "relation_text_labels", path),
                legacy_prefix=legacy_prefix,
            ),
        ),
        splits=splits,
        provenance=provenance,
    )


def split_artifacts(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationManifest | ResolvedMaterialization,
    split: str,
) -> SplitArtifacts:
    return _resolved_materialization(
        metadata_dir=metadata_dir,
        manifest=manifest,
    ).require_split(split)


def artifact_path(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationManifest | ResolvedMaterialization,
    section: str,
    key: str,
) -> Path:
    resolved = _resolved_materialization(
        metadata_dir=metadata_dir,
        manifest=manifest,
    )
    value: Path | TensorTable
    if section == "catalogs":
        value = {
            "entity_catalog": resolved.entity_catalog,
            "entity_metadata": resolved.entity_metadata,
            "relation_catalog": resolved.relation_catalog,
        }.get(key)  # type: ignore[assignment]
    elif section == "embeddings":
        value = {
            "entity_embedding_map": resolved.entity_embedding_map,
            "entity_text_embeddings": resolved.entity_text_embeddings,
            "non_text_entity_mask": resolved.non_text_entity_mask,
            "relation_embeddings": resolved.relation_embeddings,
            "relation_text_labels": resolved.relation_text_labels,
        }.get(key)  # type: ignore[assignment]
    else:
        raise KeyError(f"Unknown artifact manifest section {section!r}.")

    if value is None:
        raise KeyError(f"Unknown artifact manifest key {section}.{key}.")
    if isinstance(value, TensorTable):
        return value.path
    return value


def tensor_table(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationManifest | ResolvedMaterialization,
    section: str,
    key: str,
) -> TensorTable:
    resolved = _resolved_materialization(
        metadata_dir=metadata_dir,
        manifest=manifest,
    )
    if section != "embeddings":
        raise KeyError(f"Tensor tables are only available under 'embeddings', got {section!r}.")
    table = {
        "entity_text_embeddings": resolved.entity_text_embeddings,
        "relation_embeddings": resolved.relation_embeddings,
    }.get(key)
    if table is None:
        raise KeyError(f"Unknown tensor table key embeddings.{key}.")
    return table


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


def _resolved_materialization(
    *,
    metadata_dir: str | Path,
    manifest: Mapping[str, Any] | MaterializationManifest | ResolvedMaterialization,
) -> ResolvedMaterialization:
    if isinstance(manifest, ResolvedMaterialization):
        return manifest
    if isinstance(manifest, MaterializationManifest):
        return manifest.resolve()
    if isinstance(manifest, Mapping):
        return parse_materialization_manifest(
            manifest,
            manifest_path=manifest_path(metadata_dir),
        ).resolve()
    raise TypeError(f"Unsupported manifest type: {type(manifest)!r}")


def _materialization_dir(
    *,
    payload: Mapping[str, Any],
    manifest_dir: Path,
    generation_id: str,
    path: Path,
) -> tuple[Path, Path | None]:
    raw_dir = payload.get("materialization_dir")
    if raw_dir in (None, ""):
        legacy_prefix = Path(".materializations") / generation_id
        return manifest_dir / legacy_prefix, legacy_prefix
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError(
            f"Artifact manifest field 'materialization_dir' must be a non-empty string: {path}"
        )
    return resolve_path(metadata_dir=manifest_dir, value=raw_dir), None


def _manifest_relative_path(*, value: str, legacy_prefix: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or legacy_prefix is None:
        return path
    try:
        return path.relative_to(legacy_prefix)
    except ValueError:
        return path


def _tensor_table_spec(
    *,
    entry: Mapping[str, Any],
    path: Path,
    legacy_prefix: Path | None,
) -> TensorTableSpec:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("tensor table manifest entry requires non-empty 'path'")

    dtype = str(entry.get("dtype", "")).strip()
    if dtype != "float32":
        raise ValueError(f"unsupported tensor table dtype: {dtype!r}")

    raw_shape = entry.get("shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 2
        or not all(isinstance(value, int) for value in raw_shape)
    ):
        raise ValueError("tensor table manifest entry requires shape [rows, dim]")

    rows, dim = int(raw_shape[0]), int(raw_shape[1])
    if rows < 0:
        raise ValueError(f"tensor table rows must be nonnegative: {path}")
    if dim <= 0:
        raise ValueError(f"tensor table dim must be positive: {path}")

    return TensorTableSpec(
        path=_manifest_relative_path(value=raw_path, legacy_prefix=legacy_prefix),
        dtype=dtype,
        shape=(rows, dim),
    )


def _resolve_artifact_path(*, root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


__all__ = [
    "CatalogPaths",
    "EmbeddingPaths",
    "MANIFEST_NAME",
    "MaterializationManifest",
    "ResolvedMaterialization",
    "SplitArtifacts",
    "SplitPaths",
    "TensorTableSpec",
    "artifact_path",
    "integer",
    "load_manifest",
    "load_materialization_manifest",
    "load_materialization_manifest_from_path",
    "manifest_path",
    "mapping",
    "parse_materialization_manifest",
    "resolve_path",
    "split_artifacts",
    "string",
    "tensor_table",
]
