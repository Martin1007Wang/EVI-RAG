from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.artifacts import (
    canonicalize_materialization_manifest_payload,
    load_manifest,
    manifest_path,
)

_LEGACY_FILE_COPIES = (
    ("embeddings/entity_embedding_map.pt", "embeddings/text_row_by_entity_id.pt"),
    ("embeddings/entity_text_embeddings.f32", "embeddings/entity_text_semantic_table.f32"),
    ("embeddings/relation_embeddings.f32", "embeddings/relation_semantic_table.f32"),
)


def main() -> None:
    args = _parse_args()
    metadata_dir = args.metadata_dir.resolve()
    payload = load_manifest(metadata_dir)
    if payload is None:
        raise FileNotFoundError(f"materialization manifest not found: {manifest_path(metadata_dir)}")

    canonical = canonicalize_materialization_manifest_payload(
        payload,
        manifest_path=manifest_path(metadata_dir),
    )
    materialization_dir = _materialization_root(metadata_dir, canonical)
    copied = _ensure_canonical_files(
        materialization_dir=materialization_dir,
        write=args.write,
    )

    if args.write:
        _write_manifest(manifest_path(metadata_dir), canonical)
        mode = "updated"
    else:
        mode = "would update"

    print(f"{mode} manifest: {manifest_path(metadata_dir)}")
    print(f"materialization_dir: {materialization_dir}")
    if copied:
        for src, dst in copied:
            print(f"{'copied' if args.write else 'would copy'}: {src} -> {dst}")
    else:
        print("file rename/copy: no changes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a materialization manifest and backfill canonical artifact filenames.",
    )
    parser.add_argument(
        "metadata_dir",
        type=Path,
        help="Dataset metadata directory containing materialization_manifest.json",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply changes in place. Without this flag the script only reports what would change.",
    )
    return parser.parse_args()


def _materialization_root(metadata_dir: Path, manifest: dict[str, object]) -> Path:
    raw_dir = manifest.get("materialization_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("canonical manifest must include non-empty materialization_dir")
    path = Path(raw_dir)
    return path if path.is_absolute() else metadata_dir / path


def _ensure_canonical_files(
    *,
    materialization_dir: Path,
    write: bool,
) -> list[tuple[Path, Path]]:
    copied: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in _LEGACY_FILE_COPIES:
        src = materialization_dir / src_rel
        dst = materialization_dir / dst_rel
        if dst.exists():
            continue
        if not src.exists():
            continue
        copied.append((src, dst))
        if not write:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _fsync_file(dst)
    return copied


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
