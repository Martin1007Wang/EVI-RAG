from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.artifacts import parse_materialization_manifest
from src.data.split_index import SplitIndexWriter
from src.data.tensor_table import write_table
from src.training.config import build_training_data_config


def _write_split_index(path: Path, sample_ids: list[str]) -> None:
    with SplitIndexWriter(path, map_size=1024 * 1024, commit_frequency=10) as writer:
        for row, sample_id in enumerate(sample_ids):
            writer.put(row, sample_id)


def test_build_training_data_config_resolves_materialization_paths(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    generation_id = "gen123"
    materialization_dir = metadata_dir / ".materializations" / generation_id
    embeddings_dir = materialization_dir / "embeddings"
    materialization_metadata_dir = materialization_dir / "metadata"
    lmdb_dir = materialization_dir / "lmdb"

    embeddings_dir.mkdir(parents=True)
    materialization_metadata_dir.mkdir(parents=True)
    (lmdb_dir / "train.lmdb").mkdir(parents=True)
    (lmdb_dir / "validation.lmdb").mkdir(parents=True)
    (lmdb_dir / "test.lmdb").mkdir(parents=True)

    entity_text_embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        dim=1,
    )
    relation_embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        dim=1,
    )
    write_table(embeddings_dir / "entity_text_embeddings.f32", entity_text_embeddings)
    write_table(embeddings_dir / "relation_embeddings.f32", relation_embeddings)
    write_table(
        embeddings_dir / "train.questions.f32",
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )
    write_table(
        embeddings_dir / "validation.questions.f32",
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    )
    write_table(
        embeddings_dir / "test.questions.f32",
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )

    torch.save(
        {
            "entity_embedding_map": torch.tensor([0, 1], dtype=torch.long),
            "entity_text_row_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        materialization_metadata_dir / "entity_metadata.pt",
    )
    torch.save({"dummy": True}, materialization_metadata_dir / "entity_catalog.pt")
    torch.save({"dummy": True}, materialization_metadata_dir / "relation_catalog.pt")

    _write_split_index(materialization_metadata_dir / "train.index.lmdb", ["train-0"])
    _write_split_index(
        materialization_metadata_dir / "validation.index.lmdb",
        ["validation-0"],
    )
    _write_split_index(materialization_metadata_dir / "test.index.lmdb", ["test-0"])

    manifest = {
        "generation_id": generation_id,
        "catalogs": {
            "entity_catalog": ".materializations/gen123/metadata/entity_catalog.pt",
            "entity_metadata": ".materializations/gen123/metadata/entity_metadata.pt",
            "relation_catalog": ".materializations/gen123/metadata/relation_catalog.pt",
        },
        "embeddings": {
            "entity_embedding_map": ".materializations/gen123/embeddings/entity_embedding_map.pt",
            "entity_text_embeddings": {
                "path": ".materializations/gen123/embeddings/entity_text_embeddings.f32",
                "shape": [2, 2],
                "dtype": "float32",
            },
            "non_text_entity_mask": ".materializations/gen123/embeddings/non_text_entity_mask.pt",
            "relation_embeddings": {
                "path": ".materializations/gen123/embeddings/relation_embeddings.f32",
                "shape": [1, 2],
                "dtype": "float32",
            },
            "relation_text_labels": ".materializations/gen123/embeddings/relation_text_labels.pt",
        },
        "splits": {
            "train": {
                "lmdb_path": ".materializations/gen123/lmdb/train.lmdb",
                "index_path": ".materializations/gen123/metadata/train.index.lmdb",
                "num_samples": 1,
                "question_embeddings": {
                    "path": ".materializations/gen123/embeddings/train.questions.f32",
                    "shape": [1, 2],
                    "dtype": "float32",
                },
            },
            "validation": {
                "lmdb_path": ".materializations/gen123/lmdb/validation.lmdb",
                "index_path": ".materializations/gen123/metadata/validation.index.lmdb",
                "num_samples": 1,
                "question_embeddings": {
                    "path": ".materializations/gen123/embeddings/validation.questions.f32",
                    "shape": [1, 2],
                    "dtype": "float32",
                },
            },
            "test": {
                "lmdb_path": ".materializations/gen123/lmdb/test.lmdb",
                "index_path": ".materializations/gen123/metadata/test.index.lmdb",
                "num_samples": 1,
                "question_embeddings": {
                    "path": ".materializations/gen123/embeddings/test.questions.f32",
                    "shape": [1, 2],
                    "dtype": "float32",
                },
            },
        },
    }
    torch.save(torch.tensor([0, 1], dtype=torch.long), embeddings_dir / "entity_embedding_map.pt")
    torch.save(torch.tensor([False, False], dtype=torch.bool), embeddings_dir / "non_text_entity_mask.pt")
    torch.save(["r"], embeddings_dir / "relation_text_labels.pt")

    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    cfg = OmegaConf.create(
        {
            "dataset": {
                "paths": {
                    "metadata_dir": str(metadata_dir),
                }
            },
            "datamodule": {
                "batch_size": 4,
                "num_workers": 2,
                "eval_batch_size": 8,
                "eval_num_workers": 1,
                "pin_memory": True,
                "train_shuffle": False,
                "drop_last": False,
                "eval_drop_last": False,
                "lmdb_readahead": True,
                "max_readers": 32,
                "splits": {
                    "train": "train",
                    "validation": "validation",
                    "test": "test",
                },
            },
        }
    )

    data_config = build_training_data_config(cfg)

    assert data_config.materialization.materialization_dir == materialization_dir
    assert data_config.materialization.entity_metadata == materialization_metadata_dir / "entity_metadata.pt"
    assert data_config.materialization.require_split("train").lmdb == lmdb_dir / "train.lmdb"
    assert data_config.batch_size == 4
    assert data_config.eval_batch_size == 8
    assert data_config.train_shuffle is False


def test_parse_materialization_manifest_normalizes_legacy_relative_paths(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    manifest_path = metadata_dir / "materialization_manifest.json"
    payload = {
        "generation_id": "abc",
        "catalogs": {
            "entity_catalog": ".materializations/abc/metadata/entity_catalog.pt",
            "entity_metadata": ".materializations/abc/metadata/entity_metadata.pt",
            "relation_catalog": ".materializations/abc/metadata/relation_catalog.pt",
        },
        "embeddings": {
            "entity_embedding_map": ".materializations/abc/embeddings/entity_embedding_map.pt",
            "entity_text_embeddings": {
                "path": ".materializations/abc/embeddings/entity_text_embeddings.f32",
                "shape": [2, 2],
                "dtype": "float32",
            },
            "non_text_entity_mask": ".materializations/abc/embeddings/non_text_entity_mask.pt",
            "relation_embeddings": {
                "path": ".materializations/abc/embeddings/relation_embeddings.f32",
                "shape": [1, 2],
                "dtype": "float32",
            },
            "relation_text_labels": ".materializations/abc/embeddings/relation_text_labels.pt",
        },
        "splits": {
            "train": {
                "lmdb_path": ".materializations/abc/lmdb/train.lmdb",
                "index_path": ".materializations/abc/metadata/train.index.lmdb",
                "num_samples": 1,
                "question_embeddings": {
                    "path": ".materializations/abc/embeddings/train.questions.f32",
                    "shape": [1, 2],
                    "dtype": "float32",
                },
            }
        },
    }

    manifest = parse_materialization_manifest(payload, manifest_path=manifest_path)
    resolved = manifest.resolve()

    assert manifest.materialization_dir == metadata_dir / ".materializations" / "abc"
    assert manifest.catalogs.entity_metadata == Path("metadata/entity_metadata.pt")
    assert manifest.embeddings.entity_text_embeddings.path == Path(
        "embeddings/entity_text_embeddings.f32"
    )
    assert resolved.entity_metadata == metadata_dir / ".materializations" / "abc" / "metadata" / "entity_metadata.pt"
    assert resolved.require_split("train").lmdb == metadata_dir / ".materializations" / "abc" / "lmdb" / "train.lmdb"
