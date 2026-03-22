from __future__ import annotations

import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.io import raw_loader


def _write_split(path: Path, value: int) -> None:
    table = pa.table({"value": [value]})
    pq.write_table(table, path)


def test_import_hf_datasets_module_reuses_single_module_after_tqdm_lock(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(raw_loader, "_HF_DATASETS_MODULE", None)

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    _write_split(train_path, 1)
    _write_split(validation_path, 2)

    hf_datasets = raw_loader._import_hf_datasets_module()
    had_progress_controls = hasattr(hf_datasets, "disable_progress_bars") and hasattr(
        hf_datasets, "enable_progress_bars"
    )
    if had_progress_controls:
        hf_datasets.disable_progress_bars()

    try:
        train_ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"train": [str(train_path)]},
            split="train",
        )
        tqdm.get_lock()
        reused_module = raw_loader._import_hf_datasets_module()
        validation_ds = reused_module.load_dataset(
            "parquet",
            data_files={"validation": [str(validation_path)]},
            split="validation",
        )
    finally:
        if had_progress_controls:
            hf_datasets.enable_progress_bars()

    assert reused_module is hf_datasets
    assert len(train_ds) == 1
    assert len(validation_ds) == 1


def test_import_hf_datasets_module_replaces_local_shadow(monkeypatch) -> None:
    monkeypatch.setattr(raw_loader, "_HF_DATASETS_MODULE", None)

    shadow_module = types.ModuleType("datasets")
    shadow_module.__file__ = str(
        Path(raw_loader.__file__).resolve().parents[2] / "datasets" / "__init__.py"
    )
    monkeypatch.setitem(sys.modules, "datasets", shadow_module)

    hf_datasets = raw_loader._import_hf_datasets_module()

    assert hf_datasets is not shadow_module
    assert sys.modules["datasets"] is hf_datasets
    assert not raw_loader._is_local_datasets_shadow(hf_datasets)


def test_row_to_sample_resolves_plain_labels_after_qid_lookup() -> None:
    column_map = {
        "graph_field": "graph",
        "q_entity_field": "q_entity",
        "a_entity_field": "a_entity",
        "answer_text_field": "answer_text",
        "question_id_field": "question_id",
        "question_field": "question",
    }
    row = {
        "graph": [
            ["Alpha Entity", "rel", "Beta Entity"],
            ["Alpha Entity (Q1)", "rel", "Beta Entity (Q2)"],
        ],
        "q_entity": ["Alpha Entity"],
        "a_entity": ["Beta Entity"],
        "answer_text": ["beta"],
        "question_id": "q1",
        "question": "Which beta?",
    }

    sample = raw_loader._row_to_sample(
        row,
        dataset="unit",
        split="train",
        kb="freebase",
        column_map=column_map,
        entity_normalization="qid_in_parentheses",
    )

    assert sample.graph == [("Q1", "rel", "Q2"), ("Q1", "rel", "Q2")]
    assert sample.q_entity == ["Q1"]
    assert sample.a_entity == ["Q2"]
