from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.io.parquet_io import write_questions
from src.data.schema.constants import QuestionFields


def _base_question_row() -> dict[str, object]:
    return {
        QuestionFields.QUESTION_UID: "q_0",
        QuestionFields.DATASET: "unit",
        QuestionFields.SPLIT: "train",
        QuestionFields.KB: "kb",
        QuestionFields.QUESTION: "who directed interstellar?",
        QuestionFields.SEED_ENTITY_IDS: [1],
        QuestionFields.ANSWER_ENTITY_IDS: [2],
        QuestionFields.ANSWER_TEXTS: ["christopher nolan"],
        QuestionFields.GRAPH_ID: "g_0",
        QuestionFields.QUESTION_EMB: [0.1, 0.2, 0.3],
    }


def test_write_questions_with_token_context_roundtrip(tmp_path: Path) -> None:
    out_path = tmp_path / "questions.parquet"
    row = _base_question_row()
    row[QuestionFields.QUESTION_CTX] = [[0.1, 0.0], [0.0, 0.1]]
    row[QuestionFields.QUESTION_CTX_MASK] = [True, False]

    write_questions([row], out_path)

    table = pq.read_table(out_path)
    assert QuestionFields.QUESTION_CTX in table.schema.names
    assert QuestionFields.QUESTION_CTX_MASK in table.schema.names
    payload = table.to_pylist()[0]
    assert len(payload[QuestionFields.QUESTION_CTX]) == 2
    assert payload[QuestionFields.QUESTION_CTX_MASK] == [True, False]


def test_write_questions_rejects_unpaired_context_mask(tmp_path: Path) -> None:
    out_path = tmp_path / "questions.parquet"
    row = _base_question_row()
    row[QuestionFields.QUESTION_CTX] = [[0.1, 0.0], [0.0, 0.1]]

    with pytest.raises(ValueError, match="question_ctx_mask missing in questions while include_question_ctx is enabled"):
        write_questions([row], out_path)


def test_write_questions_rejects_context_mask_length_mismatch(tmp_path: Path) -> None:
    out_path = tmp_path / "questions.parquet"
    row = _base_question_row()
    row[QuestionFields.QUESTION_CTX] = [[0.1, 0.0], [0.0, 0.1]]
    row[QuestionFields.QUESTION_CTX_MASK] = [True]

    with pytest.raises(ValueError, match="question_ctx/question_ctx_mask length mismatch"):
        write_questions([row], out_path)
