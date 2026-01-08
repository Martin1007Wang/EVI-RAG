from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema.constants import (
    EmbeddingVocabFields,
    EntityVocabFields,
    GraphFields,
    QuestionFields,
    RelationVocabFields,
    _EMBEDDING_VOCAB_FIELDS,
    _ENTITY_VOCAB_FIELDS,
    _GRAPH_PARQUET_FIELDS,
    _QUESTION_PARQUET_FIELDS,
    _QUESTION_PARQUET_REQUIRED_FIELDS,
    _RELATION_VOCAB_FIELDS,
)
from src.data.schema.types import GraphRecord


def _validate_graph_record_schema() -> None:
    record_fields = set(GraphRecord.__dataclass_fields__)
    allowed_fields = set(_GRAPH_PARQUET_FIELDS)
    if record_fields != allowed_fields:
        unexpected = sorted(record_fields - allowed_fields)
        missing = sorted(allowed_fields - record_fields)
        raise ValueError(
            "GraphRecord schema mismatch; update _GRAPH_PARQUET_FIELDS before writing. "
            f"missing={missing} unexpected={unexpected}"
        )


def _validate_row_fields(rows: List[Dict[str, object]], *, allowed_fields: List[str], context: str) -> None:
    if not rows:
        return
    allowed = set(allowed_fields)
    for idx, row in enumerate(rows):
        extra = sorted(set(row) - allowed)
        if extra:
            raise ValueError(f"{context} row {idx} contains unexpected fields: {extra}")


def _validate_question_rows(rows: List[Dict[str, object]], *, include_question_emb: bool) -> None:
    _validate_row_fields(rows, allowed_fields=list(_QUESTION_PARQUET_FIELDS), context="questions")
    if not rows:
        return
    required = set(_QUESTION_PARQUET_REQUIRED_FIELDS)
    for idx, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"questions row {idx} missing required fields: {missing}")
    has_emb = [QuestionFields.QUESTION_EMB in row for row in rows]
    if include_question_emb:
        if not all(has_emb):
            raise ValueError("question_emb missing in questions while include_question_emb is enabled.")
    else:
        if any(has_emb):
            raise ValueError("question_emb present in questions while include_question_emb is disabled.")


def _validate_vocab_rows(rows: List[Dict[str, object]], *, allowed_fields: List[str], context: str) -> None:
    _validate_row_fields(rows, allowed_fields=allowed_fields, context=context)


@dataclass
class ParquetDatasetWriter:
    out_dir: Path
    graphs: List[GraphRecord] = field(default_factory=list)
    questions: List[Dict[str, object]] = field(default_factory=list)
    graph_writer: pq.ParquetWriter | None = None
    question_writer: pq.ParquetWriter | None = None
    include_question_emb: bool = False

    def append(self, graph: GraphRecord, question: Dict[str, object]) -> None:
        self.graphs.append(graph)
        self.questions.append(question)

    def flush(self) -> None:
        if self.graphs:
            _validate_graph_record_schema()
            table = pa.table(
                {
                    GraphFields.GRAPH_ID: pa.array([g.graph_id for g in self.graphs], type=pa.string()),
                    GraphFields.NODE_IDS: pa.array([g.node_entity_ids for g in self.graphs], type=pa.list_(pa.int64())),
                    GraphFields.NODE_EMBED_IDS: pa.array(
                        [g.node_embedding_ids for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    GraphFields.NODE_LABELS: pa.array([g.node_labels for g in self.graphs], type=pa.list_(pa.string())),
                    GraphFields.NODE_TYPE_COUNTS: pa.array(
                        [g.node_type_counts for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                    GraphFields.NODE_TYPE_IDS: pa.array([g.node_type_ids for g in self.graphs], type=pa.list_(pa.int64())),
                    GraphFields.EDGE_SRC: pa.array([g.edge_src for g in self.graphs], type=pa.list_(pa.int64())),
                    GraphFields.EDGE_DST: pa.array([g.edge_dst for g in self.graphs], type=pa.list_(pa.int64())),
                    GraphFields.EDGE_REL_IDS: pa.array(
                        [g.edge_relation_ids for g in self.graphs], type=pa.list_(pa.int64())
                    ),
                }
            )
            if self.graph_writer is None:
                self.graph_writer = pq.ParquetWriter(self.out_dir / "graphs.parquet", table.schema, compression="zstd")
            self.graph_writer.write_table(table)
            self.graphs = []

        if self.questions:
            _validate_question_rows(self.questions, include_question_emb=self.include_question_emb)
            table_q_data = {
                QuestionFields.QUESTION_UID: pa.array(
                    [row[QuestionFields.QUESTION_UID] for row in self.questions], type=pa.string()
                ),
                QuestionFields.DATASET: pa.array(
                    [row[QuestionFields.DATASET] for row in self.questions], type=pa.string()
                ),
                QuestionFields.SPLIT: pa.array([row[QuestionFields.SPLIT] for row in self.questions], type=pa.string()),
                QuestionFields.KB: pa.array([row[QuestionFields.KB] for row in self.questions], type=pa.string()),
                QuestionFields.QUESTION: pa.array(
                    [row[QuestionFields.QUESTION] for row in self.questions], type=pa.string()
                ),
                QuestionFields.SEED_ENTITY_IDS: pa.array(
                    [row[QuestionFields.SEED_ENTITY_IDS] for row in self.questions], type=pa.list_(pa.int64())
                ),
                QuestionFields.ANSWER_ENTITY_IDS: pa.array(
                    [row[QuestionFields.ANSWER_ENTITY_IDS] for row in self.questions], type=pa.list_(pa.int64())
                ),
                QuestionFields.ANSWER_TEXTS: pa.array(
                    [row[QuestionFields.ANSWER_TEXTS] for row in self.questions], type=pa.list_(pa.string())
                ),
                QuestionFields.GRAPH_ID: pa.array(
                    [row[QuestionFields.GRAPH_ID] for row in self.questions], type=pa.string()
                ),
            }
            if self.include_question_emb:
                question_embs = [row[QuestionFields.QUESTION_EMB] for row in self.questions]
                table_q_data[QuestionFields.QUESTION_EMB] = pa.array(question_embs, type=pa.list_(pa.float32()))
            table_q = pa.table(table_q_data)
            if self.question_writer is None:
                self.question_writer = pq.ParquetWriter(
                    self.out_dir / "questions.parquet", table_q.schema, compression="zstd"
                )
            self.question_writer.write_table(table_q)
            self.questions = []

    def close(self) -> None:
        self.flush()
        if self.graph_writer is not None:
            self.graph_writer.close()
        if self.question_writer is not None:
            self.question_writer.close()


def write_graphs(graphs: List[GraphRecord], output_path: Path) -> None:
    _validate_graph_record_schema()
    table = pa.table(
        {
            GraphFields.GRAPH_ID: pa.array([g.graph_id for g in graphs], type=pa.string()),
            GraphFields.NODE_IDS: pa.array([g.node_entity_ids for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.NODE_EMBED_IDS: pa.array([g.node_embedding_ids for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.NODE_LABELS: pa.array([g.node_labels for g in graphs], type=pa.list_(pa.string())),
            GraphFields.NODE_TYPE_COUNTS: pa.array([g.node_type_counts for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.NODE_TYPE_IDS: pa.array([g.node_type_ids for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.EDGE_SRC: pa.array([g.edge_src for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.EDGE_DST: pa.array([g.edge_dst for g in graphs], type=pa.list_(pa.int64())),
            GraphFields.EDGE_REL_IDS: pa.array([g.edge_relation_ids for g in graphs], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_questions(rows: List[Dict[str, object]], output_path: Path) -> None:
    has_emb = [QuestionFields.QUESTION_EMB in row for row in rows]
    include_question_emb = any(has_emb)
    _validate_question_rows(rows, include_question_emb=include_question_emb)
    table_data = {
        QuestionFields.QUESTION_UID: pa.array([row[QuestionFields.QUESTION_UID] for row in rows], type=pa.string()),
        QuestionFields.DATASET: pa.array([row[QuestionFields.DATASET] for row in rows], type=pa.string()),
        QuestionFields.SPLIT: pa.array([row[QuestionFields.SPLIT] for row in rows], type=pa.string()),
        QuestionFields.KB: pa.array([row[QuestionFields.KB] for row in rows], type=pa.string()),
        QuestionFields.QUESTION: pa.array([row[QuestionFields.QUESTION] for row in rows], type=pa.string()),
        QuestionFields.SEED_ENTITY_IDS: pa.array(
            [row[QuestionFields.SEED_ENTITY_IDS] for row in rows], type=pa.list_(pa.int64())
        ),
        QuestionFields.ANSWER_ENTITY_IDS: pa.array(
            [row[QuestionFields.ANSWER_ENTITY_IDS] for row in rows], type=pa.list_(pa.int64())
        ),
        QuestionFields.ANSWER_TEXTS: pa.array(
            [row[QuestionFields.ANSWER_TEXTS] for row in rows], type=pa.list_(pa.string())
        ),
        QuestionFields.GRAPH_ID: pa.array([row[QuestionFields.GRAPH_ID] for row in rows], type=pa.string()),
    }
    if include_question_emb:
        table_data[QuestionFields.QUESTION_EMB] = pa.array(
            [row[QuestionFields.QUESTION_EMB] for row in rows], type=pa.list_(pa.float32())
        )
    table = pa.table(table_data)
    pq.write_table(table, output_path, compression="zstd")


def write_entity_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    _validate_vocab_rows(vocab_records, allowed_fields=list(_ENTITY_VOCAB_FIELDS), context="entity_vocab")
    table = pa.table(
        {
            EntityVocabFields.ENTITY_ID: pa.array([rec[EntityVocabFields.ENTITY_ID] for rec in vocab_records], type=pa.int64()),
            EntityVocabFields.KB: pa.array([rec[EntityVocabFields.KB] for rec in vocab_records], type=pa.string()),
            EntityVocabFields.KG_ID: pa.array([rec[EntityVocabFields.KG_ID] for rec in vocab_records], type=pa.string()),
            EntityVocabFields.LABEL: pa.array(
                [rec.get(EntityVocabFields.LABEL, "") for rec in vocab_records], type=pa.string()
            ),
            EntityVocabFields.IS_TEXT: pa.array(
                [rec[EntityVocabFields.IS_TEXT] for rec in vocab_records], type=pa.bool_()
            ),
            EntityVocabFields.EMBEDDING_ID: pa.array(
                [rec[EntityVocabFields.EMBEDDING_ID] for rec in vocab_records], type=pa.int64()
            ),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_embedding_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    _validate_vocab_rows(vocab_records, allowed_fields=list(_EMBEDDING_VOCAB_FIELDS), context="embedding_vocab")
    table = pa.table(
        {
            EmbeddingVocabFields.EMBEDDING_ID: pa.array(
                [rec[EmbeddingVocabFields.EMBEDDING_ID] for rec in vocab_records], type=pa.int64()
            ),
            EmbeddingVocabFields.KB: pa.array([rec[EmbeddingVocabFields.KB] for rec in vocab_records], type=pa.string()),
            EmbeddingVocabFields.KG_ID: pa.array(
                [rec[EmbeddingVocabFields.KG_ID] for rec in vocab_records], type=pa.string()
            ),
            EmbeddingVocabFields.LABEL: pa.array(
                [rec.get(EmbeddingVocabFields.LABEL, "") for rec in vocab_records], type=pa.string()
            ),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def write_relation_vocab(vocab_records: List[Dict[str, object]], output_path: Path) -> None:
    _validate_vocab_rows(vocab_records, allowed_fields=list(_RELATION_VOCAB_FIELDS), context="relation_vocab")
    table = pa.table(
        {
            RelationVocabFields.RELATION_ID: pa.array(
                [rec[RelationVocabFields.RELATION_ID] for rec in vocab_records], type=pa.int64()
            ),
            RelationVocabFields.KB: pa.array([rec[RelationVocabFields.KB] for rec in vocab_records], type=pa.string()),
            RelationVocabFields.KG_ID: pa.array(
                [rec[RelationVocabFields.KG_ID] for rec in vocab_records], type=pa.string()
            ),
            RelationVocabFields.LABEL: pa.array(
                [rec[RelationVocabFields.LABEL] for rec in vocab_records], type=pa.string()
            ),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def _load_parquet(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pq.read_table(path)
