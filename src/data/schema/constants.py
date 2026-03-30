from __future__ import annotations

import os

_PATH_MODE_UNDIRECTED = "undirected"
_PATH_MODE_QA_DIRECTED = "qa_directed"
_PATH_MODES = (_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)
_ALLOWED_SPLITS = ("train", "validation", "test")
_NON_TEXT_EMBEDDING_ID = 0
_DIST_UNREACHABLE = -1
_DISTANCE_INT8_MAX = 127
_DISTANCE_INT16_MAX = 32767
_DISTANCE_BYTES_PER_INT8 = 1
_DISTANCE_BYTES_PER_INT16 = 2
_DISTANCE_EDGE_WEIGHT = 1
_DISTANCE_CPU_RESERVE = 1
_DISTANCE_DEFAULT_CHUNK_SIZE = 8
_DISTANCE_DEFAULT_PROGRESS_INTERVAL = 0
_REL_LABEL_SAMPLE_LIMIT = 20
_DEFAULT_BATCH_SIZE = 64
_EDGE_STAT_KEYS = (
    "raw_edges",
    "kept_edges",
    "dropped_edges",
    "self_loop_edges",
    "raw_nodes",
    "kept_nodes",
)
_FILTER_STAT_KEYS = (
    "dropped_no_question_entity",
    "dropped_no_answer",
    "dropped_no_path",
)
_FILTER_MISSING_ANCHOR_FILENAME = "filter_missing_anchor.json"
_FILTER_MISSING_ANSWER_FILENAME = "filter_missing_answer.json"
_DISTANCE_PROGRESS_DISABLED = 0
_DISTANCE_MIN_WORKERS = 1
_DISTANCE_MIN_CHUNK_SIZE = 1
_DISTANCE_LMDB_MAX_READERS = 256
_DISTANCE_DEFAULT_NUM_WORKERS = max(
    _DISTANCE_MIN_WORKERS,
    (os.cpu_count() or _DISTANCE_MIN_WORKERS) - _DISTANCE_CPU_RESERVE,
)
_MIN_CHUNK_SIZE = 1
_DISABLE_PARALLEL_WORKERS = 0
_EDGE_INDEX_MIN = 0
_BYTES_PER_GB = 1 << 30
_LMDB_MAP_SIZE_GB_MIN = 1
_LMDB_GROWTH_GB_MIN = 0
_LMDB_GROWTH_FACTOR_MIN = 1.0
_DEFAULT_LMDB_MAP_GROWTH_GB = 32
_DEFAULT_LMDB_MAP_GROWTH_FACTOR = 1.5
_VALIDATE_GRAPH_EDGES_DEFAULT = True
_REMOVE_SELF_LOOPS_DEFAULT = True
_LMDB_SHARDS_MIN = 1
_INVERSE_RELATION_SUFFIX_DEFAULT = "__inv"


class GraphFields:
    GRAPH_ID = "graph_id"
    NODE_IDS = "node_entity_ids"
    NODE_EMBED_IDS = "node_embedding_ids"
    NODE_LABELS = "node_labels"
    EDGE_SRC = "edge_src"
    EDGE_DST = "edge_dst"
    EDGE_REL_IDS = "edge_relation_ids"


class QuestionFields:
    DATASET = "dataset"
    SPLIT = "split"
    KB = "kb"
    QUESTION = "question"
    QUESTION_ENTITY_IDS = "question_entity_ids"
    ANSWER_ENTITY_IDS = "answer_entity_ids"
    ANSWER_TEXTS = "answer_texts"
    GRAPH_ID = "graph_id"
    QUESTION_EMB = "question_emb"
    QUESTION_CTX = "question_ctx"
    QUESTION_CTX_MASK = "question_ctx_mask"


class EntityVocabFields:
    ENTITY_ID = "entity_id"
    KB = "kb"
    KG_ID = "kg_id"
    LABEL = "label"
    IS_TEXT = "is_text"
    IS_CVT = "is_cvt"
    EMBEDDING_ID = "embedding_id"


class EmbeddingVocabFields:
    EMBEDDING_ID = "embedding_id"
    KB = "kb"
    KG_ID = "kg_id"
    LABEL = "label"


class RelationVocabFields:
    RELATION_ID = "relation_id"
    KB = "kb"
    KG_ID = "kg_id"
    LABEL = "label"


_GRAPH_PARQUET_FIELDS = (
    GraphFields.GRAPH_ID,
    GraphFields.NODE_IDS,
    GraphFields.NODE_EMBED_IDS,
    GraphFields.NODE_LABELS,
    GraphFields.EDGE_SRC,
    GraphFields.EDGE_DST,
    GraphFields.EDGE_REL_IDS,
)

_QUESTION_PARQUET_FIELDS = (
    QuestionFields.DATASET,
    QuestionFields.SPLIT,
    QuestionFields.KB,
    QuestionFields.QUESTION,
    QuestionFields.QUESTION_ENTITY_IDS,
    QuestionFields.ANSWER_ENTITY_IDS,
    QuestionFields.ANSWER_TEXTS,
    QuestionFields.GRAPH_ID,
    QuestionFields.QUESTION_EMB,
    QuestionFields.QUESTION_CTX,
    QuestionFields.QUESTION_CTX_MASK,
)

_QUESTION_PARQUET_REQUIRED_FIELDS = tuple(
    field
    for field in _QUESTION_PARQUET_FIELDS
    if field
    not in (
        QuestionFields.QUESTION_EMB,
        QuestionFields.QUESTION_CTX,
        QuestionFields.QUESTION_CTX_MASK,
    )
)

_ENTITY_VOCAB_FIELDS = (
    EntityVocabFields.ENTITY_ID,
    EntityVocabFields.KB,
    EntityVocabFields.KG_ID,
    EntityVocabFields.LABEL,
    EntityVocabFields.IS_TEXT,
    EntityVocabFields.IS_CVT,
    EntityVocabFields.EMBEDDING_ID,
)

_EMBEDDING_VOCAB_FIELDS = (
    EmbeddingVocabFields.EMBEDDING_ID,
    EmbeddingVocabFields.KB,
    EmbeddingVocabFields.KG_ID,
    EmbeddingVocabFields.LABEL,
)

_RELATION_VOCAB_FIELDS = (
    RelationVocabFields.RELATION_ID,
    RelationVocabFields.KB,
    RelationVocabFields.KG_ID,
    RelationVocabFields.LABEL,
)
