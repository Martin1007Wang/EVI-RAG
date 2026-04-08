# src/schema/fields.py
from __future__ import annotations
from typing import FrozenSet


class SampleFields:
    """所有 LMDB 存储字段与 PyG 数据字段的唯一声明处。"""

    # --- 图结构 ---
    EDGE_INDEX = "edge_index"
    EDGE_RELATION_IDS_GLOBAL = "edge_relation_ids_global"
    NODE_ENTITY_IDS_GLOBAL = "node_entity_ids_global"
    NUM_NODES = "num_nodes"

    # --- 问题 ---
    QUESTION_EMB = "question_emb"
    QUESTION_TEXT = "question"

    # --- 答案 / 锚点 (核心修改：从局部索引改为布尔掩码) ---
    IS_ANCHOR_MASK = "is_anchor_mask"
    IS_TARGET_MASK = "is_target_mask"  # 原 answer_local_indices
    ANSWER_ENTITY_IDS_GLOBAL = "answer_entity_ids_global"

    # --- 运行时附加（Collator 填充） ---
    NODE_TOKENS = "node_tokens"
    EDGE_RELATION_TOKENS = "edge_relation_tokens"
    IS_CVT = "is_cvt"
    HEURISTIC_LOG_V = "heuristic_log_v"

    # --- 向后兼容别名 ---
    EDGE_ATTR = EDGE_RELATION_IDS_GLOBAL
    NODE_ENTITY_IDS = NODE_ENTITY_IDS_GLOBAL
    ANSWER_IDS = ANSWER_ENTITY_IDS_GLOBAL
    RELATION_TOKENS = EDGE_RELATION_TOKENS

    # --- 防御性不自增集合 ---
    # PyG 默认只自增含 "index" 或 "face" 的张量，但为了绝对安全，
    # 我们显式声明所有作为“全局词表/实体ID”的张量严禁被加上 batch 偏移。
    NO_INCREMENT_KEYS: FrozenSet[str] = frozenset(
        {
            NODE_ENTITY_IDS_GLOBAL,
            ANSWER_ENTITY_IDS_GLOBAL,
            EDGE_RELATION_IDS_GLOBAL,
        }
    )

    # --- LMDB 存储字段 ---
    STORAGE_REQUIRED: FrozenSet[str] = frozenset(
        {
            EDGE_INDEX,
            NODE_ENTITY_IDS_GLOBAL,
            EDGE_RELATION_IDS_GLOBAL,
            QUESTION_EMB,
            IS_ANCHOR_MASK,  # 确保存入 LMDB 的已经是 1D boolean tensor
            IS_TARGET_MASK,  # 确保存入 LMDB 的已经是 1D boolean tensor
            ANSWER_ENTITY_IDS_GLOBAL,
            NUM_NODES,
        }
    )
    STORAGE_OPTIONAL: FrozenSet[str] = frozenset()
    STORAGE_ALLOWED: FrozenSet[str] = STORAGE_REQUIRED | STORAGE_OPTIONAL


class StorageSchema:
    @staticmethod
    def validate(data: dict) -> None:
        missing = SampleFields.STORAGE_REQUIRED - data.keys()
        if missing:
            raise KeyError(f"LMDB sample missing required keys: {sorted(missing)}")
        unexpected = data.keys() - SampleFields.STORAGE_ALLOWED
        if unexpected:
            raise KeyError(f"LMDB sample has unexpected keys: {sorted(unexpected)}")
