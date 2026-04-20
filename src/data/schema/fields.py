from __future__ import annotations

from typing import FrozenSet


class SampleFields:
    """LMDB 存储字段与 PyG 批字段的唯一来源。"""

    EDGE_INDEX = "edge_index"
    EDGE_RELATION_IDS_GLOBAL = "edge_relation_ids_global"
    NODE_ENTITY_IDS_GLOBAL = "node_entity_ids_global"
    NUM_NODES = "num_nodes"

    QUESTION_EMB = "question_emb"
    QUESTION_TEXT = "question"

    IS_ANCHOR_MASK = "is_anchor_mask"
    IS_TARGET_MASK = "is_target_mask"
    ANCHOR_SIGNED_DISTANCE = "anchor_signed_distance"
    ANSWER_ENTITY_IDS_GLOBAL = "answer_entity_ids_global"
    POSITIVE_EDGE_MASK = "positive_edge_mask"
    NODE_TO_TARGET_DISTANCE = "node_to_target_distance"
    SHORTEST_SUFFIX_COUNT = "shortest_suffix_count"
    BOUNDED_SUFFIX_COUNT = "bounded_suffix_count"
    MAX_PATH_LENGTH = "max_path_length"

    NODE_TOKENS = "node_tokens"
    RELATION_TOKENS = "relation_tokens"
    IS_CVT = "is_cvt"
    HEURISTIC_LOG_V = "heuristic_log_v"

    EDGE_ATTR = EDGE_RELATION_IDS_GLOBAL
    NODE_ENTITY_IDS = NODE_ENTITY_IDS_GLOBAL
    ANSWER_IDS = ANSWER_ENTITY_IDS_GLOBAL

    NO_INCREMENT_KEYS: FrozenSet[str] = frozenset(
        {
            NODE_ENTITY_IDS_GLOBAL,
            ANSWER_ENTITY_IDS_GLOBAL,
            EDGE_RELATION_IDS_GLOBAL,
            IS_ANCHOR_MASK,
            IS_TARGET_MASK,
            POSITIVE_EDGE_MASK,
            NODE_TO_TARGET_DISTANCE,
            SHORTEST_SUFFIX_COUNT,
            BOUNDED_SUFFIX_COUNT,
            MAX_PATH_LENGTH,
        }
    )

    STORAGE_REQUIRED: FrozenSet[str] = frozenset(
        {
            EDGE_INDEX,
            NODE_ENTITY_IDS_GLOBAL,
            EDGE_RELATION_IDS_GLOBAL,
            QUESTION_EMB,
            IS_ANCHOR_MASK,
            IS_TARGET_MASK,
            ANCHOR_SIGNED_DISTANCE,
            ANSWER_ENTITY_IDS_GLOBAL,
            NUM_NODES,
        }
    )

    STORAGE_OPTIONAL: FrozenSet[str] = frozenset(
        {
            POSITIVE_EDGE_MASK,
            NODE_TO_TARGET_DISTANCE,
            SHORTEST_SUFFIX_COUNT,
            BOUNDED_SUFFIX_COUNT,
            MAX_PATH_LENGTH,
        }
    )

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

        edge_index = data[SampleFields.EDGE_INDEX]
        if not hasattr(edge_index, "shape") or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"{SampleFields.EDGE_INDEX} must have shape [2, E], got {getattr(edge_index, 'shape', None)}"
            )

        num_nodes = int(data[SampleFields.NUM_NODES])
        node_ids = data[SampleFields.NODE_ENTITY_IDS_GLOBAL]
        if hasattr(node_ids, "shape") and node_ids.shape[0] != num_nodes:
            raise ValueError(
                f"{SampleFields.NODE_ENTITY_IDS_GLOBAL} length must match num_nodes: "
                f"{node_ids.shape[0]} != {num_nodes}"
            )

        anchor_mask = data[SampleFields.IS_ANCHOR_MASK]
        target_mask = data[SampleFields.IS_TARGET_MASK]
        for name, value in (
            (SampleFields.IS_ANCHOR_MASK, anchor_mask),
            (SampleFields.IS_TARGET_MASK, target_mask),
        ):
            if hasattr(value, "ndim") and value.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape {getattr(value, 'shape', None)}")

        question_emb = data[SampleFields.QUESTION_EMB]
        if hasattr(question_emb, "ndim") and question_emb.ndim != 1:
            raise ValueError(
                f"{SampleFields.QUESTION_EMB} must be 1D, got shape {getattr(question_emb, 'shape', None)}"
            )

        bounded_suffix = data.get(SampleFields.BOUNDED_SUFFIX_COUNT)
        if bounded_suffix is not None and hasattr(bounded_suffix, "ndim"):
            if bounded_suffix.ndim != 2:
                raise ValueError(
                    f"{SampleFields.BOUNDED_SUFFIX_COUNT} must be 2D, got shape "
                    f"{getattr(bounded_suffix, 'shape', None)}"
                )
            if bounded_suffix.shape[1] != num_nodes:
                raise ValueError(
                    f"{SampleFields.BOUNDED_SUFFIX_COUNT} second dim must match num_nodes: "
                    f"{bounded_suffix.shape[1]} != {num_nodes}"
                )
