from __future__ import annotations
from typing import FrozenSet
import torch


class SampleFields:
    EDGE_INDEX = "edge_index"  # LongTensor [2, num_edges]
    NODE_ENTITY_CATALOG_IDS = "node_entity_catalog_ids"  # LongTensor [num_nodes]
    EDGE_RELATION_CATALOG_IDS = "edge_relation_catalog_ids"  # LongTensor [num_edges]
    NUM_NODES = "num_nodes"  # scalar LongTensor
    NUM_EDGES = "num_edges"  # scalar LongTensor
    QUESTION_EMB = "question_emb"  # FloatTensor [hidden_dim]
    ANCHOR_NODE_IDS = "anchor_node_ids"  # LongTensor [num_anchors]
    TARGET_NODE_IDS = "target_node_ids"  # LongTensor [num_answers_in_graph]
    REACHABLE_TARGET_NODE_IDS = (
        "reachable_target_node_ids"  # LongTensor [num_reachable_targets]
    )
    ANCHOR_NODE_FORWARD_DISTANCE_FLAT = (
        "anchor_node_forward_distances_flat"  # LongTensor [num_nodes]
    )
    ANCHOR_NODE_BACKWARD_DISTANCE_FLAT = (
        "anchor_node_backward_distances_flat"  # LongTensor [num_nodes]
    )
    NODE_TARGET_DISTANCE = "node_target_distance"  # LongTensor [num_nodes]
    TARGET_NODE_DISTANCE_FLAT = (
        "target_node_distances_flat"  # LongTensor [T * num_nodes]
    )
    TARGET_SHORTEST_PATH_COUNT_FLAT = (
        "target_shortest_path_count_flat"  # FloatTensor [T * num_nodes]
    )
    TARGET_SHORTEST_PATH_EDGE_MASK_FLAT = (
        "target_shortest_path_edge_mask_flat"  # BoolTensor [T * num_edges]
    )
    NODE_ID_KEYS: FrozenSet[str] = frozenset(
        {
            ANCHOR_NODE_IDS,
            TARGET_NODE_IDS,
            REACHABLE_TARGET_NODE_IDS,
        }
    )
    NO_INCREMENT_KEYS: FrozenSet[str] = frozenset(
        {
            NODE_ENTITY_CATALOG_IDS,
            EDGE_RELATION_CATALOG_IDS,
            NUM_NODES,
            NUM_EDGES,
            QUESTION_EMB,
            ANCHOR_NODE_FORWARD_DISTANCE_FLAT,
            ANCHOR_NODE_BACKWARD_DISTANCE_FLAT,
            NODE_TARGET_DISTANCE,
            TARGET_NODE_DISTANCE_FLAT,
            TARGET_SHORTEST_PATH_COUNT_FLAT,
            TARGET_SHORTEST_PATH_EDGE_MASK_FLAT,
        }
    )
    STORAGE_REQUIRED: FrozenSet[str] = frozenset(
        {
            EDGE_INDEX,
            NODE_ENTITY_CATALOG_IDS,
            EDGE_RELATION_CATALOG_IDS,
            NUM_NODES,
            NUM_EDGES,
            QUESTION_EMB,
            ANCHOR_NODE_IDS,
            TARGET_NODE_IDS,
            REACHABLE_TARGET_NODE_IDS,
            ANCHOR_NODE_FORWARD_DISTANCE_FLAT,
            ANCHOR_NODE_BACKWARD_DISTANCE_FLAT,
            NODE_TARGET_DISTANCE,
            TARGET_NODE_DISTANCE_FLAT,
            TARGET_SHORTEST_PATH_COUNT_FLAT,
            TARGET_SHORTEST_PATH_EDGE_MASK_FLAT,
        }
    )
    STORAGE_ALLOWED: FrozenSet[str] = STORAGE_REQUIRED


class StorageSchema:
    @staticmethod
    def validate(data: dict[str, torch.Tensor]) -> None:
        missing = SampleFields.STORAGE_REQUIRED - data.keys()
        if missing:
            raise KeyError(f"Missing required keys: {sorted(missing)}")
        unexpected = data.keys() - SampleFields.STORAGE_ALLOWED
        if unexpected:
            raise KeyError(f"Unexpected keys: {sorted(unexpected)}")
        edge_index = _require_tensor(
            data[SampleFields.EDGE_INDEX],
            name=SampleFields.EDGE_INDEX,
            ndim=2,
            dtype=torch.long,
        )
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"{SampleFields.EDGE_INDEX} must have shape [2, num_edges], "
                f"got {tuple(edge_index.shape)}"
            )
        num_nodes = _scalar_int(
            data[SampleFields.NUM_NODES],
            name=SampleFields.NUM_NODES,
        )
        num_edges = _scalar_int(
            data[SampleFields.NUM_EDGES],
            name=SampleFields.NUM_EDGES,
        )
        if num_nodes <= 0:
            raise ValueError(f"{SampleFields.NUM_NODES} must be positive")
        if num_edges <= 0:
            raise ValueError(f"{SampleFields.NUM_EDGES} must be positive")
        if int(edge_index.shape[1]) != num_edges:
            raise ValueError(
                f"{SampleFields.EDGE_INDEX} edge count mismatch: "
                f"{int(edge_index.shape[1])} != {num_edges}"
            )
        _require_1d_length(
            data[SampleFields.NODE_ENTITY_CATALOG_IDS],
            name=SampleFields.NODE_ENTITY_CATALOG_IDS,
            length=num_nodes,
            dtype=torch.long,
        )
        _require_1d_length(
            data[SampleFields.EDGE_RELATION_CATALOG_IDS],
            name=SampleFields.EDGE_RELATION_CATALOG_IDS,
            length=num_edges,
            dtype=torch.long,
        )
        _require_tensor(
            data[SampleFields.QUESTION_EMB],
            name=SampleFields.QUESTION_EMB,
            ndim=1,
            dtype=torch.float32,
        )
        anchor_node_ids = _require_1d(
            data[SampleFields.ANCHOR_NODE_IDS],
            name=SampleFields.ANCHOR_NODE_IDS,
            dtype=torch.long,
        )
        target_node_ids = _require_1d(
            data[SampleFields.TARGET_NODE_IDS],
            name=SampleFields.TARGET_NODE_IDS,
            dtype=torch.long,
        )
        reachable_target_node_ids = _require_1d(
            data[SampleFields.REACHABLE_TARGET_NODE_IDS],
            name=SampleFields.REACHABLE_TARGET_NODE_IDS,
            dtype=torch.long,
        )
        _require_1d_length(
            data[SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT],
            name=SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT,
            length=num_nodes,
            dtype=torch.long,
        )
        _require_1d_length(
            data[SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT],
            name=SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT,
            length=num_nodes,
            dtype=torch.long,
        )
        _require_1d_length(
            data[SampleFields.NODE_TARGET_DISTANCE],
            name=SampleFields.NODE_TARGET_DISTANCE,
            length=num_nodes,
            dtype=torch.long,
        )
        num_reachable_targets = int(reachable_target_node_ids.numel())
        _require_1d_length(
            data[SampleFields.TARGET_NODE_DISTANCE_FLAT],
            name=SampleFields.TARGET_NODE_DISTANCE_FLAT,
            length=num_reachable_targets * num_nodes,
            dtype=torch.long,
        )
        _require_1d_length(
            data[SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT],
            name=SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT,
            length=num_reachable_targets * num_nodes,
            dtype=torch.float32,
        )
        _require_1d_length(
            data[SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT],
            name=SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT,
            length=num_reachable_targets * num_edges,
            dtype=torch.bool,
        )
        _validate_node_ids(
            anchor_node_ids,
            name=SampleFields.ANCHOR_NODE_IDS,
            num_nodes=num_nodes,
        )
        _validate_node_ids(
            target_node_ids,
            name=SampleFields.TARGET_NODE_IDS,
            num_nodes=num_nodes,
        )
        _validate_node_ids(
            reachable_target_node_ids,
            name=SampleFields.REACHABLE_TARGET_NODE_IDS,
            num_nodes=num_nodes,
        )
        _validate_subset(
            subset=reachable_target_node_ids,
            superset=target_node_ids,
            subset_name=SampleFields.REACHABLE_TARGET_NODE_IDS,
            superset_name=SampleFields.TARGET_NODE_IDS,
        )


def _require_tensor(
    value: object,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    return value


def _require_1d(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _require_tensor(value, name=name, ndim=1, dtype=dtype)


def _require_1d_length(
    value: object,
    *,
    name: str,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _require_1d(value, name=name, dtype=dtype)
    if int(tensor.numel()) != length:
        raise ValueError(f"{name} length mismatch: {int(tensor.numel())} != {length}")
    return tensor


def _scalar_int(value: object, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(
                f"{name} must be a scalar tensor, got {tuple(value.shape)}"
            )
        if value.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long, got {value.dtype}")
        return int(value.item())
    if isinstance(value, int):
        return value
    raise TypeError(
        f"{name} must be an int or scalar LongTensor, got {type(value).__name__}"
    )


def _validate_node_ids(
    value: torch.Tensor,
    *,
    name: str,
    num_nodes: int,
) -> None:
    if value.numel() == 0:
        return
    min_id = int(value.min().item())
    max_id = int(value.max().item())
    if min_id < 0 or max_id >= num_nodes:
        raise ValueError(f"{name} contains node ids outside [0, {num_nodes})")


def _validate_subset(
    *,
    subset: torch.Tensor,
    superset: torch.Tensor,
    subset_name: str,
    superset_name: str,
) -> None:
    if subset.numel() == 0:
        return
    superset_values = set(int(x) for x in superset.view(-1).tolist())
    missing = [
        int(x) for x in subset.view(-1).tolist() if int(x) not in superset_values
    ]
    if missing:
        raise ValueError(
            f"{subset_name} must be a subset of {superset_name}; "
            f"missing values: {sorted(set(missing))}"
        )
