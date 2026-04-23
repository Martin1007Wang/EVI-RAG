from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lmdb
import torch
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# 1. 核心反序列化逻辑
# ------------------------------------------------------------------ #


def deserialize_sample(payload: bytes) -> Dict[str, torch.Tensor]:
    """
    将从 LMDB 读出的 safetensors 二进制流还原为 Tensor 字典。
    采用局部 import 提高多进程加载速度并防止死锁。
    """
    from safetensors.torch import load

    data = load(payload)
    if not isinstance(data, dict):
        raise ValueError("LMDB sample payload must decode to a dict.")
    return data


def serialize_sample(sample: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a tensor-only LMDB sample payload via safetensors."""
    from safetensors.torch import save

    if not isinstance(sample, dict):
        raise TypeError(f"LMDB sample must be a dict, got {type(sample)!r}.")
    for key, value in sample.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"LMDB sample field {key!r} must be a torch.Tensor, got {type(value)!r}."
            )
    return save(_clone_shared_storage_tensors(sample))


def _clone_shared_storage_tensors(
    sample: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Detach later tensor aliases so safetensors can serialize the payload."""
    detached: Dict[str, torch.Tensor] = {}
    seen_storage_keys: set[tuple[str, Optional[int], int]] = set()

    for key, value in sample.items():
        storage_key = _tensor_storage_key(value)
        if storage_key is not None and storage_key in seen_storage_keys:
            detached[key] = value.clone()
            continue

        detached[key] = value
        if storage_key is not None:
            seen_storage_keys.add(storage_key)

    return detached


def _tensor_storage_key(tensor: torch.Tensor) -> Optional[tuple[str, Optional[int], int]]:
    if tensor.numel() == 0:
        return None

    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, int(storage.data_ptr()))


# ------------------------------------------------------------------ #
# 2. 数据发现与分片逻辑
# ------------------------------------------------------------------ #


def resolve_core_lmdb_paths(embeddings_dir: Path, split: str) -> List[Path]:
    """解析指定 split (train/validation/test) 的所有 LMDB 文件或分片路径。"""
    embeddings_dir = Path(embeddings_dir)
    # 优先查找单文件模式: split.lmdb/
    base_path = embeddings_dir / f"{split}.lmdb"
    if base_path.exists():
        return [base_path]

    # 查找分片模式: split.shard000.lmdb/, split.shard001.lmdb/ ...
    shard_paths = sorted(embeddings_dir.glob(f"{split}.shard*.lmdb"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No LMDB files found for split='{split}' in {embeddings_dir}"
        )
    return shard_paths


def get_all_keys_from_lmdb(lmdb_path: Path) -> List[str]:
    """
    遍历 LMDB 数据库并返回所有的 Key (sample_id)。
    用于 DataModule 初始化阶段扫描数据。此函数读完即关，不保持连接。
    """
    # 极简配置：只读、无锁、无 readahead
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            # 仅迭代 Key，不读取 Value 提高速度
            return [key.decode("utf-8") for key in cursor.iternext(values=False)]
    finally:
        env.close()


def assign_lmdb_shard(sample_key: str | bytes, num_shards: int) -> int:
    """根据 Sample ID 计算其应该存储/读取的 LMDB 分片索引。"""
    if num_shards <= 1:
        return 0
    if isinstance(sample_key, str):
        sample_key = sample_key.encode("utf-8")
    # 使用 zlib.crc32 保证分布的一致性与机器无关
    return int(zlib.crc32(sample_key) % num_shards)


# ------------------------------------------------------------------ #
# 3. 过滤逻辑 (用于 DataModule.setup)
# ------------------------------------------------------------------ #


def load_filter_ids(path: Path) -> set[str]:
    """从外部 JSON 列表或字典文件中加载样本 ID 过滤集合。"""
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(map(str, data))
    if isinstance(data, dict) and "sample_ids" in data:
        return set(map(str, data["sample_ids"]))

    raise ValueError(
        f"Unsupported filter format in {path}: expected list or dict with 'sample_ids'."
    )


def apply_filter_intersection(
    sample_ids: Sequence[str], filter_paths: Sequence[Path]
) -> List[str]:
    """
    对样本 ID 列表执行多重 JSON 过滤器的交集运算。
    返回过滤后仍保留的 ID 列表。
    """
    if not filter_paths:
        return list(sample_ids)

    keep_ids: Optional[set[str]] = None
    for path in filter_paths:
        current_filter = load_filter_ids(path)
        if keep_ids is None:
            keep_ids = current_filter
        else:
            keep_ids &= current_filter

    if keep_ids is None:
        return list(sample_ids)

    # 保持原始 sample_ids 的相对顺序
    return [sid for sid in sample_ids if sid in keep_ids]


# ------------------------------------------------------------------ #
# 4. 拓扑辅助工具
# ------------------------------------------------------------------ #


def local_indices(node_entity_ids: Sequence[int], targets: Sequence[int]) -> List[int]:
    """
    计算给定 target 实体在 node_entity_ids 列表中的位置索引。
    常用于将全局知识图谱实体 ID 映射为子图内部的局部 0-based 索引。
    """
    position = {nid: idx for idx, nid in enumerate(node_entity_ids)}
    return [position[t] for t in targets if t in position]


__all__ = [
    "deserialize_sample",
    "serialize_sample",
    "resolve_core_lmdb_paths",
    "get_all_keys_from_lmdb",
    "assign_lmdb_shard",
    "apply_filter_intersection",
    "local_indices",
]
