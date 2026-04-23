from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

# 内部版本号，用于防止加载了旧版本数据结构导致崩溃
_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class DatasetManifest:
    """
    数据集的全局索引清单。
    提供 O(1) 的数据集长度以及静态图特征（节点数、边数），
    支撑 DataLoader 进行零硬盘 IO 的 Smart/Dynamic Batching。
    """

    sample_ids: list[str]
    num_nodes: torch.Tensor  # [num_samples] int32
    num_edges: torch.Tensor  # [num_samples] int32

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)


def manifest_path(embeddings_dir: Path, split: str) -> Path:
    return Path(embeddings_dir) / f"{str(split)}.manifest.pt"


def save_manifest(
    path: Path,
    *,
    sample_ids: Sequence[str],
    num_nodes: Sequence[int],
    num_edges: Sequence[int],
) -> None:
    payload = _build_manifest_payload(
        sample_ids=sample_ids,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_manifest(path: Path) -> DatasetManifest:
    # 使用 weights_only=False 因为这里包含 list[str] 等基础 Python 类型，
    # 而不是纯模型权重，但由于是本地预处理生成的数据，安全性可控。
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest at {path} must decode to a dict, got {type(payload)!r}.")

    version = int(payload.get("version", -1))
    if version != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported dataset manifest version: expected {_MANIFEST_VERSION}, got {version}.")

    sample_ids = [str(value) for value in payload.get("sample_ids", [])]

    num_nodes = _coerce_int_tensor(payload.get("num_nodes"), name="num_nodes")
    num_edges = _coerce_int_tensor(payload.get("num_edges"), name="num_edges")

    _validate_lengths(
        sample_ids=sample_ids,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )
    _validate_unique_sample_ids(sample_ids)

    return DatasetManifest(
        sample_ids=sample_ids,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


def _build_manifest_payload(
    *,
    sample_ids: Sequence[str],
    num_nodes: Sequence[int],
    num_edges: Sequence[int],
) -> dict[str, Any]:
    sample_ids_list = [str(value) for value in sample_ids]

    # 强制统一使用 int32 节约内存，百万级样本也仅占用几 MB
    num_nodes_tensor = torch.as_tensor(list(num_nodes), dtype=torch.int32)
    num_edges_tensor = torch.as_tensor(list(num_edges), dtype=torch.int32)

    _validate_lengths(
        sample_ids=sample_ids_list,
        num_nodes=num_nodes_tensor,
        num_edges=num_edges_tensor,
    )
    _validate_unique_sample_ids(sample_ids_list)

    return {
        "version": _MANIFEST_VERSION,
        "sample_ids": sample_ids_list,
        "num_nodes": num_nodes_tensor,
        "num_edges": num_edges_tensor,
    }


def _coerce_int_tensor(value: Any, *, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Dataset manifest missing required field {name!r}.")
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.dim() != 1:
        raise ValueError(f"Dataset manifest field {name!r} must be 1D, got {tuple(tensor.shape)}.")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"Dataset manifest field {name!r} must be integral, got {tensor.dtype}.")
    return tensor


def _validate_lengths(
    *,
    sample_ids: list[str],
    num_nodes: torch.Tensor,
    num_edges: torch.Tensor,
) -> None:
    expected = len(sample_ids)
    actual_lengths = {
        "num_nodes": int(num_nodes.numel()),
        "num_edges": int(num_edges.numel()),
    }
    mismatched = {name: actual for name, actual in actual_lengths.items() if actual != expected}
    if mismatched:
        raise ValueError(f"Dataset manifest length mismatch: expected {expected} entries, got {mismatched}.")


def _validate_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    """
    O(1) 极速去重校验。利用 C 底层的 set() 长度判定作为 Fast-path。
    只有在发生重复时，才进入较慢的 Python for 循环寻找具体重复项进行报错。
    """
    if len(sample_ids) == len(set(sample_ids)):
        return  # Fast-path: 无重复，瞬间通过

    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen:
            if sample_id not in duplicates:
                duplicates.append(sample_id)
            if len(duplicates) >= 3:
                break
            continue
        seen.add(sample_id)

    raise ValueError(f"Dataset manifest contains duplicate sample_ids, examples: {duplicates}.")


__all__ = [
    "DatasetManifest",
    "load_manifest",
    "manifest_path",
    "save_manifest",
]
