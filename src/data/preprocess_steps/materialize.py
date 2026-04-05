from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import lmdb
import torch

from src.data.schema import SampleFields, StorageSchema
from src.utils.lmdb_utils import assign_lmdb_shard, serialize_sample

from .metadata import (
    metadata_path,
    save_metadata,
)
from .sample_types import EntityVocab, PreparedSample, RelationVocab

ALLOWED_SPLITS = ("train", "validation", "test")


def materialize_preprocessed_data(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    encoded: Dict[str, Any],  # 替换 object 为 Any，明确这是个多态字典
    embeddings_dir: Path,
    overwrite_lmdb: bool = False,
    lmdb_shards: int = 1,
    map_size_gb: float = 128,
) -> None:
    """
    将预处理好的样本、拓扑结构与高维向量特征序列化并写入 LMDB。
    同时生成训练时至关重要的 Runtime Metadata。
    """
    if lmdb_shards < 1:
        raise ValueError(f"lmdb_shards must be >= 1, got {lmdb_shards}.")
    if map_size_gb <= 0:
        raise ValueError(f"map_size_gb must be > 0, got {map_size_gb}.")

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    map_size_bytes = int(map_size_gb * (1024**3))

    # 1. 提取安全的强类型张量
    entity_metadata: Dict[str, Any] = encoded["entity_metadata"]
    question_embeddings: torch.Tensor = encoded["question_embeddings"]
    question_contexts: torch.Tensor | None = encoded.get("question_contexts")
    question_context_masks: torch.Tensor | None = encoded.get("question_context_masks")

    # 2. 保存静态词表与 Embedding
    torch.save(
        {
            "version": 1,
            "entity_embedding_map": entity_metadata["entity_embedding_map"],
            "cvt_mask": entity_metadata["cvt_mask"],
            "entity_labels": entity_metadata["entity_labels"],
            "relation_labels": encoded["relation_labels"],
        },
        embeddings_dir / "entity_metadata.pt",
    )
    torch.save(
        encoded["entity_embeddings"].contiguous(),
        embeddings_dir / "entity_embeddings.pt",
    )
    torch.save(
        encoded["relation_embeddings"].contiguous(),
        embeddings_dir / "relation_embeddings.pt",
    )

    # 3. 初始化 LMDB 环境
    envs: dict[tuple[str, int], lmdb.Environment] = {}
    for split in ALLOWED_SPLITS:
        for shard_id in range(lmdb_shards):
            path = _lmdb_path(embeddings_dir, split, shard_id, lmdb_shards)
            _reset_output_path(path, overwrite=overwrite_lmdb)
            envs[(split, shard_id)] = lmdb.open(
                str(path),
                map_size=map_size_bytes,
                subdir=True,
                lock=True,
                create=True,
                max_dbs=1,
            )

    # 4. 初始化 Metadata 统计容器 (显式类型声明)
    runtime_metadata: dict[str, dict[str, list[Any]]] = {
        split: {
            "sample_ids": [],
            "questions": [],
            "num_nodes": [],
            "num_edges": [],
            "question_tokens": [],
        }
        for split in ALLOWED_SPLITS
    }

    try:
        for idx, entry in enumerate(prepared_samples):
            sample = entry.sample
            node_index: dict[str, int] = {}
            node_entity_ids: list[int] = []
            edge_src: list[int] = []
            edge_dst: list[int] = []
            edge_relation_ids_global: list[int] = []

            # 内部辅助函数：构建局部图索引
            def _get_or_add_local_index(entity: str) -> int:
                if entity not in node_index:
                    node_index[entity] = len(node_index)
                    node_entity_ids.append(entity_vocab.entity_id(entity))
                return node_index[entity]

            # 解析边，转化为局部图的连续索引
            for head, relation, tail in entry.kept_edges:
                edge_src.append(_get_or_add_local_index(head))
                edge_dst.append(_get_or_add_local_index(tail))
                edge_relation_ids_global.append(relation_vocab.relation_id(relation))

            num_nodes = len(node_index)
            if num_nodes == 0 or not edge_relation_ids_global:
                continue

            # 构建锚点掩码 (Anchor Mask)
            is_anchor_mask = torch.zeros((num_nodes,), dtype=torch.bool)
            anchor_local_indices = [
                node_index[entity]
                for entity in _dedup_preserve_order(entry.question_entities_in_graph)
                if entity in node_index
            ]
            if anchor_local_indices:
                is_anchor_mask[
                    torch.as_tensor(anchor_local_indices, dtype=torch.long)
                ] = True

            # 如果连起点都没有，这图没法做 GFlowNet 采样，直接丢弃
            if not bool(is_anchor_mask.any().item()):
                continue

            # 构建目标答案掩码 (Target Mask)
            is_target_mask = torch.zeros((num_nodes,), dtype=torch.bool)
            legal_answer_entities = _dedup_preserve_order(entry.legal_answer_entities)
            answer_local_indices = [
                node_index[entity]
                for entity in legal_answer_entities
                if entity in node_index
            ]
            if answer_local_indices:
                is_target_mask[
                    torch.as_tensor(answer_local_indices, dtype=torch.long)
                ] = True

            # 组装 Storage Schema 所需的数据字典
            sample_dict = {
                SampleFields.EDGE_INDEX: torch.as_tensor(
                    [edge_src, edge_dst], dtype=torch.long
                ),
                SampleFields.EDGE_RELATION_IDS_GLOBAL: torch.as_tensor(
                    edge_relation_ids_global,
                    dtype=torch.long,
                ),
                SampleFields.NODE_ENTITY_IDS_GLOBAL: torch.as_tensor(
                    node_entity_ids,
                    dtype=torch.long,
                ),
                SampleFields.NUM_NODES: torch.as_tensor(num_nodes, dtype=torch.long),
                SampleFields.QUESTION_EMB: question_embeddings[idx]
                .unsqueeze(0)
                .contiguous(),
                SampleFields.IS_ANCHOR_MASK: is_anchor_mask,
                SampleFields.IS_TARGET_MASK: is_target_mask,
                SampleFields.ANSWER_ENTITY_IDS_GLOBAL: torch.as_tensor(
                    [
                        entity_vocab.entity_id(entity)
                        for entity in legal_answer_entities
                    ],
                    dtype=torch.long,
                ),
            }

            if question_contexts is not None and question_context_masks is not None:
                sample_dict[SampleFields.QUESTION_CTX] = (
                    question_contexts[idx].unsqueeze(0).contiguous()
                )
                sample_dict[SampleFields.QUESTION_CTX_MASK] = (
                    question_context_masks[idx].unsqueeze(0).contiguous()
                )

            StorageSchema.validate(sample_dict)

            # 5. 落盘写入 LMDB
            split_name = str(sample.split)
            shard_id = assign_lmdb_shard(entry.sample_id, lmdb_shards)
            with envs[(split_name, shard_id)].begin(write=True) as txn:
                txn.put(entry.sample_id.encode("utf-8"), serialize_sample(sample_dict))

            # 6. 收集 Runtime Metadata
            runtime_metadata[split_name]["sample_ids"].append(entry.sample_id)
            runtime_metadata[split_name]["questions"].append(sample.question)
            runtime_metadata[split_name]["num_nodes"].append(num_nodes)
            runtime_metadata[split_name]["num_edges"].append(
                len(edge_relation_ids_global)
            )

            # 计算有效的问题 Token 数量
            q_len = (
                int(question_context_masks[idx].sum().item())
                if question_context_masks is not None
                else 1
            )
            runtime_metadata[split_name]["question_tokens"].append(q_len)

    finally:
        # 确保安全关闭所有数据库连接
        for env in envs.values():
            env.close()

    # 7. 保存 Metadata 文件
    for split_name, metadata in runtime_metadata.items():
        save_metadata(
            metadata_path(embeddings_dir, split_name),
            split=split_name,
            sample_ids=metadata["sample_ids"],
            questions=metadata["questions"],
            num_nodes=metadata["num_nodes"],
            num_edges=metadata["num_edges"],
            question_tokens=metadata["question_tokens"],
        )


def _lmdb_path(
    embeddings_dir: Path, split: str, shard_id: int, num_shards: int
) -> Path:
    if num_shards <= 1:
        return embeddings_dir / f"{split}.lmdb"
    return embeddings_dir / f"{split}.shard{shard_id:03d}.lmdb"


def _reset_output_path(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists at {path}. Set overwrite_lmdb=true to rebuild."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _dedup_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


__all__ = ["materialize_preprocessed_data"]
