from __future__ import annotations

import logging
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import lmdb
import torch

from src.data.schema import SampleFields, StorageSchema
from src.utils.lmdb_utils import assign_lmdb_shard, serialize_sample
from src.utils.path_utils import compute_signed_anchor_distances

from .manifest import manifest_path, save_manifest
from .samples import EncodedPayload, PreparedSample
from .vocab import EntityVocab, RelationVocab

log = logging.getLogger(__name__)

ALLOWED_SPLITS = ("train", "validation", "test")


def materialize_preprocessed_data(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    payload: EncodedPayload,
    embeddings_dir: Path,
    path_mode: str = "qa_directed",
    overwrite_lmdb: bool = True,
    lmdb_shards: int = 1,
    map_size_gb: float = 128,
    commit_frequency: int = 1000,
) -> None:
    if lmdb_shards < 1:
        raise ValueError(f"lmdb_shards must be >= 1, got {lmdb_shards}.")
    if commit_frequency < 1:
        raise ValueError(f"commit_frequency must be >= 1, got {commit_frequency}.")

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    map_size_bytes = int(map_size_gb * (1024**3))

    entity_catalog = payload.entity_catalog
    question_embeddings = payload.question_embeddings
    if int(question_embeddings.size(0)) != len(prepared_samples):
        raise ValueError(
            "question_embeddings must align with prepared_samples: expected "
            f"{len(prepared_samples)}, got {int(question_embeddings.size(0))}."
        )

    log.info("Saving static assets to %s...", embeddings_dir)
    torch.save(
        {
            "version": 1,
            "entity_embedding_map": entity_catalog.entity_embedding_map,
            "cvt_mask": entity_catalog.cvt_mask,
            "entity_labels": entity_catalog.entity_labels,
            "relation_labels": payload.relation_labels,
        },
        embeddings_dir / "entity_metadata.pt",
    )
    torch.save(
        payload.entity_embeddings.contiguous(),
        embeddings_dir / "entity_embeddings.pt",
    )
    torch.save(
        payload.relation_embeddings.contiguous(),
        embeddings_dir / "relation_embeddings.pt",
    )

    runtime_manifest: dict[str, dict[str, list[Any]]] = {
        split: {
            "sample_ids": [],
            "questions": [],
            "num_nodes": [],
            "num_edges": [],
            "question_tokens": [],
        }
        for split in ALLOWED_SPLITS
    }

    with ExitStack() as stack:
        envs: dict[tuple[str, int], lmdb.Environment] = {}
        txns: dict[tuple[str, int], lmdb.Transaction] = {}
        uncommitted_counts: dict[tuple[str, int], int] = {}

        log.info("Initializing %d LMDB shards per split...", lmdb_shards)
        for split in ALLOWED_SPLITS:
            for shard_id in range(lmdb_shards):
                path = _lmdb_path(embeddings_dir, split, shard_id, lmdb_shards)
                _reset_output_path(path, overwrite=overwrite_lmdb)

                env = stack.enter_context(
                    lmdb.open(
                        str(path),
                        map_size=map_size_bytes,
                        subdir=True,
                        lock=True,
                        create=True,
                        max_dbs=1,
                    )
                )
                env_key = (split, shard_id)
                envs[env_key] = env
                txns[env_key] = env.begin(write=True)
                uncommitted_counts[env_key] = 0

        log.info("Materializing %d samples...", len(prepared_samples))
        for idx, entry in enumerate(prepared_samples):
            sample = entry.sample
            node_index: dict[str, int] = {}
            node_entity_ids: list[int] = []
            edge_src: list[int] = []
            edge_dst: list[int] = []
            edge_relation_ids_global: list[int] = []

            def _get_or_add_local_index(entity: str) -> int:
                local_idx = node_index.get(entity)
                if local_idx is None:
                    local_idx = len(node_index)
                    node_index[entity] = local_idx
                    node_entity_ids.append(entity_vocab.entity_id(entity))
                return local_idx

            for head, relation, tail in entry.kept_edges:
                edge_src.append(_get_or_add_local_index(head))
                edge_dst.append(_get_or_add_local_index(tail))
                edge_relation_ids_global.append(relation_vocab.relation_id(relation))

            num_nodes = len(node_index)
            if num_nodes == 0 or not edge_relation_ids_global:
                raise RuntimeError(
                    f"Prepared sample {entry.sample_id!r} materialized to an empty graph."
                )

            is_anchor_mask = torch.zeros((num_nodes,), dtype=torch.bool)
            anchor_local_indices = [
                node_index[ent]
                for ent in entry.question_entities_in_graph
                if ent in node_index
            ]
            if anchor_local_indices:
                is_anchor_mask[torch.as_tensor(anchor_local_indices, dtype=torch.long)] = True
            if not is_anchor_mask.any():
                raise RuntimeError(
                    f"Prepared sample {entry.sample_id!r} has no in-graph question entities to materialize."
                )

            is_target_mask = torch.zeros((num_nodes,), dtype=torch.bool)
            answer_local_indices = [
                node_index[ent]
                for ent in entry.legal_answer_entities
                if ent in node_index
            ]
            if answer_local_indices:
                is_target_mask[torch.as_tensor(answer_local_indices, dtype=torch.long)] = True

            local_edge_index = torch.as_tensor([edge_src, edge_dst], dtype=torch.long)
            anchor_signed_distance = compute_signed_anchor_distances(
                edge_index=local_edge_index,
                is_anchor_mask=is_anchor_mask,
                num_nodes=num_nodes,
                path_mode=path_mode,
            )

            sample_dict = {
                SampleFields.EDGE_INDEX: local_edge_index,
                SampleFields.EDGE_RELATION_IDS_GLOBAL: torch.as_tensor(
                    edge_relation_ids_global, dtype=torch.long
                ),
                SampleFields.NODE_ENTITY_IDS_GLOBAL: torch.as_tensor(
                    node_entity_ids, dtype=torch.long
                ),
                SampleFields.NUM_NODES: torch.as_tensor(num_nodes, dtype=torch.long),
                SampleFields.QUESTION_EMB: question_embeddings[idx].reshape(-1).contiguous(),
                SampleFields.IS_ANCHOR_MASK: is_anchor_mask,
                SampleFields.IS_TARGET_MASK: is_target_mask,
                SampleFields.ANCHOR_SIGNED_DISTANCE: anchor_signed_distance,
                SampleFields.ANSWER_ENTITY_IDS_GLOBAL: torch.as_tensor(
                    [entity_vocab.entity_id(ent) for ent in entry.legal_answer_entities],
                    dtype=torch.long,
                ),
                SampleFields.POSITIVE_EDGE_MASK: entry.positive_edge_mask,
                SampleFields.NODE_TO_TARGET_DISTANCE: entry.node_to_target_distance,
                SampleFields.SHORTEST_SUFFIX_COUNT: entry.shortest_suffix_count,
                SampleFields.BOUNDED_SUFFIX_COUNT: entry.bounded_suffix_count,
                SampleFields.MAX_PATH_LENGTH: torch.as_tensor(
                    -1 if entry.max_path_length is None else int(entry.max_path_length),
                    dtype=torch.long,
                ),
            }

            StorageSchema.validate(sample_dict)

            split_name = sample.split
            if split_name not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Unsupported split {split_name!r}; expected one of {ALLOWED_SPLITS}."
                )

            shard_id = assign_lmdb_shard(entry.sample_id, lmdb_shards)
            env_key = (split_name, shard_id)

            txn = txns[env_key]
            txn.put(entry.sample_id.encode("utf-8"), serialize_sample(sample_dict))

            uncommitted_counts[env_key] += 1
            if uncommitted_counts[env_key] >= commit_frequency:
                txn.commit()
                txns[env_key] = envs[env_key].begin(write=True)
                uncommitted_counts[env_key] = 0

            runtime_manifest[split_name]["sample_ids"].append(entry.sample_id)
            runtime_manifest[split_name]["questions"].append(sample.question)
            runtime_manifest[split_name]["num_nodes"].append(num_nodes)
            runtime_manifest[split_name]["num_edges"].append(len(edge_relation_ids_global))
            runtime_manifest[split_name]["question_tokens"].append(1)

        for txn in txns.values():
            txn.commit()

    for split_name, m in runtime_manifest.items():
        if not m["sample_ids"]:
            continue
        save_manifest(
            manifest_path(embeddings_dir, split_name),
            split=split_name,
            sample_ids=m["sample_ids"],
            questions=m["questions"],
            num_nodes=m["num_nodes"],
            num_edges=m["num_edges"],
            question_tokens=m["question_tokens"],
        )

    log.info("Materialization complete.")


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


__all__ = ["materialize_preprocessed_data"]
