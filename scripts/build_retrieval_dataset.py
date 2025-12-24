#!/usr/bin/env python3
"""Build retrieval-ready LMDB caches (with GT paths) from unified parquet dumps.

This script expects the normalized parquet files described in PREPROCESSING.md:
  - graphs.parquet
  - questions.parquet
  - entity_vocab.parquet
  - relation_vocab.parquet

It will:
  1) Write vocabulary.lmdb (entity/relation <-> id)
  2) Encode entity/relation/question texts with a HuggingFace encoder
  3) Materialize per-split LMDBs under embeddings/{split}.lmdb with fields consumed
     by src.data.g_retrieval_dataset.GRetrievalDataset and downstream g_agent/GFlowNet.

Design goals: vectorized text encoding, minimal Python overhead, and explicit GT
path materialization (avoid recomputing shortest paths later).
"""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
import lmdb
import torch
import numpy as np
try:
    from omegaconf import DictConfig
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = object  # type: ignore[assignment]
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError as exc:  # pragma: no cover - dependencies checked at runtime
    raise SystemExit("pyarrow is required for preprocessing. pip install pyarrow.") from exc

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required for text encoding. pip install transformers.") from exc


class TextEncoder:
    """Lightweight wrapper around a HuggingFace encoder."""

    def __init__(self, model_name: str, device: str, fp16: bool, progress: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.progress = progress

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None,
    ) -> torch.Tensor:
        all_embeds: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        use_progress = self.progress if show_progress is None else show_progress
        if use_progress:
            total = (len(texts) + batch_size - 1) // batch_size
            iterator = tqdm(iterator, total=total, desc=desc or "Encoding", leave=False)
        for start in iterator:
            chunk = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.to(self.dtype)
            mask = inputs["attention_mask"].to(self.dtype).unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / denom
            all_embeds.append(pooled.to("cpu", dtype=torch.float32))
        return torch.cat(all_embeds, dim=0)


def _load_parquet(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pq.read_table(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_vocab_lmdb(entity_labels: List[str], relation_labels: List[str], vocab_dir: Path, map_size_bytes: int) -> None:
    _ensure_dir(vocab_dir)
    env = lmdb.open(str(vocab_dir), map_size=map_size_bytes)
    try:
        with env.begin(write=True) as txn:
            entity_to_id = {label: idx for idx, label in enumerate(entity_labels)}
            id_to_entity = {idx: label for idx, label in enumerate(entity_labels)}
            relation_to_id = {label: idx for idx, label in enumerate(relation_labels)}
            id_to_relation = {idx: label for idx, label in enumerate(relation_labels)}
            txn.put(b"entity_to_id", pickle.dumps(entity_to_id))
            txn.put(b"id_to_entity", pickle.dumps(id_to_entity))
            txn.put(b"relation_to_id", pickle.dumps(relation_to_id))
            txn.put(b"id_to_relation", pickle.dumps(id_to_relation))
    finally:
        env.close()


def _local_indices(node_ids: Sequence[int], targets: Sequence[int]) -> List[int]:
    position = {nid: idx for idx, nid in enumerate(node_ids)}
    return [position[t] for t in targets if t in position]


def _write_sample(txn, sample_id: str, sample: Dict) -> None:
    txn.put(sample_id.encode("utf-8"), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))


def _encode_to_memmap(
    encoder: TextEncoder,
    texts: List[str],
    emb_ids: List[int],
    batch_size: int,
    max_embedding_id: int,
    out_path: Path,
    desc: str,
    show_progress: bool,
) -> torch.Tensor:
    if max_embedding_id < 0:
        return torch.empty((0, 0), dtype=torch.float32)
    if len(texts) != len(emb_ids):
        raise ValueError("texts and emb_ids must have the same length")
    if len(texts) == 0:
        tensor = torch.zeros((max_embedding_id + 1, 0), dtype=torch.float32)
        torch.save(tensor, out_path)
        return tensor
    # Encode first chunk to determine dim
    first_chunk = texts[: batch_size] if texts else []
    first_ids = emb_ids[: batch_size] if emb_ids else []
    first_emb = encoder.encode(first_chunk, batch_size, show_progress=False)
    emb_dim = first_emb.shape[1] if first_emb.numel() > 0 else 0
    mmap_path = out_path.with_suffix(out_path.suffix + ".mmap")
    mem = np.memmap(mmap_path, mode="w+", dtype="float32", shape=(max_embedding_id + 1, emb_dim))
    mem[:] = 0.0

    def _write_chunk(emb_chunk: torch.Tensor, id_chunk: List[int]) -> None:
        for emb_tensor, emb_id in zip(emb_chunk, id_chunk):
            if 0 <= emb_id <= max_embedding_id:
                mem[emb_id] = emb_tensor.cpu().numpy()

    _write_chunk(first_emb, first_ids)
    iterator = range(batch_size, len(texts), batch_size)
    if show_progress:
        total = max(0, (len(texts) - batch_size + batch_size - 1) // batch_size)
        iterator = tqdm(iterator, total=total, desc=desc, leave=False)
    for start in iterator:
        chunk_texts = texts[start : start + batch_size]
        chunk_ids = emb_ids[start : start + batch_size]
        emb_chunk = encoder.encode(chunk_texts, batch_size, show_progress=False)
        _write_chunk(emb_chunk, chunk_ids)

    mem.flush()
    tensor = torch.from_numpy(mem)
    torch.save(tensor, out_path)
    try:
        del mem
        mmap_path.unlink(missing_ok=True)
    except Exception:
        pass
    return tensor


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    _ensure_dir(link_path.parent)
    os.symlink(str(target_path), str(link_path))


def _prepare_lmdb_dir(path: Path, *, overwrite: bool) -> Path:
    """Prepare a clean temporary LMDB directory and return its path.

    We always write into a sibling ``*.tmp`` directory and atomically swap on success.
    This prevents stale keys from older builds from leaking into the current dataset.
    """
    tmp_path = Path(str(path) + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"LMDB already exists at {path}; set overwrite_lmdb=true to rebuild deterministically.")
    _ensure_dir(tmp_path)
    return tmp_path


def _finalize_lmdb_dir(*, tmp_path: Path, final_path: Path, overwrite: bool) -> None:
    if not tmp_path.exists():
        raise FileNotFoundError(f"Temporary LMDB dir missing: {tmp_path}")
    if final_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing LMDB at {final_path}")
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)


def _mean_aggregate(edge_index: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return x.new_zeros((num_nodes, x.size(1)))
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    out = x.new_zeros((num_nodes, x.size(1)))
    out.index_add_(0, dst, x[src])
    deg = x.new_zeros((num_nodes,), dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    return out / deg.clamp_min(1.0).unsqueeze(-1)


def _build_topic_pe(
    *,
    topic_one_hot: torch.Tensor,
    edge_index: torch.Tensor,
    num_rounds: int,
    num_reverse_rounds: int,
) -> torch.Tensor:
    num_nodes = int(topic_one_hot.size(0))
    if num_nodes == 0:
        return topic_one_hot.new_empty((0, topic_one_hot.size(1) * (1 + num_rounds + num_reverse_rounds)))
    features = [topic_one_hot]
    h = topic_one_hot
    for _ in range(num_rounds):
        h = _mean_aggregate(edge_index, h, num_nodes)
        features.append(h)
    rev_edge_index = edge_index.flip(0)
    h_rev = topic_one_hot
    for _ in range(num_reverse_rounds):
        h_rev = _mean_aggregate(rev_edge_index, h_rev, num_nodes)
        features.append(h_rev)
    stacked = torch.stack(features, dim=-1)
    return stacked.reshape(num_nodes, -1)


def build_dataset(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)

    # Topic features follow SubgraphRAG parity: 2-class one-hot {non-topic, topic}.
    # Topic is defined by query entities only; answers are supervision and must NOT leak into inputs.
    num_topics = int(cfg.get("num_topics", 2))
    if num_topics < 2:
        raise ValueError("num_topics must be >= 2 to build SubgraphRAG-parity topic features.")

    topic_pe_cfg = cfg.get("topic_pe") or {}
    topic_pe_precompute = bool(topic_pe_cfg.get("precompute", False))
    if topic_pe_precompute and ("num_rounds" not in topic_pe_cfg or "num_reverse_rounds" not in topic_pe_cfg):
        raise ValueError("topic_pe.precompute=true requires explicit num_rounds and num_reverse_rounds.")
    topic_pe_rounds = int(topic_pe_cfg.get("num_rounds", 0))
    topic_pe_rev_rounds = int(topic_pe_cfg.get("num_reverse_rounds", 0))
    if topic_pe_rounds < 0 or topic_pe_rev_rounds < 0:
        raise ValueError("topic_pe.num_rounds and num_reverse_rounds must be >= 0.")

    # Load vocabulary tables (small) and convert to Python dicts
    entity_vocab = _load_parquet(Path(cfg.parquet_dir) / "entity_vocab.parquet").to_pydict()
    embedding_vocab = _load_parquet(Path(cfg.parquet_dir) / "embedding_vocab.parquet").to_pydict()
    relation_vocab = _load_parquet(Path(cfg.parquet_dir) / "relation_vocab.parquet").to_pydict()

    # Sort vocab by explicit ids to ensure determinism
    entity_rows = sorted(zip(entity_vocab["entity_id"], entity_vocab["label"]), key=lambda x: x[0])
    entity_labels: List[str] = [str(label) for _, label in entity_rows]
    relation_rows = sorted(zip(relation_vocab["relation_id"], relation_vocab["label"]), key=lambda x: x[0])
    relation_labels: List[str] = [str(label) for _, label in relation_rows]

    vocab_map_size = cfg.get("vocab_map_size_gb", 1) * (1 << 30)
    _write_vocab_lmdb(
        entity_labels,
        relation_labels,
        Path(cfg.output_dir) / "vocabulary" / "vocabulary.lmdb",
        vocab_map_size,
    )

    encoder = TextEncoder(cfg.encoder, cfg.device, cfg.fp16, cfg.progress_bar)
    # Build entity embedding tensor with slot 0 reserved for non-text placeholders.
    emb_rows = sorted(zip(embedding_vocab["embedding_id"], embedding_vocab["label"]), key=lambda x: x[0])
    text_labels: List[str] = [str(label) for _, label in emb_rows]
    text_ids: List[int] = [int(eid) for eid, _ in emb_rows]
    print("Encoding textual entity labels...")
    max_embedding_id = max(entity_vocab["embedding_id"]) if entity_vocab["embedding_id"] else 0
    emb_dir = Path(cfg.output_dir) / "embeddings"
    _ensure_dir(emb_dir)
    entity_emb = _encode_to_memmap(
        encoder=encoder,
        texts=text_labels,
        emb_ids=text_ids,
        batch_size=cfg.batch_size,
        max_embedding_id=max_embedding_id,
        out_path=emb_dir / "entity_embeddings.pt",
        desc="Entities",
        show_progress=cfg.progress_bar,
    )
    print("Encoding relation labels...")
    relation_emb = encoder.encode(relation_labels, cfg.batch_size, show_progress=cfg.progress_bar, desc="Relations")
    torch.save(relation_emb, emb_dir / "relation_embeddings.pt")

    sub_cfg = cfg.get("sub", {})
    build_sub = bool(sub_cfg.get("enabled", False))
    sub_parquet_dir = Path(sub_cfg.get("parquet_dir", ""))
    sub_output_dir = Path(sub_cfg.get("output_dir", ""))
    share_artifacts = bool(sub_cfg.get("share_vocab_and_embeddings", True))
    sub_graph_ids: Set[str] = set()
    if build_sub:
        if not sub_parquet_dir.exists():
            build_sub = False
        else:
            sub_questions_path = sub_parquet_dir / "questions.parquet"
            if not sub_questions_path.exists():
                build_sub = False
            else:
                sub_graph_ids = set(pq.read_table(sub_questions_path, columns=["graph_id"]).column("graph_id").to_pylist())

    if build_sub and share_artifacts:
        _ensure_dir(sub_output_dir)
        base_vocab_dir = Path(cfg.output_dir) / "vocabulary" / "vocabulary.lmdb"
        sub_vocab_dir = sub_output_dir / "vocabulary" / "vocabulary.lmdb"
        _ensure_symlink(sub_vocab_dir, base_vocab_dir)
        sub_emb_dir = sub_output_dir / "embeddings"
        _ensure_dir(sub_emb_dir)
        _ensure_symlink(sub_emb_dir / "entity_embeddings.pt", emb_dir / "entity_embeddings.pt")
        _ensure_symlink(sub_emb_dir / "relation_embeddings.pt", emb_dir / "relation_embeddings.pt")
    elif build_sub and not share_artifacts:
        raise NotImplementedError("sub.share_vocab_and_embeddings=false is not supported; use symlinks to enforce SSOT.")

    # Load main tables as pyarrow objects without converting to dicts
    graphs_table = _load_parquet(Path(cfg.parquet_dir) / "graphs.parquet")
    questions_table = _load_parquet(Path(cfg.parquet_dir) / "questions.parquet")
    graph_ids_all = graphs_table.column("graph_id")
    # Integrity: graph_id must be unique; detect duplicates early.
    distinct = graph_ids_all.unique()
    if len(distinct) != len(graph_ids_all):
        # Find a few offending ids for debugging.
        from collections import Counter

        counts = Counter(graph_ids_all.to_pylist())
        dupes = [gid for gid, c in counts.items() if c > 1][:5]
        raise RuntimeError(f"Duplicate graph_id detected in graphs.parquet, examples: {dupes}")

    graph_id_list = graph_ids_all.to_pylist()
    graph_id_to_row: Dict[str, int] = {gid: idx for idx, gid in enumerate(graph_id_list)}
    if "positive_triple_mask" not in graphs_table.schema.names:
        raise RuntimeError(
            "graphs.parquet is missing required column `positive_triple_mask`. "
            "Re-run scripts/build_retrieval_parquet.py to regenerate normalized parquet with triple-level supervision."
        )

    def _optional_graph_col(name: str, default: object) -> List:
        if name in graphs_table.schema.names:
            return graphs_table.column(name).to_pylist()
        return [default for _ in range(graphs_table.num_rows)]

    graph_cols: Dict[str, List] = {
        "node_entity_ids": graphs_table.column("node_entity_ids").to_pylist(),
        "node_embedding_ids": graphs_table.column("node_embedding_ids").to_pylist(),
        "edge_src": graphs_table.column("edge_src").to_pylist(),
        "edge_dst": graphs_table.column("edge_dst").to_pylist(),
        "edge_relation_ids": graphs_table.column("edge_relation_ids").to_pylist(),
        # Fixed invariant: retriever supervision is ALWAYS triple-level.
        "positive_triple_mask": graphs_table.column("positive_triple_mask").to_pylist(),
        "pair_start_node_locals": _optional_graph_col("pair_start_node_locals", []),
        "pair_answer_node_locals": _optional_graph_col("pair_answer_node_locals", []),
        "pair_edge_indices": _optional_graph_col("pair_edge_indices", []),
        "pair_edge_ptr": _optional_graph_col("pair_edge_ptr", [0]),
        "gt_path_edge_indices": graphs_table.column("gt_path_edge_indices").to_pylist(),
        "gt_path_node_indices": graphs_table.column("gt_path_node_indices").to_pylist(),
    }

    questions_rows = questions_table.num_rows
    print(f"Preparing {questions_rows} samples...")

    overwrite_lmdb = bool(cfg.get("overwrite_lmdb", True))

    # Set up LMDB environments
    envs: Dict[str, lmdb.Environment] = {}
    sub_envs: Dict[str, lmdb.Environment] = {}
    map_size = cfg.map_size_gb * (1 << 30)
    all_splits = questions_table.column("split").unique().to_pylist()
    tmp_dirs: Dict[str, Path] = {}
    sub_tmp_dirs: Dict[str, Path] = {}
    for split in all_splits:
        final_dir = emb_dir / f"{split}.lmdb"
        tmp_dir = _prepare_lmdb_dir(final_dir, overwrite=overwrite_lmdb)
        tmp_dirs[str(split)] = tmp_dir
        envs[split] = lmdb.open(
            str(tmp_dir),
            map_size=map_size,
            subdir=True,
            lock=False,
        )
        if build_sub and share_artifacts:
            sub_emb_dir = sub_output_dir / "embeddings"
            sub_final_dir = sub_emb_dir / f"{split}.lmdb"
            sub_tmp_dir = _prepare_lmdb_dir(sub_final_dir, overwrite=overwrite_lmdb)
            sub_tmp_dirs[str(split)] = sub_tmp_dir
            sub_envs[split] = lmdb.open(
                str(sub_tmp_dir),
                map_size=map_size,
                subdir=True,
                lock=False,
            )

    success = False
    try:
        txn_cache: Dict[str, lmdb.Transaction] = {split: env.begin(write=True) for split, env in envs.items()}
        pending: Dict[str, int] = {split: 0 for split in envs.keys()}
        sub_txn_cache: Dict[str, lmdb.Transaction] = {split: env.begin(write=True) for split, env in sub_envs.items()}
        sub_pending: Dict[str, int] = {split: 0 for split in sub_envs.keys()}

        question_batches = questions_table.to_batches(max_chunksize=cfg.batch_size)
        total_batches = questions_table.num_rows / cfg.batch_size
        pbar = tqdm(question_batches, total=int(total_batches) + 1, desc="Writing LMDB")
        processed = 0  # running row offset since RecordBatch has no native offset attribute
        missing_graph_ids: List[str] = []

        for q_batch in pbar:
            q_batch_dict = q_batch.to_pydict()
            q_batch_texts = [str(q) for q in q_batch_dict["question"]]
            q_batch_emb = encoder.encode(q_batch_texts, cfg.batch_size, show_progress=False)

            required_indices: List[int] = []
            for gid in q_batch_dict["graph_id"]:
                idx = graph_id_to_row.get(gid)
                if idx is None:
                    missing_graph_ids.append(gid)
                else:
                    required_indices.append(idx)

            if missing_graph_ids:
                sample_ids = ", ".join(list(dict.fromkeys(missing_graph_ids))[:5])
                raise RuntimeError(f"Missing graph_id(s) in graphs.parquet, examples: {sample_ids}")

            for i in range(q_batch.num_rows):
                graph_id = q_batch_dict["graph_id"][i]
                split = q_batch_dict["split"][i]
                if split not in envs:
                    continue

                meta_val = ""
                if "metadata" in q_batch_dict:
                    meta_candidate = q_batch_dict["metadata"][i]
                    meta_val = "" if meta_candidate is None else str(meta_candidate)

                g_idx = graph_id_to_row[graph_id]
                node_entity_ids = graph_cols["node_entity_ids"][g_idx]
                node_embedding_ids = graph_cols["node_embedding_ids"][g_idx]
                edge_src = graph_cols["edge_src"][g_idx]
                edge_dst = graph_cols["edge_dst"][g_idx]
                edge_rel = graph_cols["edge_relation_ids"][g_idx]
                labels = graph_cols["positive_triple_mask"][g_idx]

                num_nodes = len(node_entity_ids)
                num_edges = len(edge_src)
                if num_edges <= 0:
                    raise ValueError(
                        f"Invalid graph with zero edges for {graph_id} (split={split}). "
                        "Fix raw parquet/filters and rebuild; empty edge_index is unsupported."
                    )
                node_global_ids = torch.tensor(node_entity_ids, dtype=torch.long)
                node_emb_ids = torch.tensor(node_embedding_ids, dtype=torch.long)
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_attr = torch.tensor(edge_rel, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.float32)
                if label_tensor.numel() != num_edges:
                    raise ValueError(
                        f"Label length mismatch for {graph_id}: labels={label_tensor.numel()} vs num_edges={num_edges}. "
                        "Rebuild normalized parquet caches to match the updated schema."
                    )

                q_entities = q_batch_dict["seed_entity_ids"][i] or []
                a_entities = q_batch_dict["answer_entity_ids"][i] or []
                q_local = _local_indices(node_entity_ids, q_entities)
                a_local = _local_indices(node_entity_ids, a_entities)

                topic_entity_mask = torch.zeros(num_nodes, dtype=torch.long)

                if q_local:
                    topic_entity_mask[torch.as_tensor(q_local, dtype=torch.long)] = 1
                topic_one_hot = torch.nn.functional.one_hot(topic_entity_mask, num_classes=num_topics).float()
                topic_pe = None
                if topic_pe_precompute:
                    topic_pe = _build_topic_pe(
                        topic_one_hot=topic_one_hot,
                        edge_index=edge_index,
                        num_rounds=topic_pe_rounds,
                        num_reverse_rounds=topic_pe_rev_rounds,
                    )
                q_emb_ids = (q_batch_dict.get("seed_embedding_ids", [[]] * (i + 1)) or [[]])[i] or []
                a_emb_ids = (q_batch_dict.get("answer_embedding_ids", [[]] * (i + 1)) or [[]])[i] or []

                path_edge_indices = graph_cols["gt_path_edge_indices"][g_idx] or []
                path_node_indices = graph_cols["gt_path_node_indices"][g_idx] or []
                pair_start_node_locals = graph_cols["pair_start_node_locals"][g_idx] or []
                pair_answer_node_locals = graph_cols["pair_answer_node_locals"][g_idx] or []
                pair_edge_indices = graph_cols["pair_edge_indices"][g_idx] or []
                pair_edge_ptr = graph_cols["pair_edge_ptr"][g_idx] or [0]

                sample = {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "labels": label_tensor,
                    "num_nodes": num_nodes,
                    "node_global_ids": node_global_ids,
                    "node_embedding_ids": node_emb_ids,
                    "question_emb": q_batch_emb[i],
                    "question": q_batch_dict["question"][i],
                    # 全局种子实体（用于下游 start_entity_ids）
                    "seed_entity_ids": torch.as_tensor(q_entities, dtype=torch.long),
                    "q_local_indices": q_local,
                    "a_local_indices": a_local,
                    "answer_entity_ids": torch.as_tensor(a_entities, dtype=torch.long),
                    "answer_entity_ids_len": torch.tensor([len(a_entities)], dtype=torch.long),
                    "q_embedding_ids": q_emb_ids,
                    "a_embedding_ids": a_emb_ids,
                    "pair_start_node_locals": pair_start_node_locals,
                    "pair_answer_node_locals": pair_answer_node_locals,
                    "pair_edge_indices": pair_edge_indices,
                    "pair_edge_ptr": pair_edge_ptr,
                    # 直接存检索图中的 GT 边索引，便于下游无需反推
                    "gt_path_edge_indices": path_edge_indices if path_edge_indices else [],
                    # 兼容性：存原始节点索引列表（非必需）
                    "gt_path_node_indices": path_node_indices if path_node_indices else [],
                    "sample_id": graph_id,
                    "idx": processed + i,
                    "metadata": meta_val,
                }
                if not topic_pe_precompute:
                    sample["topic_one_hot"] = topic_one_hot
                if topic_pe is not None:
                    sample["topic_pe"] = topic_pe

                txn = txn_cache[split]
                _write_sample(txn, graph_id, sample)
                pending[split] += 1
                if pending[split] >= cfg.txn_size:
                    txn.commit()
                    txn_cache[split] = envs[split].begin(write=True)
                    pending[split] = 0

                if sub_envs and graph_id in sub_graph_ids:
                    sub_txn = sub_txn_cache[split]
                    _write_sample(sub_txn, graph_id, sample)
                    sub_pending[split] += 1
                    if sub_pending[split] >= cfg.txn_size:
                        sub_txn.commit()
                        sub_txn_cache[split] = sub_envs[split].begin(write=True)
                        sub_pending[split] = 0
            processed += q_batch.num_rows

        for split, txn in txn_cache.items():
            if pending[split] > 0:
                txn.commit()
        for split, txn in sub_txn_cache.items():
            if sub_pending[split] > 0:
                txn.commit()
        success = True
    finally:
        for env in envs.values():
            env.close()
        for env in sub_envs.values():
            env.close()
        if success:
            for split in all_splits:
                split_key = str(split)
                final_dir = emb_dir / f"{split}.lmdb"
                _finalize_lmdb_dir(tmp_path=tmp_dirs[split_key], final_path=final_dir, overwrite=overwrite_lmdb)
                if build_sub and share_artifacts and split_key in sub_tmp_dirs:
                    sub_emb_dir = sub_output_dir / "embeddings"
                    sub_final_dir = sub_emb_dir / f"{split}.lmdb"
                    _finalize_lmdb_dir(tmp_path=sub_tmp_dirs[split_key], final_path=sub_final_dir, overwrite=overwrite_lmdb)
        else:
            # Keep tmp dirs for debugging, but avoid silently masking partial builds.
            pass


if hydra is not None:

    @hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_dataset")
    def main(cfg: DictConfig) -> None:
        _ensure_dir(Path(cfg.output_dir))
        build_dataset(cfg)

else:  # pragma: no cover

    def main(cfg: DictConfig) -> None:
        raise ModuleNotFoundError("hydra-core is required to run scripts/build_retrieval_dataset.py")


if __name__ == "__main__":
    main()
