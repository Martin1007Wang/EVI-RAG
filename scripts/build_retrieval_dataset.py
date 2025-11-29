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

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import lmdb
import torch
import numpy as np
from omegaconf import DictConfig
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


def build_dataset(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)

    num_topics = int(cfg.get("num_topics", 2))
    if num_topics <= 0:
        raise ValueError("num_topics must be positive to build topic features.")

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
    graph_cols = {
        "node_entity_ids": graphs_table.column("node_entity_ids").to_pylist(),
        "node_embedding_ids": graphs_table.column("node_embedding_ids").to_pylist(),
        "edge_src": graphs_table.column("edge_src").to_pylist(),
        "edge_dst": graphs_table.column("edge_dst").to_pylist(),
        "edge_relation_ids": graphs_table.column("edge_relation_ids").to_pylist(),
        "positive_edge_mask": graphs_table.column("positive_edge_mask").to_pylist(),
        "gt_path_edge_indices": graphs_table.column("gt_path_edge_indices").to_pylist(),
        "gt_path_node_indices": graphs_table.column("gt_path_node_indices").to_pylist(),
    }

    questions_rows = questions_table.num_rows
    print(f"Preparing {questions_rows} samples...")

    # Set up LMDB environments
    envs: Dict[str, lmdb.Environment] = {}
    map_size = cfg.map_size_gb * (1 << 30)
    all_splits = questions_table.column("split").unique().to_pylist()
    for split in all_splits:
        envs[split] = lmdb.open(
            str(emb_dir / f"{split}.lmdb"),
            map_size=map_size,
            subdir=True,
            lock=False,
        )

    try:
        txn_cache: Dict[str, lmdb.Transaction] = {split: env.begin(write=True) for split, env in envs.items()}
        pending: Dict[str, int] = {split: 0 for split in envs.keys()}

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
                labels = graph_cols["positive_edge_mask"][g_idx]

                num_nodes = len(node_entity_ids)
                num_edges = len(edge_src)
                node_global_ids = torch.tensor(node_entity_ids, dtype=torch.long)
                node_emb_ids = torch.tensor(node_embedding_ids, dtype=torch.long)
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_attr = torch.tensor(edge_rel, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.float32)
                topic_one_hot = torch.zeros((num_nodes, num_topics), dtype=torch.float32)

                q_entities = q_batch_dict["seed_entity_ids"][i] or []
                a_entities = q_batch_dict["answer_entity_ids"][i] or []
                q_local = _local_indices(node_entity_ids, q_entities)
                a_local = _local_indices(node_entity_ids, a_entities)
                if q_local and num_topics > 0:
                    topic_one_hot[torch.as_tensor(q_local, dtype=torch.long), 0] = 1.0
                if a_local and num_topics > 1:
                    topic_one_hot[torch.as_tensor(a_local, dtype=torch.long), 1] = 1.0
                q_emb_ids = (q_batch_dict.get("seed_embedding_ids", [[]] * (i + 1)) or [[]])[i] or []
                a_emb_ids = (q_batch_dict.get("answer_embedding_ids", [[]] * (i + 1)) or [[]])[i] or []

                path_edge_indices = graph_cols["gt_path_edge_indices"][g_idx] or []
                path_node_indices = graph_cols["gt_path_node_indices"][g_idx] or []

                def _edge_triples(edges: Iterable[int]) -> List[Tuple[int, int, int]]:
                    triples: List[Tuple[int, int, int]] = []
                    for e_idx in edges:
                        if 0 <= e_idx < num_edges:
                            h = node_entity_ids[edge_src[e_idx]]
                            t = node_entity_ids[edge_dst[e_idx]]
                            r = edge_rel[e_idx]
                            triples.append((h, r, t))
                    return triples

                sample = {
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "labels": label_tensor,
                    "num_nodes": num_nodes,
                    "node_global_ids": node_global_ids,
                    "node_embedding_ids": node_emb_ids,
                    "topic_one_hot": topic_one_hot,
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
                "gt_paths_nodes": [path_node_indices] if path_node_indices else [],
                "gt_paths_triples": [_edge_triples(path_edge_indices)] if path_edge_indices else [],
                # 直接存检索图中的 GT 边索引，便于下游无需反推
                "gt_path_edge_indices": path_edge_indices if path_edge_indices else [],
                # 兼容性：存原始节点索引列表（非必需）
                "gt_path_node_indices": path_node_indices if path_node_indices else [],
                "sample_id": graph_id,
                "idx": processed + i,
                "metadata": meta_val,
            }

                txn = txn_cache[split]
                _write_sample(txn, graph_id, sample)
                pending[split] += 1
                if pending[split] >= cfg.txn_size:
                    txn.commit()
                    txn_cache[split] = envs[split].begin(write=True)
                    pending[split] = 0
            processed += q_batch.num_rows

        for split, txn in txn_cache.items():
            if pending[split] > 0:
                txn.commit()
    finally:
        for env in envs.values():
            env.close()


@hydra.main(version_base=None, config_path="../configs", config_name="build_retrieval_dataset")
def main(cfg: DictConfig) -> None:
    _ensure_dir(Path(cfg.output_dir))
    build_dataset(cfg)


if __name__ == "__main__":
    main()
