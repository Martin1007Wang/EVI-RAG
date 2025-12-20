from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch_geometric.data import Batch

from src.data.g_agent_dataset import GAgentSample

from src.models.components.retriever import RetrieverOutput
from src.data.components.embedding_store import EmbeddingStore

log = logging.getLogger(__name__)


@dataclass
class GAgentSettings:
    """Declarative configuration for constructing g_agent."""

    enabled: bool = True
    # Top-K edges kept from retriever scores (per-sample, retrieval graph space).
    # Final g_agent edges = Top-K ∪ GT-path edges, then deduplicated by (h,r,t) triple.
    anchor_top_k: int = 50
    # Top-K edges incident to start entities (per-sample, retrieval graph space).
    anchor_start_k: int = 50
    output_path: Path = Path("g_agent/g_agent_samples.pt")
    force_include_gt: bool = False

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.anchor_top_k = int(self.anchor_top_k)
        self.anchor_start_k = int(self.anchor_start_k)
        self.output_path = Path(self.output_path).expanduser()
        self.force_include_gt = bool(self.force_include_gt)

    def to_metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_path"] = str(payload.get("output_path"))
        return payload


@dataclass(frozen=True)
class _GraphSlice:
    """Internal helper to hold GPU tensors for a single sample graph."""

    heads: torch.Tensor  # Local indices in G_retrieval (0 ~ N_retrieval)
    tails: torch.Tensor
    relations: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    node_global_ids: torch.Tensor  # Global Entity IDs
    node_embedding_ids: torch.Tensor  # Embedding table row per node
    # Edge indices within the per-sample retrieval graph (0 ~ E_retrieval-1).
    # Needed to align GT edge indices stored in LMDB with selected edges.
    retrieval_edge_indices: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.heads.numel())

    @property
    def num_nodes(self) -> int:
        return int(self.node_global_ids.numel())


class GAgentBuilder:
    """
    The Producer: Materialize clean GAgentSample objects from retriever scores.
    """

    def __init__(self, settings: GAgentSettings, embedding_store: Optional[EmbeddingStore] = None) -> None:
        self.cfg = settings
        self.embedding_store = embedding_store
        self.samples: List[GAgentSample] = []

        # Counters
        self.stats = {
            "num_samples": 0,
            "path_exists": 0,
            "retrieval_failed": 0,
            "gt_broken": 0,
            "edge_counts": [],
            "path_lengths": [],
        }

    def reset(self):
        self.samples = []
        for k in ["num_samples", "path_exists", "retrieval_failed", "gt_broken"]:
            self.stats[k] = 0
        self.stats["edge_counts"] = []
        self.stats["path_lengths"] = []

    def process_batch(self, batch: Batch, model_output: "RetrieverOutput") -> None:
        """
        Entry point for a batch. Slices it into individual graphs and processes them.
        """
        # 1. Extract basic info from batch
        if not hasattr(batch, "ptr"):
            raise ValueError("Batch must have 'ptr' for slicing.")

        ptr = batch.ptr
        batch_size = ptr.numel() - 1

        # Detach tensors to CPU to avoid CUDA OOM during graph algo
        scores = model_output.scores.detach().cpu()
        edge_index = batch.edge_index.detach().cpu()
        relations = batch.edge_attr.detach().cpu()
        labels = batch.labels.detach().cpu()
        node_global_ids = batch.node_global_ids.detach().cpu()

        # Sample IDs list
        sample_ids = getattr(batch, "sample_id", [])

        # Group edges by sample index (using model_output.query_ids or ptr logic)
        # Here we use query_ids for robustness if available, otherwise infer from ptr?
        # Usually retriever output aligns with edge_index.
        # But edge_index is sparse. We need to know which edges belong to which sample.
        # 'query_ids' from RetrieverOutput maps each edge to a sample index [0, B-1]
        query_ids = model_output.query_ids.detach().cpu()

        # Pre-calculate edge ranges for each sample
        # Assumes edges are sorted by query_id (which they usually are in PyG batch)
        # But let's be safe and use a grouping map
        edge_indices_by_sample = [[] for _ in range(batch_size)]
        for edge_idx, sample_idx in enumerate(query_ids.tolist()):
            if 0 <= int(sample_idx) < batch_size:
                edge_indices_by_sample[int(sample_idx)].append(edge_idx)

        for i in range(batch_size):
            # 2. Construct GraphSlice
            edge_indices = edge_indices_by_sample[i]
            if not edge_indices:
                continue

            edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)

            node_start = int(ptr[i].item())
            node_end = int(ptr[i + 1].item())
            node_slice = node_global_ids[node_start:node_end]
            if not hasattr(batch, "node_embedding_ids"):
                raise AttributeError("Batch missing node_embedding_ids; retriever dataset must provide embedding ids per node.")
            node_embedding_slice = batch.node_embedding_ids.detach().cpu()[node_start:node_end]

            # Convert global edge_index to local (0~N_retrieval)
            # Note: edge_index in batch is already offset-adjusted if using PyG Batch?
            # Usually PyG Batch stacks node indices (0~N1, N1~N2...).
            # So we subtract node_start.
            heads = edge_index[0, edge_indices_tensor] - node_start
            tails = edge_index[1, edge_indices_tensor] - node_start

            graph_slice = _GraphSlice(
                heads=heads,
                tails=tails,
                relations=relations[edge_indices_tensor],
                labels=labels[edge_indices_tensor],
                scores=scores[edge_indices_tensor],
                node_global_ids=node_slice,
                node_embedding_ids=node_embedding_slice,
                retrieval_edge_indices=edge_indices_tensor,
            )

            # 3. Process Single Sample
            try:
                sid = str(sample_ids[i])
            except IndexError:
                sid = f"unknown_{i}"

            self._build_and_add_sample(sid, graph_slice)

    def _build_and_add_sample(self, sample_id: str, graph: _GraphSlice) -> None:
        """
        Core Logic: Top-K by score -> Union GT -> Dedup by triple -> Re-index -> Create Object
        """
        # === A. Top-K Filter (retriever space) ===
        num_edges_full = graph.num_edges
        if num_edges_full <= 0:
            return
        top_k = int(self.cfg.anchor_top_k)
        if top_k <= 0:
            raise ValueError(f"anchor_top_k must be > 0, got {top_k} (sample_id={sample_id}).")
        k = min(top_k, num_edges_full)
        scores_full = graph.scores.float()
        _, top_idx = torch.topk(scores_full, k=k, largest=True, sorted=True)
        top_locals: Set[int] = {int(i) for i in top_idx.tolist()}

        # === B. Fetch Metadata from LMDB ===
        # This is crucial for "One Source of Truth"
        if self.embedding_store is None:
            raise ValueError("EmbeddingStore must be provided; builder cannot proceed without GT metadata.")

        try:
            raw_data = self.embedding_store.load_sample(sample_id)
        except KeyError:
            log.warning(f"Sample {sample_id} not found in LMDB.")
            return

        question_raw = raw_data.get("question_emb", [])
        question_emb = torch.as_tensor(question_raw, dtype=torch.float32).detach().clone()
        question_text = raw_data.get("question", "")
        # Standardize to LongTensor
        start_entity_ids = torch.as_tensor(raw_data.get("seed_entity_ids", []), dtype=torch.long).detach().clone()
        if start_entity_ids.numel() == 0:
            raise ValueError(f"Sample {sample_id} missing seed_entity_ids (start_entity_ids).")

        answer_entity_ids = torch.as_tensor(raw_data.get("answer_entity_ids", []), dtype=torch.long).detach().clone()
        if answer_entity_ids.numel() == 0:
            raise ValueError(f"Sample {sample_id} missing answer_entity_ids.")

        # === B.0 Start-edge Top-K (retrieval space) ===
        start_top_k = int(self.cfg.anchor_start_k)
        if start_top_k <= 0:
            raise ValueError(f"anchor_start_k must be > 0, got {start_top_k} (sample_id={sample_id}).")
        start_mask = torch.isin(graph.node_global_ids, start_entity_ids.view(-1))
        if not bool(start_mask.any().item()):
            raise ValueError(f"Start entities missing from retrieval graph (sample_id={sample_id}).")
        start_edge_mask = start_mask[graph.heads] | start_mask[graph.tails]
        start_edge_indices = torch.nonzero(start_edge_mask, as_tuple=False).view(-1)
        if start_edge_indices.numel() == 0:
            raise ValueError(f"No edges incident to start entities (sample_id={sample_id}).")
        k_start = min(start_top_k, int(start_edge_indices.numel()))
        start_scores = scores_full[start_edge_indices]
        _, start_rank = torch.topk(start_scores, k=k_start, largest=True, sorted=True)
        start_top_idx = start_edge_indices[start_rank]
        start_top_locals: Set[int] = {int(i) for i in start_top_idx.tolist()}

        # === B.1 Resolve GT edge ids (retrieval space) ===
        # Note: some splits/samples may not have GT paths materialized. GT is required only when we
        # explicitly force-include it (training/oracle graph); otherwise we treat it as absent.
        raw_gt_indices = raw_data.get("gt_path_edge_indices", [])  # retrieval-local or batch-global ids
        if not raw_gt_indices:
            gt_triples = raw_data.get("gt_paths_triples", [])
            if gt_triples:
                # Map (h,r,t) triples to retrieval-local edge ids using the full retrieval graph.
                edge_triples_full = []
                for e_idx in range(graph.num_edges):
                    h_global = int(graph.node_global_ids[int(graph.heads[e_idx])].item())
                    t_global = int(graph.node_global_ids[int(graph.tails[e_idx])].item())
                    r_global = int(graph.relations[e_idx].item())
                    edge_triples_full.append((h_global, r_global, t_global))
                triple_to_edge_idx_full: Dict[Tuple[int, int, int], List[int]] = {}
                for idx, triple in enumerate(edge_triples_full):
                    triple_to_edge_idx_full.setdefault(triple, []).append(idx)
                raw_gt_indices = []
                for path in gt_triples:
                    for triple in path:
                        triple_tuple = tuple(int(x) for x in triple)
                        cand = triple_to_edge_idx_full.get(triple_tuple)
                        if cand:
                            raw_gt_indices.append(cand[0])  # 取首个匹配
        if not raw_gt_indices:
            if self.cfg.force_include_gt:
                self.stats["gt_broken"] += 1
                return
            raw_gt_indices = []

        gt_locals: List[int] = []
        gt_locals_set: Set[int] = set()
        if raw_gt_indices:
            # Defensive: accept "batch-global" edge indices by mapping them back to local indices.
            global_to_local = {
                int(val.item()): i for i, val in enumerate(graph.retrieval_edge_indices.view(-1))
            }

            # Map GT edges to retrieval-local indices (0..E_retr-1 of this sample graph slice).
            for ridx in raw_gt_indices:
                r_val = int(ridx) if isinstance(ridx, int) else int(ridx.item())
                if 0 <= r_val < graph.num_edges:
                    if r_val not in gt_locals_set:
                        gt_locals.append(r_val)
                        gt_locals_set.add(r_val)
                    continue
                local_idx = global_to_local.get(r_val)
                if local_idx is not None:
                    if local_idx not in gt_locals_set:
                        gt_locals.append(local_idx)
                        gt_locals_set.add(local_idx)

            if not gt_locals:
                if self.cfg.force_include_gt:
                    self.stats["gt_broken"] += 1
                    return
        # === C. Union (Top-K ∪ GT) + Deduplicate by (h,r,t) ===
        # Order matters for determinism: first Top-K (sorted by score desc), then GT-only edges (sorted by score desc).
        selected_locals: List[int] = [int(i) for i in start_top_idx.tolist()]
        selected_locals.extend([int(i) for i in top_idx.tolist() if int(i) not in start_top_locals])
        if self.cfg.force_include_gt:
            gt_only = [int(i) for i in gt_locals if int(i) not in top_locals and int(i) not in start_top_locals]
            gt_only.sort(key=lambda idx: (-float(scores_full[idx].item()), int(idx)))
            selected_locals.extend(gt_only)

        # Dedup dictionary: triple -> aggregated attributes.
        # We aggregate score by max, label by max, and top_edge_mask as OR over Top-K membership.
        triple_to_agg: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for eidx in selected_locals:
            h_local = int(graph.heads[eidx].item())
            t_local = int(graph.tails[eidx].item())
            h_global = int(graph.node_global_ids[h_local].item())
            t_global = int(graph.node_global_ids[t_local].item())
            r_global = int(graph.relations[eidx].item())
            triple = (h_global, r_global, t_global)
            score = float(graph.scores[eidx].item())
            label = float(graph.labels[eidx].item())
            in_top = (eidx in top_locals) or (eidx in start_top_locals)
            agg = triple_to_agg.get(triple)
            if agg is None:
                triple_to_agg[triple] = {"score": score, "label": label, "in_top": in_top}
                continue
            # Aggregate
            agg["score"] = max(float(agg["score"]), score)
            agg["label"] = max(float(agg["label"]), label)
            agg["in_top"] = bool(agg["in_top"]) or in_top

        triples = list(triple_to_agg.keys())
        if not triples:
            return
        num_edges = len(triples)

        edge_heads_global = torch.tensor([t[0] for t in triples], dtype=torch.long)
        edge_relations = torch.tensor([t[1] for t in triples], dtype=torch.long)
        edge_tails_global = torch.tensor([t[2] for t in triples], dtype=torch.long)
        edge_scores = torch.tensor([float(triple_to_agg[t]["score"]) for t in triples], dtype=torch.float32)
        edge_labels = torch.tensor([float(triple_to_agg[t]["label"]) for t in triples], dtype=torch.float32)
        top_edge_mask = torch.tensor([bool(triple_to_agg[t]["in_top"]) for t in triples], dtype=torch.bool)

        # 2. Build Subgraph Topology (0 ~ N_subgraph-1)
        # Collect all unique nodes in this new small subgraph
        unique_nodes, _ = torch.sort(torch.cat([edge_heads_global, edge_tails_global]).unique())

        # Map Global ID -> Subgraph Index / Embedding Row
        node_map = {gid.item(): i for i, gid in enumerate(unique_nodes)}
        embedding_lookup = {int(g.item()): int(e.item()) for g, e in zip(graph.node_global_ids, graph.node_embedding_ids)}
        unique_embedding_ids = []
        for gid in unique_nodes.tolist():
            if gid not in embedding_lookup:
                raise ValueError(f"Missing embedding_id for global entity {gid} in sample {sample_id}")
            unique_embedding_ids.append(embedding_lookup[gid])
        node_embedding_ids = torch.tensor(unique_embedding_ids, dtype=torch.long)

        # Compute edge topology indices
        edge_head_locals = torch.tensor([node_map[h.item()] for h in edge_heads_global], dtype=torch.long)
        edge_tail_locals = torch.tensor([node_map[t.item()] for t in edge_tails_global], dtype=torch.long)

        # === D. Local indices for question/answer anchors ===
        # 起点必须来自 start_entity_ids ∩ 子图节点，缺失则直接丢弃样本。
        start_node_locals_list: List[int] = []
        for g_id in start_entity_ids.tolist():
            mapped = node_map.get(int(g_id))
            if mapped is not None:
                start_node_locals_list.append(mapped)
        if not start_node_locals_list:
            raise ValueError(f"Start entities missing from selected subgraph (sample_id={sample_id}).")
        seen_q = set()
        start_node_locals = torch.tensor(
            [x for x in start_node_locals_list if not (x in seen_q or seen_q.add(x))],
            dtype=torch.long,
        )

        # 答案实体的全局 ID 必须保留（审计字段），不可因为“不在子图中”就丢样本；
        # 否则训练/评估只在“已知可达”子集上成立，drop_unreachable 也会失效。
        #
        # 语义约束：
        # - answer_entity_ids：原始答案集合（去重、保序），与子图是否包含答案无关
        # - answer_node_locals：子图内的答案节点局部索引（可空，shape [0]）
        seen_a: set[int] = set()
        ordered_answers: List[int] = []
        for a_id in answer_entity_ids.tolist():
            a_int = int(a_id)
            if a_int in seen_a:
                continue
            seen_a.add(a_int)
            ordered_answers.append(a_int)
        answer_entity_ids = torch.tensor(ordered_answers, dtype=torch.long)

        answer_node_locals_list: List[int] = []
        for a_int in ordered_answers:
            mapped = node_map.get(int(a_int))
            if mapped is None:
                continue
            answer_node_locals_list.append(int(mapped))
        answer_node_locals = (
            torch.tensor(answer_node_locals_list, dtype=torch.long)
            if answer_node_locals_list
            else torch.empty(0, dtype=torch.long)
        )

        # === E. Ground Truth Path Mapping (by triple, after dedup) ===
        triple_to_agent = {triple: idx for idx, triple in enumerate(triples)}
        gt_triples_ordered: List[Tuple[int, int, int]] = []
        gt_triples_seen: Set[Tuple[int, int, int]] = set()
        for ridx in gt_locals:
            h_local = int(graph.heads[int(ridx)].item())
            t_local = int(graph.tails[int(ridx)].item())
            h_global = int(graph.node_global_ids[h_local].item())
            t_global = int(graph.node_global_ids[t_local].item())
            r_global = int(graph.relations[int(ridx)].item())
            triple = (h_global, r_global, t_global)
            if triple not in gt_triples_seen:
                gt_triples_ordered.append(triple)
                gt_triples_seen.add(triple)

        gt_path_indices: List[int] = []
        gt_path_seen: Set[int] = set()
        for triple in gt_triples_ordered:
            agent_idx = triple_to_agent.get(triple)
            if agent_idx is not None and agent_idx not in gt_path_seen:
                gt_path_indices.append(int(agent_idx))
                gt_path_seen.add(int(agent_idx))
        gt_path_exists = len(gt_path_indices) > 0
        if self.cfg.force_include_gt and not gt_path_exists:
            raise ValueError(f"Sample {sample_id} GT triples missing in selected subgraph after union/dedup.")

        if gt_path_exists:
            self.stats["path_exists"] += 1
            self.stats["path_lengths"].append(len(gt_path_indices))
        self.stats["edge_counts"].append(num_edges)

        # === F. Create the Object (fully specified schema) ===
        gt_path_edge_local_ids = torch.tensor(gt_path_indices, dtype=torch.long)
        gt_path_node_local_ids = torch.tensor(
            sorted(
                {int(edge_head_locals[i].item()) for i in gt_path_indices}
                | {int(edge_tail_locals[i].item()) for i in gt_path_indices}
            ),
            dtype=torch.long,
        )

        is_answer_reachable = bool(answer_node_locals.numel() > 0)

        sample = GAgentSample(
            sample_id=sample_id,
            question=question_text,
            question_emb=question_emb,
            edge_relations=edge_relations,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            node_entity_ids=unique_nodes,
            node_embedding_ids=node_embedding_ids,
            top_edge_mask=top_edge_mask,
            start_entity_ids=start_entity_ids,
            answer_entity_ids=answer_entity_ids,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            gt_path_edge_local_ids=gt_path_edge_local_ids,
            gt_path_node_local_ids=gt_path_node_local_ids,
            gt_path_exists=gt_path_exists,
            is_answer_reachable=is_answer_reachable,
        )
        # 记录但不跳过：eval 阶段允许 GT 路径被截断后不可达（train 有 force_include_gt 时仍应可达）。
        if sample.gt_path_exists and not sample.is_answer_reachable:
            self.stats["gt_broken"] += 1
            log.warning(
                "GT path exists but answer not reachable after selection (top_k=%d). sample_id=%s",
                int(self.cfg.anchor_top_k),
                sample_id,
            )

        self.samples.append(sample)
        self.stats["num_samples"] += 1

    def save(self, output_path: Path) -> Dict[str, Any] | None:
        if not self.samples:
            log.warning("No samples collected.")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute final stats
        final_stats = {
            "num_samples": self.stats["num_samples"],
            "path_exists_ratio": self.stats["path_exists"] / max(1, self.stats["num_samples"]),
            "retrieval_failed_ratio": self.stats["retrieval_failed"] / max(1, self.stats["num_samples"]),
            "avg_edges": statistics.mean(self.stats["edge_counts"]) if self.stats["edge_counts"] else 0,
            "avg_gt_len": statistics.mean(self.stats["path_lengths"]) if self.stats["path_lengths"] else 0,
        }

        records = [self._sample_to_record(s) for s in self.samples]
        payload = {
            "settings": self.cfg.to_metadata(),
            "stats": final_stats,
            "samples": records,
        }

        torch.save(payload, output_path)
        log.info(f"Saved {len(self.samples)} samples to {output_path}")
        self._log_stats(final_stats)
        return final_stats

    def _sample_to_record(self, sample: GAgentSample) -> Dict[str, Any]:
        """Convert dataclass to plain dict (deterministic schema)."""
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "question_emb": sample.question_emb,
            "edge_relations": sample.edge_relations,
            "edge_scores": sample.edge_scores,
            "edge_labels": sample.edge_labels,
            "edge_head_locals": sample.edge_head_locals,
            "edge_tail_locals": sample.edge_tail_locals,
            "node_entity_ids": sample.node_entity_ids,
            "node_embedding_ids": sample.node_embedding_ids,
            "top_edge_mask": sample.top_edge_mask.to(dtype=torch.bool),
            "start_entity_ids": sample.start_entity_ids,
            "start_node_locals": sample.start_node_locals,
            "answer_entity_ids": sample.answer_entity_ids,
            "answer_node_locals": sample.answer_node_locals,
            "gt_path_edge_local_ids": sample.gt_path_edge_local_ids,
            "gt_path_node_local_ids": sample.gt_path_node_local_ids,
            "gt_path_exists": bool(sample.gt_path_exists),
            "is_answer_reachable": bool(sample.is_answer_reachable),
        }

    def _log_stats(self, stats):
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

__all__ = ["GAgentBuilder", "GAgentSettings"]
