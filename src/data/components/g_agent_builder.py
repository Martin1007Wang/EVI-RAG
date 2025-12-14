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
    # anchors: top-K edges (by retrieval score) define the high-belief region
    anchor_top_k: int = 50
    output_path: Path = Path("g_agent/g_agent_samples.pt")
    force_include_gt: bool = False

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
    The Producer: Responsible for running Beam Search and materializing clean GAgentSample objects.
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

            # Get question local indices for seeding beam search
            # Assuming these are available in batch
            q_ptr = getattr(batch, "q_local_indices_ptr", None)
            if q_ptr is None and hasattr(batch, "_slice_dict"):
                q_ptr = batch._slice_dict.get("q_local_indices")
            q_locals = getattr(batch, "q_local_indices", None)

            start_node_locals = set()
            if q_ptr is not None and q_locals is not None:
                q_start, q_end = int(q_ptr[i]), int(q_ptr[i + 1])
                # These are local to the sample graph (0~N_retrieval)
                start_node_locals = set(q_locals[q_start:q_end].tolist())

            self._build_and_add_sample(sid, graph_slice, start_node_locals)

    def _build_and_add_sample(self, sample_id: str, graph: _GraphSlice, seeds: Set[int]):
        """
        Core Logic: Beam Search -> Re-index -> Create Object
        """
        # === A. Beam Search Filter ===
        selected_indices_set = self._select_edges_seed_anchor(
            graph=graph,
            seeds=seeds,
            anchor_top_k=self.cfg.anchor_top_k,
        )

        # Convert to sorted tensor list for deterministic indexing
        selected_indices = sorted(list(selected_indices_set))
        if not selected_indices:
            return  # Empty graph

        selected_idx_tensor = torch.tensor(selected_indices, dtype=torch.long)

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

        # === C. Construct Subgraph Data (Global & Local) ===

        # 1. Extract Edge Attributes (Global IDs for Heads/Tails)
        # heads/tails in graph_slice are local to G_retrieval (0~N_retr)
        # We map them to Global Entity IDs
        sel_heads_local = graph.heads[selected_idx_tensor]
        sel_tails_local = graph.tails[selected_idx_tensor]

        edge_heads_global = graph.node_global_ids[sel_heads_local]
        edge_tails_global = graph.node_global_ids[sel_tails_local]
        edge_relations = graph.relations[selected_idx_tensor]
        edge_scores = graph.scores[selected_idx_tensor]
        edge_labels = graph.labels[selected_idx_tensor]

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
            self.stats["retrieval_failed"] += 1
            return
        seen_q = set()
        start_node_locals = torch.tensor(
            [x for x in start_node_locals_list if not (x in seen_q or seen_q.add(x))],
            dtype=torch.long,
        )

        # 答案只保留子图中存在的实体，并与局部索引一一对齐，去重后保持输入顺序。
        answer_pairs: List[Tuple[int, int]] = []
        seen_a = set()
        for a_id in answer_entity_ids.tolist():
            a_int = int(a_id)
            if a_int in seen_a:
                continue
            mapped = node_map.get(a_int)
            if mapped is None:
                continue
            seen_a.add(a_int)
            answer_pairs.append((a_int, int(mapped)))
        if not answer_pairs:
            self.stats["retrieval_failed"] += 1
            # 严格模式：答案不在子图直接丢弃样本，避免产出不可达样本
            return
        answer_entity_ids = torch.tensor([p[0] for p in answer_pairs], dtype=torch.long)
        answer_node_locals = torch.tensor([p[1] for p in answer_pairs], dtype=torch.long)
        has_answer = True  # 非空已提前 return

        # === E. Ground Truth Path Mapping ===
        # We need to find which edges in our selected subgraph correspond to the GT path.
        # raw_data['gt_path_edge_indices'] contains indices in G_retrieval (0~E_retr-1).
        # We need to map them to indices in G_agent (0~E_agent-1).

        gt_path_indices: List[int] = []
        raw_gt_indices = raw_data.get("gt_path_edge_indices", [])  # Ensure this key matches build script
        if not raw_gt_indices:
            # 兼容 build_retrieval_dataset.py 存储的三元组路径：gt_paths_triples=[[(h,r,t),...]]
            gt_triples = raw_data.get("gt_paths_triples", [])
            # Build mapping: agent edge idx -> (h_global, r_global, t_global)
            edge_triples = []
            for e_idx in range(num_edges):
                h_global = int(edge_heads_global[e_idx].item())
                t_global = int(edge_tails_global[e_idx].item())
                r_global = int(edge_relations[e_idx].item())
                edge_triples.append((h_global, r_global, t_global))
            triple_to_edge_idx: Dict[Tuple[int, int, int], List[int]] = {}
            for idx, triple in enumerate(edge_triples):
                triple_to_edge_idx.setdefault(triple, []).append(idx)
            raw_gt_indices = []
            for path in gt_triples:
                for triple in path:
                    triple_tuple = tuple(int(x) for x in triple)
                    cand = triple_to_edge_idx.get(triple_tuple)
                    if cand:
                        raw_gt_indices.append(cand[0])  # 取首个匹配

        if raw_gt_indices:
            # Map: Retrieval Edge Index -> Agent Edge Index
            retr_to_agent = {int(ridx): aidx for aidx, ridx in enumerate(selected_indices)}
            # For clarity, also accept retrieval-edge indices stored in LMDB that refer to original retrieval graph.
            retrieval_idx_to_agent: Dict[int, int] = {}
            for aidx, ridx in enumerate(selected_indices):
                r_global = int(ridx)
                retrieval_idx_to_agent[r_global] = aidx
                if graph.retrieval_edge_indices.numel() > r_global:
                    retrieval_idx_to_agent[int(graph.retrieval_edge_indices[r_global].item())] = aidx

            for ridx in raw_gt_indices:
                # Handle tensor or int
                r_val = ridx if isinstance(ridx, int) else ridx.item()
                if r_val in retr_to_agent:
                    gt_path_indices.append(retr_to_agent[r_val])
                elif r_val in retrieval_idx_to_agent:
                    gt_path_indices.append(retrieval_idx_to_agent[r_val])
                else:
                    # Path broken! Stop.
                    self.stats["gt_broken"] += 1
                    break
        else:
            # Label-based fallback：在 g_retrieval 视图中已给出正例标签的边，直接作为 GT 路径。
            positive_edges = torch.nonzero(edge_labels > 0.5, as_tuple=False).view(-1).tolist()
            gt_path_indices = positive_edges

        gt_exists = (len(gt_path_indices) > 0) and has_answer
        if not gt_exists:
            # 严格模式：可达但无路径视为损坏样本，丢弃
            self.stats["gt_broken"] += 1
            return
        if gt_exists:
            self.stats["path_exists"] += 1
            self.stats["path_lengths"].append(len(gt_path_indices))

        self.stats["edge_counts"].append(len(selected_indices))

        # === F. Create the Object (fully specified schema) ===
        num_edges = len(selected_indices)
        top_edge_mask = torch.ones(num_edges, dtype=torch.bool)

        gt_path_edge_local_ids = torch.tensor(gt_path_indices, dtype=torch.long)
        gt_path_node_local_ids = torch.tensor(
            sorted(
                {int(edge_head_locals[i].item()) for i in gt_path_indices}
                | {int(edge_tail_locals[i].item()) for i in gt_path_indices}
            ),
            dtype=torch.long,
        )

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
            gt_path_exists=gt_exists,
            is_answer_reachable=gt_exists,
        )
        # 数据完整性：可达必有路径，不可达不得有路径
        if sample.is_answer_reachable:
            assert (
                gt_path_edge_local_ids.numel() > 0
            ), f"Sample {sample_id} is marked reachable but GT path is empty! Logic Error."
        else:
            assert gt_path_edge_local_ids.numel() == 0, f"Sample {sample_id} is unreachable but has GT path! Logic Error."

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

    # === Helper Logic (Copied & Adapted from your code) ===
    def _select_edges_seed_anchor(
        self,
        graph: _GraphSlice,
        seeds: Set[int],
        anchor_top_k: int,
    ) -> Set[int]:
        """
        Seed→Anchor 2-hop builder:
        1) Anchors: heads/tails appearing in top-K edges by retrieval score.
        2) Hop1: edges from seeds; keep direct hits to anchors; otherwise track frontier nodes.
        3) Hop2: from each frontier, keep edges hitting anchors and also keep the preceding Hop1 edge.
        Only edges that connect Seeds to Anchors (directly or via a 1-hop bridge) survive.
        """
        num_edges = graph.num_edges
        if num_edges == 0:
            return set()

        # normalize seeds to valid locals
        seeds = {int(s) for s in seeds if 0 <= int(s) < graph.num_nodes}
        scores = graph.scores.float()
        heads = graph.heads
        tails = graph.tails

        k = max(1, int(anchor_top_k))
        topk = min(k, num_edges)
        _, top_idx = torch.topk(scores, k=topk)
        anchor_nodes = {int(heads[i].item()) for i in top_idx.tolist()} | {int(tails[i].item()) for i in top_idx.tolist()}

        if not seeds:
            best_idx = int(torch.argmax(scores).item())
            seeds = {int(heads[best_idx].item()), int(tails[best_idx].item())}

        # adjacency: head -> [edge_idx]
        adj_out: Dict[int, List[int]] = {}
        for idx in range(num_edges):
            h = int(heads[idx].item())
            adj_out.setdefault(h, []).append(idx)

        selected: Set[int] = set()
        frontier_edges: Dict[int, List[int]] = {}

        # Hop1
        for s in seeds:
            for idx in adj_out.get(s, []):
                tgt = int(tails[idx].item())
                if tgt in anchor_nodes:
                    selected.add(idx)
                else:
                    frontier_edges.setdefault(tgt, []).append(idx)

        # Hop2
        for mid, hop1_edges in frontier_edges.items():
            for idx in adj_out.get(mid, []):
                tgt = int(tails[idx].item())
                if tgt in anchor_nodes:
                    selected.add(idx)
                    selected.update(hop1_edges)

        return selected


__all__ = ["GAgentBuilder", "GAgentSettings"]
