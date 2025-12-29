from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from src.data.g_agent_dataset import GAgentSample

from src.models.components.retriever import RetrieverOutput
from src.data.components.embedding_store import EmbeddingStore

log = logging.getLogger(__name__)

ZERO = 0
ONE = 1
POS_LABEL_THRESHOLD = 0.5
PAIR_STRIDE_OFFSET = 1


@dataclass
class GAgentSettings:
    """Declarative configuration for constructing g_agent."""

    enabled: bool = True
    # Global top-k edges kept after score ranking (per-sample, retrieval graph space).
    edge_top_k: int = 50
    # Only keep nodes/edges reachable from start within this hop radius (computed on the selected edge space).
    max_hops: int = 2
    # Logit calibration applied to retriever scores before caching.
    score_temperature: float = 1.0
    score_bias: float = 0.0
    output_path: Path = Path("g_agent/g_agent_samples.pt")

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.edge_top_k = int(self.edge_top_k)
        self.max_hops = int(self.max_hops)
        self.score_temperature = float(self.score_temperature)
        self.score_bias = float(self.score_bias)
        if self.max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {self.max_hops}")
        if self.edge_top_k <= 0:
            raise ValueError(f"edge_top_k must be > 0, got {self.edge_top_k}")
        if not (self.score_temperature > 0.0):
            raise ValueError(f"score_temperature must be positive, got {self.score_temperature}")
        self.output_path = Path(self.output_path).expanduser()

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
            "edge_counts": [],
            "path_lengths": [],
        }

    def reset(self):
        self.samples = []
        for k in ["num_samples", "path_exists", "retrieval_failed"]:
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
        batch_size = int(ptr.numel() - 1)

        logits = getattr(model_output, "logits", None)
        if logits is None:
            raise ValueError("Retriever output missing logits; g_agent requires logit scores.")
        scores = logits.detach().view(-1)
        if scores.numel() == 0:
            return
        if self.cfg.score_temperature != 1.0 or self.cfg.score_bias != 0.0:
            scores = scores / float(self.cfg.score_temperature) + float(self.cfg.score_bias)

        device = scores.device
        edge_index = batch.edge_index.to(device=device)
        relations = batch.edge_attr.to(device=device)
        labels = batch.labels.to(device=device)
        node_global_ids = batch.node_global_ids.to(device=device)

        if not hasattr(batch, "node_embedding_ids"):
            raise AttributeError("Batch missing node_embedding_ids; retriever dataset must provide embedding ids per node.")
        node_embedding_ids = batch.node_embedding_ids.to(device=device)

        query_ids = getattr(model_output, "query_ids", None)
        if query_ids is None:
            raise ValueError("Retriever output missing query_ids; g_agent requires per-edge graph mapping.")
        if not torch.is_tensor(query_ids):
            query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        else:
            query_ids = query_ids.to(device=device, dtype=torch.long)
        query_ids = query_ids.view(-1)
        if query_ids.numel() != scores.numel():
            raise ValueError(f"query_ids/logits shape mismatch: {query_ids.shape} vs {scores.shape}")

        num_edges = int(query_ids.numel())
        edge_order = torch.arange(num_edges, device=device, dtype=query_ids.dtype)
        sort_key = query_ids * num_edges + edge_order
        sort_idx = torch.argsort(sort_key)
        sorted_q = query_ids.index_select(0, sort_idx)

        counts = torch.zeros(batch_size, device=device, dtype=torch.long)
        counts.scatter_add_(0, sorted_q, torch.ones_like(sorted_q, dtype=torch.long))
        if int(counts.sum().item()) != num_edges:
            raise ValueError("Invalid query_ids: edge assignments exceed batch_size.")
        starts = torch.cumsum(counts, dim=0) - counts

        # Sample IDs list
        sample_ids = getattr(batch, "sample_id", [])

        for i in range(batch_size):
            count = int(counts[i].item())
            if count == 0:
                continue
            start = int(starts[i].item())
            edge_indices = sort_idx[start : start + count]

            node_start = int(ptr[i].item())
            node_end = int(ptr[i + 1].item())
            node_slice = node_global_ids[node_start:node_end]
            node_embedding_slice = node_embedding_ids[node_start:node_end]

            # Convert global edge_index to local (0~N_retrieval)
            heads = edge_index[0, edge_indices] - node_start
            tails = edge_index[1, edge_indices] - node_start

            graph_slice = _GraphSlice(
                heads=heads,
                tails=tails,
                relations=relations.index_select(0, edge_indices),
                labels=labels.index_select(0, edge_indices),
                scores=scores.index_select(0, edge_indices),
                node_global_ids=node_slice,
                node_embedding_ids=node_embedding_slice,
                retrieval_edge_indices=edge_indices,
            )

            try:
                sid = str(sample_ids[i])
            except IndexError:
                sid = f"unknown_{i}"

            self._build_and_add_sample(sid, graph_slice)

    def _build_and_add_sample(self, sample_id: str, graph: _GraphSlice) -> None:
        """
        Core Logic: Top-K by score (E_K) -> bridge shortest-path DAG (E_bridge)
        -> environment union (E_env) -> dedup -> re-index -> create sample
        """
        # === A. Global top-k edge selection (E_K; label-free) ===
        num_edges_full = graph.num_edges
        if num_edges_full <= 0:
            return
        device = graph.scores.device
        node_global_ids_cpu = graph.node_global_ids.detach().cpu()

        # === B. Fetch Metadata from LMDB ===
        # This is crucial for "One Source of Truth"
        if self.embedding_store is None:
            raise ValueError("EmbeddingStore must be provided; builder cannot proceed without per-sample metadata.")

        try:
            raw_data = self.embedding_store.load_sample(sample_id)
        except KeyError:
            log.warning(f"Sample {sample_id} not found in LMDB.")
            return

        question_raw = raw_data.get("question_emb", [])
        question_emb = torch.as_tensor(question_raw, dtype=torch.float32).detach().clone()
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)
        elif question_emb.dim() != 2:
            raise ValueError(f"question_emb must be 1D or 2D, got shape {tuple(question_emb.shape)} for {sample_id}")
        question_text = raw_data.get("question", "")
        # Standardize to LongTensor
        start_entity_ids = torch.as_tensor(raw_data.get("seed_entity_ids", []), dtype=torch.long).detach().clone()
        if start_entity_ids.numel() == 0:
            raise ValueError(f"Sample {sample_id} missing seed_entity_ids (start_entity_ids).")

        answer_entity_ids = torch.as_tensor(raw_data.get("answer_entity_ids", []), dtype=torch.long).detach().clone()
        if answer_entity_ids.numel() == 0:
            raise ValueError(f"Sample {sample_id} missing answer_entity_ids.")

        start_entity_ids_device = start_entity_ids.to(device=device)
        start_mask = torch.isin(graph.node_global_ids, start_entity_ids_device.view(-1))
        if not bool(start_mask.any().item()):
            raise ValueError(f"Start entities missing from retrieval graph (sample_id={sample_id}).")
        edge_top_k = int(self.cfg.edge_top_k)
        topk_edge_indices_cpu = self._select_topk_edges(
            edge_scores=graph.scores.detach().cpu(),
            edge_top_k=edge_top_k,
        )
        if topk_edge_indices_cpu.numel() == 0:
            self.stats["retrieval_failed"] += 1
            return
        start_node_locals_retrieval = torch.nonzero(start_mask.detach().cpu(), as_tuple=False).view(-1)
        topk_edge_heads = graph.heads.index_select(0, topk_edge_indices_cpu.to(device=device)).detach().cpu()
        topk_edge_tails = graph.tails.index_select(0, topk_edge_indices_cpu.to(device=device)).detach().cpu()
        prize_node_locals = torch.unique(torch.cat([topk_edge_heads, topk_edge_tails], dim=0))
        bridge_edge_mask = self._bridge_shortest_path_dag_mask(
            num_nodes=int(node_global_ids_cpu.numel()),
            edge_head_locals=graph.heads.detach().cpu(),
            edge_tail_locals=graph.tails.detach().cpu(),
            source_nodes=start_node_locals_retrieval,
            target_nodes=prize_node_locals,
            max_hops=int(self.cfg.max_hops),
        )
        topk_edge_mask = torch.zeros(int(graph.num_edges), dtype=torch.bool)
        topk_edge_mask[topk_edge_indices_cpu] = True
        env_edge_mask = topk_edge_mask | bridge_edge_mask
        env_edge_indices_cpu = torch.nonzero(env_edge_mask, as_tuple=False).view(-1)
        if env_edge_indices_cpu.numel() == 0:
            self.stats["retrieval_failed"] += 1
            return
        env_edge_indices = env_edge_indices_cpu.to(device=device, dtype=torch.long)

        # === C. Environment edge space (E_env) + deduplicate by (h,r,t) ===
        env_edge_heads = graph.heads.index_select(0, env_edge_indices).detach().cpu()
        env_edge_tails = graph.tails.index_select(0, env_edge_indices).detach().cpu()
        env_edge_relations = graph.relations.index_select(0, env_edge_indices).detach().cpu()
        env_edge_scores = graph.scores.index_select(0, env_edge_indices).detach().cpu()
        env_edge_labels = graph.labels.index_select(0, env_edge_indices).detach().cpu()
        # Dedup dictionary: triple -> aggregated attributes (score/label max).
        triple_to_agg: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for offset in range(int(env_edge_heads.numel())):
            h_local = int(env_edge_heads[offset].item())
            t_local = int(env_edge_tails[offset].item())
            h_global = int(node_global_ids_cpu[h_local].item())
            t_global = int(node_global_ids_cpu[t_local].item())
            r_global = int(env_edge_relations[offset].item())
            triple = (h_global, r_global, t_global)
            score = float(env_edge_scores[offset].item())
            label = float(env_edge_labels[offset].item())
            agg = triple_to_agg.get(triple)
            if agg is None:
                triple_to_agg[triple] = {"score": score, "label": label}
                continue
            # Aggregate
            agg["score"] = max(float(agg["score"]), score)
            agg["label"] = max(float(agg["label"]), label)

        triples = list(triple_to_agg.keys())
        if not triples:
            return

        edge_heads_global = torch.tensor([t[0] for t in triples], dtype=torch.long)
        edge_relations = torch.tensor([t[1] for t in triples], dtype=torch.long)
        edge_tails_global = torch.tensor([t[2] for t in triples], dtype=torch.long)
        edge_scores = torch.tensor([float(triple_to_agg[t]["score"]) for t in triples], dtype=torch.float32)
        edge_labels = torch.tensor([float(triple_to_agg[t]["label"]) for t in triples], dtype=torch.float32)

        # 2. Build subgraph topology (0 ~ N_subgraph-1)
        # Collect all unique nodes in this new small subgraph
        node_entity_ids, _ = torch.sort(torch.cat([edge_heads_global, edge_tails_global]).unique())

        # Map Global ID -> Subgraph Index / Embedding Row
        node_map = {gid.item(): i for i, gid in enumerate(node_entity_ids)}
        node_embedding_ids_cpu = graph.node_embedding_ids.detach().cpu()
        embedding_lookup = {int(g.item()): int(e.item()) for g, e in zip(node_global_ids_cpu, node_embedding_ids_cpu)}
        unique_embedding_ids = []
        for gid in node_entity_ids.tolist():
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

        # === E. Hop pruning: keep only nodes/edges reachable from start within max_hops ===
        max_hops = int(self.cfg.max_hops)
        (
            node_entity_ids,
            node_embedding_ids,
            edge_head_locals,
            edge_tail_locals,
            edge_relations,
            edge_scores,
            edge_labels,
            start_node_locals,
            answer_node_locals,
        ) = self._prune_to_hops(
            node_entity_ids=node_entity_ids,
            node_embedding_ids=node_embedding_ids,
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            edge_relations=edge_relations,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            max_hops=max_hops,
        )
        if int(edge_head_locals.numel()) == 0 or int(edge_relations.numel()) == 0:
            self.stats["retrieval_failed"] += 1
            return
        if int(start_node_locals.numel()) == 0:
            self.stats["retrieval_failed"] += 1
            return

        # === F. Path-valued supervision: undirected shortest-path DAG per (start, answer) pair ===
        (
            edge_labels,
            pair_start_node_locals,
            pair_answer_node_locals,
            pair_edge_local_ids,
            pair_edge_counts,
            pair_shortest_lengths,
        ) = self._build_pair_shortest_path_dag_csr(
            num_nodes=int(node_embedding_ids.numel()),
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
        )
        # Canonicalize positives: one edge per undirected pair, keep highest score.
        (
            edge_labels,
            pair_edge_local_ids,
            pair_edge_counts,
        ) = self._canonicalize_positive_edges_by_pair(
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
            pair_edge_local_ids=pair_edge_local_ids,
            pair_edge_counts=pair_edge_counts,
        )
        node_answer_dist = self._min_dist_to_answers(
            num_nodes=int(node_embedding_ids.numel()),
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            answer_node_locals=answer_node_locals,
            fallback_dist=int(self.cfg.max_hops) + 1,
        )
        edge_pos_mask = edge_labels > POS_LABEL_THRESHOLD
        gt_path_edge_local_ids = self._select_canonical_shortest_path_edges(
            num_nodes=int(node_embedding_ids.numel()),
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            edge_scores=edge_scores,
            edge_mask=edge_pos_mask,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
        )
        gt_path_node_local_ids = torch.empty(0, dtype=torch.long)
        gt_path_exists = False
        if gt_path_edge_local_ids.numel() > 0:
            gt_path_node_local_ids = self._build_gt_path_node_locals(
                gt_path_edge_local_ids=gt_path_edge_local_ids,
                edge_head_locals=edge_head_locals,
                edge_tail_locals=edge_tail_locals,
                start_node_locals=start_node_locals,
            )
            if gt_path_node_local_ids.numel() > 0:
                gt_path_exists = True
            else:
                gt_path_edge_local_ids = torch.empty(0, dtype=torch.long)

        if not gt_path_exists and start_node_locals.numel() > 0 and answer_node_locals.numel() > 0:
            # Zero-length shortest path: start already contains an answer.
            intersect = torch.isin(start_node_locals, answer_node_locals)
            if bool(intersect.any().item()):
                node_id = int(start_node_locals[intersect].min().item())
                gt_path_exists = True
                gt_path_edge_local_ids = torch.empty(0, dtype=torch.long)
                gt_path_node_local_ids = torch.tensor([node_id], dtype=torch.long)

        if gt_path_exists and answer_node_locals.numel() > 0:
            terminal = int(gt_path_node_local_ids[-1].item())
            if not bool((answer_node_locals == terminal).any().item()):
                gt_path_exists = False
                gt_path_edge_local_ids = torch.empty(0, dtype=torch.long)
                gt_path_node_local_ids = torch.empty(0, dtype=torch.long)

        if gt_path_exists:
            self.stats["path_exists"] += 1
            self.stats["path_lengths"].append(int(gt_path_edge_local_ids.numel()))
        self.stats["edge_counts"].append(int(edge_relations.numel()))

        is_answer_reachable = bool(pair_start_node_locals.numel() > 0)

        sample = GAgentSample(
            sample_id=sample_id,
            question=question_text,
            question_emb=question_emb,
            edge_relations=edge_relations,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            node_entity_ids=node_entity_ids,
            node_embedding_ids=node_embedding_ids,
            start_entity_ids=start_entity_ids,
            answer_entity_ids=answer_entity_ids,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            pair_start_node_locals=pair_start_node_locals,
            pair_answer_node_locals=pair_answer_node_locals,
            pair_edge_local_ids=pair_edge_local_ids,
            pair_edge_counts=pair_edge_counts,
            pair_shortest_lengths=pair_shortest_lengths,
            node_answer_dist=node_answer_dist,
            gt_path_edge_local_ids=gt_path_edge_local_ids,
            gt_path_node_local_ids=gt_path_node_local_ids,
            gt_path_exists=gt_path_exists,
            is_answer_reachable=is_answer_reachable,
        )

        self.samples.append(sample)
        self.stats["num_samples"] += 1

    def save(self, output_path: Path) -> Dict[str, Any] | None:
        if not self.samples:
            log.warning("No samples collected.")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute final stats
        attempted = int(self.stats["num_samples"]) + int(self.stats["retrieval_failed"])
        final_stats = {
            "num_samples": self.stats["num_samples"],
            "path_exists_ratio": self.stats["path_exists"] / max(1, self.stats["num_samples"]),
            "retrieval_failed_ratio": self.stats["retrieval_failed"] / max(1, attempted),
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
            "start_entity_ids": sample.start_entity_ids,
            "start_node_locals": sample.start_node_locals,
            "answer_entity_ids": sample.answer_entity_ids,
            "answer_node_locals": sample.answer_node_locals,
            "pair_start_node_locals": sample.pair_start_node_locals,
            "pair_answer_node_locals": sample.pair_answer_node_locals,
            "pair_edge_local_ids": sample.pair_edge_local_ids,
            "pair_edge_counts": sample.pair_edge_counts,
            "pair_shortest_lengths": sample.pair_shortest_lengths,
            "node_answer_dist": sample.node_answer_dist,
            "gt_path_edge_local_ids": sample.gt_path_edge_local_ids,
            "gt_path_node_local_ids": sample.gt_path_node_local_ids,
            "gt_path_exists": bool(sample.gt_path_exists),
            "is_answer_reachable": bool(sample.is_answer_reachable),
        }

    def _log_stats(self, stats):
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    @staticmethod
    def _select_topk_edges(*, edge_scores: torch.Tensor, edge_top_k: int) -> torch.Tensor:
        """Select global top-k edges by score (descending)."""
        scores = edge_scores.view(-1)
        num_edges = int(scores.numel())
        if num_edges <= 0:
            return torch.empty(0, dtype=torch.long)
        edge_top_k = int(edge_top_k)
        if edge_top_k <= 0:
            raise ValueError(f"edge_top_k must be > 0, got {edge_top_k}")
        if num_edges <= edge_top_k:
            return torch.arange(num_edges, dtype=torch.long)
        sorted_idx = torch.argsort(scores, descending=True, stable=True)
        return sorted_idx[:edge_top_k].to(dtype=torch.long)

    @staticmethod
    def _canonicalize_positive_edges_by_pair(
        *,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_labels: torch.Tensor,
        pair_edge_local_ids: torch.Tensor,
        pair_edge_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_mask = GAgentBuilder._select_best_positive_edges_by_pair(
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
        )
        if not bool(keep_mask.any().item()):
            return edge_labels, pair_edge_local_ids, pair_edge_counts
        edge_labels = torch.where(keep_mask, edge_labels, edge_labels.new_zeros(()))
        new_pair_edge_local_ids, new_pair_edge_counts = GAgentBuilder._filter_pair_edges_by_mask(
            pair_edge_local_ids=pair_edge_local_ids,
            pair_edge_counts=pair_edge_counts,
            keep_mask=keep_mask,
        )
        return edge_labels, new_pair_edge_local_ids, new_pair_edge_counts

    @staticmethod
    def _select_best_positive_edges_by_pair(
        *,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> torch.Tensor:
        num_edges = int(edge_labels.numel())
        if num_edges == ZERO:
            return torch.zeros_like(edge_labels, dtype=torch.bool)
        pos_mask = edge_labels > POS_LABEL_THRESHOLD
        if not bool(pos_mask.any().item()):
            return torch.zeros_like(edge_labels, dtype=torch.bool)
        heads = edge_head_locals.view(-1)
        tails = edge_tail_locals.view(-1)
        if heads.numel() != num_edges or tails.numel() != num_edges:
            raise ValueError("edge_head_locals/edge_tail_locals length mismatch when canonicalizing positives.")
        max_node = torch.maximum(heads, tails).max()
        stride = max_node + PAIR_STRIDE_OFFSET
        u = torch.minimum(heads, tails)
        v = torch.maximum(heads, tails)
        pair_key = u * stride + v

        pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
        pos_keys = pair_key.index_select(0, pos_idx)
        pos_scores = edge_scores.view(-1).index_select(0, pos_idx)
        order = torch.argsort(pos_idx, stable=True)
        order = order[torch.argsort(pos_scores.index_select(0, order), descending=True, stable=True)]
        order = order[torch.argsort(pos_keys.index_select(0, order), stable=True)]
        sorted_keys = pos_keys.index_select(0, order)
        is_first = torch.ones_like(sorted_keys, dtype=torch.bool)
        if int(sorted_keys.numel()) > ONE:
            is_first[ONE:] = sorted_keys[ONE:] != sorted_keys[:-ONE]
        best_pos_idx = pos_idx.index_select(0, order[is_first])
        keep_mask = torch.zeros_like(edge_labels, dtype=torch.bool)
        keep_mask[best_pos_idx] = True
        return keep_mask

    @staticmethod
    def _filter_pair_edges_by_mask(
        *,
        pair_edge_local_ids: torch.Tensor,
        pair_edge_counts: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_edge_local_ids.numel() == ZERO or pair_edge_counts.numel() == ZERO:
            return pair_edge_local_ids, pair_edge_counts
        keep_pair_mask = keep_mask[pair_edge_local_ids]
        pair_ids = torch.repeat_interleave(
            torch.arange(pair_edge_counts.numel(), device=pair_edge_counts.device),
            pair_edge_counts,
        )
        new_counts = torch.zeros_like(pair_edge_counts)
        new_counts.scatter_add_(0, pair_ids, keep_pair_mask.to(dtype=new_counts.dtype))
        new_pair_edge_local_ids = pair_edge_local_ids[keep_pair_mask]
        return new_pair_edge_local_ids, new_counts

    @staticmethod
    def _prune_to_hops(
        *,
        node_entity_ids: torch.Tensor,
        node_embedding_ids: torch.Tensor,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        edge_relations: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_labels: torch.Tensor,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        max_hops: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prune graph to the radius-`max_hops` ball around start nodes (undirected reachability)."""
        max_hops = int(max_hops)
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")

        num_nodes = int(node_entity_ids.numel())
        if num_nodes <= 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        if node_embedding_ids.numel() != num_nodes:
            raise ValueError(f"node_embedding_ids length {int(node_embedding_ids.numel())} != num_nodes {num_nodes}")

        starts = [int(x) for x in start_node_locals.view(-1).tolist()]
        if not starts:
            return (
                node_entity_ids,
                node_embedding_ids,
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        for s in starts:
            if s < 0 or s >= num_nodes:
                raise ValueError(f"start_node_locals out of range: {s} (num_nodes={num_nodes})")

        heads = [int(x) for x in edge_head_locals.view(-1).tolist()]
        tails = [int(x) for x in edge_tail_locals.view(-1).tolist()]
        num_edges = len(heads)
        if len(tails) != num_edges:
            raise ValueError("edge_head_locals/edge_tail_locals length mismatch.")
        if int(edge_relations.numel()) != num_edges:
            raise ValueError("edge_relations length mismatch with edge_head_locals.")
        if int(edge_scores.numel()) != num_edges:
            raise ValueError("edge_scores length mismatch with edge_head_locals.")
        if int(edge_labels.numel()) != num_edges:
            raise ValueError("edge_labels length mismatch with edge_head_locals.")

        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for h, t in zip(heads, tails):
            if h < 0 or h >= num_nodes or t < 0 or t >= num_nodes:
                raise ValueError(f"Edge locals out of range: h={h} t={t} num_nodes={num_nodes}")
            adj[h].append(t)
            if h != t:
                adj[t].append(h)

        dist = [-1] * num_nodes
        q: deque[int] = deque()
        for s in starts:
            if dist[s] != 0:
                dist[s] = 0
                q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= max_hops:
                continue
            for v in adj[u]:
                if dist[v] >= 0:
                    continue
                dist[v] = du + 1
                q.append(v)

        kept_nodes = [i for i, d in enumerate(dist) if d >= 0]
        if not kept_nodes:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        old_to_new = [-1] * num_nodes
        for new_i, old_i in enumerate(kept_nodes):
            old_to_new[old_i] = new_i

        kept_edges: List[int] = []
        for eid, (h, t) in enumerate(zip(heads, tails)):
            dh = dist[h]
            dt = dist[t]
            if dh < 0 or dt < 0:
                continue
            kept_edges.append(eid)

        node_entity_ids = node_entity_ids.index_select(0, torch.tensor(kept_nodes, dtype=torch.long))
        node_embedding_ids = node_embedding_ids.index_select(0, torch.tensor(kept_nodes, dtype=torch.long))

        if not kept_edges:
            return (
                node_entity_ids,
                node_embedding_ids,
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.tensor([old_to_new[s] for s in starts if old_to_new[s] >= 0], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        head_new = [old_to_new[heads[eid]] for eid in kept_edges]
        tail_new = [old_to_new[tails[eid]] for eid in kept_edges]

        kept_edges_t = torch.tensor(kept_edges, dtype=torch.long)
        edge_head_locals = torch.tensor(head_new, dtype=torch.long)
        edge_tail_locals = torch.tensor(tail_new, dtype=torch.long)
        edge_relations = edge_relations.index_select(0, kept_edges_t)
        edge_scores = edge_scores.index_select(0, kept_edges_t)
        edge_labels = edge_labels.index_select(0, kept_edges_t)
        start_new = [old_to_new[s] for s in starts if old_to_new[s] >= 0]
        start_node_locals = torch.tensor(start_new, dtype=torch.long) if start_new else torch.empty(0, dtype=torch.long)

        answers = [int(x) for x in answer_node_locals.view(-1).tolist()]
        ans_new = [old_to_new[a] for a in answers if 0 <= a < num_nodes and old_to_new[a] >= 0]
        answer_node_locals = torch.tensor(ans_new, dtype=torch.long) if ans_new else torch.empty(0, dtype=torch.long)

        return (
            node_entity_ids,
            node_embedding_ids,
            edge_head_locals,
            edge_tail_locals,
            edge_relations,
            edge_scores,
            edge_labels,
            start_node_locals,
            answer_node_locals,
        )

    @staticmethod
    def _build_gt_path_node_locals(
        *,
        gt_path_edge_local_ids: torch.Tensor,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        start_node_locals: torch.Tensor,
    ) -> torch.Tensor:
        if gt_path_edge_local_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        start_set = set(start_node_locals.view(-1).tolist())
        nodes: List[int] = []
        first_edge = int(gt_path_edge_local_ids[0].item())
        h0 = int(edge_head_locals[first_edge].item())
        t0 = int(edge_tail_locals[first_edge].item())
        if h0 in start_set:
            nodes.extend([h0, t0])
        elif t0 in start_set:
            nodes.extend([t0, h0])
        else:
            return torch.empty(0, dtype=torch.long)

        for edge_idx in gt_path_edge_local_ids[1:]:
            e = int(edge_idx.item())
            h = int(edge_head_locals[e].item())
            t = int(edge_tail_locals[e].item())
            prev = nodes[-1]
            if prev == h:
                nodes.append(t)
            elif prev == t:
                nodes.append(h)
            else:
                return torch.empty(0, dtype=torch.long)
        return torch.tensor(nodes, dtype=torch.long)

    @staticmethod
    def _bfs_dist(num_nodes: int, adjacency: list[list[int]], sources: list[int]) -> list[int]:
        dist = [-1] * num_nodes
        q: deque[int] = deque()
        for s in sources:
            if s < 0 or s >= num_nodes:
                continue
            if dist[s] == 0:
                continue
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adjacency[u]:
                if dist[v] >= 0:
                    continue
                dist[v] = du + 1
                q.append(v)
        return dist

    @classmethod
    def _min_dist_to_answers(
        cls,
        *,
        num_nodes: int,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        fallback_dist: int,
    ) -> torch.Tensor:
        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            return torch.empty(0, dtype=torch.long)
        answers = [int(x) for x in answer_node_locals.view(-1).tolist()]
        if not answers:
            return torch.full((num_nodes,), int(fallback_dist), dtype=torch.long)
        heads = [int(x) for x in edge_head_locals.view(-1).tolist()]
        tails = [int(x) for x in edge_tail_locals.view(-1).tolist()]
        if len(heads) != len(tails):
            raise ValueError("edge_head_locals/edge_tail_locals length mismatch when computing answer distances.")
        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for h, t in zip(heads, tails):
            if h < 0 or h >= num_nodes or t < 0 or t >= num_nodes:
                raise ValueError(f"Edge locals out of range: h={h} t={t} num_nodes={num_nodes}")
            adj[h].append(t)
            if h != t:
                adj[t].append(h)
        dist = cls._bfs_dist(num_nodes, adj, answers)
        fallback = int(fallback_dist)
        dist = [d if d >= 0 else fallback for d in dist]
        return torch.tensor(dist, dtype=torch.long)

    @staticmethod
    def _bridge_shortest_path_dag_mask(
        *,
        num_nodes: int,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        max_hops: int | None = None,
    ) -> torch.Tensor:
        """Return mask for edges on shortest-path DAG from sources to targets (undirected BFS)."""
        num_nodes = int(num_nodes)
        heads = [int(x) for x in edge_head_locals.view(-1).tolist()]
        tails = [int(x) for x in edge_tail_locals.view(-1).tolist()]
        num_edges = len(heads)
        if num_nodes <= 0 or num_edges == 0:
            return torch.zeros(num_edges, dtype=torch.bool)

        sources = []
        seen_s: set[int] = set()
        for s_raw in source_nodes.view(-1).tolist():
            s = int(s_raw)
            if s < 0 or s >= num_nodes or s in seen_s:
                continue
            seen_s.add(s)
            sources.append(s)
        targets = []
        seen_t: set[int] = set()
        for t_raw in target_nodes.view(-1).tolist():
            t = int(t_raw)
            if t < 0 or t >= num_nodes or t in seen_t:
                continue
            seen_t.add(t)
            targets.append(t)
        if not sources or not targets:
            return torch.zeros(num_edges, dtype=torch.bool)
        if len(tails) != num_edges:
            raise ValueError("edge_head_locals/edge_tail_locals length mismatch for DAG backtrace.")

        adj: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
        for idx, (h, t) in enumerate(zip(heads, tails)):
            if h < 0 or h >= num_nodes or t < 0 or t >= num_nodes:
                raise ValueError(f"Edge locals out of range: h={h} t={t} num_nodes={num_nodes}")
            adj[h].append((t, idx))
            if h != t:
                adj[t].append((h, idx))

        dist = [-1] * num_nodes
        q: deque[int] = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        hop_limit = None if max_hops is None else int(max_hops)
        while q:
            u = q.popleft()
            du = dist[u]
            if hop_limit is not None and du >= hop_limit:
                continue
            for v, _eid in adj[u]:
                if dist[v] >= 0:
                    continue
                dist[v] = du + 1
                q.append(v)

        reachable_targets = []
        for t in targets:
            dt = dist[t]
            if dt < 0:
                continue
            if hop_limit is not None and dt > hop_limit:
                continue
            reachable_targets.append(t)
        if not reachable_targets:
            return torch.zeros(num_edges, dtype=torch.bool)

        bridge_edge_mask = [False] * num_edges
        seen_node = [False] * num_nodes
        dq: deque[int] = deque()
        for t in reachable_targets:
            seen_node[t] = True
            dq.append(t)
        while dq:
            v = dq.popleft()
            dv = dist[v]
            if dv <= 0:
                continue
            for u, eid in adj[v]:
                if dist[u] == dv - 1:
                    bridge_edge_mask[eid] = True
                    if not seen_node[u]:
                        seen_node[u] = True
                        dq.append(u)
        return torch.tensor(bridge_edge_mask, dtype=torch.bool)

    @classmethod
    def _build_pair_shortest_path_dag_csr(
        cls,
        *,
        num_nodes: int,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build pair-wise shortest-path DAG CSR + union mask (undirected)."""
        num_nodes = int(num_nodes)
        heads = [int(x) for x in edge_head_locals.view(-1).tolist()]
        tails = [int(x) for x in edge_tail_locals.view(-1).tolist()]
        num_edges = len(heads)
        if num_nodes <= 0 or num_edges == 0:
            empty = torch.zeros(num_edges, dtype=torch.float32)
            return (
                empty,
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        starts = []
        seen_s: set[int] = set()
        for s_raw in start_node_locals.view(-1).tolist():
            s = int(s_raw)
            if s < 0 or s >= num_nodes or s in seen_s:
                continue
            seen_s.add(s)
            starts.append(s)
        answers = []
        seen_a: set[int] = set()
        for a_raw in answer_node_locals.view(-1).tolist():
            a = int(a_raw)
            if a < 0 or a >= num_nodes or a in seen_a:
                continue
            seen_a.add(a)
            answers.append(a)
        if not starts or not answers:
            empty = torch.zeros(num_edges, dtype=torch.float32)
            return (
                empty,
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        if len(tails) != num_edges:
            raise ValueError("edge_head_locals/edge_tail_locals length mismatch for DAG mask.")

        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for h, t in zip(heads, tails):
            if h < 0 or h >= num_nodes or t < 0 or t >= num_nodes:
                raise ValueError(f"Edge locals out of range: h={h} t={t} num_nodes={num_nodes}")
            adj[h].append(t)
            if h != t:
                adj[t].append(h)

        dist_from_start = {s: cls._bfs_dist(num_nodes, adj, [s]) for s in starts}
        dist_to_answer = {a: cls._bfs_dist(num_nodes, adj, [a]) for a in answers}

        mask = [False] * num_edges
        pair_start_nodes: list[int] = []
        pair_answer_nodes: list[int] = []
        pair_edge_indices: list[int] = []
        pair_edge_counts: list[int] = []
        pair_shortest_lengths: list[int] = []

        for s in starts:
            dist_s = dist_from_start[s]
            for a in answers:
                dist_a = dist_to_answer[a]
                dist_sa = dist_s[a]
                if dist_sa < 0:
                    continue
                pair_start_nodes.append(s)
                pair_answer_nodes.append(a)
                pair_shortest_lengths.append(int(dist_sa))
                before = len(pair_edge_indices)
                for idx, (u, v) in enumerate(zip(heads, tails)):
                    if dist_s[u] >= 0 and dist_a[v] >= 0 and dist_s[u] + 1 + dist_a[v] == dist_sa:
                        pair_edge_indices.append(idx)
                        mask[idx] = True
                        continue
                    if dist_s[v] >= 0 and dist_a[u] >= 0 and dist_s[v] + 1 + dist_a[u] == dist_sa:
                        pair_edge_indices.append(idx)
                        mask[idx] = True
                pair_edge_counts.append(len(pair_edge_indices) - before)

        return (
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(pair_start_nodes, dtype=torch.long),
            torch.tensor(pair_answer_nodes, dtype=torch.long),
            torch.tensor(pair_edge_indices, dtype=torch.long),
            torch.tensor(pair_edge_counts, dtype=torch.long),
            torch.tensor(pair_shortest_lengths, dtype=torch.long),
        )

    @staticmethod
    def _select_canonical_shortest_path_edges(
        *,
        num_nodes: int,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
    ) -> torch.Tensor:
        """Return a single deterministic shortest path (edge-local ids) from any start to any answer.

        Tie-break: among all shortest paths, choose the one maximizing sum of edge scores;
        break remaining ties by smaller edge id, then smaller predecessor node id (deterministic).
        """
        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            return torch.empty(0, dtype=torch.long)

        starts = [int(x) for x in start_node_locals.view(-1).tolist()]
        answers = [int(x) for x in answer_node_locals.view(-1).tolist()]
        if not starts or not answers:
            return torch.empty(0, dtype=torch.long)

        starts_set = set(starts)
        if starts_set.intersection(answers):
            # Start node already contains an answer: the optimal teacher is "stop" (empty edge path).
            return torch.empty(0, dtype=torch.long)

        heads = [int(x) for x in edge_head_locals.view(-1).tolist()]
        tails = [int(x) for x in edge_tail_locals.view(-1).tolist()]
        scores = [float(x) for x in edge_scores.view(-1).tolist()]
        num_edges = len(heads)
        if num_edges == 0:
            return torch.empty(0, dtype=torch.long)
        if len(tails) != num_edges or len(scores) != num_edges:
            raise ValueError("edge_head_locals/edge_tail_locals/edge_scores length mismatch when building shortest path.")

        edge_ids = list(range(num_edges))
        if edge_mask is not None:
            mask = edge_mask.view(-1).to(dtype=torch.bool)
            if int(mask.numel()) != num_edges:
                raise ValueError(
                    f"edge_mask length {int(mask.numel())} != num_edges {num_edges} when building shortest path."
                )
            edge_ids = [idx for idx, keep in enumerate(mask.tolist()) if keep]
            if not edge_ids:
                return torch.empty(0, dtype=torch.long)

        adj: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
        for eid in edge_ids:
            h = heads[eid]
            t = tails[eid]
            if h < 0 or h >= num_nodes or t < 0 or t >= num_nodes:
                raise ValueError(f"Edge locals out of range: eid={eid} h={h} t={t} num_nodes={num_nodes}")
            adj[h].append((t, eid))
            if h != t:
                adj[t].append((h, eid))

        # 1) Multi-source BFS for hop distance.
        dist = [-1] * num_nodes
        q: deque[int] = deque()
        for s in starts:
            if s < 0 or s >= num_nodes:
                raise ValueError(f"start_node_locals out of range: {s} (num_nodes={num_nodes})")
            if dist[s] == 0:
                continue
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            du = dist[u]
            for v, _eid in adj[u]:
                if dist[v] >= 0:
                    continue
                dist[v] = du + 1
                q.append(v)

        reachable_answers = [a for a in answers if 0 <= a < num_nodes and dist[a] >= 0]
        if not reachable_answers:
            return torch.empty(0, dtype=torch.long)
        best_dist = min(dist[a] for a in reachable_answers)
        if best_dist <= 0:
            return torch.empty(0, dtype=torch.long)

        # 2) DP over BFS layers: best_score[v] = max_{u: dist[u]=dist[v]-1} best_score[u] + score(u,v).
        best_score = [float("-inf")] * num_nodes
        parent_node = [-1] * num_nodes
        parent_edge = [-1] * num_nodes
        for s in starts:
            best_score[s] = 0.0

        nodes_order = [i for i, d in enumerate(dist) if d >= 0]
        nodes_order.sort(key=lambda i: dist[i])
        for v in nodes_order:
            dv = dist[v]
            if dv <= 0:
                continue
            best_key = (float("-inf"), 0, 0)  # (score, -edge_id, -pred_node)
            best_u = -1
            best_eid = -1
            for u, eid in adj[v]:
                if dist[u] != dv - 1:
                    continue
                cand = best_score[u] + scores[eid]
                key = (cand, -eid, -u)
                if key > best_key:
                    best_key = key
                    best_u = u
                    best_eid = eid
            if best_eid >= 0:
                best_score[v] = best_key[0]
                parent_node[v] = best_u
                parent_edge[v] = best_eid

        candidate_answers = [a for a in reachable_answers if dist[a] == best_dist]
        best_answer_key = (float("-inf"), 0)  # (score, -node_id)
        best_answer = -1
        for a in candidate_answers:
            key = (best_score[a], -a)
            if key > best_answer_key:
                best_answer_key = key
                best_answer = a
        if best_answer < 0:
            return torch.empty(0, dtype=torch.long)

        # 3) Reconstruct edge path.
        edges_rev: List[int] = []
        cur = best_answer
        while dist[cur] > 0:
            eid = parent_edge[cur]
            prev = parent_node[cur]
            if eid < 0 or prev < 0:
                return torch.empty(0, dtype=torch.long)
            edges_rev.append(eid)
            cur = prev
        edges_rev.reverse()
        if not edges_rev:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(edges_rev, dtype=torch.long)

__all__ = ["GAgentBuilder", "GAgentSettings"]
