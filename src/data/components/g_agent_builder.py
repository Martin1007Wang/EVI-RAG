from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from src.data.g_agent_dataset import GAgentSample

from src.models.components.retriever import RetrieverOutput
from src.data.components.embedding_store import EmbeddingStore

log = logging.getLogger(__name__)

ZERO = 0
HALF = 0.5
PROB_EPS = 1e-6
KEEP_RATIO_MIN = 0.0
KEEP_RATIO_MAX = 1.0
DEFAULT_START_KEEP_RATIO = 0.25
DEFAULT_START_MIN_EDGES = 1

SCORE_MODE_LOGITS = "logits"
SCORE_MODE_NODE_SOFTMAX = "node_softmax"


@dataclass
class GAgentSettings:
    """Declarative configuration for constructing g_agent."""

    enabled: bool = True
    # Keep samples with missing answer nodes (val/test only).
    allow_empty_answer: bool = False
    # Global top-k edges kept after score ranking (per-sample, retrieval graph space).
    edge_top_k: int = 50
    # Only keep nodes/edges reachable from start within this hop radius (computed on the selected edge space).
    max_hops: int = 2
    # Logit calibration applied to retriever scores before caching.
    score_temperature: float = 1.0
    score_bias: float = 0.0
    # Ensure local connectivity around start nodes (ceil(deg * ratio), capped).
    start_keep_ratio: float = DEFAULT_START_KEEP_RATIO
    start_min_edges: int = DEFAULT_START_MIN_EDGES
    start_max_edges: Optional[int] = None
    # How to store edge_scores in g_agent.
    score_mode: str = SCORE_MODE_NODE_SOFTMAX
    output_path: Path = Path("g_agent/g_agent_samples.pt")

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.allow_empty_answer = bool(self.allow_empty_answer)
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
        self.start_keep_ratio = float(self.start_keep_ratio)
        if not (KEEP_RATIO_MIN <= self.start_keep_ratio <= KEEP_RATIO_MAX):
            raise ValueError(f"start_keep_ratio must be in [0, 1], got {self.start_keep_ratio}")
        self.start_min_edges = int(self.start_min_edges)
        if self.start_min_edges < ZERO:
            raise ValueError(f"start_min_edges must be >= 0, got {self.start_min_edges}")
        if self.start_max_edges is None:
            self.start_max_edges = int(self.edge_top_k)
        else:
            self.start_max_edges = int(self.start_max_edges)
        if self.start_max_edges < ZERO:
            raise ValueError(f"start_max_edges must be >= 0, got {self.start_max_edges}")
        if self.start_max_edges != ZERO and self.start_min_edges > self.start_max_edges:
            raise ValueError(
                f"start_min_edges must be <= start_max_edges, got {self.start_min_edges} > {self.start_max_edges}"
            )
        if self.score_mode not in {SCORE_MODE_LOGITS, SCORE_MODE_NODE_SOFTMAX}:
            raise ValueError(f"score_mode must be one of {SCORE_MODE_LOGITS}/{SCORE_MODE_NODE_SOFTMAX}, got {self.score_mode}")
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

    def __init__(
        self,
        settings: GAgentSettings,
        embedding_store: Optional[EmbeddingStore] = None,
        aux_embedding_store: Optional[EmbeddingStore] = None,
    ) -> None:
        self.cfg = settings
        self.embedding_store = embedding_store
        self.aux_embedding_store = aux_embedding_store
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
        Core Logic: Top-K by score (E_env) -> dedup -> re-index -> create sample.
        Environment edges are allowed to form multiple connected components.
        """
        # === A. Global top-k edge selection (E_env; label-free) ===
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
        aux_data = None
        if self.aux_embedding_store is not None:
            aux_data = self.aux_embedding_store.load_sample(sample_id)

        question_raw = raw_data.get("question_emb", [])
        question_emb = torch.as_tensor(question_raw, dtype=torch.float32).detach().clone()
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)
        elif question_emb.dim() != 2:
            raise ValueError(f"question_emb must be 1D or 2D, got shape {tuple(question_emb.shape)} for {sample_id}")
        question_text = self._select_aux_field(aux_data, raw_data, "question", "")
        if not isinstance(question_text, str):
            raise TypeError(f"question must be string for {sample_id}, got {type(question_text).__name__}")
        # Standardize to LongTensor
        seed_raw = self._select_aux_field(aux_data, raw_data, "seed_entity_ids", [])
        start_entity_ids = torch.as_tensor(seed_raw, dtype=torch.long).detach().clone()
        if start_entity_ids.numel() == 0:
            raise ValueError(
                f"Sample {sample_id} missing seed_entity_ids (start_entity_ids). "
                "If using split LMDBs, pass aux_lmdb_path to GAgentMaterializationCallback."
            )

        answer_raw = raw_data.get("answer_entity_ids")
        if answer_raw is None:
            answer_raw = self._select_aux_field(aux_data, raw_data, "answer_entity_ids", [])
        answer_entity_ids = torch.as_tensor(answer_raw, dtype=torch.long).detach().clone()
        if answer_entity_ids.numel() == 0:
            raise ValueError(f"Sample {sample_id} missing answer_entity_ids.")

        start_entity_ids_device = start_entity_ids.to(device=device)
        start_mask = torch.isin(graph.node_global_ids, start_entity_ids_device.view(-1))
        if not bool(start_mask.any().item()):
            raise ValueError(f"Start entities missing from retrieval graph (sample_id={sample_id}).")
        start_node_locals_retrieval = torch.nonzero(start_mask, as_tuple=False).view(-1).detach().cpu()
        edge_top_k = int(self.cfg.edge_top_k)
        edge_scores_cpu = graph.scores.detach().cpu()
        heads_cpu = graph.heads.detach().cpu()
        tails_cpu = graph.tails.detach().cpu()
        relations_cpu = graph.relations.detach().cpu()
        labels_cpu = graph.labels.detach().cpu()
        select_scores_cpu = self._normalize_edge_scores(
            edge_scores=edge_scores_cpu,
            edge_head_locals=heads_cpu,
            edge_tail_locals=tails_cpu,
            num_nodes=graph.num_nodes,
        )
        topk_edge_indices_cpu = self._select_topk_edges(
            edge_scores=select_scores_cpu,
            edge_top_k=edge_top_k,
        )
        start_edge_indices_cpu = self._select_start_edges(
            heads=heads_cpu,
            tails=tails_cpu,
            edge_scores=select_scores_cpu,
            start_node_locals=start_node_locals_retrieval,
            num_nodes=graph.num_nodes,
            start_keep_ratio=float(self.cfg.start_keep_ratio),
            start_min_edges=int(self.cfg.start_min_edges),
            start_max_edges=self.cfg.start_max_edges,
        )
        if topk_edge_indices_cpu.numel() == 0:
            self.stats["retrieval_failed"] += 1
            return
        edge_candidates = [topk_edge_indices_cpu]
        if start_edge_indices_cpu.numel() > 0:
            edge_candidates.append(start_edge_indices_cpu)
        env_edge_indices_cpu = torch.unique(torch.cat(edge_candidates), sorted=True)
        if env_edge_indices_cpu.numel() == 0:
            self.stats["retrieval_failed"] += 1
            return

        # === C. Environment edge space (E_env) + deduplicate by (h,r,t) ===
        env_edge_heads = heads_cpu.index_select(0, env_edge_indices_cpu)
        env_edge_tails = tails_cpu.index_select(0, env_edge_indices_cpu)
        env_edge_relations = relations_cpu.index_select(0, env_edge_indices_cpu)
        env_edge_scores = edge_scores_cpu.index_select(0, env_edge_indices_cpu)
        env_edge_labels = labels_cpu.index_select(0, env_edge_indices_cpu)
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
        edge_scores = self._normalize_edge_scores(
            edge_scores=edge_scores,
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            num_nodes=int(node_entity_ids.numel()),
        )

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
        if answer_node_locals.numel() == ZERO:
            if not self.cfg.allow_empty_answer:
                self.stats["retrieval_failed"] += 1
                return

            num_edges = int(edge_relations.numel())
            edge_labels = torch.zeros(num_edges, dtype=torch.float32)
            empty_long = torch.empty(0, dtype=torch.long)
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
                pair_start_node_locals=empty_long,
                pair_answer_node_locals=empty_long,
                pair_edge_local_ids=empty_long,
                pair_edge_counts=empty_long,
                pair_shortest_lengths=empty_long,
                gt_path_edge_local_ids=empty_long,
                gt_path_exists=False,
                is_answer_reachable=False,
                is_dummy_agent=True,
            )
            self.samples.append(sample)
            self.stats["num_samples"] += 1
            self.stats["edge_counts"].append(num_edges)
            return

        # === E. Path supervision removed: keep label-derived DAG mask only. ===
        empty_long = torch.empty(0, dtype=torch.long)
        pair_start_node_locals = empty_long
        pair_answer_node_locals = empty_long
        pair_edge_local_ids = empty_long
        pair_edge_counts = empty_long
        pair_shortest_lengths = empty_long
        gt_path_edge_local_ids = empty_long
        gt_path_exists = False
        self.stats["edge_counts"].append(int(edge_relations.numel()))

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
            gt_path_edge_local_ids=gt_path_edge_local_ids,
            gt_path_exists=gt_path_exists,
            is_answer_reachable=is_answer_reachable,
            is_dummy_agent=False,
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
            "gt_path_edge_local_ids": sample.gt_path_edge_local_ids,
            "gt_path_exists": bool(sample.gt_path_exists),
            "is_answer_reachable": bool(sample.is_answer_reachable),
            "is_dummy_agent": bool(sample.is_dummy_agent),
        }

    def _log_stats(self, stats):
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    def _normalize_edge_scores(
        self,
        *,
        edge_scores: torch.Tensor,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if self.cfg.score_mode == SCORE_MODE_LOGITS:
            return edge_scores
        if self.cfg.score_mode != SCORE_MODE_NODE_SOFTMAX:
            raise ValueError(f"Unsupported score_mode: {self.cfg.score_mode}")
        return self._node_softmax_logit(
            edge_scores=edge_scores,
            edge_head_locals=edge_head_locals,
            edge_tail_locals=edge_tail_locals,
            num_nodes=num_nodes,
        )

    @staticmethod
    def _node_softmax_logit(
        *,
        edge_scores: torch.Tensor,
        edge_head_locals: torch.Tensor,
        edge_tail_locals: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if edge_scores.numel() == 0:
            return edge_scores
        device = edge_scores.device
        dtype = edge_scores.dtype
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
        head_idx = edge_head_locals.to(device=device, dtype=torch.long).view(-1)
        tail_idx = edge_tail_locals.to(device=device, dtype=torch.long).view(-1)

        max_head = torch.full((num_nodes,), neg_inf, device=device, dtype=dtype)
        max_head.scatter_reduce_(0, head_idx, edge_scores, reduce="amax", include_self=True)
        exp_head = torch.exp(edge_scores - max_head[head_idx])
        sum_head = torch.zeros(num_nodes, device=device, dtype=dtype)
        sum_head.scatter_add_(0, head_idx, exp_head)
        prob_head = exp_head / sum_head[head_idx].clamp_min(PROB_EPS)

        max_tail = torch.full((num_nodes,), neg_inf, device=device, dtype=dtype)
        max_tail.scatter_reduce_(0, tail_idx, edge_scores, reduce="amax", include_self=True)
        exp_tail = torch.exp(edge_scores - max_tail[tail_idx])
        sum_tail = torch.zeros(num_nodes, device=device, dtype=dtype)
        sum_tail.scatter_add_(0, tail_idx, exp_tail)
        prob_tail = exp_tail / sum_tail[tail_idx].clamp_min(PROB_EPS)

        prob = (prob_head + prob_tail) * HALF
        prob = prob.clamp(min=PROB_EPS, max=1.0 - PROB_EPS)
        return torch.log(prob) - torch.log1p(-prob)

    @staticmethod
    def _select_aux_field(
        aux_data: Optional[Dict[str, Any]],
        core_data: Dict[str, Any],
        field: str,
        default: Any,
    ) -> Any:
        if aux_data is not None and field in aux_data:
            return aux_data[field]
        return core_data.get(field, default)

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
    def _select_start_edges(
        *,
        heads: torch.Tensor,
        tails: torch.Tensor,
        edge_scores: torch.Tensor,
        start_node_locals: torch.Tensor,
        num_nodes: int,
        start_keep_ratio: float,
        start_min_edges: int,
        start_max_edges: Optional[int],
    ) -> torch.Tensor:
        start_nodes = torch.unique(start_node_locals.view(-1))
        if start_nodes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        num_edges = int(edge_scores.numel())
        if num_edges == 0:
            return torch.empty(0, dtype=torch.long)
        heads = heads.view(-1)
        tails = tails.view(-1)
        num_nodes = int(num_nodes)
        deg = torch.bincount(heads, minlength=num_nodes) + torch.bincount(tails, minlength=num_nodes)
        deg_s = deg.index_select(0, start_nodes)
        k_s = torch.ceil(deg_s.to(dtype=torch.float32) * float(start_keep_ratio)).to(dtype=torch.long)
        if start_min_edges > ZERO:
            k_s = torch.maximum(k_s, torch.full_like(k_s, int(start_min_edges)))
        if start_max_edges is not None:
            max_edges = int(start_max_edges)
            if max_edges == ZERO:
                k_s = torch.zeros_like(k_s)
            else:
                k_s = torch.minimum(k_s, torch.full_like(k_s, max_edges))
        k_s = torch.minimum(k_s, deg_s)
        if k_s.numel() == 0 or int(k_s.max().item()) == ZERO:
            return torch.empty(0, dtype=torch.long)

        edge_idx = torch.arange(num_edges, dtype=torch.long)
        incident_nodes = torch.cat([heads, tails])
        incident_edges = torch.cat([edge_idx, edge_idx])
        scores = edge_scores.view(-1)
        incident_scores = torch.cat([scores, scores])

        start_mask = torch.zeros(num_nodes, dtype=torch.bool)
        start_mask[start_nodes] = True
        keep_incident = start_mask[incident_nodes]
        if not bool(keep_incident.any().item()):
            return torch.empty(0, dtype=torch.long)
        nodes = incident_nodes[keep_incident]
        edges = incident_edges[keep_incident]
        edge_scores = incident_scores[keep_incident]

        # Stable sort by node after score order preserves per-node top-k without Python loops.
        order_score = torch.argsort(edge_scores, descending=True, stable=True)
        nodes_sorted = nodes.index_select(0, order_score)
        edges_sorted = edges.index_select(0, order_score)
        order_node = torch.argsort(nodes_sorted, stable=True)
        nodes_grouped = nodes_sorted.index_select(0, order_node)
        edges_grouped = edges_sorted.index_select(0, order_node)
        if nodes_grouped.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        counts = torch.bincount(nodes_grouped, minlength=num_nodes)
        offsets = torch.cumsum(counts, dim=0) - counts
        idx = torch.arange(nodes_grouped.numel(), dtype=torch.long)
        pos_in_group = idx - offsets[nodes_grouped]
        k_per_node = torch.zeros(num_nodes, dtype=torch.long)
        k_per_node[start_nodes] = k_s
        keep = pos_in_group < k_per_node[nodes_grouped]
        if not bool(keep.any().item()):
            return torch.empty(0, dtype=torch.long)
        return torch.unique(edges_grouped[keep], sorted=True)


__all__ = ["GAgentBuilder", "GAgentSettings"]
