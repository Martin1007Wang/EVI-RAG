from __future__ import annotations

import logging
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.embedding_store import EmbeddingStore
from src.data.components.graph_store import GraphStore
from src.utils.llm_prompting import build_triplet_prompt
from src.utils.metrics import normalize_k_values
from src.utils.text_utils import count_tokens
from datetime import datetime

logger = logging.getLogger(__name__)


class _ListDataset(Dataset[Dict[str, Any]]):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class LLMReasonerTripletDataModule(LightningDataModule):
    """Prepare prompts for LLM reasoning from g_agent/GFlowNet-selected triplets."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        score_dict_path: str,
        questions_path: str,
        entity_vocab_path: str,
        relation_vocab_path: str,
        triplet_limits: Sequence[int],
        retrieval_lmdb_dir: Optional[str] = None,
        token_budget: Optional[int] = None,
        token_budget_encoding: Optional[str] = "cl100k_base",
        system_prompt: str = "You are a concise QA agent. Use the given triplets to answer with `Ans: <entity>`.",
        num_workers: int = 0,
        prompt_tag: str = "triplets",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.score_dict_path = Path(score_dict_path).expanduser().resolve()
        self.questions_path = Path(questions_path).expanduser().resolve()
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve()
        self.triplet_limits = normalize_k_values(triplet_limits)
        if not self.triplet_limits:
            raise ValueError("triplet_limits must be a non-empty list of positive integers.")
        self.retrieval_lmdb_dir = Path(retrieval_lmdb_dir).expanduser().resolve() if retrieval_lmdb_dir else None
        if token_budget is not None and int(token_budget) <= 0:
            raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
        self.token_budget = int(token_budget) if token_budget is not None else None
        self.token_budget_encoding = token_budget_encoding
        self.system_prompt = system_prompt
        self.num_workers = int(num_workers)
        self.prompt_tag = prompt_tag
        self.data: List[Dict[str, Any]] = []
        self._graph_store: Optional[GraphStore] = None
        self._embedding_store: Optional[EmbeddingStore] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._embedding_store = None
        if self.retrieval_lmdb_dir is not None:
            lmdb_path = self._resolve_retrieval_lmdb_path(self.retrieval_lmdb_dir, split=self.split)
            self._embedding_store = EmbeddingStore(lmdb_path)
        self.data = self._build_samples()

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            _ListDataset(self.data),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )

    def _load_vocab_maps(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Use vocabulary LMDB if provided; otherwise fall back to parquet maps.
        """
        if self._graph_store is None and self.entity_vocab_path.suffix == ".lmdb":
            self._graph_store = GraphStore(self.entity_vocab_path)

        if self._graph_store is not None:
            return dict(self._graph_store.id2entity), dict(self._graph_store.id2relation)

        ent_map: Dict[int, str] = {}
        rel_map: Dict[int, str] = {}
        try:
            if self.entity_vocab_path.exists():
                ent_df = pd.read_parquet(self.entity_vocab_path)
                if "entity_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
                elif "embedding_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
            if self.relation_vocab_path.exists():
                rel_df = pd.read_parquet(self.relation_vocab_path)
                if "relation_id" in rel_df.columns and "label" in rel_df.columns:
                    rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load vocab maps: %s", exc)
        return ent_map, rel_map

    @staticmethod
    def _resolve_retrieval_lmdb_path(embeddings_dir: Path, *, split: str) -> str:
        candidate = embeddings_dir / f"{split}.lmdb"
        if candidate.exists():
            return str(candidate)
        if str(split).lower() in {"val", "valid", "validation"}:
            fallback = embeddings_dir / "val.lmdb"
            if fallback.exists():
                return str(fallback)
        return str(candidate)

    @staticmethod
    def _select_visible_prefix_by_budget(
        lines: Sequence[str],
        *,
        token_budget: int,
        encoding: Optional[str],
    ) -> Tuple[int, int, bool]:
        """
        Select the longest prefix of `lines` such that token_count(join(prefix)) <= token_budget.

        Returns:
            - visible_count: number of lines kept
            - visible_tokens: token count of the kept evidence text
            - truncated: whether any line was dropped due to budget
        """
        if not lines:
            return 0, 0, False
        if token_budget <= 0:
            return 0, 0, True

        lo, hi = 0, len(lines)
        best = 0
        best_tokens = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            text = "\n".join(lines[:mid])
            tokens = count_tokens(text, encoding=encoding)
            if tokens <= token_budget:
                best = mid
                best_tokens = tokens
                lo = mid + 1
            else:
                hi = mid - 1
        return best, best_tokens, best < len(lines)

    def _load_gt_paths_triples(self, sample_id: str) -> List[List[Tuple[int, int, int]]]:
        """
        Load GT paths as global-id triples from the retrieval LMDB (if available).

        Returns:
            List of paths, each path is a list of (head_gid, rel_gid, tail_gid).
            Empty list means GT is unavailable for this sample (or LMDB not configured).
        """
        if self._embedding_store is None:
            return []
        try:
            raw = self._embedding_store.load_sample(str(sample_id))
        except Exception as exc:  # pragma: no cover - dataset dependent
            logger.warning("Failed to load GT paths for sample_id=%s: %s", sample_id, exc)
            return []

        gt_paths = raw.get("gt_paths_triples") or []
        normalized: List[List[Tuple[int, int, int]]] = []
        if isinstance(gt_paths, (list, tuple)) and gt_paths:
            for path in gt_paths:
                if not isinstance(path, (list, tuple)) or not path:
                    continue
                triples: List[Tuple[int, int, int]] = []
                for triple in path:
                    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                        continue
                    try:
                        h, r, t = triple
                        triples.append((int(h), int(r), int(t)))
                    except Exception:
                        continue
                if triples:
                    normalized.append(triples)
        if normalized:
            return normalized

        raw_gt_indices = raw.get("gt_path_edge_indices") or []
        if not raw_gt_indices:
            return []

        try:
            edge_index = torch.as_tensor(raw.get("edge_index"), dtype=torch.long)
            edge_attr = torch.as_tensor(raw.get("edge_attr"), dtype=torch.long).view(-1)
            node_global_ids = torch.as_tensor(raw.get("node_global_ids"), dtype=torch.long).view(-1)
            gt_indices = torch.as_tensor(raw_gt_indices, dtype=torch.long).view(-1)
        except Exception:  # pragma: no cover - defensive
            return []

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return []
        num_edges = int(edge_index.size(1))
        if edge_attr.numel() != num_edges:
            return []
        if node_global_ids.numel() == 0:
            return []

        valid = (gt_indices >= 0) & (gt_indices < num_edges)
        gt_indices = gt_indices[valid]
        if gt_indices.numel() == 0:
            return []

        triples: List[Tuple[int, int, int]] = []
        for e_idx in gt_indices.tolist():
            h_local = int(edge_index[0, e_idx].item())
            t_local = int(edge_index[1, e_idx].item())
            if h_local < 0 or t_local < 0 or h_local >= int(node_global_ids.numel()) or t_local >= int(node_global_ids.numel()):
                continue
            triples.append(
                (
                    int(node_global_ids[h_local].item()),
                    int(edge_attr[e_idx].item()),
                    int(node_global_ids[t_local].item()),
                )
            )
        return [triples] if triples else []

    @staticmethod
    def _extract_edges(record: Dict[str, Any], id2entity: Dict[int, str], id2relation: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Reconstruct edges from g_agent tensors; keep full set (top-k ∪ GT), sorted by score desc.
        We preserve edge_idx (the position in the aggregated edge arrays) for hit checks.
        """
        heads_local = torch.as_tensor(record["edge_head_locals"], dtype=torch.long)
        tails_local = torch.as_tensor(record["edge_tail_locals"], dtype=torch.long)
        relations = torch.as_tensor(record["edge_relations"], dtype=torch.long)
        scores = torch.as_tensor(record.get("edge_scores", torch.zeros_like(relations)), dtype=torch.float)
        node_entity_ids = torch.as_tensor(record["node_entity_ids"], dtype=torch.long)

        edges: List[Dict[str, Any]] = []
        for idx, (h_local, t_local, rel_id, score) in enumerate(
            zip(heads_local.tolist(), tails_local.tolist(), relations.tolist(), scores.tolist())
        ):
            head_gid = int(node_entity_ids[int(h_local)].item())
            tail_gid = int(node_entity_ids[int(t_local)].item())
            rel_id_int = int(rel_id)
            edges.append(
                {
                    "edge_idx": int(idx),
                    "head_entity_id": head_gid,
                    "tail_entity_id": tail_gid,
                    "relation_id": rel_id_int,
                    "score": float(score),
                    "head_text": id2entity.get(head_gid),
                    "tail_text": id2entity.get(tail_gid),
                    "relation_text": id2relation.get(rel_id_int),
                }
            )

        edges.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        return edges

    def _derive_answers(self, record: Dict[str, Any], q_row: Optional[Any], id2entity: Dict[int, str]) -> List[str]:
        if q_row is not None and hasattr(q_row, "answer_texts") and q_row.answer_texts is not None:
            return list(q_row.answer_texts)
        answers: List[str] = []
        for aid in record.get("answer_entity_ids", []):
            aid_int = int(aid)
            answers.append(id2entity.get(aid_int, str(aid_int)))
        return answers

    def _build_samples(self) -> List[Dict[str, Any]]:
        if not self.score_dict_path.exists():
            raise FileNotFoundError(f"score_dict_path not found: {self.score_dict_path}")
        stat = self.score_dict_path.stat()
        logger.info(
            "Loading g_agent cache from %s (size=%.2f MB, ctime=%s, mtime=%s)",
            self.score_dict_path,
            stat.st_size / (1024 * 1024),
            datetime.fromtimestamp(stat.st_ctime),
            datetime.fromtimestamp(stat.st_mtime),
        )
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        cache = torch.load(self.score_dict_path, **load_kwargs)
        raw_samples = cache.get("samples") or cache
        if isinstance(raw_samples, dict):
            raw_samples = list(raw_samples.values())
        if not isinstance(raw_samples, list):
            raise ValueError("Unrecognized score dict format; expected list or mapping with 'samples' key.")

        q_df = pd.read_parquet(self.questions_path)
        questions = {row.question_uid: row for row in q_df.itertuples()}
        id2entity, id2relation = self._load_vocab_maps()

        samples: List[Dict[str, Any]] = []
        for record in raw_samples:
            sample_id = record.get("sample_id")
            if not sample_id or sample_id not in questions:
                continue
            q_row = questions[sample_id]
            question_text = record.get("question") or q_row.question
            answers_text = self._derive_answers(record, q_row, id2entity)

            edges = self._extract_edges(record, id2entity, id2relation)
            gt_path_edges = [int(x) for x in record.get("gt_path_edge_local_ids", [])]
            gt_paths_triples = self._load_gt_paths_triples(str(sample_id))
            edge_by_local_id = {int(edge["edge_idx"]): edge for edge in edges}
            triplets_all: List[Tuple[str, str, str, float, int]] = []
            for edge in edges:
                head_name = edge.get("head_text") or str(edge["head_entity_id"])
                tail_name = edge.get("tail_text") or str(edge["tail_entity_id"])
                relation_name = edge.get("relation_text") or str(edge["relation_id"])
                triplets_all.append((head_name, relation_name, tail_name, float(edge.get("score", 0.0)), int(edge["edge_idx"])))

            for k in self.triplet_limits:
                triplets = triplets_all[: int(k)]
                k_effective = len(triplets)
                retrieved_edge_ids = [edge_idx for (_, _, _, _, edge_idx) in triplets]
                retrieved_triples = {
                    (
                        int(edge_dict["head_entity_id"]),
                        int(edge_dict["relation_id"]),
                        int(edge_dict["tail_entity_id"]),
                    )
                    for edge_dict in (edge_by_local_id.get(int(edge_idx)) for edge_idx in retrieved_edge_ids)
                    if edge_dict is not None
                }
                hit_set: Optional[bool] = None
                hit_vis: Optional[bool] = None
                if gt_paths_triples:
                    hit_set = any(set(path).issubset(retrieved_triples) for path in gt_paths_triples)

                evidence_lines_all = [f"({h}, {r}, {t})" for (h, r, t, _, _) in triplets]
                if self.token_budget is None:
                    visible_count = len(evidence_lines_all)
                    visible_tokens = count_tokens("\n".join(evidence_lines_all), encoding=self.token_budget_encoding)
                    evidence_truncated = False
                else:
                    visible_count, visible_tokens, evidence_truncated = self._select_visible_prefix_by_budget(
                        evidence_lines_all,
                        token_budget=int(self.token_budget),
                        encoding=self.token_budget_encoding,
                    )

                visible_edge_ids = retrieved_edge_ids[:visible_count]
                visible_triples = {
                    (
                        int(edge_dict["head_entity_id"]),
                        int(edge_dict["relation_id"]),
                        int(edge_dict["tail_entity_id"]),
                    )
                    for edge_dict in (edge_by_local_id.get(int(edge_idx)) for edge_idx in visible_edge_ids)
                    if edge_dict is not None
                }
                if gt_paths_triples:
                    hit_vis = any(set(path).issubset(visible_triples) for path in gt_paths_triples)

                evidence_text = "\n".join(evidence_lines_all[:visible_count])
                user_prompt = build_triplet_prompt(
                    question=question_text,
                    triplets=[(head, rel, tail) for (head, rel, tail, _, _) in triplets[:visible_count]],
                    limit=int(visible_count),
                )
                samples.append(
                    {
                        "id": sample_id,
                        "question": question_text,
                        "answers": answers_text,
                        "triplets": triplets,
                        "system_prompt": self.system_prompt,
                        "user_prompt": user_prompt,
                        "prompt_tag": self.prompt_tag,
                        "window_k": int(k),
                        "k_effective": int(k_effective),
                        "retrieved_edge_ids": retrieved_edge_ids,
                        "visible_edge_ids": visible_edge_ids,
                        "gt_path_edge_local_ids": gt_path_edges,
                        "hit_set": hit_set,
                        "hit_vis": hit_vis,
                        "evidence_token_count": int(visible_tokens),
                        "prompt_token_count": count_tokens(f"{self.system_prompt}\n{user_prompt}"),
                        "token_budget": self.token_budget,
                        "evidence_truncated": bool(evidence_truncated),
                    }
                )

        logger.info("Prepared %d samples for %s/%s", len(samples), self.dataset, self.split)
        return samples


__all__ = ["LLMReasonerTripletDataModule"]
