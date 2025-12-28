from __future__ import annotations

import logging
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.graph_store import GraphStore
from src.utils.llm_prompting import build_triplet_prompt
from src.utils.metrics import normalize_k_values
from src.utils.text_utils import count_tokens
from datetime import datetime

from .answer_utils import normalize_answer_texts
logger = logging.getLogger(__name__)


class _ListDataset(Dataset[Dict[str, Any]]):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class ReasonerTripletDataModule(LightningDataModule):
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
        system_prompt: str = (
            "You are a concise QA agent. Use the given triplets to return JSON with the full answer list."
        ),
        num_workers: int = 0,
        prompt_tag: str = "triplet",
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
        if self.token_budget is not None and self.token_budget_encoding is None:
            raise ValueError("token_budget_encoding must be set when token_budget is provided.")
        self.system_prompt = system_prompt
        self.num_workers = int(num_workers)
        self.prompt_tag = prompt_tag
        self.data: List[Dict[str, Any]] = []
        self._graph_store: Optional[GraphStore] = None

    def setup(self, stage: Optional[str] = None) -> None:
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
        Use vocabulary LMDB or parquet maps (explicit, no implicit fallback).
        """
        if self.entity_vocab_path.suffix == ".lmdb" or self.relation_vocab_path.suffix == ".lmdb":
            if self.entity_vocab_path.suffix != ".lmdb" or self.relation_vocab_path.suffix != ".lmdb":
                raise ValueError("entity_vocab_path and relation_vocab_path must both be LMDB paths.")
            if self.entity_vocab_path != self.relation_vocab_path:
                raise ValueError("entity_vocab_path and relation_vocab_path must point to the same LMDB.")
            if self._graph_store is None:
                self._graph_store = GraphStore(self.entity_vocab_path)
            return dict(self._graph_store.id2entity), dict(self._graph_store.id2relation)

        if not self.entity_vocab_path.exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {self.entity_vocab_path}")
        if not self.relation_vocab_path.exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {self.relation_vocab_path}")

        ent_df = pd.read_parquet(self.entity_vocab_path)
        if "entity_id" in ent_df.columns:
            ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
        elif "embedding_id" in ent_df.columns:
            ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
        else:
            raise ValueError("entity_vocab parquet missing entity_id/embedding_id columns.")

        rel_df = pd.read_parquet(self.relation_vocab_path)
        if "relation_id" not in rel_df.columns or "label" not in rel_df.columns:
            raise ValueError("relation_vocab parquet missing relation_id/label columns.")
        rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        return ent_map, rel_map

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

    @staticmethod
    def _extract_edges(record: Dict[str, Any], id2entity: Dict[int, str], id2relation: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Reconstruct edges from g_agent tensors; keep full set (top-k ∪ GT), sorted by score desc.
        We preserve edge_idx (the position in the aggregated edge arrays) for hit checks.
        """
        heads_local = torch.as_tensor(record["edge_head_locals"], dtype=torch.long)
        tails_local = torch.as_tensor(record["edge_tail_locals"], dtype=torch.long)
        relations = torch.as_tensor(record["edge_relations"], dtype=torch.long)
        scores = torch.as_tensor(record["edge_scores"], dtype=torch.float)
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

        edges.sort(key=lambda e: float(e["score"]), reverse=True)
        return edges

    def _derive_answers(self, q_row: Optional[Any], *, sample_id: str) -> List[str]:
        if q_row is None:
            raise ValueError(f"missing question row for sample_id={sample_id}")
        if not hasattr(q_row, "answer_texts"):
            raise ValueError(f"questions.parquet missing 'answer_texts' column (sample_id={sample_id})")
        return normalize_answer_texts(q_row.answer_texts, field_name="answer_texts", sample_id=sample_id)

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
        if "samples" not in cache:
            raise ValueError("score_dict missing required 'samples' list.")
        raw_samples = cache["samples"]
        if not isinstance(raw_samples, list):
            raise ValueError("score_dict['samples'] must be a list.")

        q_df = pd.read_parquet(self.questions_path)
        questions = {row.question_uid: row for row in q_df.itertuples()}
        id2entity, id2relation = self._load_vocab_maps()

        samples: List[Dict[str, Any]] = []
        for record in raw_samples:
            sample_id = record["sample_id"]
            if sample_id not in questions:
                raise ValueError(f"questions.parquet missing sample_id={sample_id}")
            q_row = questions[sample_id]
            question_text = record["question"]
            answers_text = self._derive_answers(q_row, sample_id=str(sample_id))

            edges = self._extract_edges(record, id2entity, id2relation)
            if "gt_path_edge_local_ids" not in record:
                raise KeyError(f"gt_path_edge_local_ids missing for sample_id={sample_id}")
            gt_path_edges = [int(x) for x in record["gt_path_edge_local_ids"]]
            edge_labels = torch.as_tensor(record["edge_labels"], dtype=torch.float32).view(-1)
            if edge_labels.numel() != len(edges):
                raise ValueError(f"edge_labels length mismatch for sample_id={sample_id}")
            dag_edge_ids = {int(idx) for idx, val in enumerate(edge_labels.tolist()) if val > 0.5}
            triplets_all: List[Tuple[str, str, str, float, int]] = []
            for edge in edges:
                if edge.get("head_text") is None or edge.get("tail_text") is None or edge.get("relation_text") is None:
                    raise ValueError(f"Missing vocab text for edge {edge.get('edge_idx')} (sample_id={sample_id})")
                head_name = edge["head_text"]
                tail_name = edge["tail_text"]
                relation_name = edge["relation_text"]
                triplets_all.append((head_name, relation_name, tail_name, float(edge["score"]), int(edge["edge_idx"])))

            for k in self.triplet_limits:
                triplets = triplets_all[: int(k)]
                k_effective = len(triplets)
                retrieved_edge_ids = [edge_idx for (_, _, _, _, edge_idx) in triplets]
                hit_set: Optional[bool] = None
                hit_vis: Optional[bool] = None
                if dag_edge_ids:
                    hit_set = bool(retrieved_edge_ids) and set(retrieved_edge_ids).issubset(dag_edge_ids)

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
                if dag_edge_ids:
                    hit_vis = bool(visible_edge_ids) and set(visible_edge_ids).issubset(dag_edge_ids)

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


__all__ = ["ReasonerTripletDataModule"]
