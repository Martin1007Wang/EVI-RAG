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
            triplets_all: List[Tuple[str, str, str, float]] = []
            for edge in edges:
                head_name = edge.get("head_text") or str(edge["head_entity_id"])
                tail_name = edge.get("tail_text") or str(edge["tail_entity_id"])
                relation_name = edge.get("relation_text") or str(edge["relation_id"])
                triplets_all.append((head_name, relation_name, tail_name, float(edge.get("score", 0.0)), int(edge["edge_idx"])))

            for k in self.triplet_limits:
                triplets = triplets_all[: int(k)]
                k_effective = len(triplets)
                retrieved_edge_ids = [edge_idx for (_, _, _, _, edge_idx) in triplets]
                evidence_lines = [f"({h}, {r}, {t})" for (h, r, t, _, _) in triplets]
                evidence_text = "\n".join(evidence_lines)
                user_prompt = build_triplet_prompt(
                    question=question_text,
                    triplets=[(head, rel, tail) for (head, rel, tail, _, _) in triplets],
                    limit=int(k),
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
                        "visible_edge_ids": retrieved_edge_ids,  # no further truncation beyond K here
                        "gt_path_edge_local_ids": gt_path_edges,
                        "evidence_token_count": count_tokens(evidence_text),
                        "prompt_token_count": count_tokens(f"{self.system_prompt}\n{user_prompt}"),
                        "token_budget": None,
                        "evidence_truncated": False,
                    }
                )

        logger.info("Prepared %d samples for %s/%s", len(samples), self.dataset, self.split)
        return samples


__all__ = ["LLMReasonerTripletDataModule"]
