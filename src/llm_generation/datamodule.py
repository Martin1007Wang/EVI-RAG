from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.llm_generation.prompting import build_user_prompt

logger = logging.getLogger(__name__)


class _ListDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class LLMGenerationDataModule(LightningDataModule):
    """Prepare prompts for LLM generation from g_agent (or GFlowNet) caches."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        score_dict_path: str,
        questions_path: str,
        entity_vocab_path: str,
        relation_vocab_path: str,
        triplet_limit: int = 100,
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
        self.triplet_limit = int(triplet_limit)
        self.system_prompt = system_prompt
        self.num_workers = int(num_workers)
        self.prompt_tag = prompt_tag
        self.data: List[Dict[str, Any]] = []

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

    def _build_samples(self) -> List[Dict[str, Any]]:
        if not self.score_dict_path.exists():
            raise FileNotFoundError(f"score_dict_path not found: {self.score_dict_path}")
        cache = torch.load(self.score_dict_path, map_location="cpu")
        raw_samples = cache.get("samples") or cache
        if isinstance(raw_samples, dict):
            raw_samples = list(raw_samples.values())
        if not isinstance(raw_samples, list):
            raise ValueError("Unrecognized score dict format; expected list or mapping with 'samples' key.")

        q_df = pd.read_parquet(self.questions_path)
        questions = {row.question_uid: row for row in q_df.itertuples()}

        ent_df = pd.read_parquet(self.entity_vocab_path)
        ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
        rel_df = pd.read_parquet(self.relation_vocab_path)
        rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))

        samples: List[Dict[str, Any]] = []
        for record in raw_samples:
            sample_id = record.get("sample_id")
            if not sample_id or sample_id not in questions:
                continue
            q_row = questions[sample_id]
            question_text = record.get("question") or q_row.question
            answers_text = list(q_row.answer_texts) if hasattr(q_row, "answer_texts") else []

            edges = record.get("selected_edges") or []
            # Descending by score
            edges = sorted(edges, key=lambda e: float(e.get("score", 0.0)), reverse=True)
            triplets = []
            for edge in edges[: self.triplet_limit]:
                head_name = ent_map.get(int(edge["head_entity_id"]), str(edge["head_entity_id"]))
                tail_name = ent_map.get(int(edge["tail_entity_id"]), str(edge["tail_entity_id"]))
                relation_name = rel_map.get(int(edge["relation_id"]), str(edge["relation_id"]))
                triplets.append((head_name, relation_name, tail_name, float(edge.get("score", 0.0))))

            user_prompt = build_user_prompt(
                question=question_text,
                triplets=[(h, r, t) for (h, r, t, _) in triplets],
                limit=self.triplet_limit,
            )
            samples.append(
                {
                    "id": sample_id,
                    "question": question_text,
                    "answers": answers_text,
                    "triplets": triplets,
                    "system_prompt": self.system_prompt,
                    "user_prompt": user_prompt,
                }
            )

        logger.info("Prepared %d samples for %s/%s", len(samples), self.dataset, self.split)
        return samples
