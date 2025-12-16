from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils.metrics import normalize_k_values

logger = logging.getLogger(__name__)


class _ListDataset(Dataset[Dict[str, Any]]):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class LLMReasonerTruthDataModule(LightningDataModule):
    """Prepare ranked top-k windows for retriever-oracle evaluation.

    Input is the persisted retriever eval cache written by `RetrieverModule`
    (`eval_persist.retriever`), containing `samples[*].triplets_by_k`.
    """

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        retriever_eval_path: str,
        k_values: Sequence[int],
        num_workers: int = 0,
        prompt_tag: str = "truth",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.retriever_eval_path = Path(retriever_eval_path).expanduser().resolve()
        self.k_values = normalize_k_values(k_values)
        if not self.k_values:
            raise ValueError("k_values must be a non-empty list of positive integers.")
        self.max_k = int(max(self.k_values))
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

    def _torch_load(self, path: Path) -> Any:
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        return torch.load(path, **load_kwargs)

    @staticmethod
    def _select_triplets_by_k(triplets_by_k: Dict[Any, Any], k: int) -> List[Dict[str, Any]]:
        if k in triplets_by_k:
            key: Any = k
        elif str(k) in triplets_by_k:
            key = str(k)
        else:
            available: List[int] = []
            for raw_key in triplets_by_k.keys():
                try:
                    available.append(int(raw_key))
                except (TypeError, ValueError):
                    continue
            available.sort()
            raise ValueError(f"triplets_by_k missing k={k} (available={available})")
        edges = triplets_by_k[key]
        if not isinstance(edges, list):
            raise ValueError(f"triplets_by_k[{k}] must be a list, got {type(edges)}")
        return edges

    def _build_samples(self) -> List[Dict[str, Any]]:
        if not self.retriever_eval_path.exists():
            raise FileNotFoundError(f"retriever_eval_path not found: {self.retriever_eval_path}")
        payload = self._torch_load(self.retriever_eval_path)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"Unrecognized retriever eval payload format at {self.retriever_eval_path}")

        raw_samples = payload.get("samples") or []
        if isinstance(raw_samples, dict):
            raw_samples = list(raw_samples.values())
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError(f"Empty retriever eval samples in {self.retriever_eval_path}")

        samples: List[Dict[str, Any]] = []
        dropped_no_answer = 0
        for record in raw_samples:
            if not isinstance(record, dict):
                continue
            sample_id = record.get("sample_id")
            if sample_id in (None, ""):
                continue
            answers = record.get("answer_entity_ids") or []
            if not answers:
                dropped_no_answer += 1
                continue
            triplets_by_k = record.get("triplets_by_k")
            if not isinstance(triplets_by_k, dict):
                raise ValueError(f"Sample {sample_id} missing triplets_by_k dict.")

            edges = self._select_triplets_by_k(triplets_by_k, self.max_k)
            head_ids = [int(e.get("head_entity_id", -1)) for e in edges]
            tail_ids = [int(e.get("tail_entity_id", -1)) for e in edges]
            samples.append(
                {
                    "id": str(sample_id),
                    "question": str(record.get("question", "")),
                    "answer_entity_ids": [int(x) for x in answers],
                    "head_entity_ids": head_ids,
                    "tail_entity_ids": tail_ids,
                }
            )

        logger.info(
            "Prepared %d samples for %s/%s from %s (dropped_no_answer=%d, max_k=%d).",
            len(samples),
            self.dataset,
            self.split,
            self.retriever_eval_path,
            dropped_no_answer,
            self.max_k,
        )
        return samples


__all__ = ["LLMReasonerTruthDataModule"]

