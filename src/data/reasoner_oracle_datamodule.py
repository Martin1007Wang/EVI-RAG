from __future__ import annotations

import inspect
import json
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


class ReasonerOracleDataModule(LightningDataModule):
    """Prepare ranked top-k windows for retriever-oracle evaluation.

    Input is the persisted eval_retriever cache written by `RetrieverModule`
    (`eval_persist.retriever`), containing `samples[*].triplets_by_k`.
    """

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        eval_retriever_path: str,
        k_values: Sequence[int],
        artifact_name: str = "eval_retriever",
        schema_version: int = 1,
        num_workers: int = 0,
        prompt_tag: str = "oracle",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.eval_retriever_path = Path(eval_retriever_path).expanduser().resolve()
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer.")
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
            available = sorted(str(raw_key) for raw_key in triplets_by_k.keys())
            raise ValueError(f"triplets_by_k missing k={k} (available={available})")
        edges = triplets_by_k[key]
        if not isinstance(edges, list):
            raise ValueError(f"triplets_by_k[{k}] must be a list, got {type(edges)}")
        return edges

    def _build_samples(self) -> List[Dict[str, Any]]:
        if not self.eval_retriever_path.exists():
            self._raise_missing_eval_retriever()
        self._validate_manifest(self.eval_retriever_path)
        payload = self._torch_load(self.eval_retriever_path)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"Unrecognized eval_retriever payload format at {self.eval_retriever_path}")

        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError(f"Empty eval_retriever samples in {self.eval_retriever_path}")

        samples: List[Dict[str, Any]] = []
        for record in raw_samples:
            if not isinstance(record, dict):
                raise ValueError("eval_retriever samples must be dicts.")
            sample_id = record["sample_id"]
            if sample_id in (None, ""):
                raise ValueError("sample_id is empty.")
            answers = record["answer_entity_ids"]
            if not answers:
                raise ValueError(f"answer_entity_ids empty for sample_id={sample_id}")
            triplets_by_k = record["triplets_by_k"]
            if not isinstance(triplets_by_k, dict):
                raise ValueError(f"Sample {sample_id} missing triplets_by_k dict.")

            edges = self._select_triplets_by_k(triplets_by_k, self.max_k)
            head_ids = [int(e["head_entity_id"]) for e in edges]
            tail_ids = [int(e["tail_entity_id"]) for e in edges]
            samples.append(
                {
                    "id": str(sample_id),
                    "question": str(record["question"]),
                    "answer_entity_ids": [int(x) for x in answers],
                    "head_entity_ids": head_ids,
                    "tail_entity_ids": tail_ids,
                }
            )

        logger.info(
            "Prepared %d samples for %s/%s from %s (max_k=%d).",
            len(samples),
            self.dataset,
            self.split,
            self.eval_retriever_path,
            self.max_k,
        )
        return samples

    def _validate_manifest(self, data_path: Path) -> None:
        manifest_path = data_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_retriever manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "artifact" not in manifest:
            raise ValueError("eval_retriever manifest missing artifact.")
        if manifest["artifact"] != self.artifact_name:
            raise ValueError(
                f"eval_retriever manifest artifact mismatch: expected '{self.artifact_name}', "
                f"got '{manifest['artifact']}'."
            )
        if "schema_version" not in manifest:
            raise ValueError("eval_retriever manifest missing schema_version.")
        if int(manifest["schema_version"]) != self.schema_version:
            raise ValueError(
                f"eval_retriever manifest schema_version mismatch: expected {self.schema_version}, "
                f"got {manifest['schema_version']}."
            )
        if "file" not in manifest:
            raise ValueError("eval_retriever manifest missing file.")
        if manifest["file"] != data_path.name:
            raise ValueError(
                f"eval_retriever manifest file mismatch: expected '{data_path.name}', "
                f"got '{manifest['file']}'."
            )

    def _raise_missing_eval_retriever(self) -> None:
        split = str(self.split)
        base_dir = self.eval_retriever_path.parent
        legacy_candidates = [
            base_dir / f"{split}_eval_retriever.pt",
            base_dir / f"{split}_retriever_eval.pt",
        ]
        existing = [str(p) for p in legacy_candidates if p.exists()]
        if existing:
            raise FileNotFoundError(
                "eval_retriever artifact naming mismatch. "
                f"Expected {self.eval_retriever_path} but found legacy file(s): {existing}. "
                "Rename them to '<split>.pt' or re-run eval_retriever."
            )
        raise FileNotFoundError(f"eval_retriever_path not found: {self.eval_retriever_path}")


__all__ = ["ReasonerOracleDataModule"]
