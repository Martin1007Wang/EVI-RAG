from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.g_agent_dataset import GAgentSample, load_g_agent_samples
from src.utils.metrics import normalize_k_values

logger = logging.getLogger(__name__)


class _ListDataset(Dataset[Dict[str, Any]]):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class SelectorEvalDataModule(LightningDataModule):
    """Prepare g_agent samples + selector-specific evidence for evaluation."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        g_agent_path: str,
        selector: str,
        k_values: List[int],
        eval_gflownet_path: Optional[str] = None,
        eval_gflownet_artifact_name: str = "eval_gflownet",
        eval_gflownet_schema_version: int = 1,
        drop_unreachable: bool = False,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.g_agent_path = Path(g_agent_path).expanduser().resolve()
        self.selector = str(selector).strip().lower()
        self.k_values = normalize_k_values(k_values)
        if not self.k_values:
            raise ValueError("k_values must be a non-empty list of positive integers.")
        self.eval_gflownet_path = Path(eval_gflownet_path).expanduser().resolve() if eval_gflownet_path else None
        self.eval_gflownet_artifact_name = str(eval_gflownet_artifact_name).strip()
        if not self.eval_gflownet_artifact_name:
            raise ValueError("eval_gflownet_artifact_name must be a non-empty string.")
        self.eval_gflownet_schema_version = int(eval_gflownet_schema_version)
        if self.eval_gflownet_schema_version <= 0:
            raise ValueError("eval_gflownet_schema_version must be a positive integer.")
        self.drop_unreachable = bool(drop_unreachable)
        self.num_workers = int(num_workers)
        self.data: List[Dict[str, Any]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.g_agent_path.exists():
            raise FileNotFoundError(f"g_agent_path not found: {self.g_agent_path}")
        samples = load_g_agent_samples(self.g_agent_path, drop_unreachable=self.drop_unreachable)
        rollouts_map: Dict[str, List[List[int]]] = {}
        if self.selector == "gflownet_rollouts":
            if self.eval_gflownet_path is None:
                raise ValueError("eval_gflownet_path must be set for selector=gflownet_rollouts.")
            self._validate_eval_gflownet_manifest(self.eval_gflownet_path)
            rollouts_map = self._load_rollouts(self.eval_gflownet_path)
        elif self.selector != "retriever_topk":
            raise ValueError(f"Unsupported selector={self.selector!r}. Use 'retriever_topk' or 'gflownet_rollouts'.")

        records: List[Dict[str, Any]] = []
        missing_rollouts = 0
        for sample in samples:
            rollouts = None
            if self.selector == "gflownet_rollouts":
                rollouts = rollouts_map.get(sample.sample_id)
                if rollouts is None:
                    missing_rollouts += 1
                    continue
            records.append(self._sample_to_record(sample, rollouts))

        if self.selector == "gflownet_rollouts" and missing_rollouts > 0:
            raise ValueError(f"Missing rollouts for {missing_rollouts} g_agent samples.")

        self.data = records
        logger.info(
            "Prepared %d selector-eval samples for %s/%s (selector=%s).",
            len(self.data),
            self.dataset,
            self.split,
            self.selector,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            _ListDataset(self.data),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: batch,
        )

    @staticmethod
    def _sample_to_record(sample: GAgentSample, rollouts: Optional[List[List[int]]]) -> Dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "edge_head_locals": sample.edge_head_locals,
            "edge_tail_locals": sample.edge_tail_locals,
            "edge_scores": sample.edge_scores,
            "edge_labels": sample.edge_labels,
            "start_node_locals": sample.start_node_locals,
            "answer_node_locals": sample.answer_node_locals,
            "num_nodes": int(sample.node_entity_ids.numel()),
            "rollout_edge_ids": rollouts,
        }

    @staticmethod
    def _load_rollouts(path: Path) -> Dict[str, List[List[int]]]:
        if path.suffix == ".jsonl":
            raw_samples = []
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if isinstance(record, dict):
                    raw_samples.append(record)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
        else:
            load_kwargs = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
            payload = torch.load(path, **load_kwargs)
            raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(raw_samples, list):
            raise ValueError(f"Unrecognized eval_gflownet format at {path}")

        rollouts_map: Dict[str, List[List[int]]] = {}
        for record in raw_samples:
            if not isinstance(record, dict):
                continue
            sample_id = record.get("sample_id")
            if sample_id in (None, ""):
                continue
            rollouts = record.get("rollouts")
            if not isinstance(rollouts, list):
                raise ValueError(f"Sample {sample_id} missing rollouts list in {path}")
            edge_lists: List[List[int]] = []
            for ridx, rollout in enumerate(rollouts):
                if not isinstance(rollout, dict):
                    raise ValueError(f"Rollout {ridx} must be dict for sample_id={sample_id}")
                if "edge_ids" in rollout:
                    edge_ids_raw = rollout.get("edge_ids")
                    if not isinstance(edge_ids_raw, list):
                        raise ValueError(f"Rollout {ridx} edge_ids must be list for sample_id={sample_id}")
                    edge_ids: List[int] = []
                    for eidx, edge_id in enumerate(edge_ids_raw):
                        if not isinstance(edge_id, int):
                            raise ValueError(
                                f"Rollout {ridx} edge_ids[{eidx}] must be int for sample_id={sample_id}"
                            )
                        edge_ids.append(int(edge_id))
                    edge_lists.append(edge_ids)
                    continue

                edges = rollout.get("edges")
                if edges is None:
                    raise ValueError(f"Rollout {ridx} missing edges/edge_ids for sample_id={sample_id}")
                if not isinstance(edges, list):
                    raise ValueError(f"Rollout {ridx} edges must be list for sample_id={sample_id}")
                edge_ids: List[int] = []
                for eidx, edge in enumerate(edges):
                    if not isinstance(edge, dict):
                        raise ValueError(f"Edge {eidx} must be dict for sample_id={sample_id}")
                    if "edge_id" not in edge:
                        raise ValueError(f"Edge {eidx} missing edge_id for sample_id={sample_id}")
                    edge_ids.append(int(edge["edge_id"]))
                edge_lists.append(edge_ids)
            rollouts_map[str(sample_id)] = edge_lists
        if not rollouts_map:
            raise ValueError(f"No rollouts loaded from {path}")
        return rollouts_map

    def _validate_eval_gflownet_manifest(self, data_path: Path) -> None:
        if not data_path.exists():
            self._raise_missing_eval_gflownet(data_path)
        manifest_path = data_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_gflownet manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact") != self.eval_gflownet_artifact_name:
            raise ValueError(
                f"eval_gflownet manifest artifact mismatch: expected '{self.eval_gflownet_artifact_name}', "
                f"got '{manifest.get('artifact')}'."
            )
        if int(manifest.get("schema_version", -1)) != self.eval_gflownet_schema_version:
            raise ValueError(
                f"eval_gflownet manifest schema_version mismatch: expected {self.eval_gflownet_schema_version}, "
                f"got {manifest.get('schema_version')}."
            )
        if manifest.get("file") != data_path.name:
            raise ValueError(
                f"eval_gflownet manifest file mismatch: expected '{data_path.name}', "
                f"got '{manifest.get('file')}'."
            )

    @staticmethod
    def _raise_missing_eval_gflownet(data_path: Path) -> None:
        split = data_path.stem
        base_dir = data_path.parent
        legacy_candidates = [
            base_dir / f"{split}_eval_gflownet.jsonl",
            base_dir / f"{split}_gflownet_eval.jsonl",
        ]
        existing = [str(p) for p in legacy_candidates if p.exists()]
        if existing:
            raise FileNotFoundError(
                "eval_gflownet artifact naming mismatch. "
                f"Expected {data_path} but found legacy file(s): {existing}. "
                "Rename them to '<split>.jsonl' or re-run eval_gflownet."
            )
        raise FileNotFoundError(f"eval_gflownet_path not found: {data_path}")


__all__ = ["SelectorEvalDataModule"]
