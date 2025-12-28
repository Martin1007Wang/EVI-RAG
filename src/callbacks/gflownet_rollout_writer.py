from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter


class GFlowNetRolloutWriter(BasePredictionWriter):
    """Stream GFlowNet rollouts to JSONL during predict()."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str = "predict",
        enabled: bool = True,
        artifact_name: str = "eval_gflownet",
        schema_version: int = 1,
        textualize: bool = False,
        entity_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(write_interval="batch")
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer.")
        self.textualize = bool(textualize)
        self.entity_vocab_path = entity_vocab_path
        self.relation_vocab_path = relation_vocab_path
        self.overwrite = bool(overwrite)
        self._processor: Optional[_EvalPersistProcessor] = None
        self._output_path: Optional[Path] = None
        self._manifest_path: Optional[Path] = None

    def on_predict_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / f"{self.split}.jsonl"
        self._manifest_path = self.output_dir / f"{self.split}.manifest.json"
        if self.overwrite and trainer.global_rank == 0:
            self._output_path.write_text("", encoding="utf-8")
        if self.textualize:
            self._processor = _EvalPersistProcessor(
                {
                    "textualize": True,
                    "entity_vocab_path": self.entity_vocab_path,
                    "relation_vocab_path": self.relation_vocab_path,
                }
            )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.enabled or prediction is None:
            return
        records = prediction
        if not isinstance(records, list) or not records:
            return
        records = self._gather_records(records)
        if not records:
            return
        if self._processor is not None:
            records = self._processor.process(records)
        if trainer.global_rank != 0:
            return
        if self._output_path is None:
            raise RuntimeError("GFlowNetRolloutWriter missing output path; on_predict_start was not called.")
        with self._output_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or trainer.global_rank != 0:
            return
        if self._manifest_path is None or self._output_path is None:
            raise RuntimeError("GFlowNetRolloutWriter missing manifest/output path; on_predict_start was not called.")
        manifest = {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": self._output_path.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "gflownet_rollout_writer",
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _gather_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: list[Optional[list]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if dist.get_rank() != 0:
                return []
            merged: list[Dict[str, Any]] = []
            for part in gathered:
                if part:
                    merged.extend(part)
            return merged
        return records


@dataclass
class _EvalPersistProcessor:
    """Textualize edges and aggregate rollouts into candidate chains."""

    cfg: Dict[str, Any]

    def __post_init__(self) -> None:
        self.ent_map, self.rel_map = self._resolve_vocab_maps(self.cfg)

    def process(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self.cfg.get("textualize"):
            self._require_edge_metadata(records)
        if self.ent_map is not None or self.rel_map is not None:
            self._inject_text(records)
        for sample in records:
            sample["candidate_chains"] = self._build_candidate_paths(sample)
        return records

    def _inject_text(self, records: list[Dict[str, Any]]) -> None:
        for sample in records:
            for rollout in sample.get("rollouts", []):
                for edge in rollout.get("edges", []):
                    h = edge.get("head_entity_id")
                    r = edge.get("relation_id")
                    t = edge.get("tail_entity_id")
                    s_id = edge.get("src_entity_id")
                    d_id = edge.get("dst_entity_id")
                    if self.ent_map is not None:
                        edge["head_text"] = self.ent_map.get(h, str(h) if h is not None else None)
                        edge["tail_text"] = self.ent_map.get(t, str(t) if t is not None else None)
                        edge["src_text"] = self.ent_map.get(s_id, str(s_id) if s_id is not None else None)
                        edge["dst_text"] = self.ent_map.get(d_id, str(d_id) if d_id is not None else None)
                    if self.rel_map is not None:
                        edge["relation_text"] = self.rel_map.get(r, str(r) if r is not None else None)

    def _require_edge_metadata(self, records: list[Dict[str, Any]]) -> None:
        for sample in records:
            sample_id = sample.get("sample_id", "<unknown>")
            rollouts = sample.get("rollouts")
            if not isinstance(rollouts, list) or not rollouts:
                raise ValueError(f"textualize=true requires non-empty rollouts (sample_id={sample_id}).")
            for ridx, rollout in enumerate(rollouts):
                if "edges" not in rollout:
                    raise ValueError(
                        "textualize=true requires rollout edge metadata; "
                        f"missing 'edges' in sample_id={sample_id}, rollout_index={ridx}."
                    )

    def _build_candidate_paths(self, sample: Dict[str, Any]) -> list[Dict[str, Any]]:
        chain_stats: Dict[tuple, Dict[str, Any]] = {}
        for rollout in sample.get("rollouts", []):
            ridx = int(rollout.get("rollout_index", 0) or 0)
            path = rollout.get("edges") or []
            sig = []
            for e in path:
                src = e.get("src_entity_id")
                if src is None:
                    src = e.get("head_entity_id")
                dst = e.get("dst_entity_id")
                if dst is None:
                    dst = e.get("tail_entity_id")
                sig.append((src, e.get("relation_id"), dst))
            sig = tuple(sig)
            if not sig:
                continue
            stat = chain_stats.setdefault(
                sig,
                {
                    "frequency": 0,
                    "from_rollouts": set(),
                    "example_edges": path,
                },
            )
            stat["frequency"] += 1
            stat["from_rollouts"].add(ridx)

        candidates: list[Dict[str, Any]] = []
        for sig, stat in chain_stats.items():
            edges = stat["example_edges"]
            chain_text = " -> ".join(self._fmt_edge(e) for e in edges)
            candidates.append(
                {
                    "signature": sig,
                    "length": len(edges),
                    "frequency": stat["frequency"],
                    "from_rollouts": sorted(stat["from_rollouts"]),
                    "chain_edges": [
                        {
                            "head_entity_id": e.get("head_entity_id"),
                            "relation_id": e.get("relation_id"),
                            "tail_entity_id": e.get("tail_entity_id"),
                            "head_text": e.get("head_text"),
                            "relation_text": e.get("relation_text"),
                            "tail_text": e.get("tail_text"),
                            "src_entity_id": e.get("src_entity_id"),
                            "dst_entity_id": e.get("dst_entity_id"),
                        }
                        for e in edges
                    ],
                    "chain_text": chain_text,
                }
            )

        candidates.sort(key=lambda c: (-c["frequency"], -c["length"]))
        for i, c in enumerate(candidates, 1):
            c["rank"] = i
        return candidates

    @staticmethod
    def _fmt_edge(e: Dict[str, Any]) -> str:
        def _txt(val_text: Any, val_id: Any) -> str:
            if val_text is not None:
                return str(val_text)
            if val_id is None:
                return "UNK"
            return str(val_id)

        h = _txt(e.get("src_text"), e.get("src_entity_id"))
        t = _txt(e.get("dst_text"), e.get("dst_entity_id"))
        if h == "UNK" and t == "UNK":
            h = _txt(e.get("head_text"), e.get("head_entity_id"))
            t = _txt(e.get("tail_text"), e.get("tail_entity_id"))
        r = _txt(e.get("relation_text"), e.get("relation_id"))
        return f"{h} -[{r}]-> {t}"

    def _resolve_vocab_maps(self, cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            raise ValueError("textualize=true requires both entity_vocab_path and relation_vocab_path.")
        if not Path(entity_path).exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {entity_path}")
        if not Path(relation_path).exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {relation_path}")
        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        if entity_path and Path(entity_path).exists():
            ent_df = pd.read_parquet(entity_path)
            if "entity_id" in ent_df.columns:
                ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
            elif "embedding_id" in ent_df.columns:
                ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
        if relation_path and Path(relation_path).exists():
            rel_df = pd.read_parquet(relation_path)
            rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        return ent_map, rel_map


__all__ = ["GFlowNetRolloutWriter"]
