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


_ZERO = 0
_ONE = 1
_DEFAULT_SPLIT = "predict"
_DEFAULT_ARTIFACT_NAME = "eval_gflownet"
_DEFAULT_SCHEMA_VERSION = _ONE
_WRITE_INTERVAL = "batch"
_UNKNOWN_TEXT = "UNK"
_DEFAULT_ROLLOUT_INDEX = _ZERO
_RANK_OFFSET = _ONE


class GFlowNetRolloutArtifactWriter(BasePredictionWriter):
    """Persist GFlowNet rollout artifacts during predict()."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str = _DEFAULT_SPLIT,
        enabled: bool = True,
        artifact_name: str = _DEFAULT_ARTIFACT_NAME,
        schema_version: int = _DEFAULT_SCHEMA_VERSION,
        textualize: bool = False,
        entity_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(write_interval=_WRITE_INTERVAL)
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= _ZERO:
            raise ValueError("schema_version must be a positive integer.")
        self.textualize = bool(textualize)
        self.entity_vocab_path = entity_vocab_path
        self.relation_vocab_path = relation_vocab_path
        self.overwrite = bool(overwrite)
        self._processor: Optional[_RolloutArtifactProcessor] = None
        self._output_path: Optional[Path] = None
        self._manifest_path: Optional[Path] = None

    def on_predict_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / f"{self.split}.jsonl"
        self._manifest_path = self.output_dir / f"{self.split}.manifest.json"
        if self.overwrite and trainer.global_rank == _ZERO:
            self._output_path.write_text("", encoding="utf-8")
        if self.textualize:
            self._processor = _RolloutArtifactProcessor(
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
        records = self._apply_processor(records)
        if trainer.global_rank != _ZERO:
            return
        self._append_records(records)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or trainer.global_rank != _ZERO:
            return
        if self._manifest_path is None or self._output_path is None:
            raise RuntimeError("GFlowNetRolloutArtifactWriter missing manifest/output path; on_predict_start was not called.")
        manifest = self._build_manifest(self._output_path.name)
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _apply_processor(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self._processor is None:
            return records
        return self._processor.process(records)

    def _append_records(self, records: list[Dict[str, Any]]) -> None:
        if self._output_path is None:
            raise RuntimeError("GFlowNetRolloutArtifactWriter missing output path; on_predict_start was not called.")
        with self._output_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_manifest(self, file_name: str) -> Dict[str, Any]:
        return {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": file_name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "gflownet_rollout_artifact_writer",
        }

    @staticmethod
    def _gather_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: list[Optional[list]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if dist.get_rank() != _ZERO:
                return []
            merged: list[Dict[str, Any]] = []
            for part in gathered:
                if part:
                    merged.extend(part)
            return merged
        return records


@dataclass
class _RolloutArtifactProcessor:
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
        chain_stats = self._collect_chain_stats(sample)
        if not chain_stats:
            return []
        candidates = self._build_candidates(chain_stats)
        return self._rank_candidates(candidates)

    def _collect_chain_stats(self, sample: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
        chain_stats: Dict[tuple, Dict[str, Any]] = {}
        for rollout in sample.get("rollouts", []):
            ridx = int(rollout.get("rollout_index", _DEFAULT_ROLLOUT_INDEX) or _DEFAULT_ROLLOUT_INDEX)
            path = rollout.get("edges") or []
            sig = self._signature_from_edges(path)
            if not sig:
                continue
            stat = chain_stats.setdefault(
                sig,
                {
                    "frequency": _ZERO,
                    "from_rollouts": set(),
                    "example_edges": path,
                },
            )
            stat["frequency"] += _ONE
            stat["from_rollouts"].add(ridx)
        return chain_stats

    def _build_candidates(self, chain_stats: Dict[tuple, Dict[str, Any]]) -> list[Dict[str, Any]]:
        candidates: list[Dict[str, Any]] = []
        for sig, stat in chain_stats.items():
            edges = stat["example_edges"]
            candidates.append(self._build_candidate(sig, stat, edges))
        return candidates

    def _build_candidate(self, signature: tuple, stat: Dict[str, Any], edges: list[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "signature": signature,
            "length": len(edges),
            "frequency": stat["frequency"],
            "from_rollouts": sorted(stat["from_rollouts"]),
            "chain_edges": self._format_chain_edges(edges),
            "chain_text": self._format_chain_text(edges),
        }

    def _format_chain_edges(self, edges: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [
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
        ]

    def _format_chain_text(self, edges: list[Dict[str, Any]]) -> str:
        return " -> ".join(self._fmt_edge(e) for e in edges)

    def _rank_candidates(self, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        candidates.sort(key=lambda c: (-c["frequency"], -c["length"]))
        for i, candidate in enumerate(candidates, _RANK_OFFSET):
            candidate["rank"] = i
        return candidates

    def _signature_from_edges(self, path: list[Dict[str, Any]]) -> tuple:
        signature: list[tuple] = []
        for edge in path:
            src = edge.get("src_entity_id")
            if src is None:
                src = edge.get("head_entity_id")
            dst = edge.get("dst_entity_id")
            if dst is None:
                dst = edge.get("tail_entity_id")
            signature.append((src, edge.get("relation_id"), dst))
        return tuple(signature)

    @staticmethod
    def _fmt_edge(e: Dict[str, Any]) -> str:
        def _txt(val_text: Any, val_id: Any) -> str:
            if val_text is not None:
                return str(val_text)
            if val_id is None:
                return _UNKNOWN_TEXT
            return str(val_id)

        h = _txt(e.get("src_text"), e.get("src_entity_id"))
        t = _txt(e.get("dst_text"), e.get("dst_entity_id"))
        if h == _UNKNOWN_TEXT and t == _UNKNOWN_TEXT:
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


__all__ = ["GFlowNetRolloutArtifactWriter"]
