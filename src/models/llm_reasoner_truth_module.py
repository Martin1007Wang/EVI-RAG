from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.utils.metrics import normalize_k_values


def _oracle_metrics_for_sample(
    *,
    head_entity_ids: torch.Tensor,
    tail_entity_ids: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    k_values: Sequence[int],
) -> Dict[str, float]:
    answers = torch.unique(answer_entity_ids.to(dtype=torch.long).view(-1))
    if answers.numel() == 0:
        out: Dict[str, float] = {f"answer_hit@{k}": 0.0 for k in k_values}
        out.update({f"answer_recall@{k}": 0.0 for k in k_values})
        return out

    heads = head_entity_ids.to(dtype=torch.long).view(-1)
    tails = tail_entity_ids.to(dtype=torch.long).view(-1)
    num_edges = int(heads.numel())
    if num_edges == 0:
        out: Dict[str, float] = {f"answer_hit@{k}": 0.0 for k in k_values}
        out.update({f"answer_recall@{k}": 0.0 for k in k_values})
        return out

    head_hit = torch.isin(heads, answers)
    tail_hit = torch.isin(tails, answers)

    out: Dict[str, float] = {}
    found_any = False
    found_set: set[int] = set()
    max_k = max(int(k) for k in k_values) if k_values else 0
    # Monotone scan over ranks up to max_k to avoid repeated slicing work.
    max_scan = min(num_edges, max_k)
    k_pointer = 0
    ks = list(k_values)
    for rank_idx in range(1, max_scan + 1):
        e_idx = rank_idx - 1
        if bool(head_hit[e_idx].item()):
            found_set.add(int(heads[e_idx].item()))
            found_any = True
        if bool(tail_hit[e_idx].item()):
            found_set.add(int(tails[e_idx].item()))
            found_any = True

        while k_pointer < len(ks) and rank_idx == int(ks[k_pointer]):
            out[f"answer_hit@{int(ks[k_pointer])}"] = 1.0 if found_any else 0.0
            out[f"answer_recall@{int(ks[k_pointer])}"] = float(len(found_set) / int(answers.numel()))
            k_pointer += 1

    # For k beyond available edges, repeat last value (same convention as retriever metrics).
    last_hit = 1.0 if found_any else 0.0
    last_recall = float(len(found_set) / int(answers.numel()))
    while k_pointer < len(ks):
        out[f"answer_hit@{int(ks[k_pointer])}"] = last_hit
        out[f"answer_recall@{int(ks[k_pointer])}"] = last_recall
        k_pointer += 1
    return out


class LLMReasonerTruthModule(LightningModule):
    """Prediction-only module that computes retriever-oracle upper bounds."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        output_dir: str,
        k_values: Sequence[int],
        prompt_tag: str = "truth",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.prompt_tag = prompt_tag
        self.k_values = normalize_k_values(k_values)
        if not self.k_values:
            raise ValueError("k_values must be a non-empty list of positive integers.")

    def predict_step(self, batch: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for sample in batch:
            head_ids = torch.as_tensor(sample.get("head_entity_ids", []), dtype=torch.long)
            tail_ids = torch.as_tensor(sample.get("tail_entity_ids", []), dtype=torch.long)
            answer_ids = torch.as_tensor(sample.get("answer_entity_ids", []), dtype=torch.long)
            oracle = _oracle_metrics_for_sample(
                head_entity_ids=head_ids,
                tail_entity_ids=tail_ids,
                answer_entity_ids=answer_ids,
                k_values=self.k_values,
            )
            outputs.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question", ""),
                    "answer_entity_ids": sample.get("answer_entity_ids", []),
                    "oracle": oracle,
                }
            )
        return outputs

    @staticmethod
    def _aggregate_metrics(rows: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
        total = 0
        sums: Dict[str, float] = {f"answer_hit@{k}": 0.0 for k in k_values}
        sums.update({f"answer_recall@{k}": 0.0 for k in k_values})
        for row in rows:
            oracle = row.get("oracle")
            if not isinstance(oracle, dict):
                continue
            total += 1
            for k in k_values:
                hit_key = f"answer_hit@{k}"
                rec_key = f"answer_recall@{k}"
                sums[hit_key] += float(oracle.get(hit_key, 0.0))
                sums[rec_key] += float(oracle.get(rec_key, 0.0))
        denom = float(max(total, 1))
        metrics: Dict[str, float] = {
            "oracle/total": float(total),
        }
        for k in k_values:
            metrics[f"oracle/answer_hit@{k}"] = sums[f"answer_hit@{k}"] / denom
            metrics[f"oracle/answer_recall@{k}"] = sums[f"answer_recall@{k}"] / denom
        return metrics

    def on_predict_epoch_end(self, results: Optional[List[List[Dict[str, Any]]]] = None) -> None:
        batches: Optional[List[Any]] = results
        if batches is None and self.trainer is not None:
            predict_loop = getattr(self.trainer, "predict_loop", None)
            if predict_loop is not None:
                batches = getattr(predict_loop, "predictions", None)

        flat: List[Dict[str, Any]] = []
        if batches:
            for batch in batches:
                if isinstance(batch, list):
                    flat.extend(batch)
                elif batch is not None:
                    flat.append(batch)  # pragma: no cover

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        merged = flat
        if world_size > 1:
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, flat)
            if rank != 0:
                return
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self._prediction_path()
        with pred_path.open("w") as f:
            for row in merged:
                f.write(json.dumps(row) + "\n")

        metrics = self._aggregate_metrics(merged, k_values=self.k_values)
        metrics_path = pred_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))

        rank_zero_info(f"Predictions saved to {pred_path}")
        rank_zero_info(f"Metrics saved to {metrics_path}")

        if self.trainer is not None and self.trainer.logger is not None:
            self.trainer.logger.log_metrics(metrics, step=0)

    def _prediction_path(self) -> Path:
        fname = f"{self.dataset}-{self.prompt_tag}-{self.split}.jsonl"
        return self.output_dir / fname

    def configure_optimizers(self) -> Optional[Any]:
        return None


__all__ = ["LLMReasonerTruthModule"]
