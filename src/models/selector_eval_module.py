from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.utils.metrics import normalize_k_values
from src.utils.selector_metrics import edge_precision_recall_f1, path_hit_from_selected, ensure_int_list


class SelectorEvalModule(LightningModule):
    """Prediction-only module that evaluates evidence selection against DAG mask."""

    def __init__(
        self,
        *,
        dataset: str,
        split: str,
        output_dir: str,
        selector: str,
        k_values: Sequence[int],
        allow_empty_dag: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.selector = str(selector).strip().lower()
        self.k_values = normalize_k_values(k_values)
        if not self.k_values:
            raise ValueError("k_values must be a non-empty list of positive integers.")
        self.allow_empty_dag = bool(allow_empty_dag)
        self._records: List[Dict[str, Any]] = []
        if self.selector not in {"retriever_topk", "gflownet_rollouts"}:
            raise ValueError(f"Unsupported selector={self.selector!r}. Use 'retriever_topk' or 'gflownet_rollouts'.")

    def predict_step(self, batch: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for sample in batch:
            outputs.append(self._evaluate_sample(sample))
        return outputs

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
                    flat.append(batch)

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
        self.predict_metrics = metrics

        rank_zero_info(f"Predictions saved to {pred_path}")
        rank_zero_info(f"Metrics saved to {metrics_path}")

        if self.trainer is not None and self.trainer.logger is not None:
            self.trainer.logger.log_metrics(metrics, step=0)

    def _evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(sample.get("sample_id", "unknown"))
        edge_head_locals = torch.as_tensor(sample.get("edge_head_locals"), dtype=torch.long).view(-1)
        edge_tail_locals = torch.as_tensor(sample.get("edge_tail_locals"), dtype=torch.long).view(-1)
        edge_scores = torch.as_tensor(sample.get("edge_scores"), dtype=torch.float32).view(-1)
        edge_labels = torch.as_tensor(sample.get("edge_labels"), dtype=torch.float32).view(-1)
        start_nodes = torch.as_tensor(sample.get("start_node_locals"), dtype=torch.long).view(-1)
        answer_nodes = torch.as_tensor(sample.get("answer_node_locals"), dtype=torch.long).view(-1)
        num_nodes = int(sample.get("num_nodes", 0))

        if edge_head_locals.numel() != edge_tail_locals.numel():
            raise ValueError(f"edge_head_locals/edge_tail_locals length mismatch for {sample_id}")
        if edge_labels.numel() != edge_head_locals.numel():
            raise ValueError(f"edge_labels length mismatch for {sample_id}")
        if edge_scores.numel() != edge_head_locals.numel():
            raise ValueError(f"edge_scores length mismatch for {sample_id}")
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive for {sample_id}")

        num_edges = int(edge_labels.numel())
        if num_edges <= 0:
            raise ValueError(f"edge list empty for {sample_id}")

        pos_mask = edge_labels > 0.5
        if not bool(pos_mask.any().item()) and not self.allow_empty_dag:
            start_hit = bool(torch.isin(start_nodes, answer_nodes).any().item()) if start_nodes.numel() > 0 else False
            raise ValueError(
                f"edge_labels has no positives for {sample_id} (start_hits_answer={start_hit}); "
                "set allow_empty_dag=true to accept empty DAG."
            )

        rollout_edge_ids: Optional[List[List[int]]] = sample.get("rollout_edge_ids")
        if self.selector == "gflownet_rollouts":
            if rollout_edge_ids is None:
                raise ValueError(f"rollout_edge_ids missing for {sample_id}")
            if not isinstance(rollout_edge_ids, list):
                raise ValueError(f"rollout_edge_ids must be list for {sample_id}")
            max_k = max(self.k_values)
            if len(rollout_edge_ids) < max_k:
                raise ValueError(
                    f"rollout_edge_ids length {len(rollout_edge_ids)} < max_k {max_k} for {sample_id}"
                )

        metrics_by_k: Dict[str, Dict[str, float]] = {}
        if self.selector == "retriever_topk":
            order = torch.argsort(edge_scores, descending=True)

        for k in self.k_values:
            if self.selector == "retriever_topk":
                k_use = min(int(k), num_edges)
                selected_ids = order[:k_use].tolist() if k_use > 0 else []
            else:
                prefix = rollout_edge_ids[: int(k)]
                selected_ids = []
                for ridx, rollout in enumerate(prefix):
                    selected_ids.extend(
                        ensure_int_list(rollout, sample_id=sample_id, field=f"rollout_edge_ids[{ridx}]")
                    )

            selected_ids = [int(eid) for eid in selected_ids]
            for eid in selected_ids:
                if eid < 0 or eid >= num_edges:
                    raise ValueError(f"edge_id {eid} out of range for {sample_id}")
            selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=pos_mask.device)
            if selected_ids:
                selected_mask[selected_ids] = True

            precision, recall, f1 = edge_precision_recall_f1(
                selected_mask=selected_mask,
                positive_mask=pos_mask,
                allow_empty_positive=self.allow_empty_dag,
                sample_id=sample_id,
            )
            hit = path_hit_from_selected(
                edge_head_locals=edge_head_locals,
                edge_tail_locals=edge_tail_locals,
                selected_mask=selected_mask,
                positive_mask=pos_mask,
                start_node_locals=start_nodes,
                answer_node_locals=answer_nodes,
                num_nodes=num_nodes,
                sample_id=sample_id,
            )
            metrics_by_k[str(k)] = {
                "edge_precision": precision,
                "edge_recall": recall,
                "edge_f1": f1,
                "path_hit": 1.0 if hit else 0.0,
            }

        return {
            "sample_id": sample_id,
            "metrics": metrics_by_k,
        }

    @staticmethod
    def _aggregate_metrics(rows: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
        sums: Dict[int, Dict[str, float]] = {}
        counts: Dict[int, int] = {}
        for k in k_values:
            sums[int(k)] = {"edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0, "path_hit": 0.0}
            counts[int(k)] = 0

        for row in rows:
            metrics = row.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            for k in k_values:
                m = metrics.get(str(k))
                if not isinstance(m, dict):
                    continue
                sums[int(k)]["edge_precision"] += float(m.get("edge_precision", 0.0))
                sums[int(k)]["edge_recall"] += float(m.get("edge_recall", 0.0))
                sums[int(k)]["edge_f1"] += float(m.get("edge_f1", 0.0))
                sums[int(k)]["path_hit"] += float(m.get("path_hit", 0.0))
                counts[int(k)] += 1

        metrics_out: Dict[str, float] = {
            "selector/total": float(len(rows)),
        }
        for k in k_values:
            denom = float(max(counts[int(k)], 1))
            metrics_out[f"selector/edge_precision@{k}"] = sums[int(k)]["edge_precision"] / denom
            metrics_out[f"selector/edge_recall@{k}"] = sums[int(k)]["edge_recall"] / denom
            metrics_out[f"selector/edge_f1@{k}"] = sums[int(k)]["edge_f1"] / denom
            metrics_out[f"selector/path_hit@{k}"] = sums[int(k)]["path_hit"] / denom
        return metrics_out

    def _prediction_path(self) -> Path:
        fname = f"{self.dataset}-{self.selector}-{self.split}.jsonl"
        return self.output_dir / fname

    def configure_optimizers(self) -> Optional[Any]:
        return None


__all__ = ["SelectorEvalModule"]
