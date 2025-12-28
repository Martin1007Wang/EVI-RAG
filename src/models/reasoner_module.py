from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.utils.llm_client import init_llm, run_chat
from src.utils.llm_metrics import evaluate_predictions
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

    last_hit = 1.0 if found_any else 0.0
    last_recall = float(len(found_set) / int(answers.numel()))
    while k_pointer < len(ks):
        out[f"answer_hit@{int(ks[k_pointer])}"] = last_hit
        out[f"answer_recall@{int(ks[k_pointer])}"] = last_recall
        k_pointer += 1
    return out


class ReasonerModule(LightningModule):
    """Reasoner module with two modes: LLM inference or oracle upper bound."""

    def __init__(
        self,
        *,
        mode: str = "llm",
        model_name: Optional[str] = None,
        tensor_parallel_size: int = 1,
        temperature: float = 0.0,
        frequency_penalty: float = 0.0,
        max_seq_len: int = 4096,
        max_tokens: int = 1024,
        seed: int = 0,
        output_dir: str,
        dataset: str,
        split: str,
        prompt_tag: str = "triplet",
        backend: str = "auto",
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: float = 120.0,
        k_values: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"llm", "oracle"}:
            raise ValueError(f"mode must be 'llm' or 'oracle', got {mode!r}")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.dataset = dataset
        self.split = split
        self.prompt_tag = prompt_tag
        self.predict_metrics: Dict[str, Any] = {}

        if self.mode == "llm":
            if not model_name:
                raise ValueError("model_name must be set when mode='llm'.")
            self.model_name = model_name
            self.token_budget = int(max_seq_len)
            self.llm, self._is_openai = init_llm(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_seq_len=max_seq_len,
                max_tokens=max_tokens,
                seed=seed,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                backend=backend,
                ollama_base_url=ollama_base_url,
                ollama_timeout=ollama_timeout,
            )
            self.k_values = []
        else:
            self.model_name = None
            self.token_budget = None
            self.llm = None
            self._is_openai = False
            self.k_values = normalize_k_values(k_values)
            if not self.k_values:
                raise ValueError("k_values must be a non-empty list of positive integers for mode='oracle'.")

    def predict_step(self, batch: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        if self.mode == "oracle":
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
                        "id": sample["id"],
                        "question": sample["question"],
                        "answer_entity_ids": sample["answer_entity_ids"],
                        "oracle": oracle,
                    }
                )
            return outputs

        outputs: List[Dict[str, Any]] = []
        for sample in batch:
            messages = [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["user_prompt"]},
            ]
            prediction = run_chat(self.llm, messages, is_openai=self._is_openai)
            prompt_token_count = sample["prompt_token_count"]
            evidence_token_count = sample["evidence_token_count"]
            token_budget = sample["token_budget"]
            evidence_truncated = bool(sample["evidence_truncated"])
            outputs.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "answers": sample["answers"],
                    "prediction": prediction,
                    "triplets": sample["triplets"],
                    "paths": sample["paths"],
                    "window_k": sample["window_k"],
                    "k_effective": sample["k_effective"],
                    "retrieved_edge_ids": sample["retrieved_edge_ids"],
                    "visible_edge_ids": sample["visible_edge_ids"],
                    "gt_path_edge_local_ids": sample["gt_path_edge_local_ids"],
                    "hit_set": sample["hit_set"],
                    "hit_vis": sample["hit_vis"],
                    "evidence_token_count": evidence_token_count,
                    "prompt_token_count": prompt_token_count,
                    "token_budget": token_budget,
                    "evidence_truncated": evidence_truncated,
                }
            )
        return outputs

    @staticmethod
    def _aggregate_oracle_metrics(rows: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
        total = 0
        sums: Dict[str, float] = {f"answer_hit@{k}": 0.0 for k in k_values}
        sums.update({f"answer_recall@{k}": 0.0 for k in k_values})
        for row in rows:
            if "oracle" not in row:
                raise ValueError("oracle metrics missing from prediction row.")
            oracle = row["oracle"]
            if not isinstance(oracle, dict):
                raise ValueError("oracle metrics must be a dict.")
            total += 1
            for k in k_values:
                hit_key = f"answer_hit@{k}"
                rec_key = f"answer_recall@{k}"
                if hit_key not in oracle or rec_key not in oracle:
                    raise ValueError("oracle metrics missing required keys.")
                sums[hit_key] += float(oracle[hit_key])
                sums[rec_key] += float(oracle[rec_key])
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

        if self.mode == "llm":
            dedup: Dict[str, Dict[str, Any]] = {}
            for row in merged:
                row_id = str(row.get("id", len(dedup)))
                if row_id not in dedup:
                    dedup[row_id] = row
            if len(dedup) != len(merged):
                rank_zero_info(f"Deduplicated predictions: kept {len(dedup)} unique ids from {len(merged)} rows.")
            merged = list(dedup.values())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self._prediction_path()
        with pred_path.open("w") as f:
            for row in merged:
                f.write(json.dumps(row) + "\n")

        if self.mode == "oracle":
            metrics = self._aggregate_oracle_metrics(merged, k_values=self.k_values)
        else:
            metrics = evaluate_predictions(merged)
        self.predict_metrics = metrics

        metrics_path = pred_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))

        rank_zero_info(f"Predictions saved to {pred_path}")
        rank_zero_info(f"Metrics saved to {metrics_path}")

        if self.trainer is not None and self.trainer.logger is not None:
            self.trainer.logger.log_metrics(metrics, step=0)

    def _prediction_path(self) -> Path:
        if self.mode == "oracle":
            fname = f"{self.dataset}-{self.prompt_tag}-{self.split}.jsonl"
        else:
            safe_model = str(self.model_name).replace("/", "_")
            fname = f"{self.dataset}-{self.prompt_tag}-{safe_model}-{self.split}.jsonl"
        return self.output_dir / fname

    def configure_optimizers(self) -> Optional[Any]:
        return None


__all__ = ["ReasonerModule"]
