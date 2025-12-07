from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.utils.llm_client import init_llm, run_chat
from src.utils.llm_metrics import evaluate_predictions


class LLMReasonerModule(LightningModule):
    """Prediction-only module that runs LLMs over prepared prompts."""

    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        frequency_penalty: float,
        max_seq_len: int,
        max_tokens: int,
        seed: int,
        output_dir: str,
        dataset: str,
        split: str,
        prompt_tag: str = "triplets",
        backend: str = "auto",
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.dataset = dataset
        self.split = split
        self.prompt_tag = prompt_tag

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

    def predict_step(self, batch: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for sample in batch:
            messages = [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["user_prompt"]},
            ]
            prediction = run_chat(self.llm, messages, is_openai=self._is_openai)
            outputs.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "answers": sample.get("answers", []),
                    "prediction": prediction,
                    "triplets": sample.get("triplets", []),
                    "paths": sample.get("paths", []),
                }
            )
        return outputs

    def on_predict_epoch_end(self, results: Optional[List[List[Dict[str, Any]]]] = None) -> None:
        """
        Lightning does not pass predict outputs into this hook; we must pull from trainer.predict_loop.predictions.
        """
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
                    flat.append(batch)  # pragma: no cover - defensive branch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self._prediction_path()
        with pred_path.open("w") as f:
            for row in flat:
                f.write(json.dumps(row) + "\n")

        metrics = evaluate_predictions(flat)
        metrics_path = pred_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2))

        rank_zero_info(f"Predictions saved to {pred_path}")
        rank_zero_info(f"Metrics saved to {metrics_path}")

        if self.trainer is not None and self.trainer.logger is not None:
            self.trainer.logger.log_metrics(metrics, step=0)

    def _prediction_path(self) -> Path:
        safe_model = self.model_name.replace("/", "_")
        fname = f"{self.dataset}-{self.prompt_tag}-{safe_model}-{self.split}.jsonl"
        return self.output_dir / fname

    def configure_optimizers(self) -> Optional[Any]:
        return None


__all__ = ["LLMReasonerModule"]
