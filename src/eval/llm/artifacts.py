from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.io.serialization import write_metrics_json, write_metrics_jsonl

from .metrics import compute_llm_metrics


def write_llm_metrics(
    *,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    split: str,
    provider: str,
    top_k: int,
    answer_key: str,
    answer_separator: str,
    metrics_filename_template: Optional[str] = None,
    input_labels_path: Optional[Path] = None,
) -> tuple[Path, dict[str, Any]]:
    template = str(metrics_filename_template or "{split}_k{k}_{provider}.metrics.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / template.format(split=split, k=int(top_k), provider=provider)
    metrics = compute_llm_metrics(
        input_path=input_path,
        input_labels_path=input_labels_path,
        output_path=output_path,
        split=split,
        provider=provider,
        top_k=top_k,
        answer_key=answer_key,
        answer_separator=answer_separator,
    )
    return write_metrics_json(path=metrics_path, metrics=metrics), metrics


def write_llm_metrics_artifacts(
    *,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    split: str,
    provider: str,
    top_k: int,
    answer_key: str,
    answer_separator: str,
    metrics_filename_template: Optional[str] = None,
    input_labels_path: Optional[Path] = None,
    metrics_log_dir: Optional[Path] = None,
    metrics_jsonl_name: str = "llm.jsonl",
    dataset_name: str = "",
    dataset_scope: str = "",
) -> tuple[Path, dict[str, Any]]:
    metrics_path, metrics = write_llm_metrics(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
        split=split,
        provider=provider,
        top_k=top_k,
        answer_key=answer_key,
        answer_separator=answer_separator,
        metrics_filename_template=metrics_filename_template,
        input_labels_path=input_labels_path,
    )
    _append_llm_metrics_jsonl(
        metrics=metrics,
        metrics_log_dir=metrics_log_dir,
        metrics_jsonl_name=metrics_jsonl_name,
        dataset_name=dataset_name,
        dataset_scope=dataset_scope,
        split=split,
        provider=provider,
        top_k=top_k,
        input_path=input_path,
        input_labels_path=input_labels_path,
        output_path=output_path,
    )
    return metrics_path, metrics


def _append_llm_metrics_jsonl(
    *,
    metrics: dict[str, Any],
    metrics_log_dir: Optional[Path],
    metrics_jsonl_name: str,
    dataset_name: str,
    dataset_scope: str,
    split: str,
    provider: str,
    top_k: int,
    input_path: Path,
    input_labels_path: Optional[Path],
    output_path: Path,
) -> None:
    if metrics_log_dir is None:
        return
    metrics_log_dir.mkdir(parents=True, exist_ok=True)
    metric_payload = {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and key.startswith("llm/")
    }
    if not metric_payload:
        return
    write_metrics_jsonl(
        path=metrics_log_dir / metrics_jsonl_name,
        stage="llm",
        metrics=metric_payload,
        step=0,
        epoch=None,
        metadata={
            "dataset_name": str(dataset_name or ""),
            "dataset_scope": str(dataset_scope or ""),
            "split": split,
            "provider": provider,
            "top_k": int(top_k),
            "input_path": str(input_path),
            "input_labels_path": str(input_labels_path) if input_labels_path is not None else "",
            "output_path": str(output_path),
        },
    )


__all__ = ["write_llm_metrics", "write_llm_metrics_artifacts"]
