from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

_metadata_VERSION = 1


@dataclass(frozen=True)
class RuntimeSampleMetadata:
    split: str
    sample_ids: list[str]
    questions: list[str]
    num_nodes: torch.Tensor
    num_edges: torch.Tensor
    question_tokens: torch.Tensor

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)


def metadata_path(embeddings_dir: Path, split: str) -> Path:
    return Path(embeddings_dir) / f"{str(split)}.metadata.pt"


def save_metadata(
    path: Path,
    *,
    split: str,
    sample_ids: Sequence[str],
    questions: Sequence[str],
    num_nodes: Sequence[int],
    num_edges: Sequence[int],
    question_tokens: Sequence[int],
) -> None:
    payload = _build_metadata_payload(
        split=split,
        sample_ids=sample_ids,
        questions=questions,
        num_nodes=num_nodes,
        num_edges=num_edges,
        question_tokens=question_tokens,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_metadata(path: Path) -> RuntimeSampleMetadata:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime sample metadata at {path} must decode to a dict, got {type(payload)!r}.")
    version = int(payload.get("version", -1))
    if version != _metadata_VERSION:
        raise ValueError(
            "Unsupported runtime sample metadata version: " f"expected {_metadata_VERSION}, got {version}."
        )
    split = str(payload.get("split", ""))
    sample_ids = [str(value) for value in list(payload.get("sample_ids", []))]
    questions = [str(value) for value in list(payload.get("questions", []))]
    num_nodes = _coerce_int_tensor(payload.get("num_nodes"), name="num_nodes")
    num_edges = _coerce_int_tensor(payload.get("num_edges"), name="num_edges")
    question_tokens = _coerce_int_tensor(payload.get("question_tokens"), name="question_tokens")
    _validate_lengths(
        sample_ids=sample_ids,
        questions=questions,
        num_nodes=num_nodes,
        num_edges=num_edges,
        question_tokens=question_tokens,
    )
    _validate_unique_sample_ids(sample_ids)
    return RuntimeSampleMetadata(
        split=split,
        sample_ids=sample_ids,
        questions=questions,
        num_nodes=num_nodes,
        num_edges=num_edges,
        question_tokens=question_tokens,
    )


def _build_metadata_payload(
    *,
    split: str,
    sample_ids: Sequence[str],
    questions: Sequence[str],
    num_nodes: Sequence[int],
    num_edges: Sequence[int],
    question_tokens: Sequence[int],
) -> dict[str, object]:
    sample_ids = [str(value) for value in sample_ids]
    questions = [str(value) for value in questions]
    num_nodes_tensor = torch.as_tensor(list(num_nodes), dtype=torch.int32)
    num_edges_tensor = torch.as_tensor(list(num_edges), dtype=torch.int32)
    question_tokens_tensor = torch.as_tensor(list(question_tokens), dtype=torch.int32)
    _validate_lengths(
        sample_ids=sample_ids,
        questions=questions,
        num_nodes=num_nodes_tensor,
        num_edges=num_edges_tensor,
        question_tokens=question_tokens_tensor,
    )
    _validate_unique_sample_ids(sample_ids)
    return {
        "version": _metadata_VERSION,
        "split": str(split),
        "sample_ids": sample_ids,
        "questions": questions,
        "num_nodes": num_nodes_tensor,
        "num_edges": num_edges_tensor,
        "question_tokens": question_tokens_tensor,
    }


def _coerce_int_tensor(value: object, *, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Runtime sample metadata missing required field {name!r}.")
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.dim() != 1:
        raise ValueError(f"Runtime sample metadata field {name!r} must be 1D, got {tuple(tensor.shape)}.")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"Runtime sample metadata field {name!r} must be integral, got {tensor.dtype}.")
    return tensor


def _validate_lengths(
    *,
    sample_ids: list[str],
    questions: list[str],
    num_nodes: torch.Tensor,
    num_edges: torch.Tensor,
    question_tokens: torch.Tensor,
) -> None:
    expected = len(sample_ids)
    actual_lengths = {
        "questions": len(questions),
        "num_nodes": int(num_nodes.numel()),
        "num_edges": int(num_edges.numel()),
        "question_tokens": int(question_tokens.numel()),
    }
    mismatched = {name: actual for name, actual in actual_lengths.items() if actual != expected}
    if mismatched:
        raise ValueError("Runtime sample metadata length mismatch: " f"expected {expected} entries, got {mismatched}.")


def _validate_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen:
            duplicates.append(sample_id)
            if len(duplicates) >= 3:
                break
            continue
        seen.add(sample_id)
    if duplicates:
        raise ValueError("Runtime sample metadata contains duplicate sample_ids, examples: " f"{duplicates}.")


__all__ = [
    "RuntimeSampleMetadata",
    "load_metadata",
    "metadata_path",
    "save_metadata",
]
