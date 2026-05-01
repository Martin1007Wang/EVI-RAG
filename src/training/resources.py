from __future__ import annotations

from typing import Any

import torch
from lightning import LightningDataModule


_EMBEDDING_NORM_ATOL = 1.0e-3


def setup_datamodule(
    datamodule: LightningDataModule,
    stage: str | None = "fit",
) -> Any:
    datamodule.prepare_data()
    datamodule.setup(stage)

    resources = getattr(datamodule, "model_resources", None)
    if resources is None:
        raise AttributeError("Datamodule must expose `model_resources` after setup('fit').")

    validate_model_resources(resources)
    return resources


def validate_model_resources(resources: Any) -> None:
    required = {
        "entity_text_embeddings": 2,
        "entity_embedding_map": 1,
        "relation_embeddings": 2,
    }

    for name, ndim in required.items():
        if not hasattr(resources, name):
            raise AttributeError(f"model_resources is missing field: {name!r}.")

        value = getattr(resources, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"model_resources.{name} must be a torch.Tensor, got {type(value)!r}."
            )

        if value.ndim != ndim:
            raise ValueError(
                f"model_resources.{name} must be {ndim}D, got shape={tuple(value.shape)}."
            )

    entity_dim = int(resources.entity_text_embeddings.size(1))
    relation_dim = int(resources.relation_embeddings.size(1))
    if entity_dim != relation_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"entity_text_embeddings={entity_dim}, relation_embeddings={relation_dim}."
        )

    _validate_l2_normalized_rows(
        resources.entity_text_embeddings,
        name="model_resources.entity_text_embeddings",
    )
    _validate_l2_normalized_rows(
        resources.relation_embeddings,
        name="model_resources.relation_embeddings",
    )


def _validate_l2_normalized_rows(
    tensor: torch.Tensor,
    *,
    name: str,
    atol: float = _EMBEDDING_NORM_ATOL,
) -> None:
    if tensor.numel() == 0:
        return

    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")

    norms = torch.linalg.vector_norm(tensor.to(dtype=torch.float32), ord=2, dim=1)
    deviation = (norms - 1.0).abs()
    max_deviation = float(deviation.max().item())
    if max_deviation <= float(atol):
        return

    row = int(deviation.argmax().item())
    norm = float(norms[row].item())
    raise ValueError(
        f"{name} rows must be L2-normalized within atol={float(atol):g}; "
        f"row {row} has norm={norm:.6g}. Rebuild embeddings with "
        "`python src/preprocess.py dataset=<name>`."
    )
