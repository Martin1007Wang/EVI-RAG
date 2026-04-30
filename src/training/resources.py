from __future__ import annotations

from typing import Any

import torch
from lightning import LightningDataModule


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
