from __future__ import annotations

from pathlib import Path
import torch


class EmbeddingStore:
    def __init__(self, embeddings_dir: Path) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        entity_path = self.embeddings_dir / "entity_embeddings.pt"
        relation_path = self.embeddings_dir / "relation_embeddings.pt"

        if not entity_path.exists():
            raise FileNotFoundError(f"missing entity embeddings: {entity_path}")
        if not relation_path.exists():
            raise FileNotFoundError(f"missing relation embeddings: {relation_path}")

        self.entity_embeddings = torch.load(entity_path, map_location="cpu", mmap=True)
        self.relation_embeddings = torch.load(relation_path, map_location="cpu", mmap=True)

        if not torch.is_tensor(self.entity_embeddings) or self.entity_embeddings.ndim != 2:
            raise ValueError("entity_embeddings must be a 2D tensor.")
        if not torch.is_tensor(self.relation_embeddings) or self.relation_embeddings.ndim != 2:
            raise ValueError("relation_embeddings must be a 2D tensor.")

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        entity_ids = entity_ids.long()
        embeddings = self.entity_embeddings.index_select(0, entity_ids)

        non_text_mask = entity_ids.eq(0)
        if bool(non_text_mask.any()):
            embeddings = embeddings.clone()
            embeddings[non_text_mask] = 0.0

        return embeddings

    def get_relation_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.relation_embeddings.index_select(0, relation_ids.long())

    def close(self) -> None:
        pass
