from pathlib import Path
import torch


class EmbeddingStore:
    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = Path(embeddings_dir)
        self.entity_embeddings = torch.load(self.embeddings_dir / "entity_embeddings.pt", map_location="cpu", mmap=True)
        self.relation_embeddings = torch.load(self.embeddings_dir / "relation_embeddings.pt", map_location="cpu", mmap=True)

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings.index_select(0, entity_ids)

    def get_relation_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.relation_embeddings.index_select(0, relation_ids)
