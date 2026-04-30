from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize: bool = True,
        progress_bar: bool = True,
    ) -> None:
        if device in {"", "auto"}:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)
        self.normalize = normalize
        self.progress_bar = progress_bar
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def _forward_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0, :]
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.to(dtype=torch.float32, device="cpu")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        desc: str = "Encode",
        query_prefix: str = "",
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        prefixed = [f"{query_prefix}{t}" for t in texts] if query_prefix else list(texts)
        outputs = []
        iterator = range(0, len(prefixed), batch_size)
        for start in tqdm(iterator, desc=desc, disable=not self.progress_bar):
            batch = prefixed[start : start + batch_size]
            outputs.append(self._forward_batch(batch))
        return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class EncodedFeatures:
    entity_text_embeddings: torch.Tensor  # [num_text_entities, dim]
    relation_embeddings: torch.Tensor  # [num_relations, dim]
    question_embeddings: torch.Tensor  # [num_samples, dim]


def encode_text_features(
    *,
    entity_text_labels: list[str],
    relation_text_labels: list[str],
    question_texts: list[str],
    encoder_name: str,
    device: str = "auto",
    batch_size: int | None = None,
    progress_bar: bool = True,
) -> EncodedFeatures:
    resolved_batch_size = batch_size or (256 if torch.cuda.is_available() else 16)
    encoder = TextEncoder(
        model_name=encoder_name,
        device=device,
        progress_bar=progress_bar,
    )
    entity_embs = encoder.encode(
        entity_text_labels,
        resolved_batch_size,
        desc="Entities",
    )
    relation_embs = encoder.encode(
        relation_text_labels,
        resolved_batch_size,
        desc="Relations",
    )
    question_embs = encoder.encode(
        question_texts,
        resolved_batch_size,
        desc="Questions",
        query_prefix="Represent this sentence: ",
    )
    return EncodedFeatures(
        entity_text_embeddings=entity_embs,
        relation_embeddings=relation_embs,
        question_embeddings=question_embs,
    )
