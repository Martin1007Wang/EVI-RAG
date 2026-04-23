from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .samples import EncodedPayload, PreparedSample
from .vocab import EntityVocab, RelationVocab

log = logging.getLogger(__name__)

_DEFAULT_BATCH_GPU = 256
_DEFAULT_BATCH_CPU = 64
_BGE_QUERY_PREFIX = "Represent this sentence: "


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
        self.model_name = model_name
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
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)

        prefixed = [f"{query_prefix}{text}" for text in texts] if query_prefix else list(texts)

        outputs: list[torch.Tensor] = []
        iterator = range(0, len(prefixed), batch_size)

        for start in tqdm(iterator, desc=desc, disable=not self.progress_bar):
            batch = prefixed[start : start + batch_size]
            outputs.append(self._forward_batch(batch))

        return torch.cat(outputs, dim=0)

    def encode_one(
        self,
        text: str,
        *,
        query_prefix: str = "",
    ) -> torch.Tensor:
        return self.encode(
            [text],
            batch_size=1,
            desc="EncodeOne",
            query_prefix=query_prefix,
        )[0]


def encode_preprocessed_features(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    embeddings_dir: Path,
    encoder_name: str,
    device: str = "auto",
    batch_size: int | None = None,
    progress_bar: bool = True,
) -> EncodedPayload:
    del embeddings_dir

    resolved_batch_size = batch_size or (
        _DEFAULT_BATCH_GPU if torch.cuda.is_available() else _DEFAULT_BATCH_CPU
    )

    entity_catalog = entity_vocab.build_catalog()
    relation_labels = relation_vocab.labels()
    question_texts = [sample.sample.question for sample in prepared_samples]

    encoder = TextEncoder(
        model_name=encoder_name,
        device=device,
        progress_bar=progress_bar,
    )

    flat_entity_embs = encoder.encode(
        entity_catalog.text_labels,
        resolved_batch_size,
        desc="Entities",
    )

    entity_embeddings = torch.full(
        (entity_catalog.max_embedding_id + 1, encoder.hidden_size),
        float("nan"),
        dtype=torch.float32,
    )
    # embedding_id=0 is the reserved non-text sentinel. Keep it finite so old-style
    # placeholder lookups never inject NaNs into the batch.
    entity_embeddings[0].zero_()
    entity_embeddings[entity_catalog.text_embedding_ids] = flat_entity_embs

    relation_embeddings = encoder.encode(
        relation_labels,
        resolved_batch_size,
        desc="Relations",
    )

    question_embeddings = encoder.encode(
        question_texts,
        resolved_batch_size,
        desc="Questions",
        query_prefix=_BGE_QUERY_PREFIX,
    )

    if question_embeddings.ndim != 2 or question_embeddings.shape[1] != encoder.hidden_size:
        raise ValueError(
            f"question_embeddings must have shape [N, {encoder.hidden_size}], "
            f"got {tuple(question_embeddings.shape)}"
        )

    return EncodedPayload(
        entity_catalog=entity_catalog,
        relation_labels=relation_labels,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        question_embeddings=question_embeddings,
    )
