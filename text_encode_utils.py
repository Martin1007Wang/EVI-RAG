from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def _iter_batches(
    items: Sequence[str], batch_size: int
) -> Iterable[tuple[int, Sequence[str]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


@dataclass
class TextEncoder:
    model_name: str
    device: str = "cpu"
    fp16: bool = False
    progress_bar: bool = True

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        model_kwargs: dict[str, object] = {"trust_remote_code": True}
        if self._device.type == "cuda" and self.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self._device)
        self.model.eval()
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError(
                f"Unable to infer hidden size for encoder={self.model_name!r}."
            )
        self.hidden_size = int(hidden_size)

    def _forward_batch(
        self,
        texts: Sequence[str],
        *,
        max_tokens: int | None,
        pad_to_max: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not texts:
            empty_embeddings = torch.empty((0, self.hidden_size), dtype=torch.float32)
            empty_context = torch.empty((0, 0, self.hidden_size), dtype=torch.float32)
            empty_mask = torch.empty((0, 0), dtype=torch.bool)
            return empty_embeddings, empty_context, empty_mask

        tokenizer_kwargs: dict[str, object] = {
            "padding": "max_length" if pad_to_max else True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if max_tokens is not None:
            tokenizer_kwargs["max_length"] = int(max_tokens)
        batch = self.tokenizer(list(texts), **tokenizer_kwargs)
        model_inputs = {
            key: value.to(self._device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = model_inputs["attention_mask"]
            pooled = _mean_pool(last_hidden_state, attention_mask)
        return (
            pooled.detach().to(dtype=torch.float32, device="cpu"),
            last_hidden_state.detach().to(dtype=torch.float32, device="cpu"),
            attention_mask.detach().to(dtype=torch.bool, device="cpu"),
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        *,
        show_progress: bool = False,
        desc: str | None = None,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        iterator = list(_iter_batches(texts, batch_size))
        progress = tqdm(
            iterator,
            desc=desc or "Encode",
            disable=not (show_progress and self.progress_bar),
        )
        for _, chunk in progress:
            pooled, _, _ = self._forward_batch(
                chunk,
                max_tokens=None,
                pad_to_max=False,
            )
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def encode_with_context(
        self,
        texts: Sequence[str],
        batch_size: int,
        *,
        max_tokens: int,
        show_progress: bool = False,
        desc: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}.")
        if not texts:
            empty_embeddings = torch.empty((0, self.hidden_size), dtype=torch.float32)
            empty_context = torch.empty(
                (0, max_tokens, self.hidden_size), dtype=torch.float32
            )
            empty_mask = torch.empty((0, max_tokens), dtype=torch.bool)
            return empty_embeddings, empty_context, empty_mask
        pooled_outputs: list[torch.Tensor] = []
        context_outputs: list[torch.Tensor] = []
        mask_outputs: list[torch.Tensor] = []
        iterator = list(_iter_batches(texts, batch_size))
        progress = tqdm(
            iterator,
            desc=desc or "EncodeWithContext",
            disable=not (show_progress and self.progress_bar),
        )
        for _, chunk in progress:
            pooled, context, mask = self._forward_batch(
                chunk,
                max_tokens=max_tokens,
                pad_to_max=True,
            )
            pooled_outputs.append(pooled)
            context_outputs.append(context)
            mask_outputs.append(mask)
        return (
            torch.cat(pooled_outputs, dim=0),
            torch.cat(context_outputs, dim=0),
            torch.cat(mask_outputs, dim=0),
        )


def encode_to_memmap(
    *,
    encoder: TextEncoder,
    texts: Sequence[str],
    emb_ids: Sequence[int],
    batch_size: int,
    max_embedding_id: int,
    out_path: str | Path,
    desc: str | None = None,
    show_progress: bool = False,
) -> Path:
    if len(texts) != len(emb_ids):
        raise ValueError(
            "texts and emb_ids must have the same length: "
            f"texts={len(texts)} emb_ids={len(emb_ids)}."
        )
    if max_embedding_id < 0:
        raise ValueError(f"max_embedding_id must be >= 0, got {max_embedding_id}.")
    out_path = Path(out_path)
    embeddings = torch.zeros(
        (int(max_embedding_id) + 1, int(encoder.hidden_size)),
        dtype=torch.float32,
    )
    if texts:
        iterator = list(_iter_batches(list(texts), batch_size))
        progress = tqdm(
            iterator,
            desc=desc or "EncodeToTensor",
            disable=not (show_progress and encoder.progress_bar),
        )
        for start, chunk in progress:
            chunk_emb = encoder.encode(
                chunk,
                batch_size=len(chunk),
                show_progress=False,
            )
            for row_offset, emb_id in enumerate(emb_ids[start : start + len(chunk)]):
                embeddings[int(emb_id)] = chunk_emb[row_offset]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.contiguous(), out_path)
    return out_path


__all__ = ["TextEncoder", "encode_to_memmap"]
