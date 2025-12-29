from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

ENCODER_EPS = 1e-6


class TextEncoder:
    """Lightweight wrapper around a HuggingFace encoder."""

    def __init__(self, model_name: str, device: str, fp16: bool, progress: bool) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("transformers is required for text encoding. pip install transformers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.progress = progress

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        all_embeds: List[torch.Tensor] = []
        iterator = _iter_batches(len(texts), batch_size)
        use_progress = self.progress if show_progress is None else show_progress
        if use_progress:
            total = (len(texts) + batch_size - 1) // batch_size
            iterator = tqdm(iterator, total=total, desc=desc or "Encoding", leave=False)
        for start, end in iterator:
            chunk = list(texts[start:end])
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.to(self.dtype)
            mask = inputs["attention_mask"].to(self.dtype).unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=ENCODER_EPS)
            pooled = summed / denom
            all_embeds.append(pooled.to("cpu", dtype=torch.float32))
        return torch.cat(all_embeds, dim=0)


def encode_to_memmap(
    encoder: TextEncoder,
    texts: Sequence[str],
    emb_ids: Sequence[int],
    batch_size: int,
    max_embedding_id: int,
    out_path: Path,
    desc: str,
    show_progress: bool,
) -> torch.Tensor:
    if max_embedding_id < 0:
        return torch.empty((0, 0), dtype=torch.float32)
    if len(texts) != len(emb_ids):
        raise ValueError("texts and emb_ids must have the same length")
    if not texts:
        tensor = torch.zeros((max_embedding_id + 1, 0), dtype=torch.float32)
        torch.save(tensor, out_path)
        return tensor

    first_texts = list(texts[:batch_size])
    first_ids = list(emb_ids[:batch_size])
    first_emb = encoder.encode(first_texts, batch_size, show_progress=False)
    emb_dim = _embedding_dim(first_emb)
    mem, mmap_path = _init_memmap(out_path, max_embedding_id, emb_dim)
    _write_chunk(mem, first_emb, first_ids, max_embedding_id)

    iterator = _iter_batches(len(texts), batch_size, offset=batch_size)
    if show_progress:
        total = max(0, (len(texts) - batch_size + batch_size - 1) // batch_size)
        iterator = tqdm(iterator, total=total, desc=desc, leave=False)
    for start, end in iterator:
        chunk_texts = list(texts[start:end])
        chunk_ids = list(emb_ids[start:end])
        emb_chunk = encoder.encode(chunk_texts, batch_size, show_progress=False)
        _write_chunk(mem, emb_chunk, chunk_ids, max_embedding_id)

    mem.flush()
    tensor = torch.from_numpy(mem)
    torch.save(tensor, out_path)
    _cleanup_memmap(mmap_path)
    return tensor


def _iter_batches(total: int, batch_size: int, *, offset: int = 0) -> Iterable[Tuple[int, int]]:
    for start in range(offset, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def _embedding_dim(emb: torch.Tensor) -> int:
    return int(emb.shape[1]) if emb.numel() > 0 else 0


def _init_memmap(out_path: Path, max_embedding_id: int, emb_dim: int) -> Tuple[np.memmap, Path]:
    mmap_path = out_path.with_suffix(out_path.suffix + ".mmap")
    mem = np.memmap(mmap_path, mode="w+", dtype="float32", shape=(max_embedding_id + 1, emb_dim))
    mem[:] = 0.0
    return mem, mmap_path


def _write_chunk(mem: np.memmap, emb_chunk: torch.Tensor, id_chunk: Sequence[int], max_embedding_id: int) -> None:
    for emb_tensor, emb_id in zip(emb_chunk, id_chunk):
        if 0 <= int(emb_id) <= max_embedding_id:
            mem[int(emb_id)] = emb_tensor.cpu().numpy()


def _cleanup_memmap(mmap_path: Path) -> None:
    try:
        mmap_path.unlink(missing_ok=True)
    except Exception:
        pass
