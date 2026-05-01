from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

TEXT_ENCODE_CACHE_SCHEMA_VERSION = 1


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        progress_bar: bool = True,
    ) -> None:
        if device in {"", "auto"}:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)
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
    cache_dir: str | Path | None = None,
) -> EncodedFeatures:
    resolved_batch_size = batch_size or (256 if torch.cuda.is_available() else 16)
    encoder: TextEncoder | None = None

    def get_encoder() -> TextEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = TextEncoder(
                model_name=encoder_name,
                device=device,
                progress_bar=progress_bar,
            )
        return encoder

    resolved_cache_dir = Path(cache_dir) if cache_dir not in (None, "") else None
    entity_embs = _encode_text_table(
        texts=entity_text_labels,
        encoder_factory=get_encoder,
        batch_size=resolved_batch_size,
        desc="Entities",
        encoder_name=encoder_name,
        cache_dir=resolved_cache_dir,
        cache_kind="entities",
    )
    relation_embs = _encode_text_table(
        texts=relation_text_labels,
        encoder_factory=get_encoder,
        batch_size=resolved_batch_size,
        desc="Relations",
        encoder_name=encoder_name,
        cache_dir=resolved_cache_dir,
        cache_kind="relations",
    )
    question_embs = _encode_text_table(
        texts=question_texts,
        encoder_factory=get_encoder,
        batch_size=resolved_batch_size,
        desc="Questions",
        query_prefix="Represent this sentence: ",
        encoder_name=encoder_name,
        cache_dir=resolved_cache_dir,
        cache_kind="questions",
    )
    return EncodedFeatures(
        entity_text_embeddings=entity_embs,
        relation_embeddings=relation_embs,
        question_embeddings=question_embs,
    )


def _encode_text_table(
    texts: list[str],
    *,
    encoder_factory: Callable[[], TextEncoder],
    batch_size: int,
    desc: str,
    encoder_name: str,
    cache_dir: Path | None,
    cache_kind: str,
    query_prefix: str = "",
) -> torch.Tensor:
    cache_path = _text_cache_path(
        cache_dir=cache_dir,
        kind=cache_kind,
        encoder_name=encoder_name,
        texts=texts,
        query_prefix=query_prefix,
    )
    if cache_path is not None:
        cached = _load_cached_embeddings(cache_path, expected_rows=len(texts))
        if cached is not None:
            log.info("Text encode cache hit: kind=%s rows=%d", cache_kind, len(texts))
            return cached

    encoder = encoder_factory()
    embeddings = encoder.encode(
        texts,
        batch_size,
        desc=desc,
        query_prefix=query_prefix,
    )
    embeddings = embeddings.to(dtype=torch.float32, device="cpu").contiguous()
    if cache_path is not None:
        _write_cached_embeddings(cache_path, embeddings=embeddings)
    return embeddings


def _text_cache_path(
    *,
    cache_dir: Path | None,
    kind: str,
    encoder_name: str,
    texts: list[str],
    query_prefix: str,
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {
        "schema_version": TEXT_ENCODE_CACHE_SCHEMA_VERSION,
        "kind": kind,
        "encoder_name": encoder_name,
        "query_prefix": query_prefix,
        "texts": texts,
    }
    cache_key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{kind}-{cache_key}.pt"


def _load_cached_embeddings(
    path: Path,
    *,
    expected_rows: int,
) -> torch.Tensor | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        log.warning("Ignoring unreadable text encode cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        log.warning("Ignoring malformed text encode cache %s: expected dict.", path)
        return None
    if payload.get("schema_version") != TEXT_ENCODE_CACHE_SCHEMA_VERSION:
        return None
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        log.warning("Ignoring malformed text encode cache %s: missing tensor.", path)
        return None
    if embeddings.ndim != 2 or int(embeddings.size(0)) != int(expected_rows):
        log.warning(
            "Ignoring text encode cache %s: expected %d rows, got shape=%s.",
            path,
            expected_rows,
            tuple(embeddings.shape),
        )
        return None
    if not embeddings.dtype.is_floating_point:
        log.warning("Ignoring text encode cache %s: dtype=%s.", path, embeddings.dtype)
        return None
    if not bool(torch.isfinite(embeddings).all()):
        log.warning("Ignoring text encode cache %s: contains non-finite values.", path)
        return None
    return embeddings.to(dtype=torch.float32, device="cpu").contiguous()


def _write_cached_embeddings(path: Path, *, embeddings: torch.Tensor) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": TEXT_ENCODE_CACHE_SCHEMA_VERSION,
                "embeddings": embeddings.to(
                    dtype=torch.float32,
                    device="cpu",
                ).contiguous(),
            },
            tmp_path,
        )
        tmp_path.replace(path)
    except Exception as exc:
        log.warning("Failed to write text encode cache %s: %s", path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
