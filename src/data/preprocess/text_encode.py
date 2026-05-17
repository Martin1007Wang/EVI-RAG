from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

_pooling = "cls"
_normalization = "l2"
_output_dtype = "float32"


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "auto",
        progress_bar: bool = True,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        max_length: int | None = None,
    ) -> None:
        model_name = str(model_name).strip()
        revision = str(revision).strip()
        resolved_tokenizer_name = str(tokenizer_name or model_name).strip()
        resolved_tokenizer_revision = str(tokenizer_revision or revision).strip()
        if not model_name:
            raise ValueError("TextEncoder requires a non-empty model_name.")
        if not revision:
            raise ValueError("TextEncoder requires a non-empty revision.")
        if not resolved_tokenizer_name:
            raise ValueError("TextEncoder requires a non-empty tokenizer_name.")
        if not resolved_tokenizer_revision:
            raise ValueError("TextEncoder requires a non-empty tokenizer_revision.")
        if max_length is not None and int(max_length) <= 0:
            raise ValueError(f"max_length must be positive or None, got {max_length}.")
        if device in {"", "auto"}:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.model_name = model_name
        self.revision = revision
        self.tokenizer_name = resolved_tokenizer_name
        self.tokenizer_revision = resolved_tokenizer_revision
        self.max_length = None if max_length is None else int(max_length)
        self.device = torch.device(resolved_device)
        self.progress_bar = progress_bar
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            revision=self.tokenizer_revision,
        )
        self.model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        self.model.to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def _forward_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
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
        batch_size = _resolve_batch_size(batch_size)
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        outputs = []
        iterator = range(0, len(texts), batch_size)
        for start in tqdm(iterator, desc=desc, disable=not self.progress_bar):
            raw_batch = texts[start : start + batch_size]
            batch = (
                [f"{query_prefix}{t}" for t in raw_batch]
                if query_prefix
                else list(raw_batch)
            )
            outputs.append(self._forward_batch(batch))
        return torch.cat(outputs, dim=0)


def encode_text_table(
    *,
    texts: list[str],
    encoder_name: str,
    encoder_revision: str,
    device: str = "auto",
    batch_size: int | None = None,
    progress_bar: bool = True,
    cache_dir: str | Path | None = None,
    cache_kind: str,
    desc: str,
    query_prefix: str = "",
    encoder: TextEncoder | None = None,
    tokenizer_name: str | None = None,
    tokenizer_revision: str | None = None,
    max_length: int | None = None,
) -> torch.Tensor:
    resolved_batch_size = _resolve_batch_size(batch_size)
    resolved_cache_dir = Path(cache_dir) if cache_dir not in (None, "") else None
    provenance = text_encoder_provenance(
        encoder_name=encoder_name,
        encoder_revision=encoder_revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        max_length=max_length,
    )
    owned_encoder: TextEncoder | None = encoder
    resolved_encoder_name = cast(str, provenance["encoder_name"])
    resolved_encoder_revision = cast(str, provenance["encoder_revision"])
    resolved_tokenizer_name = cast(str, provenance["tokenizer_name"])
    resolved_tokenizer_revision = cast(str, provenance["tokenizer_revision"])
    resolved_max_length = cast(int | None, provenance["max_length"])

    def get_encoder() -> TextEncoder:
        nonlocal owned_encoder
        if owned_encoder is None:
            owned_encoder = TextEncoder(
                model_name=resolved_encoder_name,
                revision=resolved_encoder_revision,
                device=device,
                progress_bar=progress_bar,
                tokenizer_name=resolved_tokenizer_name,
                tokenizer_revision=resolved_tokenizer_revision,
                max_length=resolved_max_length,
            )
        return owned_encoder

    return _encode_text_table(
        texts=texts,
        encoder_factory=get_encoder,
        batch_size=resolved_batch_size,
        desc=desc,
        provenance=provenance,
        cache_dir=resolved_cache_dir,
        cache_kind=cache_kind,
        query_prefix=query_prefix,
    )


def _encode_text_table(
    texts: list[str],
    *,
    encoder_factory: Callable[[], TextEncoder],
    batch_size: int,
    desc: str,
    provenance: Mapping[str, object],
    cache_dir: Path | None,
    cache_kind: str,
    query_prefix: str = "",
) -> torch.Tensor:
    cache_path = _text_cache_path(
        cache_dir=cache_dir,
        kind=cache_kind,
        provenance=provenance,
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
        _write_cached_embeddings(cache_path, embeddings=embeddings, provenance=provenance)
    return embeddings


def text_encoder_provenance(
    *,
    encoder_name: str,
    encoder_revision: str,
    tokenizer_name: str | None = None,
    tokenizer_revision: str | None = None,
    max_length: int | None = None,
) -> dict[str, object]:
    resolved_encoder_name = str(encoder_name).strip()
    resolved_encoder_revision = str(encoder_revision).strip()
    resolved_tokenizer_name = str(tokenizer_name or resolved_encoder_name).strip()
    resolved_tokenizer_revision = str(
        tokenizer_revision or resolved_encoder_revision
    ).strip()
    if not resolved_encoder_name:
        raise ValueError("encoder_name must be non-empty.")
    if not resolved_encoder_revision:
        raise ValueError("encoder_revision must be non-empty.")
    if not resolved_tokenizer_name:
        raise ValueError("tokenizer_name must be non-empty.")
    if not resolved_tokenizer_revision:
        raise ValueError("tokenizer_revision must be non-empty.")
    if max_length is not None and int(max_length) <= 0:
        raise ValueError(f"max_length must be positive or None, got {max_length}.")
    return {
        "encoder_name": resolved_encoder_name,
        "encoder_revision": resolved_encoder_revision,
        "tokenizer_name": resolved_tokenizer_name,
        "tokenizer_revision": resolved_tokenizer_revision,
        "max_length": None if max_length is None else int(max_length),
        "truncation": True,
        "pooling": _pooling,
        "normalize": _normalization,
        "output_dtype": _output_dtype,
    }


def _resolve_batch_size(batch_size: int | None) -> int:
    value = 256 if batch_size is None and torch.cuda.is_available() else batch_size
    if value is None:
        value = 16
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return resolved


def _text_cache_path(
    *,
    cache_dir: Path | None,
    kind: str,
    provenance: Mapping[str, object],
    texts: list[str],
    query_prefix: str,
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {
        "kind": kind,
        "provenance": dict(provenance),
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
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("Ignoring unreadable text encode cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        log.warning("Ignoring malformed text encode cache %s: expected dict.", path)
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


def _write_cached_embeddings(
    path: Path,
    *,
    embeddings: torch.Tensor,
    provenance: Mapping[str, object],
) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "provenance": dict(provenance),
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
