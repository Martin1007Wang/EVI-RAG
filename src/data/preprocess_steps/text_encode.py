from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 假设这些是从你的本地模块导入的
from .sample_types import EntityVocab, PreparedSample, RelationVocab

log = logging.getLogger(__name__)


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling, ensuring numerical stability."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return masked.sum(dim=1) / denom


@dataclass
class TextEncoder:
    model_name: str
    device: str = "auto"
    precision: str = "bf16"
    progress_bar: bool = True

    def __post_init__(self) -> None:
        self._device = self._resolve_device(self.device)
        self._torch_dtype = self._resolve_dtype(self.precision)

        log.info(f"Loading TextEncoder: {self.model_name} on {self._device} (precision={self.precision})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self._torch_dtype,
        )
        self.model.to(self._device)
        self.model.eval()

        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Unable to infer hidden size for encoder={self.model_name!r}.")
        self.hidden_size = int(hidden_size)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        device_str = device_str.strip().lower()
        if device_str in {"", "auto"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "gpu":
            device_str = "cuda"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available.")
        return torch.device(device_str)

    @staticmethod
    def _resolve_dtype(precision: str) -> torch.dtype:
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in precision_map:
            raise ValueError(f"Unsupported precision {precision}. Must be one of {list(precision_map.keys())}")
        return precision_map[precision]

    def _forward_batch(
        self,
        texts: Sequence[str],
        *,
        max_tokens: int | None,
        pad_to_max: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        tokenizer_kwargs: dict[str, Any] = {
            "padding": "max_length" if pad_to_max else True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if max_tokens is not None:
            tokenizer_kwargs["max_length"] = int(max_tokens)

        batch = self.tokenizer(list(texts), **tokenizer_kwargs)
        model_inputs = {k: v.to(self._device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # 核心实证支持：使用 autocast 保证数值稳定与速度，安全回退到 CPU 的 fp32
        enable_autocast = self._device.type == "cuda" and self.precision != "fp32"

        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type, dtype=self._torch_dtype, enabled=enable_autocast
        ):
            outputs = self.model(**model_inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = model_inputs["attention_mask"]
            pooled = _mean_pool(last_hidden_state, attention_mask)

        # 统一以 fp32 驻留内存，防止下游操作精度不匹配
        return (
            pooled.to(dtype=torch.float32, device="cpu"),
            last_hidden_state.to(dtype=torch.float32, device="cpu"),
            attention_mask.to(dtype=torch.bool, device="cpu"),
        )

    def encode(self, texts: Sequence[str], batch_size: int, *, desc: str | None = None) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)

        outputs: list[torch.Tensor] = []
        progress = tqdm(
            _iter_batches(texts, batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=desc or "Encode",
            disable=not self.progress_bar,
        )
        for _, chunk in progress:
            pooled, _, _ = self._forward_batch(chunk, max_tokens=None, pad_to_max=False)
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    def encode_with_token_states(
        self, texts: Sequence[str], batch_size: int, *, max_tokens: int, desc: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not texts:
            return (
                torch.empty((0, self.hidden_size), dtype=torch.float32),
                torch.empty((0, max_tokens, self.hidden_size), dtype=torch.float32),
                torch.empty((0, max_tokens), dtype=torch.bool),
            )

        pooled_list, context_list, mask_list = [], [], []
        progress = tqdm(
            _iter_batches(texts, batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=desc or "EncodeTokens",
            disable=not self.progress_bar,
        )
        for _, chunk in progress:
            pooled, context, mask = self._forward_batch(chunk, max_tokens=max_tokens, pad_to_max=True)
            pooled_list.append(pooled)
            context_list.append(context)
            mask_list.append(mask)

        return torch.cat(pooled_list, dim=0), torch.cat(context_list, dim=0), torch.cat(mask_list, dim=0)


# =============================================================================
# 流水线与缓存逻辑
# =============================================================================


def encode_preprocessed_features(
    *,
    prepared_samples: Sequence[PreparedSample],
    entity_vocab: EntityVocab,
    relation_vocab: RelationVocab,
    embeddings_dir: Path,
    encoder_name: str,
    device: str = "auto",
    batch_size: int | None = None,
    precision: str = "bf16",  # 对齐 Lightning
    progress_bar: bool = True,
    question_ctx_max_tokens: int = 0,
    reuse_embeddings_if_exists: bool = False,
) -> Dict[str, Any]:

    if not encoder_name:
        raise ValueError("encoder_name cannot be empty.")

    resolved_batch_size = batch_size or (256 if torch.cuda.is_available() else 256)

    entity_metadata = entity_vocab.build_entity_metadata()
    relation_labels = relation_vocab.labels()
    question_texts = [entry.sample.question for entry in prepared_samples]

    cache_payload = _try_load_reusable_embeddings(
        embeddings_dir=embeddings_dir,
        entity_metadata=entity_metadata,
        relation_labels=relation_labels,
        question_texts=question_texts,
        question_ctx_max_tokens=question_ctx_max_tokens,
        enabled=reuse_embeddings_if_exists,
    )

    if cache_payload is not None:
        log.info("Cache hit! Loaded entities, relations, and questions from disk.")
        return {"entity_metadata": entity_metadata, "relation_labels": relation_labels, **cache_payload}

    log.info("Computing dense embeddings...")
    encoder = TextEncoder(
        model_name=encoder_name,
        device=device,
        precision=precision,
        progress_bar=progress_bar,
    )

    # 1. 编码实体 (去除了冗余的内部 chunking)
    raw_texts = cast(List[str], entity_metadata["text_labels"])
    raw_emb_ids = cast(List[int], entity_metadata["text_embedding_ids"])
    max_id = int(cast(int, entity_metadata["max_embedding_id"]))

    log.info("Encoding Entities...")
    flat_entity_embs = encoder.encode(raw_texts, batch_size=resolved_batch_size, desc="Entities")

    # 采用高级索引直接赋值，时间复杂度最优
    entity_embeddings = torch.zeros((max_id + 1, encoder.hidden_size), dtype=torch.float32)
    entity_embeddings[raw_emb_ids] = flat_entity_embs

    # 2. 编码关系
    relation_embeddings = encoder.encode(relation_labels, batch_size=resolved_batch_size, desc="Relations")

    # 3. 编码问题
    if question_ctx_max_tokens > 0:
        question_embeddings, question_contexts, question_context_masks = encoder.encode_with_token_states(
            question_texts, batch_size=resolved_batch_size, max_tokens=question_ctx_max_tokens, desc="Questions(Tokens)"
        )
    else:
        question_embeddings = encoder.encode(question_texts, batch_size=resolved_batch_size, desc="Questions")
        question_contexts, question_context_masks = None, None

    output_payload = {
        "entity_embeddings": entity_embeddings,
        "relation_embeddings": relation_embeddings,
        "question_embeddings": question_embeddings,
        "question_contexts": question_contexts,
        "question_context_masks": question_context_masks,
    }

    if reuse_embeddings_if_exists:
        _save_cache(embeddings_dir, output_payload, entity_metadata, relation_labels, question_texts, question_ctx_max_tokens)

    return {"entity_metadata": entity_metadata, "relation_labels": relation_labels, **output_payload}


def _try_load_reusable_embeddings(
    *,
    embeddings_dir: Path,
    entity_metadata: Dict[str, Any],
    relation_labels: Sequence[str],
    question_texts: Sequence[str],
    question_ctx_max_tokens: int,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None

    cache_file = embeddings_dir / "pipeline_cache.pt"
    if not cache_file.exists():
        return None

    try:
        # 移除 mmap 保证多版本兼容性，保留 weights_only 防止反序列化漏洞
        cache = torch.load(cache_file, map_location="cpu", weights_only=True)
    except Exception as e:
        log.warning(f"Failed to load cache: {e}. Recomputing...")
        return None

    # 形式化检验假设：元数据必须完全匹配
    if cache.get("relation_labels") != list(relation_labels):
        return None
    if cache.get("question_texts") != list(question_texts):
        return None
    if cache.get("question_ctx_max_tokens") != question_ctx_max_tokens:
        return None
    if cache.get("entity_labels") != entity_metadata.get("entity_labels"):
        return None

    for key in ["entity_embedding_map", "cvt_mask"]:
        payload_tensor = cache.get(key)
        current_tensor = entity_metadata.get(key)
        if not (torch.is_tensor(payload_tensor) and torch.is_tensor(current_tensor)):
            return None
        if not torch.equal(payload_tensor, current_tensor):
            return None

    return cache.get("tensors")


def _save_cache(
    embeddings_dir: Path,
    tensors: Dict[str, Any],
    entity_metadata: Dict[str, Any],
    relation_labels: Sequence[str],
    question_texts: Sequence[str],
    question_ctx_max_tokens: int,
) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    cache = {
        "relation_labels": list(relation_labels),
        "question_texts": list(question_texts),
        "question_ctx_max_tokens": question_ctx_max_tokens,
        "entity_labels": entity_metadata.get("entity_labels"),
        "entity_embedding_map": entity_metadata.get("entity_embedding_map"),
        "cvt_mask": entity_metadata.get("cvt_mask"),
        "tensors": tensors,
    }
    torch.save(cache, embeddings_dir / "pipeline_cache.pt")
    log.info(f"Saved computed embeddings to {embeddings_dir / 'pipeline_cache.pt'}")
