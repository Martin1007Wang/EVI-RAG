from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.data.components import SharedDataResources
from src.models.components.graph import DDE
from src.models.components.projections import EmbeddingProjector

logger = logging.getLogger(__name__)

HASH_CHUNK_SIZE = 4 * 1024 * 1024


class GraphEmbedder(nn.Module):
    """
    GFlowNet 图嵌入器：训练要求 retriever ckpt，评估可从 gflownet ckpt 元数据恢复结构信息。

    结构特征（topic/DDE）由 retriever ckpt 或 gflownet ckpt 中的 parity_meta 决定。
    严格保持 g_agent 的 SSOT 字段，不引入额外缓存字段。
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        proj_dropout: float,
        projector_checkpoint: str | None = None,
        projector_key_prefixes: Optional[Sequence[str]] = None,
        allow_deferred_init: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj_dropout = float(proj_dropout)
        ckpt_raw = "" if projector_checkpoint is None else str(projector_checkpoint).strip()
        if ckpt_raw.lower() in {"null", "none"}:
            ckpt_raw = ""
        self.projector_checkpoint = ckpt_raw
        self.allow_deferred_init = bool(allow_deferred_init)
        if not self.projector_checkpoint and not self.allow_deferred_init:
            raise ValueError(
                "GraphEmbedder requires retriever ckpt. "
                "Provide `ckpt.retriever` or enable deferred init for gflownet-eval."
            )

        self.projector_key_prefixes = self._normalize_prefixes(projector_key_prefixes or ["model._orig_mod", "model", ""])

        self._shared_resources: Optional[SharedDataResources] = None
        self._global_embeddings = None
        self._entity_dim: Optional[int] = None
        self._relation_dim: Optional[int] = None
        self._num_entities: Optional[int] = None
        self._num_relations: Optional[int] = None

        self._edge_in_dim: Optional[int] = None
        self._use_topic_pe: bool = False
        self._num_topics: int = 0
        self._struct_dim: int = 0
        self._dde: Optional[DDE] = None

        self.entity_proj: Optional[EmbeddingProjector] = None
        self.relation_proj: Optional[EmbeddingProjector] = None
        self.query_proj: Optional[EmbeddingProjector] = None
        self.non_text_entity_emb: Optional[nn.Embedding] = None
        self.edge_adapter: Optional[nn.Module] = None
        self.edge_score_proj: Optional[nn.Linear] = None
        self._components_initialized = False
        self._retriever_meta_source: Optional[str] = None
        self._retriever_ckpt_hash: Optional[str] = None
        self._retriever_ckpt_basename: Optional[str] = None

        # 构造所有可训练子模块，确保 strict load 不依赖外部 datamodule/setup。
        if self.projector_checkpoint:
            state_dict = self._load_checkpoint_state(self.projector_checkpoint)
            self._load_retriever_components(state_dict)
            self._components_initialized = True
            self._retriever_meta_source = "retriever_ckpt"
            self._retriever_ckpt_basename = Path(self.projector_checkpoint).name
        elif self.allow_deferred_init:
            self._retriever_meta_source = "deferred"

    @staticmethod
    def _normalize_prefixes(prefixes: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for p in prefixes:
            p_clean = str(p).rstrip(".")
            if p_clean not in seen:
                ordered.append(p_clean)
                seen.add(p_clean)
        augmented: list[str] = list(ordered)
        for p in ordered:
            retr = f"retriever.{p}" if p else "retriever"
            if retr not in seen:
                augmented.append(retr)
                seen.add(retr)
        if "" not in seen:
            augmented.append("")
        return augmented

    def setup(self, resources_cfg: SharedDataResources | Dict[str, Any], device: Optional[torch.device] = None) -> None:
        if self._shared_resources is not None:
            return
        if isinstance(resources_cfg, SharedDataResources):
            self._shared_resources = resources_cfg
        else:
            self._shared_resources = SharedDataResources(**resources_cfg)

        self._global_embeddings = self._shared_resources.global_embeddings
        self._entity_dim = int(self._global_embeddings.entity_embeddings.size(1))
        self._relation_dim = int(self._global_embeddings.relation_embeddings.size(1))
        self._num_entities = int(self._global_embeddings.entity_embeddings.size(0))
        self._num_relations = int(self._global_embeddings.relation_embeddings.size(0))

        if self.hidden_dim != self._entity_dim or self.hidden_dim != self._relation_dim:
            raise ValueError(
                f"GraphEmbedder requires hidden_dim==retriever emb dim; got hidden_dim={self.hidden_dim}, "
                f"entity_dim={self._entity_dim}, relation_dim={self._relation_dim}."
            )

        if device is not None:
            self.to(device)

    def embed_batch(self, batch: Any, *, device: torch.device) -> "EmbedOutputs":
        if not self._components_initialized:
            raise RuntimeError("GraphEmbedder is not initialized; load a gflownet ckpt with retriever_meta first.")
        if self._shared_resources is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before embed_batch.")
        if self.edge_adapter is None or self.entity_proj is None or self.relation_proj is None or self.query_proj is None:
            raise RuntimeError("GraphEmbedder not initialized; call setup() first.")
        if self.hidden_dim != self._entity_dim or self.hidden_dim != self._relation_dim:
            raise ValueError(
                f"GraphEmbedder hidden_dim mismatch with resources (hidden_dim={self.hidden_dim}, "
                f"entity_dim={self._entity_dim}, relation_dim={self._relation_dim})."
            )

        edge_index = batch.edge_index.to(device)
        node_ptr = batch.ptr.to(device)
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("Batch must contain at least one graph.")

        edge_batch, edge_ptr = self._compute_edge_batch(edge_index, node_ptr=node_ptr, num_graphs=num_graphs, device=device)
        edge_relations_raw = batch.edge_attr
        edge_relations = edge_relations_raw.to(device)

        question_raw = self._prepare_question_embeddings(batch.question_emb, batch_size=num_graphs, device=device)
        question_tokens = self.query_proj(question_raw)

        node_embedding_ids = getattr(batch, "node_embedding_ids", None)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if node_embedding_ids is None or node_global_ids is None:
            raise ValueError("Batch missing node_embedding_ids/node_global_ids; g_agent cache must include them.")
        if node_embedding_ids.shape != node_global_ids.shape:
            raise ValueError("node_embedding_ids shape mismatch with node_global_ids in g_agent batch.")

        node_global_ids = node_global_ids.to(device)
        node_embeddings = self._lookup_entities(node_embedding_ids, device=device)
        node_tokens = self.entity_proj(node_embeddings)
        non_text_mask = (node_embedding_ids == 0).to(device=device, dtype=torch.bool)
        node_tokens = self._apply_non_text_override(node_tokens=node_tokens, non_text_mask=non_text_mask)

        relation_embeddings = self._lookup_relations(edge_relations_raw, device=device)
        relation_tokens = self.relation_proj(relation_embeddings)

        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]

        # Interaction: Linear([Question, Head, Relation, Tail] (+struct)) without materializing the giant concat.
        q_edge = question_tokens[edge_batch]
        head_edge = node_tokens[edge_index[0]]
        tail_edge = node_tokens[edge_index[1]]
        struct: Optional[torch.Tensor] = None
        if self._use_topic_pe:
            struct = self._build_structure_features(
                batch=batch,
                edge_index=edge_index,
                num_nodes=int(node_tokens.size(0)),
                device=device,
                dtype=q_edge.dtype,
            )
        edge_tokens = self._edge_adapter_forward_parts(
            q_edge=q_edge,
            head_edge=head_edge,
            relation_edge=relation_tokens,
            tail_edge=tail_edge,
            struct_edge=struct,
        )
        edge_scores = getattr(batch, "edge_scores", None)
        if edge_scores is None:
            raise ValueError("Batch missing edge_scores; g_agent cache must include per-edge retriever scores.")
        edge_scores = edge_scores.to(device=device, dtype=edge_tokens.dtype).view(-1, 1)
        if edge_scores.size(0) != edge_tokens.size(0):
            raise ValueError(f"edge_scores length {edge_scores.size(0)} != num_edges {edge_tokens.size(0)}")
        if self.edge_score_proj is None:
            raise RuntimeError("edge_score_proj is not initialized; call setup() first.")
        edge_tokens = edge_tokens + self.edge_score_proj(edge_scores)

        start_entity_ids, start_entity_mask = self._pad_start_entities(
            batch.start_entity_ids.to(device),
            batch._slice_dict["start_entity_ids"].to(device),
            num_graphs=num_graphs,
            device=device,
        )
        outputs = EmbedOutputs(
            edge_tokens=edge_tokens,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            heads_global=heads_global,
            tails_global=tails_global,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            start_entity_ids=start_entity_ids,
            start_entity_mask=start_entity_mask,
        )
        outputs.validate_device(device)
        return outputs

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def _load_checkpoint_state(self, ckpt_path: str) -> Dict[str, torch.Tensor]:
        ckpt_path = Path(ckpt_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"projector_checkpoint not found: {ckpt_path}")
        load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        checkpoint = torch.load(str(ckpt_path), **load_kwargs)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint format at {ckpt_path}; expected mapping but got {type(state_dict)}")
        return state_dict

    def _find_first_match(self, state_dict: Dict[str, torch.Tensor], suffix: str) -> str:
        candidates: list[str] = []
        for prefix in self.projector_key_prefixes:
            prefix_clean = prefix.rstrip(".")
            key = f"{prefix_clean}.{suffix}" if prefix_clean else suffix
            candidates.append(key)
            if key in state_dict:
                return key
        raise KeyError(f"Missing key '{suffix}' in checkpoint; tried: {candidates}")

    def _find_pair(self, state_dict: Dict[str, torch.Tensor], suffix: str) -> tuple[str, str]:
        w = self._find_first_match(state_dict, f"{suffix}.weight")
        b = self._find_first_match(state_dict, f"{suffix}.bias")
        return w, b

    def _load_projector(self, state_dict: Dict[str, torch.Tensor], *, name: str) -> EmbeddingProjector:
        weight_key, bias_key = self._find_pair(state_dict, f"{name}.network.0")
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        if out_dim != self.hidden_dim or in_dim != self.hidden_dim:
            raise ValueError(f"{name} projector shape mismatch: ckpt=({out_dim},{in_dim}) vs expected=({self.hidden_dim},{self.hidden_dim})")
        projector = EmbeddingProjector(output_dim=out_dim, input_dim=in_dim, finetune=True)
        load_state = {"network.0.weight": weight, "network.0.bias": bias}
        missing, unexpected = projector.load_state_dict(load_state, strict=False)
        if missing:
            raise RuntimeError(f"{name} projector missing keys: {missing}")
        if unexpected and any("network.0" not in key for key in unexpected):
            raise RuntimeError(f"{name} projector unexpected keys: {unexpected}")
        return projector

    def _load_retriever_components(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.entity_proj = self._load_projector(state_dict, name="entity_proj")
        self.relation_proj = self._load_projector(state_dict, name="relation_proj")
        self.query_proj = self._load_projector(state_dict, name="query_proj")

        non_text_key = self._find_first_match(state_dict, "non_text_entity_emb.weight")
        non_text_weight = state_dict[non_text_key]
        if non_text_weight.shape != (1, self.hidden_dim):
            raise ValueError(
                f"non_text_entity_emb.weight shape mismatch: got {tuple(non_text_weight.shape)} expected {(1, self.hidden_dim)}"
            )
        self.non_text_entity_emb = nn.Embedding(1, self.hidden_dim)
        self.non_text_entity_emb.weight.data.copy_(non_text_weight)

        w0_key = self._find_first_match(state_dict, "feature_extractor.network.0.weight")
        b0_key = self._find_first_match(state_dict, "feature_extractor.network.0.bias")
        w1_key = self._find_first_match(state_dict, "feature_extractor.network.3.weight")
        b1_key = self._find_first_match(state_dict, "feature_extractor.network.3.bias")
        w0, b0, w1, b1 = state_dict[w0_key], state_dict[b0_key], state_dict[w1_key], state_dict[b1_key]
        emb_dim = int(w0.shape[0])
        edge_in_dim = int(w0.shape[1])
        hidden_dim_ckpt = int(w1.shape[0])
        if hidden_dim_ckpt != self.hidden_dim:
            raise ValueError(f"feature_extractor hidden_dim mismatch: ckpt={hidden_dim_ckpt} vs expected={self.hidden_dim}")
        if int(w1.shape[1]) != emb_dim:
            raise ValueError(f"feature_extractor layer1 in-dim mismatch: ckpt={tuple(w1.shape)}")

        semantic_dim = 4 * self.hidden_dim
        use_topic_pe, num_topics, num_rounds, num_rev = self._load_retriever_parity_meta(
            state_dict,
            edge_in_dim=edge_in_dim,
            semantic_dim=semantic_dim,
        )
        struct_dim = 0
        if use_topic_pe:
            if num_topics <= 1:
                raise ValueError(f"Invalid retriever parity_meta: num_topics must be >= 2 when topic_pe=1, got {num_topics}")
            struct_dim = 2 * int(num_topics) * (1 + int(num_rounds) + int(num_rev))
        expected_in_dim = semantic_dim + struct_dim
        if edge_in_dim != expected_in_dim:
            raise ValueError(
                f"feature_extractor input_dim mismatch: ckpt={edge_in_dim} vs expected={expected_in_dim} "
                f"(semantic={semantic_dim}, struct={struct_dim}; topic_pe={int(use_topic_pe)}, "
                f"num_topics={int(num_topics)}, dde_rounds={int(num_rounds)}, dde_reverse_rounds={int(num_rev)})."
            )
        self._edge_in_dim = edge_in_dim
        self._use_topic_pe = bool(use_topic_pe)
        self._num_topics = int(num_topics) if use_topic_pe else 0
        self._struct_dim = int(struct_dim)
        self._dde = DDE(num_rounds=int(num_rounds), num_reverse_rounds=int(num_rev)) if use_topic_pe else None

        # Adapter: Linear -> LayerNorm -> GELU -> Linear, warm-start from retriever DenseFeatureExtractor linears.
        self.edge_adapter = nn.Sequential(
            nn.Linear(edge_in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, self.hidden_dim),
        )
        with torch.no_grad():
            self.edge_adapter[0].weight.copy_(w0)
            self.edge_adapter[0].bias.copy_(b0)
            self.edge_adapter[3].weight.copy_(w1)
            self.edge_adapter[3].bias.copy_(b1)
        self.edge_score_proj = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.edge_score_proj.weight)
        self._components_initialized = True

    def _configure_struct_features(
        self,
        *,
        use_topic_pe: bool,
        num_topics: int,
        num_rounds: int,
        num_reverse_rounds: int,
    ) -> None:
        struct_dim = 0
        if use_topic_pe:
            if num_topics <= 1:
                raise ValueError(f"Invalid retriever_meta: num_topics must be >= 2 when topic_pe=1, got {num_topics}")
            struct_dim = 2 * int(num_topics) * (1 + int(num_rounds) + int(num_reverse_rounds))
        self._edge_in_dim = 4 * self.hidden_dim + struct_dim
        self._use_topic_pe = bool(use_topic_pe)
        self._num_topics = int(num_topics) if use_topic_pe else 0
        self._struct_dim = int(struct_dim)
        self._dde = DDE(num_rounds=int(num_rounds), num_reverse_rounds=int(num_reverse_rounds)) if use_topic_pe else None

    def _init_fresh_components(self) -> None:
        """Initialize projector/adapter from scratch after struct configuration."""
        if self._edge_in_dim is None:
            raise RuntimeError("edge_in_dim is not initialized; configure struct features first.")
        self.entity_proj = EmbeddingProjector(output_dim=self.hidden_dim, input_dim=self.hidden_dim, finetune=True)
        self.relation_proj = EmbeddingProjector(output_dim=self.hidden_dim, input_dim=self.hidden_dim, finetune=True)
        self.query_proj = EmbeddingProjector(output_dim=self.hidden_dim, input_dim=self.hidden_dim, finetune=True)
        self.non_text_entity_emb = nn.Embedding(1, self.hidden_dim)
        nn.init.zeros_(self.non_text_entity_emb.weight)

        self.edge_adapter = nn.Sequential(
            nn.Linear(self._edge_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_score_proj = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.edge_score_proj.weight)
        self._components_initialized = True

    def _load_retriever_parity_meta(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        edge_in_dim: int,
        semantic_dim: int,
    ) -> tuple[bool, int, int, int]:
        """Load minimal feature parity metadata from retriever checkpoint.

        Returns (use_topic_pe, num_topics, num_rounds, num_reverse_rounds). If metadata is missing,
        semantic-only checkpoints (edge_in_dim==semantic_dim) are accepted; otherwise we fail fast.
        """
        try:
            meta_key = self._find_first_match(state_dict, "parity_meta")
        except KeyError:
            if int(edge_in_dim) == int(semantic_dim):
                return False, 0, 0, 0
            raise ValueError(
                "Retriever checkpoint is missing parity_meta required to disambiguate structural features "
                f"(edge_in_dim={edge_in_dim}, semantic_dim={semantic_dim}). Re-export/retrain retriever with updated code."
            )
        meta = state_dict[meta_key]
        if not torch.is_tensor(meta):
            raise ValueError(f"Invalid parity_meta type: expected torch.Tensor but got {type(meta)}")
        meta = meta.to(dtype=torch.long).view(-1)
        if meta.numel() < 4:
            raise ValueError(f"Invalid parity_meta length: expected >=4 but got {int(meta.numel())}")
        use_topic_pe = bool(int(meta[0].item()))
        num_topics = int(meta[1].item())
        num_rounds = int(meta[2].item())
        num_rev = int(meta[3].item())
        return use_topic_pe, num_topics, num_rounds, num_rev

    def init_from_retriever_meta(self, meta: Dict[str, Any]) -> None:
        """Initialize missing components from retriever meta stored in gflownet checkpoints."""
        use_topic_pe, num_topics, num_rounds, num_rev = self._parse_retriever_meta(meta)
        if self._components_initialized:
            self._validate_retriever_meta(use_topic_pe, num_topics, num_rounds, num_rev, meta)
            self._record_retriever_meta(meta, source="gflownet_ckpt", allow_override=False)
            return
        self._configure_struct_features(
            use_topic_pe=use_topic_pe,
            num_topics=num_topics,
            num_rounds=num_rounds,
            num_reverse_rounds=num_rev,
        )
        self._init_fresh_components()
        self._record_retriever_meta(meta, source="gflownet_ckpt", allow_override=True)

    def export_retriever_meta(self) -> Dict[str, Any]:
        if not self._components_initialized:
            raise RuntimeError("GraphEmbedder is not initialized; cannot export retriever meta.")
        use_topic_pe = bool(self._use_topic_pe)
        num_topics = int(self._num_topics) if use_topic_pe else 0
        num_rounds = int(self._dde.num_rounds) if self._dde is not None else 0
        num_rev = int(self._dde.num_reverse_rounds) if self._dde is not None else 0
        parity_meta = [int(use_topic_pe), int(num_topics), int(num_rounds), int(num_rev)]
        meta: Dict[str, Any] = {
            "parity_meta": parity_meta,
            "use_topic_pe": bool(use_topic_pe),
            "num_topics": int(num_topics),
            "num_rounds": int(num_rounds),
            "num_reverse_rounds": int(num_rev),
            "struct_dim": int(self._struct_dim or 0),
            "edge_in_dim": int(self._edge_in_dim or 0),
            "hidden_dim": int(self.hidden_dim),
        }
        if self.projector_checkpoint and self._retriever_ckpt_hash is None:
            self._retriever_ckpt_hash = self._compute_ckpt_hash(self.projector_checkpoint)
            self._retriever_ckpt_basename = Path(self.projector_checkpoint).name
        if self._retriever_ckpt_hash:
            meta["retriever_ckpt_hash"] = self._retriever_ckpt_hash
        if self._retriever_ckpt_basename:
            meta["retriever_ckpt_basename"] = self._retriever_ckpt_basename
        if self._retriever_meta_source:
            meta["source"] = self._retriever_meta_source
        return meta

    def _compute_ckpt_hash(self, ckpt_path: str) -> Optional[str]:
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            return None
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _record_retriever_meta(self, meta: Dict[str, Any], *, source: str, allow_override: bool) -> None:
        if allow_override or self._retriever_meta_source is None or self._retriever_meta_source == "deferred":
            self._retriever_meta_source = source
        ckpt_hash = meta.get("retriever_ckpt_hash")
        ckpt_name = meta.get("retriever_ckpt_basename")
        if ckpt_hash and (allow_override or self._retriever_ckpt_hash is None):
            self._retriever_ckpt_hash = str(ckpt_hash)
        if ckpt_name and (allow_override or self._retriever_ckpt_basename is None):
            self._retriever_ckpt_basename = str(ckpt_name)

    def _parse_retriever_meta(self, meta: Dict[str, Any]) -> tuple[bool, int, int, int]:
        if not isinstance(meta, dict):
            raise TypeError("retriever_meta must be a dict.")
        parity_meta = meta.get("parity_meta")
        if parity_meta is not None:
            if isinstance(parity_meta, torch.Tensor):
                parity_meta = parity_meta.detach().cpu().tolist()
            if len(parity_meta) != 4:
                raise ValueError(f"parity_meta must have 4 values; got {parity_meta}")
            use_topic_pe, num_topics, num_rounds, num_rev = [int(v) for v in parity_meta]
        else:
            use_topic_pe = int(meta.get("use_topic_pe", 0))
            num_topics = int(meta.get("num_topics", 0))
            num_rounds = int(meta.get("num_rounds", 0))
            num_rev = int(meta.get("num_reverse_rounds", 0))
        if use_topic_pe and num_topics <= 1:
            raise ValueError(f"Invalid retriever_meta: num_topics must be >= 2 when topic_pe=1, got {num_topics}")
        struct_dim = 2 * int(num_topics) * (1 + int(num_rounds) + int(num_rev)) if use_topic_pe else 0
        edge_in_dim = 4 * self.hidden_dim + struct_dim
        if "edge_in_dim" in meta and int(meta["edge_in_dim"]) != edge_in_dim:
            raise ValueError(f"retriever_meta edge_in_dim mismatch: meta={meta['edge_in_dim']} expected={edge_in_dim}")
        if "struct_dim" in meta and int(meta["struct_dim"]) != struct_dim:
            raise ValueError(f"retriever_meta struct_dim mismatch: meta={meta['struct_dim']} expected={struct_dim}")
        return bool(use_topic_pe), int(num_topics), int(num_rounds), int(num_rev)

    def _validate_retriever_meta(
        self,
        use_topic_pe: bool,
        num_topics: int,
        num_rounds: int,
        num_rev: int,
        meta: Dict[str, Any],
    ) -> None:
        current_use = bool(self._use_topic_pe)
        current_topics = int(self._num_topics) if self._use_topic_pe else 0
        current_rounds = int(self._dde.num_rounds) if self._dde is not None else 0
        current_rev = int(self._dde.num_reverse_rounds) if self._dde is not None else 0
        if (current_use, current_topics, current_rounds, current_rev) != (
            bool(use_topic_pe),
            int(num_topics),
            int(num_rounds),
            int(num_rev),
        ):
            raise ValueError(
                "retriever_meta mismatch with existing embedder configuration: "
                f"meta=({use_topic_pe},{num_topics},{num_rounds},{num_rev}) "
                f"current=({current_use},{current_topics},{current_rounds},{current_rev})"
            )
        _ = self._parse_retriever_meta(meta)

    def _build_structure_features(
        self,
        *,
        batch: Any,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self._use_topic_pe or self._num_topics <= 0:
            raise RuntimeError("_build_structure_features called but topic_pe is disabled.")
        start_nodes = getattr(batch, "start_node_locals", None)
        if start_nodes is None:
            raise ValueError("topic_pe is enabled in retriever checkpoint but batch.start_node_locals is missing.")
        start_nodes = start_nodes.to(device=device, dtype=torch.long).view(-1)
        if start_nodes.numel() > 0:
            if (start_nodes < 0).any() or (start_nodes >= num_nodes).any():
                raise ValueError("start_node_locals out of range for topic_pe feature construction.")

        topic_entity_mask = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if start_nodes.numel() > 0:
            topic_entity_mask[start_nodes] = 1
        topic_one_hot = F.one_hot(topic_entity_mask, num_classes=int(self._num_topics)).to(dtype=dtype)

        feats: list[torch.Tensor] = [topic_one_hot]
        if self._dde is not None:
            reverse_edge_index = getattr(batch, "reverse_edge_index", None)
            if reverse_edge_index is not None:
                reverse_edge_index = reverse_edge_index.to(device=device)
            feats.extend(self._dde(topic_one_hot, edge_index, reverse_edge_index))

        stacked = torch.stack(feats, dim=-1)
        node_struct = stacked.reshape(num_nodes, -1)
        head_struct = node_struct[edge_index[0]]
        tail_struct = node_struct[edge_index[1]]
        return torch.cat([head_struct, tail_struct], dim=-1)

    # ------------------------------------------------------------------ #
    # Embedding + feature construction
    # ------------------------------------------------------------------ #
    def _lookup_entities(self, embedding_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if self._num_entities is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        return self._global_embeddings.get_entity_embeddings(embedding_ids, device=device)  # type: ignore[call-arg]

    def _lookup_relations(self, ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if self._num_relations is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id >= self._num_relations:
                raise ValueError(f"Relation ids out of range: min={min_id} max={max_id} valid=[0,{self._num_relations - 1}]")
        return self._global_embeddings.get_relation_embeddings(ids, device=device)  # type: ignore[call-arg]

    def _edge_adapter_forward_parts(
        self,
        *,
        q_edge: torch.Tensor,         # [E_total, H]
        head_edge: torch.Tensor,      # [E_total, H]
        relation_edge: torch.Tensor,  # [E_total, H]
        tail_edge: torch.Tensor,      # [E_total, H]
        struct_edge: Optional[torch.Tensor],  # [E_total, S] or None
    ) -> torch.Tensor:
        if self.edge_adapter is None:
            raise RuntimeError("edge_adapter is not initialized; call setup() first.")
        if self._edge_in_dim is None:
            raise RuntimeError("edge_in_dim is not initialized; setup() must load retriever adapter weights.")

        if not isinstance(self.edge_adapter, nn.Sequential) or len(self.edge_adapter) != 4:
            raise TypeError("edge_adapter must be a 4-layer Sequential: Linear -> LayerNorm -> GELU -> Linear.")
        linear0, norm0, act0, linear1 = self.edge_adapter
        if not isinstance(linear0, nn.Linear) or not isinstance(linear1, nn.Linear):
            raise TypeError("edge_adapter linear layers are malformed; expected nn.Linear at positions 0 and 3.")

        hidden_dim = int(self.hidden_dim)
        in_dim = int(linear0.in_features)
        expected_min = 4 * hidden_dim
        if in_dim < expected_min:
            raise ValueError(f"edge_adapter in_features={in_dim} < semantic_dim={expected_min}")

        w = linear0.weight
        b = linear0.bias
        offset = 0
        w_q = w[:, offset : offset + hidden_dim]
        offset += hidden_dim
        w_h = w[:, offset : offset + hidden_dim]
        offset += hidden_dim
        w_r = w[:, offset : offset + hidden_dim]
        offset += hidden_dim
        w_t = w[:, offset : offset + hidden_dim]
        offset += hidden_dim

        out0 = (
            F.linear(q_edge, w_q)
            + F.linear(head_edge, w_h)
            + F.linear(relation_edge, w_r)
            + F.linear(tail_edge, w_t)
        )
        if struct_edge is not None:
            struct_dim = int(struct_edge.size(1))
            if in_dim != offset + struct_dim:
                raise ValueError(
                    f"edge_adapter input dim mismatch: in_features={in_dim}, semantic={offset}, struct={struct_dim}"
                )
            w_s = w[:, offset:]
            out0 = out0 + F.linear(struct_edge, w_s)
        else:
            if in_dim != offset:
                raise ValueError(
                    f"edge_adapter expects struct features (in_features={in_dim}, semantic_only={offset}) "
                    "but struct_edge is None; ensure retriever parity_meta matches g_agent feature construction."
                )
        if b is not None:
            out0 = out0 + b

        out = norm0(out0)
        out = act0(out)
        return linear1(out)

    def _apply_non_text_override(
        self,
        *,
        node_tokens: torch.Tensor,
        non_text_mask: Optional[torch.Tensor] = None,
        node_embedding_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.non_text_entity_emb is None or self.entity_proj is None:
            return node_tokens
        if non_text_mask is None:
            if node_embedding_ids is None:
                return node_tokens
            non_text_mask = node_embedding_ids == 0
        non_text_mask = non_text_mask.to(device=node_tokens.device, dtype=torch.bool)
        if not bool(non_text_mask.any().item()):
            return node_tokens
        non_text_proj = self.entity_proj(self.non_text_entity_emb.weight.to(device=node_tokens.device))[0].to(dtype=node_tokens.dtype)
        return torch.where(non_text_mask.unsqueeze(-1), non_text_proj.unsqueeze(0), node_tokens)

    @staticmethod
    def _prepare_question_embeddings(question_emb: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if question_emb is None or question_emb.numel() == 0:
            raise ValueError("question_emb is missing or empty; g_agent cache must provide question embeddings for every graph.")
        if question_emb.dim() == 1:
            if question_emb.numel() % max(batch_size, 1) != 0:
                raise ValueError(f"question_emb flatten length {question_emb.numel()} not divisible by batch_size={batch_size}")
            dim = question_emb.numel() // max(batch_size, 1)
            question_emb = question_emb.view(batch_size, dim)
        elif question_emb.dim() == 2:
            if question_emb.size(0) == 1 and batch_size > 1:
                question_emb = question_emb.expand(batch_size, -1)
            elif question_emb.size(0) != batch_size:
                raise ValueError(f"question_emb batch mismatch: {question_emb.size(0)} vs {batch_size}")
        else:
            raise ValueError(f"Unsupported question_emb rank={question_emb.dim()}")
        return question_emb.to(device)

    @staticmethod
    def _compute_edge_batch(
        edge_index: torch.Tensor,
        *,
        node_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_ptr.numel() != num_graphs + 1:
            raise ValueError(f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}")
        # Map each edge to its graph id via head node global index and node_ptr boundaries.
        # NOTE: `right=True` is required to assign boundary nodes correctly:
        # for a prefix-sum ptr, node indices in graph g satisfy ptr[g] <= i < ptr[g+1].
        # If we used `right=False`, a head index exactly equal to ptr[g] (the first node of graph g)
        # would be bucketized into graph g-1, breaking the edge_ptr segmentation invariant.
        edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
        if edge_batch.numel() > 1 and not bool((edge_batch[:-1] <= edge_batch[1:]).all().item()):
            inv = torch.nonzero(edge_batch[:-1] > edge_batch[1:], as_tuple=False).view(-1)
            preview = inv[:5].detach().cpu().tolist()
            raise ValueError(
                "edge_batch is not non-decreasing along the flattened edge list, which breaks the per-graph "
                "edge_ptr slice semantics required by GraphEnv; "
                f"first_inversions={preview}. "
                "Ensure edges are concatenated per-graph (PyG Batch) or sort edges by graph before building edge_ptr."
            )
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        edge_ptr[1:] = edge_counts.cumsum(0)
        return edge_batch, edge_ptr

    @staticmethod
    def _pad_start_entities(
        start_entity_ids: torch.Tensor,
        start_ptr: torch.Tensor,
        *,
        num_graphs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if start_ptr.numel() != num_graphs + 1:
            raise ValueError("start_entity_ptr length mismatch; expected one offset per graph.")
        counts = start_ptr[1:] - start_ptr[:-1]
        if counts.numel() != num_graphs:
            raise ValueError("start_entity_ptr length mismatch; expected one count per graph.")
        if (counts <= 0).any():
            missing = torch.nonzero(counts <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(f"start_entity_ids must be non-empty per graph; missing at indices {missing}")
        max_start = int(counts.max().item()) if num_graphs > 0 else 0
        padded = torch.full((num_graphs, max_start), -1, dtype=torch.long, device=device)
        mask = torch.zeros((num_graphs, max_start), dtype=torch.bool, device=device)
        batch_ids = torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)
        local_idx = torch.arange(start_entity_ids.numel(), device=device) - start_ptr[batch_ids]
        padded[batch_ids, local_idx] = start_entity_ids
        mask[batch_ids, local_idx] = True
        return padded, mask


@dataclass(frozen=True)
class EmbedOutputs:
    edge_tokens: torch.Tensor
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    heads_global: torch.Tensor
    tails_global: torch.Tensor
    node_tokens: torch.Tensor
    question_tokens: torch.Tensor
    start_entity_ids: torch.Tensor
    start_entity_mask: torch.Tensor

    def validate_device(self, device: torch.device) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                raise RuntimeError(f"EmbedOutputs device mismatch: {name}.device={value.device}, expected={device}")


__all__ = ["GraphEmbedder", "EmbedOutputs"]
