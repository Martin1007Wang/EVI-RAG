from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.data.components import SharedDataResources
from src.models.components.projections import EmbeddingProjector

logger = logging.getLogger(__name__)


class GraphEmbedder(nn.Module):
    """负责实体/关系/问题投影，输出确定性的 edge/question/start 表示。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        proj_dropout: float,
        projector_checkpoint: Optional[str],
        freeze_projectors: bool,
        kge_interaction: str = "concat",
        projector_key_prefixes: Optional[list[str]] = None,
        use_gfn_projectors: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj_dropout = float(proj_dropout)
        self.projector_checkpoint = projector_checkpoint
        self.freeze_projectors = bool(freeze_projectors)
        self.kge_interaction = str(kge_interaction).strip().lower()
        self.use_gfn_projectors = bool(use_gfn_projectors)
        base_prefixes = projector_key_prefixes or [
            "model._orig_mod",
            "model",
            "",
        ]
        self.projector_key_prefixes = self._augment_prefixes_with_retriever(base_prefixes)

        self._shared_resources: Optional[SharedDataResources] = None
        self._global_embeddings = None
        self._entity_dim: Optional[int] = None
        self._relation_dim: Optional[int] = None
        self._question_dim: Optional[int] = None
        self._num_entities: Optional[int] = None
        self._num_relations: Optional[int] = None

        self._retriever_extractor_frozen: bool = False
        self.retriever_entity_projector: Optional[EmbeddingProjector] = None
        self.retriever_relation_projector: Optional[EmbeddingProjector] = None
        self.retriever_query_projector: Optional[EmbeddingProjector] = None
        self.entity_projector: Optional[nn.Module] = None
        self.relation_projector: Optional[nn.Module] = None
        self.query_projector: Optional[nn.Module] = None
        self.retriever_extractor: Optional[nn.Module] = None
        self.edge_adapter: Optional[nn.Module] = None

    @staticmethod
    def _is_trainable(module: Optional[nn.Module]) -> bool:
        return module is not None and any(param.requires_grad for param in module.parameters())

    @staticmethod
    def _augment_prefixes_with_retriever(prefixes: list[str]) -> list[str]:
        """Normalize prefixes and ensure retriever.* variants are also tried."""
        normalized: list[str] = []
        for p in prefixes:
            p_clean = p.rstrip(".")
            if p_clean not in normalized:
                normalized.append(p_clean)
        augmented = list(normalized)
        for p in normalized:
            retriever_pref = f"retriever.{p}" if p else "retriever"
            if retriever_pref not in augmented:
                augmented.append(retriever_pref)
        return augmented

    @property
    def entity_dim(self) -> int:
        if self._entity_dim is None:
            raise RuntimeError("GraphEmbedder not initialized; call setup() first.")
        return self._entity_dim

    @property
    def relation_dim(self) -> int:
        if self._relation_dim is None:
            raise RuntimeError("GraphEmbedder not initialized; call setup() first.")
        return self._relation_dim

    def setup(self, resources_cfg: SharedDataResources | Dict[str, Any], device: Optional[torch.device] = None) -> None:
        if self._shared_resources is not None:
            return
        if isinstance(resources_cfg, SharedDataResources):
            self._shared_resources = resources_cfg
        else:
            self._shared_resources = SharedDataResources(**resources_cfg)
        self._global_embeddings = self._shared_resources.global_embeddings
        self._entity_dim = self._global_embeddings.entity_embeddings.size(1)
        self._relation_dim = self._global_embeddings.relation_embeddings.size(1)
        if self.kge_interaction not in {"concat", "add", "mul", "sub"}:
            raise ValueError(f"Unsupported kge_interaction={self.kge_interaction}; expected concat/add/mul/sub.")
        self._num_entities = self._global_embeddings.entity_embeddings.size(0)
        self._num_relations = self._global_embeddings.relation_embeddings.size(0)
        self._init_projection_layers()
        self._load_projector_checkpoint()
        if self.freeze_projectors:
            self._freeze_retriever_projectors()
            # 不切换 eval 模式，仅冻结梯度，避免下游 module.eval() 警告
        if device is not None:
            self.to(device)

    def embed_batch(self, batch: Any, *, device: torch.device) -> "EmbedOutputs":
        if self._shared_resources is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before embed_batch.")

        edge_index = batch.edge_index
        node_ptr = batch.ptr
        num_graphs = int(node_ptr.numel() - 1)

        edge_batch, edge_ptr = self._compute_edge_batch(edge_index, batch=batch, num_graphs=num_graphs, device=device)
        edge_relations = batch.edge_attr
        edge_labels = batch.edge_labels
        edge_scores = getattr(batch, "edge_scores", None)
        if edge_scores is None:
            raise ValueError("Batch missing edge_scores; g_agent cache must include retriever scores for soft prior.")
        path_mask, path_exists = self._build_path_mask(batch=batch, edge_index=edge_index, device=device)

        question_raw = self._prepare_question_raw(batch.question_emb, batch_size=num_graphs, device=device)
        node_global_ids = batch.node_global_ids
        node_embedding_ids = getattr(batch, "node_embedding_ids", None)
        node_embedding_ids = node_embedding_ids
        if node_embedding_ids.shape != node_global_ids.shape:
            raise ValueError("node_embedding_ids shape mismatch with node_global_ids in g_agent batch.")
        node_tokens = self._lookup_entities(node_embedding_ids)
        node_raw_tokens = self._lookup_entities_raw(node_embedding_ids)
        question_tokens = self._project_question_tokens(question_raw)
        if torch.is_grad_enabled() and self._is_trainable(self.entity_projector) and not node_tokens.requires_grad:
            raise RuntimeError("node_tokens does not require grad; entity_projector must remain trainable.")
        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]
        edge_tokens = self._build_edge_tokens(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            question_raw=question_raw,
            node_raw_tokens=node_raw_tokens,
            question_tokens=question_tokens,
            node_tokens=node_tokens,
        )
        if torch.is_grad_enabled() and self._is_trainable(self.edge_adapter) and not edge_tokens.requires_grad:
            raise RuntimeError("edge_tokens does not require grad; ensure retriever_extractor/edge_adapter are not under no_grad.")
        start_entity_ids, start_entity_mask = self._pad_start_entities(
            batch.start_entity_ids.to(device),
            batch._slice_dict["start_entity_ids"].to(device),
            num_graphs=num_graphs,
            device=device,
        )
        start_node_ptr = batch._slice_dict.get("start_node_locals")
        if start_node_ptr is None:
            raise ValueError("Batch missing start_node_locals slice info; g_agent cache may be corrupt.")
        return EmbedOutputs(
            edge_tokens=edge_tokens,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_labels=edge_labels,
            edge_scores=edge_scores,
            path_mask=path_mask,
            path_exists=path_exists,
            heads_global=heads_global,
            tails_global=tails_global,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            start_entity_ids=start_entity_ids,
            start_entity_mask=start_entity_mask,
        )

    # --- internals -------------------------------------------------------
    def _init_projection_layers(self) -> None:
        if self._entity_dim is None or self._relation_dim is None:
            raise ValueError("Embedding dimensions are not initialized; call setup() first.")
        if self.use_gfn_projectors:
            self.entity_projector = self._build_projection_head(self._entity_dim)
            self.relation_projector = self._build_projection_head(self._relation_dim)
        else:
            if self.hidden_dim != self._entity_dim or self.hidden_dim != self._relation_dim:
                raise ValueError(
                    f"use_gfn_projectors=False 需要 hidden_dim 与 retriever projector 维度一致，"
                    f"当前 hidden_dim={self.hidden_dim}, entity_dim={self._entity_dim}, relation_dim={self._relation_dim}"
                )
            self.entity_projector = nn.Identity()
            self.relation_projector = nn.Identity()
        edge_in_dim = self._edge_feature_dim()
        if self.use_gfn_projectors:
            # 直接使用 GFN 投影后的 q/h/r/t 组合特征，edge_adapter 负责可训练映射。
            self.retriever_extractor = None
            self.edge_adapter = nn.Sequential(
                nn.LayerNorm(edge_in_dim),
                nn.Linear(edge_in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.proj_dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            # Frozen retriever extractor (结构与 retriever DenseFeatureExtractor 对齐)。
            self.retriever_extractor = nn.Sequential(
                nn.Linear(edge_in_dim, self._entity_dim),
                nn.ReLU(),
                nn.Dropout(self.proj_dropout),
                nn.Linear(self._entity_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.proj_dropout),
            )
            for p in self.retriever_extractor.parameters():
                p.requires_grad = False
            self._retriever_extractor_frozen = True
            # retriever_extractor 已产出 hidden_dim 特征，避免重复映射。
            self.edge_adapter = nn.Identity()

    def _init_query_projection(self, question_dim: int) -> None:
        if question_dim <= 0:
            raise ValueError(f"Invalid question_dim={question_dim}")
        if self.use_gfn_projectors:
            self.query_projector = self._build_projection_head(question_dim)
        else:
            if question_dim != self.hidden_dim:
                raise ValueError(
                    f"use_gfn_projectors=False 需要 question_dim==hidden_dim，当前 question_dim={question_dim}, hidden_dim={self.hidden_dim}"
                )
            self.query_projector = nn.Identity()

    def _build_projection_head(self, in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.proj_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _compute_edge_batch(
        self,
        edge_index: torch.Tensor,
        *,
        batch: Any,
        num_graphs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_batch = batch.batch[edge_index[0]].to(device)
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        edge_ptr[1:] = edge_counts.cumsum(0)
        return edge_batch, edge_ptr

    def _build_path_mask(
        self,
        *,
        batch: Any,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        path_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
        path_mask[batch.gt_path_edge_local_ids.to(device)] = True
        path_exists = batch.gt_path_exists.view(-1)
        return path_mask, path_exists

    def _edge_feature_dim(self) -> int:
        base_dim = self.hidden_dim if self.use_gfn_projectors else self._entity_dim
        if self.kge_interaction == "concat":
            return 4 * base_dim  # q, h, r, t
        if self.kge_interaction in {"add", "mul", "sub"}:
            return base_dim  # 同维度运算
        raise ValueError(f"Unsupported kge_interaction={self.kge_interaction}")

    def _build_edge_tokens(
        self,
        *,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        question_raw: torch.Tensor,
        node_raw_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # question_raw 已按图顺序排列，edge_batch 对应 edge_index[0] 的图归属。
        if self.use_gfn_projectors:
            head_emb = node_tokens[edge_index[0]]
            tail_emb = node_tokens[edge_index[1]]
            rel_emb = self._lookup_relations(edge_relations)
            edge_feats = self._compose_edge_features(question_tokens[edge_batch], head_emb, rel_emb, tail_emb)
            return self.edge_adapter(edge_feats)

        head_emb = node_raw_tokens[edge_index[0]]
        tail_emb = node_raw_tokens[edge_index[1]]
        rel_emb = self._lookup_relations_raw(edge_relations)
        edge_feats = self._compose_edge_features(question_raw[edge_batch], head_emb, rel_emb, tail_emb)
        if self.retriever_extractor is None:
            raise RuntimeError("retriever_extractor is not initialized.")
        frozen_feats = self.retriever_extractor(edge_feats)
        return self.edge_adapter(frozen_feats)

    def _compose_edge_features(
        self,
        question: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        if self.kge_interaction == "concat":
            return torch.cat([question, head, relation, tail], dim=-1)
        if self.kge_interaction == "add":
            return question + head + relation + tail
        if self.kge_interaction == "mul":
            return question * head * relation * tail
        if self.kge_interaction == "sub":
            return question + head + relation - tail
        raise ValueError(f"Unsupported kge_interaction={self.kge_interaction}")

    def _prepare_question_raw(self, question_emb: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
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

        tokens = question_emb.to(device)
        if self.retriever_query_projector is not None:
            tokens = self.retriever_query_projector(tokens)
        if tokens.size(1) != self._entity_dim:
            raise ValueError(f"question_emb dim mismatch: got {tokens.size(1)}, expected {self._entity_dim} (retriever space).")
        return tokens

    def _project_question_tokens(self, question_raw: torch.Tensor) -> torch.Tensor:
        if self.query_projector is None:
            self._init_query_projection(question_dim=int(question_raw.size(1)))
        if self.query_projector is not None and next(self.query_projector.parameters(), None) is not None:
            self.query_projector = self.query_projector.to(question_raw.device)
        if self._question_dim is None:
            self._question_dim = int(question_raw.size(1))
        elif int(question_raw.size(1)) != self._question_dim:
            raise ValueError(f"Inconsistent question_dim: current {question_raw.size(1)} vs expected {self._question_dim}")
        return self.query_projector(question_raw)  # type: ignore[operator]

    def _load_projector_checkpoint(self) -> None:
        if self.projector_checkpoint is None:
            raise ValueError("GraphEmbedder 需要 projector_checkpoint 指向 retriever 投影权重。")
        ckpt_path = Path(self.projector_checkpoint).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"projector_checkpoint 不存在: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.retriever_entity_projector = self._load_single_projector(
            state_dict,
            name="entity_proj",
            prefixes=self.projector_key_prefixes,
            expected_in_dim=self._entity_dim,
            expected_out_dim=self._entity_dim,
        )
        self.retriever_relation_projector = self._load_single_projector(
            state_dict,
            name="relation_proj",
            prefixes=self.projector_key_prefixes,
            expected_in_dim=self._relation_dim,
            expected_out_dim=self._relation_dim,
        )
        self.retriever_query_projector = self._load_single_projector(
            state_dict,
            name="query_proj",
            prefixes=self.projector_key_prefixes,
            expected_in_dim=None,
            expected_out_dim=None,
        )
        if not self.use_gfn_projectors:
            self._load_edge_projector_from_ckpt(state_dict)

    def _load_single_projector(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        name: str,
        prefixes: list[str],
        expected_in_dim: Optional[int],
        expected_out_dim: Optional[int],
    ) -> EmbeddingProjector:
        weight_key = None
        bias_key = None
        for prefix in prefixes:
            prefix_clean = prefix.rstrip(".")
            candidate_weight = f"{prefix_clean}.{name}.network.0.weight" if prefix_clean else f"{name}.network.0.weight"
            candidate_bias = f"{prefix_clean}.{name}.network.0.bias" if prefix_clean else f"{name}.network.0.bias"
            if candidate_weight in state_dict and candidate_bias in state_dict:
                weight_key, bias_key = candidate_weight, candidate_bias
                break
        if weight_key is None or bias_key is None:
            raise KeyError(f"在 checkpoint 中未找到 {name} 权重，prefixes={prefixes}")
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        if expected_out_dim is not None and out_dim != expected_out_dim:
            raise ValueError(f"{name} projector 输出维度不匹配: ckpt={out_dim} vs expected={expected_out_dim}")
        if expected_in_dim is not None and in_dim != expected_in_dim:
            raise ValueError(f"{name} projector 输入维度不匹配: ckpt={in_dim} vs expected={expected_in_dim}")
        projector = EmbeddingProjector(output_dim=out_dim, input_dim=in_dim, finetune=False)
        load_state = {"network.0.weight": weight, "network.0.bias": bias}
        missing, unexpected = projector.load_state_dict(load_state, strict=False)
        if missing:
            raise RuntimeError(f"{name} projector 缺失权重: {missing}")
        if unexpected and any("network.0" not in key for key in unexpected):
            raise RuntimeError(f"{name} projector 出现多余键: {unexpected}")
        projector.eval()
        return projector

    def _freeze_retriever_projectors(self) -> None:
        for module in [self.retriever_entity_projector, self.retriever_relation_projector, self.retriever_query_projector]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def _load_edge_projector_from_ckpt(self, state_dict: Dict[str, torch.Tensor]) -> None:
        edge_in_dim = self._edge_feature_dim()
        w0_key = b0_key = w1_key = b1_key = None
        for prefix in self.projector_key_prefixes:
            prefix_clean = prefix.rstrip(".")
            base = f"{prefix_clean}.feature_extractor.network" if prefix_clean else "feature_extractor.network"
            candidate = (
                f"{base}.0.weight",
                f"{base}.0.bias",
                f"{base}.3.weight",
                f"{base}.3.bias",
            )
            if all(k in state_dict for k in candidate):
                w0_key, b0_key, w1_key, b1_key = candidate
                break
        if w0_key is None:
            logger.warning(
                "edge_projector weights not found in retriever checkpoint (prefixes=%s); "
                "falling back to trainable edge_projector.",
                self.projector_key_prefixes,
            )
            return  # Optional: keep trainable edge_projector when ckpt missing

        w0, b0, w1, b1 = state_dict[w0_key], state_dict[b0_key], state_dict[w1_key], state_dict[b1_key]
        if w0.shape[1] != edge_in_dim or w0.shape[0] != self._entity_dim:
            raise ValueError(
                f"edge_projector layer0 shape mismatch: ckpt {tuple(w0.shape)} vs expected ({self._entity_dim},{edge_in_dim})"
            )
        if w1.shape[1] != self._entity_dim or w1.shape[0] != self.hidden_dim:
            raise ValueError(
                f"edge_projector layer1 shape mismatch: ckpt {tuple(w1.shape)} vs expected ({self.hidden_dim},{self._entity_dim})"
            )

        edge_state = self.retriever_extractor.state_dict()
        edge_state["0.weight"] = w0
        edge_state["0.bias"] = b0
        edge_state["3.weight"] = w1
        edge_state["3.bias"] = b1
        missing, unexpected = self.retriever_extractor.load_state_dict(edge_state, strict=False)
        if missing:
            raise RuntimeError(f"edge_projector missing keys when loading retriever dense head: {missing}")
        if unexpected and any("network" not in k for k in unexpected):
            raise RuntimeError(f"edge_projector unexpected keys: {unexpected}")
        self.retriever_extractor.eval()
        for p in self.retriever_extractor.parameters():
            p.requires_grad = False
        self._retriever_extractor_frozen = True
        logger.info("edge_projector loaded and frozen from retriever checkpoint using prefix=%s", prefix_clean)

    def _lookup_entities_raw(self, embedding_ids: torch.Tensor) -> torch.Tensor:
        if self._num_entities is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        embeddings = self._global_embeddings.get_entity_embeddings(embedding_ids)  # type: ignore[call-arg]
        if self.retriever_entity_projector is not None:
            embeddings = self.retriever_entity_projector(embeddings)
        return embeddings

    def _lookup_entities(self, embedding_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self._lookup_entities_raw(embedding_ids)
        return self.entity_projector(embeddings)  # type: ignore[operator]

    def _lookup_relations(self, ids: torch.Tensor) -> torch.Tensor:
        embeddings = self._lookup_relations_raw(ids)
        return self.relation_projector(embeddings)  # type: ignore[operator]

    def _lookup_relations_raw(self, ids: torch.Tensor) -> torch.Tensor:
        if self._num_relations is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id >= self._num_relations:
                raise ValueError(f"Relation ids out of range: min={min_id} max={max_id} valid=[0,{self._num_relations - 1}]")
        embeddings = self._global_embeddings.get_relation_embeddings(ids)  # type: ignore[call-arg]
        if self.retriever_relation_projector is not None:
            embeddings = self.retriever_relation_projector(embeddings)
        return embeddings

    def _pad_start_entities(
        self,
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
    edge_labels: torch.Tensor
    edge_scores: torch.Tensor
    path_mask: torch.Tensor
    path_exists: torch.Tensor
    heads_global: torch.Tensor
    tails_global: torch.Tensor
    node_tokens: torch.Tensor
    question_tokens: torch.Tensor
    start_entity_ids: torch.Tensor
    start_entity_mask: torch.Tensor


__all__ = ["GraphEmbedder", "EmbedOutputs"]
