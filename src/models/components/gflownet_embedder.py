from __future__ import annotations

import logging
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from src.data.components import SharedDataResources
from src.models.components.graph import DDE
from src.models.components.heads import DenseFeatureExtractor
from src.models.components.projections import EmbeddingProjector

logger = logging.getLogger(__name__)


class GraphEmbedder(nn.Module):
    """
    GFlowNet 图嵌入器：复用 Retriever checkpoint 中的投影器 + DenseFeatureExtractor 权重，
    并严格对齐 SubgraphRAG parity 的结构特征（topic seed + DDE forward/reverse）。
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        proj_dropout: float,
        projector_checkpoint: str,
        projector_key_prefixes: Optional[Sequence[str]] = None,
        freeze_retriever: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.proj_dropout = float(proj_dropout)
        self.projector_checkpoint = str(projector_checkpoint)
        self.freeze_retriever = bool(freeze_retriever)

        self.projector_key_prefixes = self._normalize_prefixes(projector_key_prefixes or ["model._orig_mod", "model", ""])

        self._shared_resources: Optional[SharedDataResources] = None
        self._global_embeddings = None
        self._entity_dim: Optional[int] = None
        self._relation_dim: Optional[int] = None
        self._num_entities: Optional[int] = None
        self._num_relations: Optional[int] = None

        self.use_topic_pe: bool = True
        self.num_topics: int = 2
        self.num_rounds: int = 0
        self.num_reverse_rounds: int = 0
        self._edge_in_dim: Optional[int] = None

        self.entity_proj: Optional[EmbeddingProjector] = None
        self.relation_proj: Optional[EmbeddingProjector] = None
        self.query_proj: Optional[EmbeddingProjector] = None
        self.non_text_entity_emb: Optional[nn.Embedding] = None
        self.dde: Optional[DDE] = None
        self.fused_dropout: Optional[nn.Dropout] = None
        self.feature_extractor: Optional[DenseFeatureExtractor] = None

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

        state_dict = self._load_checkpoint_state()
        self._load_retriever_components(state_dict)
        if self.freeze_retriever:
            self._freeze_modules()
        if device is not None:
            self.to(device)

    def embed_batch(self, batch: Any, *, device: torch.device) -> "EmbedOutputs":
        if self._shared_resources is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before embed_batch.")
        if self.feature_extractor is None or self.entity_proj is None or self.relation_proj is None or self.query_proj is None:
            raise RuntimeError("GraphEmbedder not initialized; call setup() first.")

        edge_index = batch.edge_index.to(device)
        node_ptr = batch.ptr.to(device)
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("Batch must contain at least one graph.")

        edge_batch, edge_ptr = self._compute_edge_batch(edge_index, batch=batch, num_graphs=num_graphs, device=device)
        edge_relations = batch.edge_attr.to(device)
        edge_labels = batch.edge_labels.to(device)
        edge_scores = batch.edge_scores.to(device)
        path_mask, path_exists = self._build_path_mask(batch=batch, edge_index=edge_index, device=device)

        question_raw = self._prepare_question_embeddings(batch.question_emb, batch_size=num_graphs, device=device)
        question_tokens = self.query_proj(question_raw)

        node_embedding_ids = getattr(batch, "node_embedding_ids", None)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if node_embedding_ids is None or node_global_ids is None:
            raise ValueError("Batch missing node_embedding_ids/node_global_ids; g_agent cache must include them.")
        node_embedding_ids = node_embedding_ids.to(device)
        node_global_ids = node_global_ids.to(device)
        if node_embedding_ids.shape != node_global_ids.shape:
            raise ValueError("node_embedding_ids shape mismatch with node_global_ids in g_agent batch.")

        node_embeddings = self._lookup_entities(node_embedding_ids)
        node_tokens = self.entity_proj(node_embeddings)
        node_tokens = self._apply_non_text_override(node_tokens=node_tokens, node_embedding_ids=node_embedding_ids)

        relation_embeddings = self._lookup_relations(edge_relations)
        relation_tokens = self.relation_proj(relation_embeddings)

        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]

        semantic = torch.cat(
            [
                question_tokens[edge_batch],
                node_tokens[edge_index[0]],
                relation_tokens,
                node_tokens[edge_index[1]],
            ],
            dim=-1,
        )
        if self.dde is not None:
            topic_one_hot = self._build_topic_one_hot(batch=batch, num_nodes_total=int(node_global_ids.numel()), device=device)
            struct = self._build_structure_features(
                topic_one_hot=topic_one_hot,
                edge_index=edge_index,
                head_idx=edge_index[0],
                tail_idx=edge_index[1],
                device=device,
            )
            fused = torch.cat([semantic, struct], dim=-1)
        else:
            fused = semantic

        if self._edge_in_dim is None:
            raise RuntimeError("edge_in_dim is not initialized; setup() must load retriever feature_extractor weights.")
        if int(fused.size(1)) != int(self._edge_in_dim):
            raise ValueError(f"edge feature dim mismatch: got {int(fused.size(1))}, expected {int(self._edge_in_dim)}")

        if self.fused_dropout is None:
            raise RuntimeError("fused_dropout is not initialized.")
        edge_tokens = self.feature_extractor(self.fused_dropout(fused))

        start_entity_ids, start_entity_mask = self._pad_start_entities(
            batch.start_entity_ids.to(device),
            batch._slice_dict["start_entity_ids"].to(device),
            num_graphs=num_graphs,
            device=device,
        )
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

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def _load_checkpoint_state(self) -> Dict[str, torch.Tensor]:
        ckpt_path = Path(self.projector_checkpoint).expanduser()
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
        meta_key = self._find_first_match(state_dict, "parity_meta")
        meta = state_dict[meta_key].view(-1).to(dtype=torch.long)
        if meta.numel() != 4:
            raise ValueError(f"parity_meta must have 4 entries, got shape {tuple(meta.shape)}")
        self.use_topic_pe = bool(int(meta[0].item()))
        self.num_topics = int(meta[1].item())
        self.num_rounds = int(meta[2].item())
        self.num_reverse_rounds = int(meta[3].item())
        if self.use_topic_pe:
            self.dde = DDE(num_rounds=self.num_rounds, num_reverse_rounds=self.num_reverse_rounds)
        else:
            self.dde = None

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
        struct_dim_expected = 0
        if self.use_topic_pe:
            struct_dim_expected = 2 * self.num_topics * (1 + self.num_rounds + self.num_reverse_rounds)
        if edge_in_dim != semantic_dim + struct_dim_expected:
            raise ValueError(
                f"feature_extractor input_dim mismatch: ckpt={edge_in_dim} vs expected={semantic_dim + struct_dim_expected} "
                f"(semantic={semantic_dim}, struct={struct_dim_expected})"
            )
        self._edge_in_dim = edge_in_dim
        self.fused_dropout = nn.Dropout(self.proj_dropout)
        self.feature_extractor = DenseFeatureExtractor(
            input_dim=edge_in_dim,
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            dropout_p=self.proj_dropout,
        )
        fe_state = self.feature_extractor.state_dict()
        fe_state["network.0.weight"] = w0
        fe_state["network.0.bias"] = b0
        fe_state["network.3.weight"] = w1
        fe_state["network.3.bias"] = b1
        missing, unexpected = self.feature_extractor.load_state_dict(fe_state, strict=False)
        if missing:
            raise RuntimeError(f"feature_extractor missing keys: {missing}")
        if unexpected and any("network" not in k for k in unexpected):
            raise RuntimeError(f"feature_extractor unexpected keys: {unexpected}")

    def _freeze_modules(self) -> None:
        for module in (self.entity_proj, self.relation_proj, self.query_proj, self.non_text_entity_emb, self.feature_extractor):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Embedding + feature construction
    # ------------------------------------------------------------------ #
    def _lookup_entities(self, embedding_ids: torch.Tensor) -> torch.Tensor:
        if self._num_entities is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        return self._global_embeddings.get_entity_embeddings(embedding_ids)  # type: ignore[call-arg]

    def _lookup_relations(self, ids: torch.Tensor) -> torch.Tensor:
        if self._num_relations is None:
            raise RuntimeError("GraphEmbedder.setup() must be called before use.")
        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id >= self._num_relations:
                raise ValueError(f"Relation ids out of range: min={min_id} max={max_id} valid=[0,{self._num_relations - 1}]")
        return self._global_embeddings.get_relation_embeddings(ids)  # type: ignore[call-arg]

    def _apply_non_text_override(self, *, node_tokens: torch.Tensor, node_embedding_ids: torch.Tensor) -> torch.Tensor:
        if self.non_text_entity_emb is None or self.entity_proj is None:
            return node_tokens
        non_text_mask = node_embedding_ids == 0
        if not bool(non_text_mask.any().item()):
            return node_tokens
        non_text_proj = self.entity_proj(self.non_text_entity_emb.weight.to(device=node_tokens.device))[0].to(dtype=node_tokens.dtype)
        return torch.where(non_text_mask.unsqueeze(-1), non_text_proj.unsqueeze(0), node_tokens)

    def _build_topic_one_hot(self, *, batch: Any, num_nodes_total: int, device: torch.device) -> torch.Tensor:
        topic = torch.zeros((num_nodes_total, self.num_topics), device=device, dtype=torch.float32)
        start_node_locals = getattr(batch, "start_node_locals", None)
        if start_node_locals is None or start_node_locals.numel() == 0:
            raise ValueError("g_agent batch missing non-empty start_node_locals; cannot build topic_one_hot.")
        topic[start_node_locals.to(device).long(), 0] = 1.0
        return topic

    def _build_structure_features(
        self,
        *,
        topic_one_hot: torch.Tensor,
        edge_index: torch.Tensor,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.dde is None:
            raise RuntimeError("DDE is not initialized but structure features were requested.")
        feats = [topic_one_hot]
        feats.extend(self.dde(topic_one_hot, edge_index, edge_index.flip(0)))
        stacked = torch.stack(feats, dim=-1)
        node_struct = stacked.reshape(stacked.size(0), -1)
        head_struct = node_struct[head_idx]
        tail_struct = node_struct[tail_idx]
        return torch.cat([head_struct, tail_struct], dim=-1)

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
        batch: Any,
        num_graphs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_batch = batch.batch[edge_index[0]].to(device)
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        edge_ptr[1:] = edge_counts.cumsum(0)
        return edge_batch, edge_ptr

    @staticmethod
    def _build_path_mask(
        *,
        batch: Any,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        path_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
        path_mask[batch.gt_path_edge_local_ids.to(device)] = True
        path_exists = batch.gt_path_exists.view(-1)
        return path_mask, path_exists

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
