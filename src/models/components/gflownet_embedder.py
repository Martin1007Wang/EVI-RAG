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
from src.models.components.graph import (
    DDE,
    STRUCT_MODE_DIFFUSION,
    STRUCT_MODE_DISTANCE,
    STRUCT_MODE_NONE,
)
from src.utils.graph_utils import compute_edge_batch
from src.models.components.projections import EmbeddingProjector

logger = logging.getLogger(__name__)

HASH_CHUNK_SIZE = 4 * 1024 * 1024

_EDGE_MODE_CONCAT = "concat"
_EDGE_MODE_GEOMETRY = "geometry"
_DIST_SCALAR_DIM = 1
_STATE_NET_DROPOUT_P = 0.0
_EDGE_DIR_COUNT = 2


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
        projector_checkpoint: str | None = None,
        projector_key_prefixes: Optional[Sequence[str]] = None,
        allow_deferred_init: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
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
        self._struct_mode: int = STRUCT_MODE_NONE
        self._num_topics: int = 0
        self._struct_dim: int = 0
        self._dde: Optional[DDE] = None
        self._distance_max_hops: int = 0
        self._distance_emb_dim: int = 0

        self.entity_proj: Optional[EmbeddingProjector] = None
        self.relation_proj: Optional[EmbeddingProjector] = None
        self.query_proj: Optional[EmbeddingProjector] = None
        self.non_text_entity_emb: Optional[nn.Embedding] = None
        self.q_gate: Optional[nn.Sequential] = None
        self.q_bias: Optional[nn.Sequential] = None
        self.struct_proj: Optional[nn.Sequential] = None
        self.struct_gate_net: Optional[nn.Sequential] = None
        self.state_net: Optional[nn.Sequential] = None
        self.edge_adapter: Optional[nn.Module] = None
        self.edge_adapter_residual: Optional[nn.Linear] = None
        self.edge_adapter_residual_gate: Optional[nn.Parameter] = None
        self.edge_score_proj: Optional[nn.Linear] = None
        self._edge_mode: str = _EDGE_MODE_CONCAT
        self._use_distmult: bool = False
        self._use_transe: bool = False
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
        if self.entity_proj is None or self.relation_proj is None or self.query_proj is None:
            raise RuntimeError("GraphEmbedder not initialized; call setup() first.")
        if self._edge_mode == _EDGE_MODE_CONCAT and self.edge_adapter is None:
            raise RuntimeError("GraphEmbedder not initialized; edge_adapter missing for concat mode.")
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
        node_struct_raw: Optional[torch.Tensor] = None
        if self._use_topic_pe:
            struct, node_struct_raw = self._build_structure_features(
                batch=batch,
                edge_index=edge_index,
                num_nodes=int(node_tokens.size(0)),
                device=device,
                dtype=q_edge.dtype,
            )
        struct_swap = self._swap_struct_edge(struct) if struct is not None else None
        if self._edge_mode == _EDGE_MODE_GEOMETRY:
            edge_tokens_fwd = self._edge_tokens_from_geometry(
                q_edge=q_edge,
                head_edge=head_edge,
                relation_edge=relation_tokens,
                tail_edge=tail_edge,
                struct_edge=struct,
            )
            edge_tokens_bwd = self._edge_tokens_from_geometry(
                q_edge=q_edge,
                head_edge=tail_edge,
                relation_edge=relation_tokens,
                tail_edge=head_edge,
                struct_edge=struct_swap,
            )
        else:
            edge_tokens_fwd = self._edge_adapter_forward_parts(
                q_edge=q_edge,
                head_edge=head_edge,
                relation_edge=relation_tokens,
                tail_edge=tail_edge,
                struct_edge=struct,
            )
            edge_tokens_bwd = self._edge_adapter_forward_parts(
                q_edge=q_edge,
                head_edge=tail_edge,
                relation_edge=relation_tokens,
                tail_edge=head_edge,
                struct_edge=struct_swap,
            )
        edge_tokens = self._combine_undirected_edge_tokens(edge_tokens_fwd, edge_tokens_bwd)
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
            node_struct_raw=node_struct_raw,
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

    def _load_non_text_entity_emb(self, state_dict: Dict[str, torch.Tensor]) -> None:
        non_text_key = self._find_first_match(state_dict, "non_text_entity_emb.weight")
        non_text_weight = state_dict[non_text_key]
        expected_shape = (1, self.hidden_dim)
        if tuple(non_text_weight.shape) != expected_shape:
            raise ValueError(
                f"non_text_entity_emb.weight shape mismatch: got {tuple(non_text_weight.shape)} expected {expected_shape}"
            )
        self.non_text_entity_emb = nn.Embedding(1, self.hidden_dim)
        self.non_text_entity_emb.weight.data.copy_(non_text_weight)

    def _try_load_feature_extractor(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]]:
        try:
            w0_key = self._find_first_match(state_dict, "feature_extractor.network.0.weight")
            b0_key = self._find_first_match(state_dict, "feature_extractor.network.0.bias")
            w1_key = self._find_first_match(state_dict, "feature_extractor.network.3.weight")
            b1_key = self._find_first_match(state_dict, "feature_extractor.network.3.bias")
        except KeyError:
            return None
        w0, b0, w1, b1 = state_dict[w0_key], state_dict[b0_key], state_dict[w1_key], state_dict[b1_key]
        emb_dim = int(w0.shape[0])
        edge_in_dim = int(w0.shape[1])
        hidden_dim_ckpt = int(w1.shape[0])
        return w0, b0, w1, b1, emb_dim, edge_in_dim, hidden_dim_ckpt

    def _load_linear_layer(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        suffix: str,
        in_dim: int,
        out_dim: int,
        name: str,
    ) -> nn.Linear:
        w_key, b_key = self._find_pair(state_dict, suffix)
        weight = state_dict[w_key]
        bias = state_dict[b_key]
        if tuple(weight.shape) != (out_dim, in_dim):
            raise ValueError(f"{name} weight shape mismatch: got {tuple(weight.shape)} expected {(out_dim, in_dim)}")
        layer = nn.Linear(in_dim, out_dim)
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        return layer

    def _load_struct_proj(self, state_dict: Dict[str, torch.Tensor]) -> nn.Sequential:
        w_key, b_key = self._find_pair(state_dict, "struct_proj.0")
        weight = state_dict[w_key]
        bias = state_dict[b_key]
        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        norm = nn.LayerNorm(out_dim)
        n_w_key, n_b_key = self._find_pair(state_dict, "struct_proj.1")
        norm.weight.data.copy_(state_dict[n_w_key])
        norm.bias.data.copy_(state_dict[n_b_key])
        return nn.Sequential(linear, norm, nn.GELU())

    def _load_state_net(self, state_dict: Dict[str, torch.Tensor]) -> nn.Sequential:
        w0_key, b0_key = self._find_pair(state_dict, "state_net.0")
        w1_key, b1_key = self._find_pair(state_dict, "state_net.4")
        w0 = state_dict[w0_key]
        b0 = state_dict[b0_key]
        w1 = state_dict[w1_key]
        b1 = state_dict[b1_key]
        in_dim = int(w0.shape[1])
        hidden_dim = int(w0.shape[0])
        if int(w1.shape[1]) != hidden_dim or int(w1.shape[0]) != hidden_dim:
            raise ValueError(f"state_net shape mismatch: got {tuple(w1.shape)} expected {(hidden_dim, hidden_dim)}")
        linear0 = nn.Linear(in_dim, hidden_dim)
        linear0.weight.data.copy_(w0)
        linear0.bias.data.copy_(b0)
        norm = nn.LayerNorm(hidden_dim)
        n_w_key, n_b_key = self._find_pair(state_dict, "state_net.1")
        norm.weight.data.copy_(state_dict[n_w_key])
        norm.bias.data.copy_(state_dict[n_b_key])
        linear1 = nn.Linear(hidden_dim, hidden_dim)
        linear1.weight.data.copy_(w1)
        linear1.bias.data.copy_(b1)
        return nn.Sequential(linear0, norm, nn.GELU(), nn.Dropout(p=_STATE_NET_DROPOUT_P), linear1)

    def _state_net_input_dim(self) -> int:
        if self.state_net is None:
            raise RuntimeError("state_net is not initialized.")
        if not isinstance(self.state_net, nn.Sequential) or len(self.state_net) < 1:
            raise TypeError("state_net must be a Sequential with Linear at index 0.")
        linear0 = self.state_net[0]
        if not isinstance(linear0, nn.Linear):
            raise TypeError("state_net[0] must be nn.Linear.")
        return int(linear0.in_features)

    def _validate_struct_proj(self) -> None:
        if self.struct_proj is None:
            raise RuntimeError("struct_proj is not initialized.")
        if not isinstance(self.struct_proj, nn.Sequential) or len(self.struct_proj) < 1:
            raise TypeError("struct_proj must be a Sequential with Linear at index 0.")
        linear0 = self.struct_proj[0]
        if not isinstance(linear0, nn.Linear):
            raise TypeError("struct_proj[0] must be nn.Linear.")
        if int(linear0.in_features) != int(self._struct_dim):
            raise ValueError(
                f"struct_proj input_dim mismatch: got {int(linear0.in_features)} expected {int(self._struct_dim)}"
            )
        if int(linear0.out_features) != int(self.hidden_dim):
            raise ValueError(
                f"struct_proj output_dim mismatch: got {int(linear0.out_features)} expected {int(self.hidden_dim)}"
            )

    def _load_concat_edge_components(
        self,
        state_dict: Dict[str, torch.Tensor],
        feature_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int],
    ) -> None:
        w0, b0, w1, b1, emb_dim, edge_in_dim, hidden_dim_ckpt = feature_data
        if hidden_dim_ckpt != self.hidden_dim:
            raise ValueError(f"feature_extractor hidden_dim mismatch: ckpt={hidden_dim_ckpt} vs expected={self.hidden_dim}")
        if int(w1.shape[1]) != emb_dim:
            raise ValueError(f"feature_extractor layer1 in-dim mismatch: ckpt={tuple(w1.shape)}")

        semantic_dim = 4 * self.hidden_dim
        struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim = self._load_retriever_parity_meta(
            state_dict,
            edge_in_dim=edge_in_dim,
            semantic_dim=semantic_dim,
        )
        self._configure_struct_features(
            struct_mode=struct_mode,
            num_topics=num_topics,
            num_rounds=num_rounds,
            num_reverse_rounds=num_rev,
            distance_max_hops=max_hops,
            distance_emb_dim=dist_dim,
        )
        if int(edge_in_dim) != int(self._edge_in_dim):
            raise ValueError(
                f"feature_extractor input_dim mismatch: ckpt={edge_in_dim} vs expected={int(self._edge_in_dim)}"
            )

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
        self._init_edge_adapter_residual(edge_in_dim)
        self.edge_score_proj = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.edge_score_proj.weight)
        self._edge_mode = _EDGE_MODE_CONCAT
        self._components_initialized = True

    def _load_geometry_edge_components(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self._edge_mode = _EDGE_MODE_GEOMETRY
        self.q_gate = nn.Sequential(
            self._load_linear_layer(
                state_dict, suffix="q_gate.0", in_dim=self.hidden_dim, out_dim=self.hidden_dim, name="q_gate"
            ),
            nn.Sigmoid(),
        )
        self.q_bias = nn.Sequential(
            self._load_linear_layer(
                state_dict, suffix="q_bias.0", in_dim=self.hidden_dim, out_dim=self.hidden_dim, name="q_bias"
            ),
            nn.Tanh(),
        )
        self.struct_proj = self._load_struct_proj(state_dict)
        self.struct_gate_net = nn.Sequential(
            self._load_linear_layer(
                state_dict, suffix="struct_gate_net.0", in_dim=self.hidden_dim, out_dim=1, name="struct_gate_net"
            ),
            nn.Sigmoid(),
        )
        self.state_net = self._load_state_net(state_dict)

        semantic_dim = 4 * self.hidden_dim
        struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim = self._load_retriever_parity_meta(
            state_dict,
            edge_in_dim=semantic_dim,
            semantic_dim=semantic_dim,
            require_meta=True,
        )
        self._configure_struct_features(
            struct_mode=struct_mode,
            num_topics=num_topics,
            num_rounds=num_rounds,
            num_reverse_rounds=num_rev,
            distance_max_hops=max_hops,
            distance_emb_dim=dist_dim,
        )
        self._validate_struct_proj()

        state_in_dim = self._state_net_input_dim()
        self._infer_geometry_modes(state_in_dim)
        self.edge_score_proj = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.edge_score_proj.weight)
        self._components_initialized = True

    def _infer_geometry_modes(self, edge_in_dim: int) -> None:
        hidden_dim = int(self.hidden_dim)
        distmult_only = 2 * hidden_dim
        transe_only = 2 * hidden_dim + _DIST_SCALAR_DIM
        both = 3 * hidden_dim + _DIST_SCALAR_DIM
        if edge_in_dim == both:
            self._use_distmult = True
            self._use_transe = True
            return
        if edge_in_dim == transe_only:
            self._use_distmult = False
            self._use_transe = True
            return
        if edge_in_dim == distmult_only:
            self._use_distmult = True
            self._use_transe = False
            return
        raise ValueError(
            "state_net input dim mismatch for geometry retriever: "
            f"edge_in_dim={edge_in_dim}, hidden_dim={hidden_dim}."
        )

    def _load_retriever_components(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.entity_proj = self._load_projector(state_dict, name="entity_proj")
        self.relation_proj = self._load_projector(state_dict, name="relation_proj")
        self.query_proj = self._load_projector(state_dict, name="query_proj")
        self._load_non_text_entity_emb(state_dict)

        feature_data = self._try_load_feature_extractor(state_dict)
        if feature_data is None:
            self._load_geometry_edge_components(state_dict)
            return
        self._load_concat_edge_components(state_dict, feature_data)

    def _configure_struct_features(
        self,
        *,
        struct_mode: int,
        num_topics: int,
        num_rounds: int,
        num_reverse_rounds: int,
        distance_max_hops: int,
        distance_emb_dim: int,
    ) -> None:
        struct_dim = 0
        if struct_mode == STRUCT_MODE_DIFFUSION:
            if num_topics <= 1:
                raise ValueError(f"Invalid retriever_meta: num_topics must be >= 2 when struct_mode=diffusion, got {num_topics}")
            struct_dim = 2 * int(num_topics) * (1 + int(num_rounds) + int(num_reverse_rounds))
        elif struct_mode == STRUCT_MODE_DISTANCE:
            if distance_emb_dim <= 0:
                raise ValueError("Invalid retriever_meta: distance_emb_dim must be > 0 when struct_mode=distance.")
            if distance_max_hops <= 0:
                raise ValueError("Invalid retriever_meta: distance_max_hops must be > 0 when struct_mode=distance.")
            struct_dim = 2 * int(distance_emb_dim)
        elif struct_mode != STRUCT_MODE_NONE:
            raise ValueError(f"Invalid retriever_meta: unknown struct_mode={struct_mode}")
        self._edge_in_dim = 4 * self.hidden_dim + struct_dim
        self._use_topic_pe = bool(struct_mode != STRUCT_MODE_NONE)
        self._struct_mode = int(struct_mode)
        self._num_topics = int(num_topics) if struct_mode == STRUCT_MODE_DIFFUSION else 0
        self._struct_dim = int(struct_dim)
        self._dde = (
            DDE(num_rounds=int(num_rounds), num_reverse_rounds=int(num_reverse_rounds))
            if struct_mode == STRUCT_MODE_DIFFUSION
            else None
        )
        self._distance_max_hops = int(distance_max_hops) if struct_mode == STRUCT_MODE_DISTANCE else 0
        self._distance_emb_dim = int(distance_emb_dim) if struct_mode == STRUCT_MODE_DISTANCE else 0

    def _init_edge_adapter_residual(self, edge_in_dim: int) -> None:
        self.edge_adapter_residual = nn.Linear(int(edge_in_dim), self.hidden_dim)
        nn.init.zeros_(self.edge_adapter_residual.weight)
        if self.edge_adapter_residual.bias is not None:
            nn.init.zeros_(self.edge_adapter_residual.bias)
        # Gate starts at 0 to preserve the original adapter behavior.
        self.edge_adapter_residual_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

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
        self._init_edge_adapter_residual(self._edge_in_dim)
        self.edge_score_proj = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.edge_score_proj.weight)
        self._components_initialized = True

    def _load_retriever_parity_meta(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        edge_in_dim: int,
        semantic_dim: int,
        require_meta: bool = False,
    ) -> tuple[int, int, int, int, int, int]:
        """Load minimal feature parity metadata from retriever checkpoint.

        Returns (struct_mode, num_topics, num_rounds, num_reverse_rounds, distance_max_hops, distance_emb_dim).
        Distance struct_mode is no longer supported. Non-zero distance fields are ignored when struct_mode is not distance.
        If metadata is missing and require_meta is False, semantic-only checkpoints (edge_in_dim==semantic_dim) are accepted;
        otherwise we fail fast.
        """
        try:
            meta_key = self._find_first_match(state_dict, "parity_meta")
        except KeyError:
            if require_meta:
                raise ValueError("Retriever checkpoint is missing parity_meta required for geometry features.") from None
            if int(edge_in_dim) == int(semantic_dim):
                return STRUCT_MODE_NONE, 0, 0, 0, 0, 0
            raise ValueError(
                "Retriever checkpoint is missing parity_meta required to disambiguate structural features "
                f"(edge_in_dim={edge_in_dim}, semantic_dim={semantic_dim}). Re-export/retrain retriever with updated code."
            )
        meta = state_dict[meta_key]
        if not torch.is_tensor(meta):
            raise ValueError(f"Invalid parity_meta type: expected torch.Tensor but got {type(meta)}")
        meta = meta.to(dtype=torch.long).view(-1)
        if meta.numel() == 4:
            use_topic_pe = bool(int(meta[0].item()))
            num_topics = int(meta[1].item())
            num_rounds = int(meta[2].item())
            num_rev = int(meta[3].item())
            struct_mode = STRUCT_MODE_DIFFUSION if use_topic_pe else STRUCT_MODE_NONE
            max_hops = 0
            dist_dim = 0
            return struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim
        if meta.numel() < 6:
            raise ValueError(f"Invalid parity_meta length: expected 4 or >=6 but got {int(meta.numel())}")
        struct_mode = int(meta[0].item())
        num_topics = int(meta[1].item())
        num_rounds = int(meta[2].item())
        num_rev = int(meta[3].item())
        max_hops = int(meta[4].item())
        dist_dim = int(meta[5].item())
        if struct_mode == STRUCT_MODE_DISTANCE:
            raise ValueError(
                "Unsupported struct_mode in parity_meta: distance mode has been removed. "
                "Re-export retriever without distance features."
            )
        if struct_mode not in (STRUCT_MODE_NONE, STRUCT_MODE_DIFFUSION):
            raise ValueError(f"Unsupported struct_mode in parity_meta: {struct_mode}.")
        if max_hops != 0 or dist_dim != 0:
            logger.warning(
                "Ignoring distance fields in parity_meta for retriever checkpoint (max_hops=%d, dist_dim=%d).",
                max_hops,
                dist_dim,
            )
            max_hops = 0
            dist_dim = 0
        return struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim

    def init_from_retriever_meta(self, meta: Dict[str, Any]) -> None:
        """Initialize missing components from retriever meta stored in gflownet checkpoints."""
        struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim = self._parse_retriever_meta(meta)
        if self._components_initialized:
            self._validate_retriever_meta(struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim, meta)
            self._record_retriever_meta(meta, source="gflownet_ckpt", allow_override=False)
            return
        self._configure_struct_features(
            struct_mode=struct_mode,
            num_topics=num_topics,
            num_rounds=num_rounds,
            num_reverse_rounds=num_rev,
            distance_max_hops=max_hops,
            distance_emb_dim=dist_dim,
        )
        self._init_fresh_components()
        self._record_retriever_meta(meta, source="gflownet_ckpt", allow_override=True)

    def export_retriever_meta(self) -> Dict[str, Any]:
        if not self._components_initialized:
            raise RuntimeError("GraphEmbedder is not initialized; cannot export retriever meta.")
        struct_mode = int(self._struct_mode)
        use_topic_pe = bool(struct_mode != STRUCT_MODE_NONE)
        num_topics = int(self._num_topics) if struct_mode == STRUCT_MODE_DIFFUSION else 0
        num_rounds = int(self._dde.num_rounds) if self._dde is not None else 0
        num_rev = int(self._dde.num_reverse_rounds) if self._dde is not None else 0
        max_hops = 0
        dist_dim = 0
        parity_meta = [struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim]
        meta: Dict[str, Any] = {
            "parity_meta": parity_meta,
            "struct_mode": int(struct_mode),
            "use_topic_pe": bool(use_topic_pe),
            "num_topics": int(num_topics),
            "num_rounds": int(num_rounds),
            "num_reverse_rounds": int(num_rev),
            "distance_max_hops": int(max_hops),
            "distance_emb_dim": int(dist_dim),
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

    @staticmethod
    def _normalize_struct_mode(mode: Any) -> int:
        if isinstance(mode, bool):
            return STRUCT_MODE_DIFFUSION if mode else STRUCT_MODE_NONE
        if isinstance(mode, int):
            if mode in (STRUCT_MODE_NONE, STRUCT_MODE_DIFFUSION):
                return int(mode)
            raise ValueError(f"Unknown struct_mode id: {mode}")
        mode_clean = str(mode).strip().lower()
        if mode_clean in {"", "diffusion", "precomputed", "topic_pe"}:
            return STRUCT_MODE_DIFFUSION
        if mode_clean in {"distance", "dist"}:
            raise ValueError("Distance struct_mode has been removed.")
        if mode_clean in {"none", "off", "disabled"}:
            return STRUCT_MODE_NONE
        raise ValueError(f"Unknown struct_mode: {mode}")

    def _parse_retriever_meta(self, meta: Dict[str, Any]) -> tuple[int, int, int, int, int, int]:
        if not isinstance(meta, dict):
            raise TypeError("retriever_meta must be a dict.")
        parity_meta = meta.get("parity_meta")
        if parity_meta is not None:
            if isinstance(parity_meta, torch.Tensor):
                parity_meta = parity_meta.detach().cpu().tolist()
            if len(parity_meta) == 4:
                use_topic_pe, num_topics, num_rounds, num_rev = [int(v) for v in parity_meta]
                struct_mode = STRUCT_MODE_DIFFUSION if use_topic_pe else STRUCT_MODE_NONE
                max_hops = 0
                dist_dim = 0
            elif len(parity_meta) >= 6:
                struct_mode, num_topics, num_rounds, num_rev, max_hops, dist_dim = [int(v) for v in parity_meta[:6]]
            else:
                raise ValueError(f"parity_meta must have 4 or >=6 values; got {parity_meta}")
        else:
            struct_mode_raw = meta.get("struct_mode", None)
            if struct_mode_raw is None:
                use_topic_pe = int(meta.get("use_topic_pe", 0))
                struct_mode = STRUCT_MODE_DIFFUSION if use_topic_pe else STRUCT_MODE_NONE
            else:
                struct_mode = self._normalize_struct_mode(struct_mode_raw)
            num_topics = int(meta.get("num_topics", 0))
            num_rounds = int(meta.get("num_rounds", 0))
            num_rev = int(meta.get("num_reverse_rounds", 0))
            max_hops = int(meta.get("distance_max_hops", 0))
            dist_dim = int(meta.get("distance_emb_dim", 0))

        if struct_mode == STRUCT_MODE_DIFFUSION and num_topics <= 1:
            raise ValueError(f"Invalid retriever_meta: num_topics must be >= 2 when struct_mode=diffusion, got {num_topics}")
        if struct_mode == STRUCT_MODE_DISTANCE:
            raise ValueError(
                "Invalid retriever_meta: distance struct_mode is no longer supported. "
                "Re-export retriever without distance features."
            )
        if struct_mode not in (STRUCT_MODE_NONE, STRUCT_MODE_DIFFUSION):
            raise ValueError(f"Invalid retriever_meta: unknown struct_mode={struct_mode}.")
        if max_hops != 0 or dist_dim != 0:
            logger.warning(
                "Ignoring distance fields in retriever_meta (max_hops=%d, dist_dim=%d).",
                max_hops,
                dist_dim,
            )
            max_hops = 0
            dist_dim = 0

        if struct_mode == STRUCT_MODE_DIFFUSION:
            struct_dim = 2 * int(num_topics) * (1 + int(num_rounds) + int(num_rev))
        else:
            struct_dim = 0
        edge_in_dim = 4 * self.hidden_dim + struct_dim
        if "edge_in_dim" in meta and int(meta["edge_in_dim"]) != edge_in_dim:
            raise ValueError(f"retriever_meta edge_in_dim mismatch: meta={meta['edge_in_dim']} expected={edge_in_dim}")
        if "struct_dim" in meta and int(meta["struct_dim"]) != struct_dim:
            raise ValueError(f"retriever_meta struct_dim mismatch: meta={meta['struct_dim']} expected={struct_dim}")
        return int(struct_mode), int(num_topics), int(num_rounds), int(num_rev), int(max_hops), int(dist_dim)

    def _validate_retriever_meta(
        self,
        struct_mode: int,
        num_topics: int,
        num_rounds: int,
        num_rev: int,
        max_hops: int,
        dist_dim: int,
        meta: Dict[str, Any],
    ) -> None:
        current_mode = int(self._struct_mode)
        current_topics = int(self._num_topics) if self._struct_mode == STRUCT_MODE_DIFFUSION else 0
        current_rounds = int(self._dde.num_rounds) if self._dde is not None else 0
        current_rev = int(self._dde.num_reverse_rounds) if self._dde is not None else 0
        if (current_mode, current_topics, current_rounds, current_rev) != (
            int(struct_mode),
            int(num_topics),
            int(num_rounds),
            int(num_rev),
        ):
            raise ValueError(
                "retriever_meta mismatch with existing embedder configuration: "
                f"meta=({struct_mode},{num_topics},{num_rounds},{num_rev}) "
                f"current=({current_mode},{current_topics},{current_rounds},{current_rev})"
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._struct_mode == STRUCT_MODE_NONE:
            raise RuntimeError("_build_structure_features called but struct_mode is none.")
        start_nodes = getattr(batch, "start_node_locals", None)
        if start_nodes is None:
            raise ValueError("struct features require batch.start_node_locals but it is missing.")
        start_nodes = start_nodes.to(device=device, dtype=torch.long).view(-1)
        if start_nodes.numel() > 0:
            if (start_nodes < 0).any() or (start_nodes >= num_nodes).any():
                raise ValueError("start_node_locals out of range for struct feature construction.")

        if self._struct_mode == STRUCT_MODE_DIFFUSION:
            if self._num_topics <= 0:
                raise RuntimeError("struct_mode=diffusion requires num_topics > 0.")
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
            edge_struct = torch.cat([head_struct, tail_struct], dim=-1)
            return edge_struct, node_struct

        raise RuntimeError(f"Unknown struct_mode={self._struct_mode} in _build_structure_features.")

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
        out = linear1(out)

        if self.edge_adapter_residual is None or self.edge_adapter_residual_gate is None:
            return out
        residual_linear = self.edge_adapter_residual
        if not isinstance(residual_linear, nn.Linear):
            raise TypeError("edge_adapter_residual must be an nn.Linear.")
        if int(residual_linear.in_features) != in_dim:
            raise ValueError(
                f"edge_adapter_residual in_features={int(residual_linear.in_features)} "
                f"!= edge_adapter in_features={in_dim}"
            )

        w_res = residual_linear.weight
        b_res = residual_linear.bias
        offset = 0
        wq_res = w_res[:, offset : offset + hidden_dim]
        offset += hidden_dim
        wh_res = w_res[:, offset : offset + hidden_dim]
        offset += hidden_dim
        wr_res = w_res[:, offset : offset + hidden_dim]
        offset += hidden_dim
        wt_res = w_res[:, offset : offset + hidden_dim]
        offset += hidden_dim
        residual = (
            F.linear(q_edge, wq_res)
            + F.linear(head_edge, wh_res)
            + F.linear(relation_edge, wr_res)
            + F.linear(tail_edge, wt_res)
        )
        if struct_edge is not None:
            w_s = w_res[:, offset:]
            residual = residual + F.linear(struct_edge, w_s)
        if b_res is not None:
            residual = residual + b_res
        gate = self.edge_adapter_residual_gate.to(device=residual.device, dtype=residual.dtype)
        return out + gate * residual

    def _edge_tokens_from_geometry(
        self,
        *,
        q_edge: torch.Tensor,         # [E_total, H]
        head_edge: torch.Tensor,      # [E_total, H]
        relation_edge: torch.Tensor,  # [E_total, H]
        tail_edge: torch.Tensor,      # [E_total, H]
        struct_edge: Optional[torch.Tensor],  # [E_total, S] or None
    ) -> torch.Tensor:
        if self.q_gate is None or self.q_bias is None or self.struct_proj is None:
            raise RuntimeError("Geometry edge tokens require q_gate/q_bias/struct_proj to be initialized.")
        if self.struct_gate_net is None or self.state_net is None:
            raise RuntimeError("Geometry edge tokens require struct_gate_net/state_net to be initialized.")
        if struct_edge is None:
            raise ValueError("Geometry edge tokens require struct_edge but it is None.")

        r_ctx = relation_edge * self.q_gate(q_edge) + self.q_bias(q_edge)
        struct_ctx = self.struct_proj(struct_edge)
        nav_gate = self.struct_gate_net(struct_ctx)

        parts = []
        if self._use_distmult:
            interaction_vec = head_edge * r_ctx * tail_edge
            parts.append(interaction_vec * nav_gate)
        parts.append(struct_ctx)
        if self._use_transe:
            error_vec = head_edge + r_ctx - tail_edge
            dist_scalar = -torch.norm(error_vec, p=2, dim=-1, keepdim=True)
            parts.append(error_vec)
            parts.append(dist_scalar)
        if not parts:
            raise RuntimeError("Geometry edge tokens have no features to combine.")
        combined = torch.cat(parts, dim=-1)
        return self.state_net(combined)

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
    def _swap_struct_edge(struct_edge: torch.Tensor) -> torch.Tensor:
        struct_dim = int(struct_edge.size(1))
        if struct_dim % _EDGE_DIR_COUNT != 0:
            raise ValueError("struct_edge feature dim must be even for undirected swap.")
        half = struct_dim // _EDGE_DIR_COUNT
        return torch.cat([struct_edge[:, half:], struct_edge[:, :half]], dim=-1)

    @staticmethod
    def _combine_undirected_edge_tokens(
        edge_tokens_fwd: torch.Tensor,
        edge_tokens_bwd: torch.Tensor,
    ) -> torch.Tensor:
        if edge_tokens_fwd.shape != edge_tokens_bwd.shape:
            raise ValueError("edge_tokens_fwd/bwd shape mismatch for undirected combine.")
        return (edge_tokens_fwd + edge_tokens_bwd) / float(_EDGE_DIR_COUNT)

    @staticmethod
    def _prepare_question_embeddings(question_emb: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if question_emb is None or question_emb.numel() == 0:
            raise ValueError("question_emb is missing or empty; g_agent cache must provide question embeddings for every graph.")
        if question_emb.dim() != 2:
            raise ValueError(f"question_emb must be 2D [B, D], got rank={question_emb.dim()}")
        if question_emb.size(0) != batch_size:
            raise ValueError(f"question_emb batch mismatch: {question_emb.size(0)} vs {batch_size}")
        return question_emb.to(device)

    @staticmethod
    def _compute_edge_batch(
        edge_index: torch.Tensor,
        *,
        node_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return compute_edge_batch(
            edge_index,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=device,
        )

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
    node_struct_raw: Optional[torch.Tensor]

    def validate_device(self, device: torch.device) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                raise RuntimeError(f"EmbedOutputs device mismatch: {name}.device={value.device}, expected={device}")


__all__ = ["GraphEmbedder", "EmbedOutputs"]
