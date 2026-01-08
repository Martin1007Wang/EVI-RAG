from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn



_GUMBEL_EPS = 1e-10
_ZERO = 0
_ONE = 1
_TWO = 2
_NEG_ONE = -1
_PTR_MIN_LEN = 2


def neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.dim() != 1 or segment_ids.dim() != 1:
        raise ValueError("logits and segment_ids must be 1D tensors.")
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have the same length.")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if logits.numel() == 0:
        return torch.full((num_segments,), neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    calc_dtype = logits.dtype
    if calc_dtype in (torch.float16, torch.bfloat16):
        calc_dtype = torch.float32
    logits = logits.to(dtype=calc_dtype)
    neg_inf = torch.finfo(calc_dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=calc_dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=calc_dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(calc_dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


def compute_policy_log_probs(
    *,
    edge_logits: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
    edge_batch: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    temperature: float,
    allow_stop: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if num_graphs <= 0:
        device = edge_logits.device
        zeros = torch.zeros(0, device=device)
        has_edge = torch.zeros(0, device=device, dtype=torch.bool)
        if stop_logits is None:
            return zeros, None, zeros, has_edge
        return zeros, zeros, zeros, has_edge
    if edge_logits.dtype != torch.float32:
        edge_logits = edge_logits.to(dtype=torch.float32)
    if stop_logits is not None and stop_logits.dtype != torch.float32:
        stop_logits = stop_logits.to(dtype=torch.float32)

    stop_scaled = None
    if stop_logits is not None:
        stop_scaled = stop_logits / float(temperature)
    if edge_logits.numel() == 0:
        log_denom = edge_logits.new_full((num_graphs,), neg_inf_value(edge_logits))
        has_edge = torch.zeros(num_graphs, device=edge_logits.device, dtype=torch.bool)
        if stop_scaled is None:
            return edge_logits, None, log_denom, has_edge
        log_denom = stop_scaled
        return edge_logits, stop_scaled - log_denom, log_denom, has_edge

    edge_scaled = edge_logits / float(temperature)
    valid_edges = valid_edges.to(device=edge_scaled.device, dtype=torch.bool)
    logsumexp_edges = segment_logsumexp_1d(edge_scaled[valid_edges], edge_batch[valid_edges], num_graphs)
    has_edge = logsumexp_edges > neg_inf_value(logsumexp_edges)
    if stop_scaled is None:
        if allow_stop is not None:
            raise ValueError("allow_stop requires stop_logits.")
        log_denom = logsumexp_edges
        log_prob_edge = edge_scaled - log_denom[edge_batch]
        log_prob_edge = torch.where(
            valid_edges,
            log_prob_edge,
            torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
        )
        return log_prob_edge, None, log_denom, has_edge
    if allow_stop is None:
        allow_stop_mask = torch.ones(num_graphs, device=stop_logits.device, dtype=torch.bool)
    else:
        allow_stop_mask = allow_stop.to(device=stop_logits.device, dtype=torch.bool).view(-1)
        if allow_stop_mask.numel() != num_graphs:
            raise ValueError("allow_stop length mismatch with batch size.")
    allow_stop_mask = allow_stop_mask | (~has_edge)
    log_denom = torch.where(
        allow_stop_mask,
        torch.logaddexp(logsumexp_edges, stop_scaled),
        logsumexp_edges,
    )
    log_prob_edge = edge_scaled - log_denom[edge_batch]
    log_prob_edge = torch.where(
        valid_edges,
        log_prob_edge,
        torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
    )
    log_prob_stop = stop_scaled - log_denom
    log_prob_stop = torch.where(
        allow_stop_mask,
        log_prob_stop,
        torch.full_like(log_prob_stop, neg_inf_value(log_prob_stop)),
    )
    return log_prob_edge, log_prob_stop, log_denom, has_edge


def gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u.clamp(min=_GUMBEL_EPS, max=1.0 - _GUMBEL_EPS)))


def pool_nodes_mean_by_ptr(
    *,
    node_tokens: torch.Tensor,
    node_locals: torch.Tensor,
    ptr: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if node_tokens.dim() != 2:
        raise ValueError("node_tokens must be [N, H] for pooling.")
    node_locals = node_locals.to(device=node_tokens.device, dtype=torch.long).view(-1)
    dim = int(node_tokens.size(-1))
    pooled = torch.zeros((num_graphs, dim), device=node_tokens.device, dtype=node_tokens.dtype)
    if node_locals.numel() == 0:
        return pooled
    positions = torch.arange(node_locals.numel(), device=node_tokens.device, dtype=ptr.dtype)
    batch_ids = torch.bucketize(positions, ptr[1:], right=True)
    pooled.index_add_(0, batch_ids, node_tokens[node_locals])
    counts = (ptr[1:] - ptr[:-1]).clamp(min=_ONE).to(dtype=node_tokens.dtype)
    return pooled / counts.unsqueeze(-1)


@dataclass(frozen=True)
class CommonBatchTensors:
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    question_emb: torch.Tensor
    node_embeddings: torch.Tensor
    node_embedding_ids: torch.Tensor
    edge_embeddings: torch.Tensor


@dataclass(frozen=True)
class RolloutBatchTensors:
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    node_min_dists: torch.Tensor
    start_ptr: torch.Tensor
    answer_ptr: torch.Tensor
    node_in_degree: torch.Tensor | None


@dataclass(frozen=True)
class CommonBatchStats:
    num_graphs: int
    num_nodes_total: int
    num_edges: int
    node_counts: torch.Tensor
    node_ptr: torch.Tensor


class GFlowNetInputValidator:
    def __init__(self, *, validate_edge_batch: bool = True) -> None:
        self.validate_edge_batch = bool(validate_edge_batch)

    def validate_edge_batch(self, batch: Any, *, device: torch.device) -> None:
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)

    def validate_rollout_batch(self, batch: Any, *, device: torch.device, is_training: bool) -> None:
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)
        rollout = self._collect_rollout_tensors(batch, device=device)
        self._validate_rollout_anchors(rollout, stats, is_training=is_training)
        self._validate_node_min_dists(rollout.node_min_dists, stats.num_nodes_total)
        self._validate_node_in_degree(rollout.node_in_degree, stats.num_nodes_total)

    @staticmethod
    def _require_tensor_attr(batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        value = getattr(batch, name, None)
        if not torch.is_tensor(value):
            raise ValueError(f"Batch missing tensor field: {name}.")
        return value.to(device=device, non_blocking=True)

    @staticmethod
    def _require_slice_ptr(batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        slice_dict = getattr(batch, "_slice_dict", None)
        if not isinstance(slice_dict, dict):
            raise ValueError("Batch missing _slice_dict required for packed pointers.")
        ptr = slice_dict.get(name)
        if ptr is None:
            raise ValueError(f"Batch missing _slice_dict entry: {name}.")
        return torch.as_tensor(ptr, dtype=torch.long, device=device).view(_NEG_ONE)

    def _collect_common_tensors(self, batch: Any, device: torch.device) -> CommonBatchTensors:
        return CommonBatchTensors(
            node_ptr=self._require_tensor_attr(batch, "ptr", device=device),
            edge_index=self._require_tensor_attr(batch, "edge_index", device=device),
            edge_attr=self._require_tensor_attr(batch, "edge_attr", device=device),
            question_emb=self._require_tensor_attr(batch, "question_emb", device=device),
            node_embeddings=self._require_tensor_attr(batch, "node_embeddings", device=device),
            node_embedding_ids=self._require_tensor_attr(batch, "node_embedding_ids", device=device),
            edge_embeddings=self._require_tensor_attr(batch, "edge_embeddings", device=device),
        )

    def _collect_rollout_tensors(self, batch: Any, *, device: torch.device) -> RolloutBatchTensors:
        node_in_degree = getattr(batch, "node_in_degree", None)
        if torch.is_tensor(node_in_degree):
            node_in_degree = node_in_degree.to(device=device, non_blocking=True)
        return RolloutBatchTensors(
            q_local_indices=self._require_tensor_attr(batch, "q_local_indices", device=device),
            a_local_indices=self._require_tensor_attr(batch, "a_local_indices", device=device),
            node_min_dists=self._require_tensor_attr(batch, "node_min_dists", device=device),
            start_ptr=self._require_slice_ptr(batch, "q_local_indices", device=device),
            answer_ptr=self._require_slice_ptr(batch, "a_local_indices", device=device),
            node_in_degree=node_in_degree if torch.is_tensor(node_in_degree) else None,
        )

    def _validate_common(self, tensors: CommonBatchTensors) -> CommonBatchStats:
        self._validate_integer_dtype(tensors.node_ptr, "ptr")
        num_graphs, num_nodes_total, node_counts = self._validate_node_ptr(tensors.node_ptr)
        self._validate_integer_dtype(tensors.edge_index, "edge_index")
        num_edges = self._validate_edge_index(tensors.edge_index, tensors.node_ptr, num_nodes_total, num_graphs)
        self._validate_integer_dtype(tensors.edge_attr, "edge_attr")
        self._validate_edge_attr(tensors.edge_attr, num_edges)
        return CommonBatchStats(
            num_graphs=num_graphs,
            num_nodes_total=num_nodes_total,
            num_edges=num_edges,
            node_counts=node_counts,
            node_ptr=tensors.node_ptr,
        )

    @staticmethod
    def _validate_integer_dtype(tensor: torch.Tensor, name: str) -> None:
        if tensor.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long, got {tensor.dtype}.")

    @staticmethod
    def _validate_floating_dtype(tensor: torch.Tensor, name: str) -> None:
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}.")

    @staticmethod
    def _validate_node_ptr(node_ptr: torch.Tensor) -> tuple[int, int, torch.Tensor]:
        if node_ptr.dim() != 1:
            raise ValueError(f"ptr must be 1D, got shape={tuple(node_ptr.shape)}.")
        if node_ptr.numel() < _PTR_MIN_LEN:
            raise ValueError("ptr must have at least two offsets.")
        if int(node_ptr[0].item()) != _ZERO:
            raise ValueError("ptr must start at 0.")
        if bool((node_ptr[1:] < node_ptr[:-1]).any().item()):
            raise ValueError("ptr must be non-decreasing.")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            raise ValueError("ptr must encode at least one graph.")
        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= _ZERO:
            raise ValueError("ptr must encode positive node counts.")
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=_ZERO)
        return num_graphs, num_nodes_total, node_counts

    def _validate_edge_index(
        self,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        num_nodes_total: int,
        num_graphs: int,
    ) -> int:
        if edge_index.dim() != _TWO or edge_index.size(0) != _TWO:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        num_edges = int(edge_index.size(1))
        if edge_index.numel() == _ZERO:
            return num_edges
        if bool((edge_index < _ZERO).any().item()):
            raise ValueError("edge_index contains negative node ids.")
        if bool((edge_index >= num_nodes_total).any().item()):
            raise ValueError("edge_index contains node ids out of range.")
        if not self.validate_edge_batch:
            return num_edges
        edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
        tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
        if bool((edge_batch != tail_batch).any().item()):
            raise ValueError("edge_index crosses graph boundaries; head/tail graph assignments differ.")
        if num_edges > _ONE and bool((edge_batch[:-1] > edge_batch[1:]).any().item()):
            raise ValueError("edge_index must be grouped by graph for packed slicing.")
        if bool((edge_batch < _ZERO).any().item()) or bool((edge_batch >= num_graphs).any().item()):
            raise ValueError("edge_index induces out-of-range graph ids.")
        return num_edges

    @staticmethod
    def _validate_edge_attr(edge_attr: torch.Tensor, num_edges: int) -> None:
        if edge_attr.dim() != 1:
            raise ValueError(f"edge_attr must be 1D, got shape={tuple(edge_attr.shape)}.")
        if edge_attr.numel() != num_edges:
            raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges}.")
        if edge_attr.numel() > _ZERO and bool((edge_attr < _ZERO).any().item()):
            raise ValueError("edge_attr contains negative relation ids (stop edges are disallowed).")

    def _validate_embeddings(self, common: CommonBatchTensors, stats: CommonBatchStats) -> None:
        self._validate_floating_dtype(common.question_emb, "question_emb")
        self._validate_floating_dtype(common.node_embeddings, "node_embeddings")
        self._validate_floating_dtype(common.edge_embeddings, "edge_embeddings")
        if common.question_emb.dim() != _TWO:
            raise ValueError(f"question_emb must be [B, D], got shape={tuple(common.question_emb.shape)}.")
        if common.node_embeddings.dim() != _TWO:
            raise ValueError(f"node_embeddings must be [N, D], got shape={tuple(common.node_embeddings.shape)}.")
        if common.edge_embeddings.dim() != _TWO:
            raise ValueError(f"edge_embeddings must be [E, D], got shape={tuple(common.edge_embeddings.shape)}.")
        if common.question_emb.size(0) != stats.num_graphs:
            raise ValueError("question_emb batch size mismatch with ptr.")
        if common.node_embeddings.size(0) != stats.num_nodes_total:
            raise ValueError("node_embeddings length mismatch with ptr.")
        if common.edge_embeddings.size(0) != stats.num_edges:
            raise ValueError("edge_embeddings length mismatch with edge_index.")
        self._ensure_finite(common.question_emb, "question_emb")
        self._ensure_finite(common.node_embeddings, "node_embeddings")
        self._ensure_finite(common.edge_embeddings, "edge_embeddings")

    def _validate_node_embedding_ids(self, node_embedding_ids: torch.Tensor, num_nodes_total: int) -> None:
        self._validate_integer_dtype(node_embedding_ids, "node_embedding_ids")
        if node_embedding_ids.dim() != 1:
            raise ValueError(f"node_embedding_ids must be 1D, got shape={tuple(node_embedding_ids.shape)}.")
        if node_embedding_ids.numel() != num_nodes_total:
            raise ValueError("node_embedding_ids length mismatch with ptr.")

    def _validate_rollout_anchors(
        self,
        rollout: RolloutBatchTensors,
        stats: CommonBatchStats,
        *,
        is_training: bool,
    ) -> None:
        q_local = rollout.q_local_indices.view(_NEG_ONE)
        a_local = rollout.a_local_indices.view(_NEG_ONE)
        self._validate_integer_dtype(q_local, "q_local_indices")
        self._validate_integer_dtype(a_local, "a_local_indices")
        self._validate_integer_dtype(rollout.start_ptr, "q_local_indices_ptr")
        self._validate_integer_dtype(rollout.answer_ptr, "a_local_indices_ptr")
        self._validate_ptr(rollout.start_ptr, stats.num_graphs, q_local.numel(), "q_local_indices")
        self._validate_ptr(rollout.answer_ptr, stats.num_graphs, a_local.numel(), "a_local_indices")
        start_counts = rollout.start_ptr[1:] - rollout.start_ptr[:-1]
        missing_start = start_counts == _ZERO
        if bool(missing_start.any().item()):
            raise ValueError("Start nodes missing from batch; sub dataset should exclude these samples.")
        answer_counts = rollout.answer_ptr[1:] - rollout.answer_ptr[:-1]
        missing_answer = answer_counts == _ZERO
        if bool(missing_answer.any().item()):
            raise ValueError("Answer nodes missing in batch; filter unreachable samples before GFlowNet.")
        self._validate_local_indices(q_local, rollout.start_ptr, stats.node_ptr, "q_local_indices")
        self._validate_local_indices(a_local, rollout.answer_ptr, stats.node_ptr, "a_local_indices")

    @staticmethod
    def _validate_ptr(ptr: torch.Tensor, num_graphs: int, total: int, name: str) -> None:
        if ptr.dim() != 1:
            raise ValueError(f"{name}_ptr must be 1D, got shape={tuple(ptr.shape)}.")
        if ptr.numel() < _PTR_MIN_LEN:
            raise ValueError(f"{name}_ptr must have at least two offsets.")
        if ptr.numel() != num_graphs + 1:
            raise ValueError(f"{name}_ptr length mismatch with num_graphs.")
        if int(ptr[0].item()) != _ZERO:
            raise ValueError(f"{name}_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError(f"{name}_ptr must be non-decreasing.")
        if int(ptr[-1].item()) != total:
            raise ValueError(f"{name}_ptr must end at {total}, got {int(ptr[-1].item())}.")

    @staticmethod
    def _validate_local_indices(
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        name: str,
    ) -> None:
        if local_indices.numel() == _ZERO:
            return
        device = node_ptr.device
        local_indices = local_indices.to(device=device, non_blocking=True)
        ptr = ptr.to(device=device, non_blocking=True)
        node_ptr = node_ptr.to(device=device, non_blocking=True)
        positions = torch.arange(local_indices.numel(), device=device, dtype=ptr.dtype)
        graph_ids = torch.bucketize(positions, ptr[1:], right=True)
        if bool((local_indices < _ZERO).any().item()):
            raise ValueError(f"{name} contains negative indices.")
        node_start = node_ptr[graph_ids]
        node_end = node_ptr[graph_ids + _ONE]
        in_range = (local_indices >= node_start) & (local_indices < node_end)
        if bool((~in_range).any().item()):
            raise ValueError(f"{name} indices fall outside per-graph node ranges.")

    @staticmethod
    def _validate_node_min_dists(node_min_dists: torch.Tensor, num_nodes_total: int) -> None:
        if node_min_dists.dim() != 1:
            raise ValueError(f"node_min_dists must be 1D, got shape={tuple(node_min_dists.shape)}.")
        if node_min_dists.numel() != num_nodes_total:
            raise ValueError("node_min_dists length mismatch with ptr.")

    @staticmethod
    def _validate_node_in_degree(node_in_degree: torch.Tensor | None, num_nodes_total: int) -> None:
        if node_in_degree is None:
            return
        if node_in_degree.dim() != 1:
            raise ValueError(f"node_in_degree must be 1D, got shape={tuple(node_in_degree.shape)}.")
        if node_in_degree.numel() != num_nodes_total:
            raise ValueError("node_in_degree length mismatch with ptr.")

    @staticmethod
    def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values.")


@dataclass(frozen=True)
class EdgeTokenInputs:
    edge_batch: torch.Tensor
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor


@dataclass(frozen=True)
class RolloutInputs:
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    reward_node_embeddings: torch.Tensor
    reward_question_emb: torch.Tensor
    node_in_degree: torch.Tensor
    node_min_dists: torch.Tensor
    start_node_locals: torch.Tensor
    answer_node_locals: torch.Tensor
    start_ptr: torch.Tensor
    answer_ptr: torch.Tensor
    dummy_mask: torch.Tensor


class GFlowNetBatchProcessor:
    """Prepare rollout inputs and graph caches for EB-GFN batches."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
    ) -> None:
        self.backbone = backbone

    @staticmethod
    def _get_node_ptr(batch: Any, device: torch.device) -> torch.Tensor:
        return batch.ptr.to(device=device, non_blocking=True)

    @staticmethod
    def _get_start_answer_ptrs(
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_node_locals = batch.q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        answer_node_locals = batch.a_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = batch._slice_dict
        start_ptr = torch.as_tensor(slice_dict["q_local_indices"], dtype=torch.long, device=device)
        answer_ptr = torch.as_tensor(slice_dict["a_local_indices"], dtype=torch.long, device=device)
        return start_node_locals, answer_node_locals, start_ptr, answer_ptr

    @staticmethod
    def build_dummy_mask(
        *,
        answer_ptr: torch.Tensor,
    ) -> torch.Tensor:
        answer_counts = answer_ptr[1:] - answer_ptr[:-1]
        missing = answer_counts == _ZERO
        if bool(missing.any().item()):
            raise ValueError("Answer nodes missing in batch; filter missing answers before GFlowNet.")
        return missing.to(dtype=torch.bool)

    def resolve_edge_batch(
        self,
        *,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
        return edge_batch.to(device=device, dtype=torch.long).view(-1)

    def encode_batch_tokens(
        self,
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens, relation_tokens, question_tokens = self.backbone(batch)
        return (
            node_tokens.to(device=device, dtype=torch.float32),
            relation_tokens.to(device=device, dtype=torch.float32),
            question_tokens.to(device=device, dtype=torch.float32),
        )

    @staticmethod
    def _get_reward_embeddings(
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_embeddings = getattr(batch, "node_embeddings", None)
        question_emb = getattr(batch, "question_emb", None)
        if not torch.is_tensor(node_embeddings) or not torch.is_tensor(question_emb):
            raise ValueError("Batch missing node_embeddings/question_emb required for reward embeddings.")
        return (
            node_embeddings.to(device=device, dtype=torch.float32, non_blocking=True),
            question_emb.to(device=device, dtype=torch.float32, non_blocking=True),
        )

    def prepare_edge_token_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> EdgeTokenInputs:
        node_ptr = self._get_node_ptr(batch, device)
        edge_index = batch.edge_index.to(device=device, non_blocking=True)
        edge_relations = batch.edge_attr.to(device=device, non_blocking=True)
        edge_batch_full = self.resolve_edge_batch(
            edge_index=edge_index,
            node_ptr=node_ptr,
            device=device,
        )
        node_tokens, relation_tokens, question_tokens = self.encode_batch_tokens(batch, device=device)
        return EdgeTokenInputs(
            edge_batch=edge_batch_full,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
        )

    @staticmethod
    def _compute_node_in_degree(
        *,
        edge_index: torch.Tensor,
        num_nodes_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_nodes_total <= _ZERO:
            return torch.zeros((_ZERO,), device=device, dtype=torch.long)
        node_in_degree = torch.bincount(edge_index.view(-1), minlength=num_nodes_total)
        return node_in_degree.to(device=device, dtype=torch.long)

    @staticmethod
    def _get_node_min_dists(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.node_min_dists.to(device=device, dtype=torch.long, non_blocking=True)

    @staticmethod
    def build_edge_ptr(edge_batch: torch.Tensor, num_graphs: int, device: torch.device) -> torch.Tensor:
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=device)
        edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        edge_ptr[1:] = counts.cumsum(0)
        return edge_ptr

    @staticmethod
    def _build_rollout_inputs(
        *,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        reward_node_embeddings: torch.Tensor,
        reward_question_emb: torch.Tensor,
        node_in_degree: torch.Tensor,
        node_min_dists: torch.Tensor,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        answer_ptr: torch.Tensor,
        dummy_mask: torch.Tensor,
    ) -> RolloutInputs:
        return RolloutInputs(
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
            reward_node_embeddings=reward_node_embeddings,
            reward_question_emb=reward_question_emb,
            node_in_degree=node_in_degree,
            node_min_dists=node_min_dists,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            start_ptr=start_ptr,
            answer_ptr=answer_ptr,
            dummy_mask=dummy_mask,
        )

    def prepare_full_rollout_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> RolloutInputs:
        token_inputs = self.prepare_edge_token_inputs(batch, device)
        node_ptr = token_inputs.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        edge_index_base = batch.edge_index.to(device=device, non_blocking=True)
        node_in_degree = self._compute_node_in_degree(
            edge_index=edge_index_base, num_nodes_total=num_nodes_total, device=device
        )
        node_min_dists = self._get_node_min_dists(batch, device=device)
        reward_node_embeddings, reward_question_emb = self._get_reward_embeddings(batch, device=device)

        start_node_locals, answer_node_locals, start_ptr, answer_ptr = self._get_start_answer_ptrs(
            batch, device=device
        )
        dummy_mask = self.build_dummy_mask(answer_ptr=answer_ptr)
        return self._build_rollout_inputs(
            edge_batch=token_inputs.edge_batch,
            edge_ptr=self.build_edge_ptr(token_inputs.edge_batch, num_graphs, device),
            node_ptr=node_ptr,
            edge_index=token_inputs.edge_index,
            edge_relations=token_inputs.edge_relations,
            node_tokens=token_inputs.node_tokens,
            relation_tokens=token_inputs.relation_tokens,
            question_tokens=token_inputs.question_tokens,
            reward_node_embeddings=reward_node_embeddings,
            reward_question_emb=reward_question_emb,
            node_in_degree=node_in_degree,
            node_min_dists=node_min_dists,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            start_ptr=start_ptr,
            answer_ptr=answer_ptr,
            dummy_mask=dummy_mask,
        )

    def prepare_rollout_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> RolloutInputs:
        return self.prepare_full_rollout_inputs(batch, device)

    @staticmethod
    def _repeat_ptr(counts: torch.Tensor, repeats: int) -> torch.Tensor:
        counts_rep = counts.repeat(repeats)
        ptr = torch.zeros(counts_rep.numel() + 1, dtype=counts.dtype, device=counts.device)
        if counts_rep.numel() > 0:
            ptr[1:] = counts_rep.cumsum(0)
        return ptr

    @staticmethod
    def _repeat_edges(
        *,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        num_rollouts: int,
        num_graphs: int,
        total_nodes: int,
        total_edges: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_offsets = (
            torch.arange(num_rollouts, device=device, dtype=edge_index.dtype) * total_nodes
        ).repeat_interleave(total_edges)
        edge_index_rep = edge_index.repeat(1, num_rollouts)
        if edge_offsets.numel() > 0:
            edge_index_rep = edge_index_rep + edge_offsets
        edge_relations_rep = edge_relations.repeat(num_rollouts)
        relation_tokens_rep = relation_tokens.repeat(num_rollouts, 1)
        edge_batch_offsets = (
            torch.arange(num_rollouts, device=device, dtype=edge_batch.dtype) * num_graphs
        ).repeat_interleave(total_edges)
        edge_batch_rep = edge_batch.repeat(num_rollouts)
        if edge_batch_offsets.numel() > 0:
            edge_batch_rep = edge_batch_rep + edge_batch_offsets
        return edge_index_rep, edge_relations_rep, relation_tokens_rep, edge_batch_rep

    @staticmethod
    def _repeat_nodes(
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        reward_node_embeddings: torch.Tensor,
        reward_question_emb: torch.Tensor,
        node_in_degree: torch.Tensor,
        node_min_dists: torch.Tensor,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens_rep = node_tokens.repeat(num_rollouts, 1)
        question_tokens_rep = question_tokens.repeat(num_rollouts, 1)
        reward_node_embeddings_rep = reward_node_embeddings.repeat(num_rollouts, 1)
        reward_question_emb_rep = reward_question_emb.repeat(num_rollouts, 1)
        node_in_degree_rep = node_in_degree.repeat(num_rollouts)
        node_min_dists_rep = node_min_dists.repeat(num_rollouts)
        return (
            node_tokens_rep,
            question_tokens_rep,
            reward_node_embeddings_rep,
            reward_question_emb_rep,
            node_in_degree_rep,
            node_min_dists_rep,
        )

    @classmethod
    def _repeat_start_answer(
        cls,
        *,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        answer_ptr: torch.Tensor,
        total_nodes: int,
        num_rollouts: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_counts = (start_ptr[1:] - start_ptr[:-1]).clamp(min=0)
        answer_counts = (answer_ptr[1:] - answer_ptr[:-1]).clamp(min=0)
        start_ptr_rep = cls._repeat_ptr(start_counts, num_rollouts)
        answer_ptr_rep = cls._repeat_ptr(answer_counts, num_rollouts)
        start_offsets = (
            torch.arange(num_rollouts, device=device, dtype=start_node_locals.dtype) * total_nodes
        ).repeat_interleave(start_node_locals.numel())
        answer_offsets = (
            torch.arange(num_rollouts, device=device, dtype=answer_node_locals.dtype) * total_nodes
        ).repeat_interleave(answer_node_locals.numel())
        start_locals_rep = start_node_locals.repeat(num_rollouts)
        answer_locals_rep = answer_node_locals.repeat(num_rollouts)
        if start_offsets.numel() > 0:
            start_locals_rep = start_locals_rep + start_offsets
        if answer_offsets.numel() > 0:
            answer_locals_rep = answer_locals_rep + answer_offsets
        return start_locals_rep, answer_locals_rep, start_ptr_rep, answer_ptr_rep

    def _repeat_rollout_meta(
        self,
        inputs: RolloutInputs,
        *,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, torch.device]:
        node_ptr = inputs.node_ptr
        edge_ptr = inputs.edge_ptr
        num_graphs = int(node_ptr.numel() - 1)
        total_nodes = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        total_edges = int(edge_ptr[-1].item()) if edge_ptr.numel() > 0 else 0
        device = node_ptr.device
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        edge_counts = (edge_ptr[1:] - edge_ptr[:-1]).clamp(min=0)
        node_ptr_rep = self._repeat_ptr(node_counts, num_rollouts)
        edge_ptr_rep = self._repeat_ptr(edge_counts, num_rollouts)
        return node_ptr_rep, edge_ptr_rep, num_graphs, total_nodes, total_edges, device

    def _repeat_rollout_payload(
        self,
        inputs: RolloutInputs,
        *,
        num_rollouts: int,
        num_graphs: int,
        total_nodes: int,
        total_edges: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        edges = self._repeat_edges(
            edge_index=inputs.edge_index,
            edge_relations=inputs.edge_relations,
            relation_tokens=inputs.relation_tokens,
            edge_batch=inputs.edge_batch,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            total_nodes=total_nodes,
            total_edges=total_edges,
            device=device,
        )
        nodes = self._repeat_nodes(
            node_tokens=inputs.node_tokens,
            question_tokens=inputs.question_tokens,
            reward_node_embeddings=inputs.reward_node_embeddings,
            reward_question_emb=inputs.reward_question_emb,
            node_in_degree=inputs.node_in_degree,
            node_min_dists=inputs.node_min_dists,
            num_rollouts=num_rollouts,
        )
        start = self._repeat_start_answer(
            start_node_locals=inputs.start_node_locals,
            answer_node_locals=inputs.answer_node_locals,
            start_ptr=inputs.start_ptr,
            answer_ptr=inputs.answer_ptr,
            total_nodes=total_nodes,
            num_rollouts=num_rollouts,
            device=device,
        )
        return self._assemble_repeat_payload(
            edges,
            nodes,
            start,
            dummy_mask=inputs.dummy_mask.repeat(num_rollouts),
        )

    @staticmethod
    def _assemble_repeat_payload(
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        nodes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        start: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        dummy_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        edge_index_rep, edge_relations_rep, relation_tokens_rep, edge_batch_rep = edges
        (
            node_tokens_rep,
            question_tokens_rep,
            reward_node_embeddings_rep,
            reward_question_emb_rep,
            node_in_degree_rep,
            node_min_dists_rep,
        ) = nodes
        start_locals_rep, answer_locals_rep, start_ptr_rep, answer_ptr_rep = start
        return {
            "edge_index": edge_index_rep,
            "edge_relations": edge_relations_rep,
            "relation_tokens": relation_tokens_rep,
            "edge_batch": edge_batch_rep,
            "node_tokens": node_tokens_rep,
            "question_tokens": question_tokens_rep,
            "reward_node_embeddings": reward_node_embeddings_rep,
            "reward_question_emb": reward_question_emb_rep,
            "node_in_degree": node_in_degree_rep,
            "node_min_dists": node_min_dists_rep,
            "start_node_locals": start_locals_rep,
            "answer_node_locals": answer_locals_rep,
            "start_ptr": start_ptr_rep,
            "answer_ptr": answer_ptr_rep,
            "dummy_mask": dummy_mask,
        }

    def repeat_rollout_inputs(self, inputs: RolloutInputs, num_rollouts: int) -> RolloutInputs:
        if num_rollouts <= 1:
            return inputs
        node_ptr_rep, edge_ptr_rep, num_graphs, total_nodes, total_edges, device = self._repeat_rollout_meta(
            inputs,
            num_rollouts=num_rollouts,
        )
        payload = self._repeat_rollout_payload(
            inputs,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            total_nodes=total_nodes,
            total_edges=total_edges,
            device=device,
        )

        return self._build_rollout_inputs(
            edge_batch=payload["edge_batch"],
            edge_ptr=edge_ptr_rep,
            node_ptr=node_ptr_rep,
            edge_index=payload["edge_index"],
            edge_relations=payload["edge_relations"],
            node_tokens=payload["node_tokens"],
            relation_tokens=payload["relation_tokens"],
            question_tokens=payload["question_tokens"],
            reward_node_embeddings=payload["reward_node_embeddings"],
            reward_question_emb=payload["reward_question_emb"],
            node_in_degree=payload["node_in_degree"],
            node_min_dists=payload["node_min_dists"],
            start_node_locals=payload["start_node_locals"],
            answer_node_locals=payload["answer_node_locals"],
            start_ptr=payload["start_ptr"],
            answer_ptr=payload["answer_ptr"],
            dummy_mask=payload["dummy_mask"],
        )

    @staticmethod
    def compute_node_batch(node_ptr: torch.Tensor, num_graphs: int, device: torch.device) -> torch.Tensor:
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        return torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)

    @staticmethod
    def compute_node_flags(
        num_nodes_total: int,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals] = True
        return node_is_start, node_is_answer

    def build_graph_cache(self, inputs: RolloutInputs, *, device: torch.device) -> Dict[str, torch.Tensor]:
        node_ptr = inputs.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        node_batch = self.compute_node_batch(node_ptr, num_graphs, device)
        node_is_start, node_is_answer = self.compute_node_flags(
            num_nodes_total,
            inputs.start_node_locals,
            inputs.answer_node_locals,
            device,
        )
        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": inputs.edge_index,
            "edge_batch": inputs.edge_batch,
            "edge_relations": inputs.edge_relations,
            "node_ptr": node_ptr,
            "edge_ptr": inputs.edge_ptr,
            "node_tokens": inputs.node_tokens,
            "relation_tokens": inputs.relation_tokens,
            "question_tokens": inputs.question_tokens,
            "node_batch": node_batch,
            "node_is_start": node_is_start,
            "node_is_answer": node_is_answer,
            "node_in_degree": inputs.node_in_degree,
            "start_node_locals": inputs.start_node_locals,
            "start_ptr": inputs.start_ptr,
            "answer_node_locals": inputs.answer_node_locals,
            "answer_ptr": inputs.answer_ptr,
            "dummy_mask": inputs.dummy_mask,
        }
        return graph_cache


__all__ = [
    "compute_policy_log_probs",
    "gumbel_noise_like",
    "neg_inf_value",
    "pool_nodes_mean_by_ptr",
    "segment_logsumexp_1d",
    "GFlowNetInputValidator",
    "EdgeTokenInputs",
    "GFlowNetBatchProcessor",
    "RolloutInputs",
]
