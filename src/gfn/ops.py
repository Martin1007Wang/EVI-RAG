from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from src.utils.graph import compute_edge_batch

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


def scatter_logsumexp(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    return segment_logsumexp_1d(logits, segment_ids, num_segments)


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
        if allow_stop is None:
            allow_stop_mask = torch.ones(num_graphs, device=stop_scaled.device, dtype=torch.bool)
        else:
            allow_stop_mask = allow_stop.to(device=stop_scaled.device, dtype=torch.bool).view(-1)
            if allow_stop_mask.numel() != num_graphs:
                raise ValueError("allow_stop length mismatch with batch size.")
        log_denom = torch.where(allow_stop_mask, stop_scaled, log_denom)
        neg_inf = neg_inf_value(stop_scaled)
        log_prob_stop = torch.where(
            allow_stop_mask,
            stop_scaled - log_denom,
            torch.full_like(stop_scaled, neg_inf),
        )
        return edge_logits, log_prob_stop, log_denom, has_edge

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


@dataclass(frozen=True)
class CommonBatchTensors:
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    question_emb: torch.Tensor
    node_embeddings: torch.Tensor
    node_embedding_ids: torch.Tensor
    edge_embeddings: torch.Tensor
    slice_dict: dict | None


@dataclass(frozen=True)
class RolloutBatchTensors:
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    start_ptr: torch.Tensor
    answer_ptr: torch.Tensor
    retrieval_failure: torch.Tensor | None


@dataclass(frozen=True)
class CommonBatchStats:
    num_graphs: int
    num_nodes_total: int
    num_edges: int
    node_counts: torch.Tensor
    node_ptr: torch.Tensor


class GFlowNetInputValidator:
    def __init__(
        self,
        *,
        validate_edge_batch: bool = True,
        validate_rollout_batch: bool = True,
        move_to_device: bool = True,
    ) -> None:
        self.validate_edge_batch_enabled = bool(validate_edge_batch)
        self.validate_rollout_batch_enabled = bool(validate_rollout_batch)
        self.move_to_device = bool(move_to_device)

    def validate_edge_batch(self, batch: Any, *, device: torch.device) -> None:
        if not self.validate_edge_batch_enabled:
            return
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)

    def validate_rollout_batch(self, batch: Any, *, device: torch.device, is_training: bool) -> None:
        if not self.validate_rollout_batch_enabled:
            return
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)
        rollout = self._collect_rollout_tensors(batch, device=common.node_ptr.device)
        self._validate_rollout_anchors(rollout, stats, is_training=is_training)

    def _require_tensor_attr(self, batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        value = getattr(batch, name, None)
        if not torch.is_tensor(value):
            raise ValueError(f"Batch missing tensor field: {name}.")
        target_device = device if self.move_to_device else value.device
        if value.device != target_device:
            return value.to(device=target_device, non_blocking=True)
        return value

    def _require_slice_ptr(self, batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        slice_dict = getattr(batch, "_slice_dict", None)
        if not isinstance(slice_dict, dict):
            raise ValueError("Batch missing _slice_dict required for packed pointers.")
        ptr = slice_dict.get(name)
        if ptr is None:
            raise ValueError(f"Batch missing _slice_dict entry: {name}.")
        return torch.as_tensor(ptr, dtype=torch.long, device=device).view(_NEG_ONE)

    def _collect_common_tensors(self, batch: Any, device: torch.device) -> CommonBatchTensors:
        node_ptr = self._require_tensor_attr(batch, "ptr", device=device)
        device = node_ptr.device
        return CommonBatchTensors(
            node_ptr=node_ptr,
            edge_index=self._require_tensor_attr(batch, "edge_index", device=device),
            edge_attr=self._require_tensor_attr(batch, "edge_attr", device=device),
            question_emb=self._require_tensor_attr(batch, "question_emb", device=device),
            node_embeddings=self._require_tensor_attr(batch, "node_embeddings", device=device),
            node_embedding_ids=self._require_tensor_attr(batch, "node_embedding_ids", device=device),
            edge_embeddings=self._require_tensor_attr(batch, "edge_embeddings", device=device),
            slice_dict=getattr(batch, "_slice_dict", None),
        )

    def _collect_rollout_tensors(self, batch: Any, *, device: torch.device) -> RolloutBatchTensors:
        retrieval_failure = getattr(batch, "retrieval_failure", None)
        if retrieval_failure is not None:
            retrieval_failure = torch.as_tensor(retrieval_failure, dtype=torch.bool, device=device).view(_NEG_ONE)
        q_local_indices = self._require_tensor_attr(batch, "q_local_indices", device=device)
        device = q_local_indices.device
        a_local_indices = self._require_tensor_attr(batch, "a_local_indices", device=device)
        return RolloutBatchTensors(
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            start_ptr=self._require_slice_ptr(batch, "q_local_indices", device=device),
            answer_ptr=self._require_slice_ptr(batch, "a_local_indices", device=device),
            retrieval_failure=retrieval_failure if torch.is_tensor(retrieval_failure) else None,
        )

    def _validate_common(self, tensors: CommonBatchTensors) -> CommonBatchStats:
        self._validate_integer_dtype(tensors.node_ptr, "ptr")
        num_graphs, num_nodes_total, node_counts = self._validate_node_ptr(tensors.node_ptr)
        self._validate_integer_dtype(tensors.edge_index, "edge_index")
        num_edges = self._validate_edge_index(tensors.edge_index, tensors.node_ptr, num_graphs)
        self._validate_edge_ptr_from_slice(
            tensors=tensors,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )
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
        if int(node_ptr[0].detach().tolist()) != _ZERO:
            raise ValueError("ptr must start at 0.")
        if bool((node_ptr[1:] < node_ptr[:-1]).any().detach().tolist()):
            raise ValueError("ptr must be non-decreasing.")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            raise ValueError("ptr must encode at least one graph.")
        num_nodes_total = int(node_ptr[-1].detach().tolist())
        if num_nodes_total <= _ZERO:
            raise ValueError("ptr must encode positive node counts.")
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=_ZERO)
        return num_graphs, num_nodes_total, node_counts

    def _validate_edge_index(
        self,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        num_graphs: int,
    ) -> int:
        if edge_index.dim() != _TWO or edge_index.size(0) != _TWO:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        num_edges = int(edge_index.size(1))
        if edge_index.numel() == _ZERO:
            return num_edges
        if not self.validate_edge_batch_enabled:
            return num_edges
        compute_edge_batch(
            edge_index,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=edge_index.device,
            validate=True,
        )
        return num_edges

    @staticmethod
    def _validate_edge_attr(edge_attr: torch.Tensor, num_edges: int) -> None:
        if edge_attr.dim() != 1:
            raise ValueError(f"edge_attr must be 1D, got shape={tuple(edge_attr.shape)}.")
        if edge_attr.numel() != num_edges:
            raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges}.")
        if edge_attr.numel() > _ZERO and bool((edge_attr < _ZERO).any().detach().tolist()):
            raise ValueError("edge_attr contains negative relation ids (stop edges are disallowed).")

    def _validate_edge_ptr_from_slice(
        self,
        *,
        tensors: CommonBatchTensors,
        num_edges: int,
        num_graphs: int,
    ) -> None:
        if not self.validate_edge_batch_enabled:
            return
        slice_dict = tensors.slice_dict
        if slice_dict is None:
            return
        edge_ptr_raw = slice_dict.get("edge_index") if isinstance(slice_dict, dict) else None
        if edge_ptr_raw is None:
            return
        edge_ptr = torch.as_tensor(edge_ptr_raw, dtype=torch.long, device=tensors.node_ptr.device).view(_NEG_ONE)
        if edge_ptr.dim() != 1:
            raise ValueError(f"edge_index_ptr must be 1D, got shape={tuple(edge_ptr.shape)}.")
        if edge_ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"edge_index_ptr length {edge_ptr.numel()} != num_graphs+1 ({num_graphs + _ONE}).")
        if int(edge_ptr[0].detach().tolist()) != _ZERO or int(edge_ptr[-_ONE].detach().tolist()) != num_edges:
            raise ValueError(
                f"edge_index_ptr must start at 0 and end at {num_edges}, got start={int(edge_ptr[0].detach().tolist())} "
                f"end={int(edge_ptr[-_ONE].detach().tolist())}."
            )
        if bool((edge_ptr[1:] < edge_ptr[:-1]).any().detach().tolist()):
            raise ValueError("edge_index_ptr must be non-decreasing.")
        edge_batch, edge_ptr_expected = compute_edge_batch(
            tensors.edge_index,
            node_ptr=tensors.node_ptr,
            num_graphs=num_graphs,
            device=tensors.edge_index.device,
            validate=True,
        )
        if edge_batch.numel() != num_edges:
            raise ValueError("edge_batch length mismatch during edge_index_ptr validation.")
        if not torch.equal(edge_ptr_expected.to(device=edge_ptr.device), edge_ptr):
            raise ValueError("edge_index_ptr from _slice_dict is inconsistent with edge_index/node_ptr.")

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
        if bool(missing_start.any().detach().tolist()):
            raise ValueError("Start nodes missing from batch; sub dataset should exclude these samples.")
        answer_counts = rollout.answer_ptr[1:] - rollout.answer_ptr[:-1]
        missing_answer = answer_counts == _ZERO
        if bool(missing_answer.any().detach().tolist()):
            if rollout.retrieval_failure is None:
                raise ValueError("Answer nodes missing in batch; filter unreachable samples before GFlowNet.")
            retrieval_mask = rollout.retrieval_failure.to(device=missing_answer.device, dtype=torch.bool)
            if retrieval_mask.numel() != stats.num_graphs:
                raise ValueError("retrieval_failure length mismatch with num_graphs.")
            active_missing = missing_answer & (~retrieval_mask)
            if bool(active_missing.any().detach().tolist()):
                raise ValueError("Answer nodes missing without retrieval_failure flag; data inconsistent.")
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
        if int(ptr[0].detach().tolist()) != _ZERO:
            raise ValueError(f"{name}_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().detach().tolist()):
            raise ValueError(f"{name}_ptr must be non-decreasing.")
        if int(ptr[-1].detach().tolist()) != total:
            raise ValueError(f"{name}_ptr must end at {total}, got {int(ptr[-1].detach().tolist())}.")

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
        if bool((local_indices < _ZERO).any().detach().tolist()):
            raise ValueError(f"{name} contains negative indices.")
        node_start = node_ptr[graph_ids]
        node_end = node_ptr[graph_ids + _ONE]
        in_range = (local_indices >= node_start) & (local_indices < node_end)
        if bool((~in_range).any().detach().tolist()):
            raise ValueError(f"{name} indices fall outside per-graph node ranges.")

    @staticmethod
    def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().detach().tolist()
            raise ValueError(f"{name} contains {bad} non-finite values.")


@dataclass(frozen=True)
class EdgeTokenInputs:
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
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
    node_embedding_ids: torch.Tensor
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
        cvt_init: nn.Module,
        cvt_mask: Optional[torch.Tensor] = None,
        require_precomputed_edge_batch: bool = True,
    ) -> None:
        self.backbone = backbone
        self.cvt_init = cvt_init
        self._cvt_mask = cvt_mask
        self.require_precomputed_edge_batch = bool(require_precomputed_edge_batch)

    @staticmethod
    def _get_node_ptr(batch: Any, device: torch.device) -> torch.Tensor:
        return batch.ptr.to(device=device, non_blocking=True)

    @staticmethod
    def _get_start_answer_ptrs(
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        override_nodes = getattr(batch, "start_override_locals", None)
        override_ptr = getattr(batch, "start_override_ptr", None)
        if torch.is_tensor(override_nodes) or torch.is_tensor(override_ptr):
            if not torch.is_tensor(override_nodes) or not torch.is_tensor(override_ptr):
                raise ValueError("start_override_locals/start_override_ptr must be provided together.")
            start_node_locals = override_nodes.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
            start_ptr = override_ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        else:
            start_node_locals = batch.q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
            slice_dict = batch._slice_dict
            start_ptr = torch.as_tensor(slice_dict["q_local_indices"], dtype=torch.long, device=device)
        answer_node_locals = batch.a_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = batch._slice_dict
        answer_ptr = torch.as_tensor(slice_dict["a_local_indices"], dtype=torch.long, device=device)
        return start_node_locals, answer_node_locals, start_ptr, answer_ptr

    @staticmethod
    def _get_retrieval_failure(
        batch: Any,
        *,
        device: torch.device,
        num_graphs: int,
    ) -> torch.Tensor:
        flag = getattr(batch, "retrieval_failure", None)
        if flag is None:
            return torch.zeros((num_graphs,), device=device, dtype=torch.bool)
        flag = torch.as_tensor(flag, dtype=torch.bool, device=device).view(-1)
        if flag.numel() != num_graphs:
            raise ValueError("retrieval_failure length mismatch with num_graphs.")
        return flag

    @staticmethod
    def _get_q_local_indices(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)

    @staticmethod
    def build_dummy_mask(
        *,
        answer_ptr: torch.Tensor,
        retrieval_failure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        answer_counts = answer_ptr[1:] - answer_ptr[:-1]
        missing = answer_counts == _ZERO
        if retrieval_failure is not None:
            retrieval_failure = retrieval_failure.to(device=answer_ptr.device, dtype=torch.bool).view(-1)
            if retrieval_failure.numel() != int(answer_ptr.numel() - 1):
                raise ValueError("retrieval_failure length mismatch with batch size.")
            # 对于检索失败的样本，不作为 dummy；无标记的缺失仍视为错误。
            unchecked = missing & (~retrieval_failure)
            if bool(unchecked.any().detach().tolist()):
                raise ValueError("Answer nodes missing without retrieval_failure flag; filter data.")
            missing = torch.zeros_like(missing, dtype=torch.bool, device=answer_ptr.device)
        else:
            if bool(missing.any().detach().tolist()):
                raise ValueError("Answer nodes missing in batch; filter missing answers before GFlowNet.")
        return missing.to(dtype=torch.bool)

    def _resolve_edge_batch_and_ptr(
        self,
        batch: Any,
        *,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_edges = int(edge_index.size(1))
        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None and edge_ptr is None:
            if self.require_precomputed_edge_batch:
                raise ValueError("Batch missing edge_batch/edge_ptr; enable precompute_edge_batch.")
            edge_batch, edge_ptr = compute_edge_batch(
                edge_index,
                node_ptr=node_ptr,
                num_graphs=num_graphs,
                device=edge_index.device,
                validate=False,
            )
            return (
                edge_batch.to(device=edge_index.device, dtype=torch.long).view(-1),
                edge_ptr.to(device=edge_index.device, dtype=torch.long).view(-1),
            )
        if not (torch.is_tensor(edge_batch) and torch.is_tensor(edge_ptr)):
            raise ValueError("edge_batch and edge_ptr must be provided together.")
        edge_batch = edge_batch.to(device=edge_index.device, dtype=torch.long, non_blocking=True).view(-1)
        edge_ptr = edge_ptr.to(device=edge_index.device, dtype=torch.long, non_blocking=True).view(-1)
        if edge_batch.numel() != num_edges:
            raise ValueError("edge_batch length mismatch with edge_index.")
        if edge_ptr.numel() != num_graphs + _ONE:
            raise ValueError("edge_ptr length mismatch with num_graphs.")
        return edge_batch, edge_ptr

    def _resolve_node_is_cvt(
        self,
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self._cvt_mask is None:
            return None
        node_global_ids = getattr(batch, "node_global_ids", None)
        if node_global_ids is None:
            return None
        node_global_ids = torch.as_tensor(node_global_ids, dtype=torch.long, device=self._cvt_mask.device)
        if node_global_ids.numel() == _ZERO:
            return torch.zeros_like(node_global_ids, dtype=torch.bool, device=device)
        max_id = int(node_global_ids.max().detach().tolist())
        if max_id >= int(self._cvt_mask.numel()):
            raise ValueError("node_global_ids exceeds entity_vocab size for CVT mask.")
        node_is_cvt = self._cvt_mask.index_select(0, node_global_ids)
        return node_is_cvt.to(device=device, dtype=torch.bool, non_blocking=True).view(-1)

    def _apply_cvt_init(
        self,
        batch: Any,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        node_is_cvt = self._resolve_node_is_cvt(batch, device=device)
        if node_is_cvt is None:
            available = None
            if hasattr(batch, "keys"):
                try:
                    available = list(batch.keys())
                except Exception:  # pragma: no cover - best-effort diagnostics
                    available = None
            batch_type = type(batch).__name__
            has_ptr = hasattr(batch, "ptr")
            has_node_global_ids = hasattr(batch, "node_global_ids")
            raise ValueError(
                "node_is_cvt missing; CVT initialization requires cvt_mask + node_global_ids from entity_vocab. "
                f"batch_type={batch_type} has_ptr={has_ptr} has_node_global_ids={has_node_global_ids} "
                f"available_fields={available}"
            )
        return self.cvt_init(
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_index=edge_index,
            node_is_cvt=node_is_cvt,
        )

    def encode_batch_tokens(
        self,
        batch: Any,
        *,
        device: torch.device,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens, relation_tokens, question_tokens = self.backbone(batch)
        node_tokens = node_tokens.to(device=device, non_blocking=True)
        relation_tokens = relation_tokens.to(device=device, non_blocking=True)
        question_tokens = question_tokens.to(device=device, non_blocking=True)
        node_tokens = self._apply_cvt_init(
            batch,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_index=edge_index,
            device=device,
        )
        target_dtype = node_tokens.dtype
        if relation_tokens.dtype != target_dtype:
            relation_tokens = relation_tokens.to(dtype=target_dtype)
        if question_tokens.dtype != target_dtype:
            question_tokens = question_tokens.to(dtype=target_dtype)
        return node_tokens, relation_tokens, question_tokens

    def prepare_edge_token_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> EdgeTokenInputs:
        node_ptr = self._get_node_ptr(batch, device)
        edge_index = batch.edge_index.to(device=device, non_blocking=True)
        edge_relations = batch.edge_attr.to(device=device, non_blocking=True)
        edge_batch_full, edge_ptr = self._resolve_edge_batch_and_ptr(
            batch,
            edge_index=edge_index,
            node_ptr=node_ptr,
            num_graphs=int(node_ptr.numel() - 1),
        )
        node_tokens, relation_tokens, question_tokens = self.encode_batch_tokens(
            batch,
            device=device,
            edge_index=edge_index,
            edge_batch=edge_batch_full,
        )
        return EdgeTokenInputs(
            edge_batch=edge_batch_full,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
        )

    @staticmethod
    def _get_node_embedding_ids(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.node_embedding_ids.to(device=device, dtype=torch.long, non_blocking=True)

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
        node_embedding_ids: torch.Tensor,
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
            node_embedding_ids=node_embedding_ids,
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
        node_embedding_ids = self._get_node_embedding_ids(batch, device=device)

        start_node_locals, answer_node_locals, start_ptr, answer_ptr = self._get_start_answer_ptrs(
            batch, device=device
        )
        retrieval_failure = self._get_retrieval_failure(batch, device=device, num_graphs=num_graphs)
        dummy_mask = self.build_dummy_mask(answer_ptr=answer_ptr, retrieval_failure=retrieval_failure)
        edge_ptr = token_inputs.edge_ptr
        return self._build_rollout_inputs(
            edge_batch=token_inputs.edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=token_inputs.edge_index,
            edge_relations=token_inputs.edge_relations,
            node_tokens=token_inputs.node_tokens,
            relation_tokens=token_inputs.relation_tokens,
            question_tokens=token_inputs.question_tokens,
            node_embedding_ids=node_embedding_ids,
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

    @staticmethod
    def compute_min_start_nodes(
        *,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_node_locals = start_node_locals.to(device=device, dtype=torch.long).view(-1)
        start_ptr = start_ptr.to(device=device, dtype=torch.long).view(-1)
        if num_graphs <= _ZERO:
            empty = torch.zeros((_ZERO,), device=device, dtype=torch.long)
            return empty, empty.to(dtype=torch.bool)
        start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).clamp(min=_ZERO)
        has_start = start_counts > _ZERO
        if start_node_locals.numel() == _ZERO:
            return torch.zeros((num_graphs,), device=device, dtype=torch.long), has_start
        graph_ids = torch.repeat_interleave(
            torch.arange(num_graphs, device=device),
            start_counts,
        )
        sentinel = torch.full(
            (num_graphs,),
            torch.iinfo(start_node_locals.dtype).max,
            device=device,
            dtype=start_node_locals.dtype,
        )
        min_start = sentinel.scatter_reduce_(0, graph_ids, start_node_locals, reduce="amin", include_self=True)
        min_start = torch.where(has_start, min_start, torch.zeros_like(min_start))
        return min_start, has_start

    @staticmethod
    def compute_single_start_nodes(
        *,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_node_locals = start_node_locals.to(device=device, dtype=torch.long).view(-1)
        start_ptr = start_ptr.to(device=device, dtype=torch.long).view(-1)
        if num_graphs <= _ZERO:
            empty = torch.zeros((_ZERO,), device=device, dtype=torch.long)
            return empty, empty.to(dtype=torch.bool)
        if start_ptr.numel() != num_graphs + _ONE:
            raise ValueError("start_ptr length mismatch for single start selection.")
        start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).clamp(min=_ZERO)
        has_start = start_counts > _ZERO
        if start_counts.numel() > _ZERO and bool((start_counts > _ONE).any().detach().tolist()):
            raise ValueError("Multiple start nodes per graph; enable expand_multi_start to flatten starts.")
        if start_node_locals.numel() == _ZERO:
            return torch.zeros((num_graphs,), device=device, dtype=torch.long), has_start
        total = int(start_counts.sum().detach().tolist())
        if total != int(start_node_locals.numel()):
            raise ValueError("start_node_locals length mismatch with start_ptr counts.")
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=device), start_counts)
        start_nodes = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        if start_node_locals.numel() > _ZERO:
            start_nodes.index_copy_(0, graph_ids, start_node_locals)
        return start_nodes, has_start

    def build_graph_cache(self, inputs: RolloutInputs, *, device: torch.device) -> Dict[str, torch.Tensor]:
        node_ptr = inputs.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].detach().tolist()) if node_ptr.numel() > 0 else 0
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
            "node_embedding_ids": inputs.node_embedding_ids,
            "node_batch": node_batch,
            "node_is_start": node_is_start,
            "node_is_answer": node_is_answer,
            # target aliases for unified naming
            "node_is_target": node_is_answer,
            "start_node_locals": inputs.start_node_locals,
            "start_ptr": inputs.start_ptr,
            "answer_node_locals": inputs.answer_node_locals,
            "answer_ptr": inputs.answer_ptr,
            "target_node_locals": inputs.answer_node_locals,
            "target_ptr": inputs.answer_ptr,
            "dummy_mask": inputs.dummy_mask,
        }
        return graph_cache


__all__ = [
    "compute_policy_log_probs",
    "gumbel_noise_like",
    "neg_inf_value",
    "segment_logsumexp_1d",
    "scatter_logsumexp",
    "GFlowNetInputValidator",
    "EdgeTokenInputs",
    "GFlowNetBatchProcessor",
    "RolloutInputs",
]
