from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch

from src.metrics.common import extract_sample_ids
from src.models.components.graph_ops import (
    build_edge_head_csr_from_mask,
    build_edge_tail_csr_from_mask,
    gumbel_noise_like,
    segment_logsumexp_1d,
    segment_max,
)
from .constants import (
    _NEG_ONE,
    _ANSWER_POOLINGS,
    _SELF_RELATION_ID,
)
from .types import _PreparedBatch


class DualFlowBatchMixin:
    @staticmethod
    def _compute_log_denom(*, logits: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if num_graphs <= 0:
            return torch.zeros((0,), device=logits.device, dtype=logits.dtype)
        edge_batch = edge_batch.view(-1)
        if edge_batch.device != logits.device:
            edge_batch = edge_batch.to(device=logits.device)
        if edge_batch.dtype != torch.long:
            edge_batch = edge_batch.to(dtype=torch.long)
        counts = torch.bincount(edge_batch, minlength=num_graphs)
        log_denom = segment_logsumexp_1d(logits, edge_batch, num_graphs)
        neg_inf = torch.finfo(logits.dtype).min
        return torch.where(counts > 0, log_denom, torch.full_like(log_denom, neg_inf))

    @staticmethod
    def _ensure_tensor(
        value: torch.Tensor,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)
        tensor = value
        if tensor.device != device:
            return tensor.to(device=device, dtype=dtype or tensor.dtype, non_blocking=non_blocking)
        if dtype is not None and tensor.dtype != dtype:
            return tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _build_dummy_mask(*, answer_ptr: torch.Tensor) -> torch.Tensor:
        return answer_ptr[1:] == answer_ptr[:-1]

    @staticmethod
    def _build_node_batch(*, node_ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
        num_graphs = node_ptr.numel() - 1
        if num_graphs <= 0:
            return torch.zeros((0,), device=device, dtype=torch.long)
        counts = node_ptr[1:] - node_ptr[:-1]
        return torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)

    def _resolve_node_is_cvt(
        self,
        node_global_ids: torch.Tensor,
        *,
        num_nodes_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self._cvt_enabled:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        cvt_mask = self._cvt_mask
        if cvt_mask is None:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        node_global_ids = node_global_ids.view(-1)
        if node_global_ids.numel() != num_nodes_total:
            raise ValueError("node_global_ids length mismatch with ptr.")
        return cvt_mask.to(device=device, dtype=torch.bool).index_select(0, node_global_ids)

    @staticmethod
    def _build_node_mask(num_nodes_total: int, indices: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((num_nodes_total,), device=indices.device, dtype=torch.bool)
        if indices.numel() > 0:
            valid = indices >= 0
            safe = indices[valid]
            if safe.numel() > 0:
                mask[safe] = True
        return mask

    @staticmethod
    def _build_anchor_tokens(*, node_tokens: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.view(-1)
        if node_ids.device != node_tokens.device:
            node_ids = node_ids.to(device=node_tokens.device)
        if node_ids.dtype != torch.long:
            node_ids = node_ids.to(dtype=torch.long)
        if node_ids.numel() == 0:
            return torch.zeros((0, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        safe = node_ids.clamp(min=0)
        tokens = node_tokens.index_select(0, safe)
        valid = node_ids >= 0
        return torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

    @staticmethod
    def _pool_answer_tokens(
        *,
        node_tokens: torch.Tensor,
        answer_indices: torch.Tensor,
        answer_ptr: torch.Tensor,
        num_graphs: int,
        pooling: str,
    ) -> torch.Tensor:
        if num_graphs <= 0:
            return torch.zeros((0, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        answer_indices = answer_indices.view(-1)
        answer_ptr = answer_ptr.view(-1)
        if answer_ptr.numel() != num_graphs + 1:
            raise ValueError("answer_ptr length mismatch with num_graphs.")
        out = torch.zeros((num_graphs, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        counts = (answer_ptr[1:] - answer_ptr[:-1]).clamp(min=0)
        if answer_indices.numel() == 0:
            return out
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=node_tokens.device), counts)
        answer_sel = node_tokens.index_select(0, answer_indices)
        pooling = str(pooling).strip().lower()
        if pooling not in _ANSWER_POOLINGS:
            raise ValueError(f"answer pooling must be one of {sorted(_ANSWER_POOLINGS)}, got {pooling!r}.")
        if pooling == "mean":
            out.index_add_(0, graph_ids, answer_sel)
            denom = counts.to(dtype=out.dtype).clamp(min=1).unsqueeze(-1)
            return out / denom
        if pooling == "max":
            neg_inf = torch.finfo(out.dtype).min
            out.fill_(neg_inf)
            index = graph_ids.view(-1, 1).expand(-1, answer_sel.size(-1))
            out.scatter_reduce_(0, index, answer_sel, reduce="amax", include_self=True)
            return torch.where(counts.view(-1, 1) > 0, out, torch.zeros_like(out))
        if pooling == "logsumexp":
            neg_inf = torch.finfo(out.dtype).min
            max_per = torch.full_like(out, neg_inf)
            index = graph_ids.view(-1, 1).expand(-1, answer_sel.size(-1))
            max_per.scatter_reduce_(0, index, answer_sel, reduce="amax", include_self=True)
            max_sel = max_per.index_select(0, graph_ids)
            exp = torch.exp(answer_sel - max_sel)
            sum_per = torch.zeros_like(out)
            sum_per.index_add_(0, graph_ids, exp)
            eps = torch.finfo(out.dtype).eps
            logsumexp = torch.log(sum_per.clamp(min=eps)) + max_per
            return torch.where(counts.view(-1, 1) > 0, logsumexp, torch.zeros_like(out))
        return out

    @staticmethod
    def _edge_reorder_perm(
        *,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        node_ptr: torch.Tensor,
        num_edges_before: int,
    ) -> Optional[torch.Tensor]:
        if edge_index.size(1) != num_edges_before:
            raise ValueError("edge_index length mismatch before reorder.")
        if edge_relations.numel() != num_edges_before:
            raise ValueError("edge_relations length mismatch before reorder.")
        if edge_batch.numel() != num_edges_before:
            raise ValueError("edge_batch length mismatch before reorder.")
        if edge_index.numel() == 0:
            return None
        num_graphs = node_ptr.numel() - 1
        if num_graphs <= 0:
            return None
        edge_batch = edge_batch.view(-1)
        if edge_batch.numel() <= 1:
            return None
        if (edge_batch[:-1] <= edge_batch[1:]).all().item():
            return None
        order = torch.argsort(edge_batch)
        return order

    @staticmethod
    def _reorder_edge_inverse_map(
        *,
        edge_inverse_map: torch.Tensor,
        perm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if perm is None or edge_inverse_map.numel() == 0:
            return edge_inverse_map
        if edge_inverse_map.numel() != perm.numel():
            raise ValueError("edge_inverse_map length mismatch with perm.")
        perm = perm.view(-1)
        edge_inverse_map = edge_inverse_map.index_select(0, perm)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        valid = edge_inverse_map >= 0
        safe = torch.where(valid, edge_inverse_map, torch.zeros_like(edge_inverse_map))
        mapped = inv_perm.index_select(0, safe)
        return torch.where(valid, mapped, edge_inverse_map)

    @staticmethod
    def _resolve_context_tokens(context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.dim() == 2:
            return context_tokens
        if context_tokens.dim() == 3 and context_tokens.size(1) == 1:
            return context_tokens.squeeze(1)
        raise ValueError("context_tokens must be [num_graphs, hidden_dim].")

    def _build_forward_context(
        self,
        *,
        question_tokens: torch.Tensor,
        start_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.size(0) != start_tokens.size(0):
            raise ValueError("question_tokens and start_tokens must align on batch dimension.")
        fused = torch.cat((question_tokens, start_tokens), dim=-1)
        return self.forward_ctx_proj(fused)

    def _build_backward_context(
        self,
        *,
        question_tokens: torch.Tensor,
        start_tokens: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.size(0) != start_tokens.size(0) or question_tokens.size(0) != answer_tokens.size(0):
            raise ValueError("question_tokens, start_tokens, and answer_tokens must align on batch dimension.")
        fused = torch.cat((question_tokens, start_tokens, answer_tokens), dim=-1)
        return self.backward_ctx_proj(fused)

    @staticmethod
    def _build_step_ids(*, num_graphs: int, step: int, device: torch.device) -> torch.Tensor:
        return torch.full((num_graphs,), step, device=device, dtype=torch.long)

    def _select_start_nodes(
        self,
        *,
        question_tokens: torch.Tensor,
        node_tokens_fwd: torch.Tensor,
        node_tokens_bwd: torch.Tensor,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = ptr.view(-1)
        local_indices = local_indices.view(-1)
        counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
        if not allow_empty:
            torch._assert((counts > 0).all(), f"{name} missing in batch; filter data.")
        num_graphs = counts.numel()
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        hidden_dim = node_tokens_fwd.size(-1)
        if local_indices.numel() == 0 or num_graphs == 0:
            zeros = torch.zeros((num_graphs, hidden_dim), device=node_tokens_fwd.device, dtype=node_tokens_fwd.dtype)
            return out, zeros, zeros
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        question_tokens = self._resolve_context_tokens(question_tokens)
        question_sel = question_tokens.index_select(0, graph_ids)
        node_sel_fwd = node_tokens_fwd.index_select(0, local_indices)
        node_sel_bwd = node_tokens_bwd.index_select(0, local_indices)
        if temperature <= 0:
            raise ValueError("start_selector temperature must be > 0.")
        logits = self.start_selector(torch.cat((question_sel, node_sel_fwd), dim=-1)).view(-1)
        logits_scaled = logits / temperature
        log_denom = segment_logsumexp_1d(logits_scaled, graph_ids, num_graphs)
        soft_weights = torch.exp(logits_scaled - log_denom.index_select(0, graph_ids))
        noise = gumbel_noise_like(torch.zeros_like(logits_scaled, dtype=torch.float32))
        scores = logits_scaled + noise.to(dtype=logits_scaled.dtype)
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > 0
        hard_weights = torch.zeros_like(logits)
        argmax_valid = argmax[valid]
        if argmax_valid.numel() > 0:
            hard_weights.index_put_((argmax_valid,), torch.ones_like(argmax_valid, dtype=logits.dtype))
        # Straight-through: hard selection forward, soft gradients backward.
        weights = hard_weights - soft_weights.detach() + soft_weights
        start_nodes = torch.where(valid, local_indices.index_select(0, argmax), out)
        start_tokens_fwd = torch.zeros((num_graphs, hidden_dim), device=node_sel_fwd.device, dtype=node_sel_fwd.dtype)
        start_tokens_fwd.index_add_(0, graph_ids, node_sel_fwd * weights.unsqueeze(-1))
        start_tokens_bwd = torch.zeros((num_graphs, hidden_dim), device=node_sel_bwd.device, dtype=node_sel_bwd.dtype)
        start_tokens_bwd.index_add_(0, graph_ids, node_sel_bwd * weights.unsqueeze(-1))
        return start_nodes, start_tokens_fwd, start_tokens_bwd

    @staticmethod
    def _sample_nodes_uniform(
        *,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
    ) -> torch.Tensor:
        ptr = ptr.view(-1)
        local_indices = local_indices.view(-1)
        counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
        if not allow_empty:
            torch._assert((counts > 0).all(), f"{name} missing in batch; filter data.")
        num_graphs = counts.numel()
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        if local_indices.numel() == 0 or num_graphs == 0:
            return out
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        scores = gumbel_noise_like(torch.zeros_like(local_indices, dtype=torch.float32))
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > 0
        out = torch.where(valid, local_indices.index_select(0, argmax), out)
        return out

    @staticmethod
    def _extract_graph_stats(batch: Any) -> tuple[int, int]:
        num_graphs = getattr(batch, "num_graphs", None)
        num_nodes_total = getattr(batch, "num_nodes_total", None)
        if num_graphs is None or num_nodes_total is None:
            raise AttributeError("Batch missing num_graphs/num_nodes_total; ensure collate precomputes graph stats.")
        return int(num_graphs), int(num_nodes_total)

    def _extract_graph_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        node_ptr = getattr(batch, "ptr", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        if not torch.is_tensor(node_ptr) or not torch.is_tensor(edge_index) or not torch.is_tensor(edge_attr):
            raise AttributeError("Batch missing ptr/edge_index/edge_attr required for DualFlow.")
        node_ptr = self._ensure_tensor(node_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_index = self._ensure_tensor(edge_index, device=device, dtype=torch.long, non_blocking=True)
        edge_relations = self._ensure_tensor(edge_attr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None or edge_ptr is None:
            raise AttributeError(
                "Batch missing edge_batch/edge_ptr; enable data.precompute_edge_batch in the collator."
            )
        edge_batch = self._ensure_tensor(edge_batch, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_ptr = self._ensure_tensor(edge_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        return node_ptr, edge_index, edge_relations, edge_batch, edge_ptr

    def _extract_index_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        q_local_indices = getattr(batch, "q_local_indices", None)
        a_local_indices = getattr(batch, "a_local_indices", None)
        if not torch.is_tensor(q_local_indices) or not torch.is_tensor(a_local_indices):
            raise AttributeError("Batch missing q_local_indices/a_local_indices required for DualFlow.")
        q_local_indices = self._ensure_tensor(q_local_indices, device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_local_indices = self._ensure_tensor(a_local_indices, device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = getattr(batch, "_slice_dict")
        q_ptr = self._ensure_tensor(slice_dict["q_local_indices"], device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_ptr = self._ensure_tensor(slice_dict["a_local_indices"], device=device, dtype=torch.long, non_blocking=True).view(-1)
        answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr is None and hasattr(batch, "_slice_dict"):
            answer_ptr = batch._slice_dict.get("answer_entity_ids")
        if answer_ptr is None:
            raise AttributeError("Batch missing answer_entity_ids_ptr required for DualFlow.")
        answer_ptr = self._ensure_tensor(answer_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        return q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr

    def _extract_embeddings(
        self,
        batch: Any,
        *,
        edge_index: torch.Tensor,
        node_is_cvt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        question_emb = getattr(batch, "question_emb", None)
        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if not torch.is_tensor(question_emb):
            raise AttributeError("Batch missing question_emb required for DualFlow.")
        if not torch.is_tensor(node_embeddings) or not torch.is_tensor(edge_embeddings):
            raise AttributeError("Batch missing node_embeddings/edge_embeddings required for DualFlow.")
        question_emb = self._ensure_tensor(question_emb, device=device, non_blocking=True)
        node_embeddings = self._ensure_tensor(node_embeddings, device=device, non_blocking=True)
        edge_embeddings = self._ensure_tensor(edge_embeddings, device=device, non_blocking=True)
        return question_emb, node_embeddings, edge_embeddings

    def _prepare_batch(self, batch: Any, *, build_bwd: bool = True) -> tuple[_PreparedBatch, Optional[_PreparedBatch]]:
        num_graphs, num_nodes_total = self._extract_graph_stats(batch)
        node_ptr, edge_index, edge_relations, edge_batch, edge_ptr = self._extract_graph_tensors(batch)
        q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr = self._extract_index_tensors(batch)
        if edge_index.numel() > 0:
            torch._assert((edge_index >= 0).all(), "edge_index contains negative values.")
            torch._assert((edge_index < num_nodes_total).all(), "edge_index out of range for num_nodes_total.")
        if q_local_indices.numel() > 0:
            torch._assert((q_local_indices >= 0).all(), "q_local_indices contains negative values.")
            torch._assert((q_local_indices < num_nodes_total).all(), "q_local_indices out of range.")
        if a_local_indices.numel() > 0:
            torch._assert((a_local_indices >= 0).all(), "a_local_indices contains negative values.")
            torch._assert((a_local_indices < num_nodes_total).all(), "a_local_indices out of range.")
        node_global_ids = getattr(batch, "node_global_ids", None)
        if not torch.is_tensor(node_global_ids):
            raise AttributeError("Batch missing node_global_ids required for DualFlow.")
        node_global_ids = self._ensure_tensor(
            node_global_ids, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        edge_inverse_map = getattr(batch, "edge_inverse_map", None)
        if not torch.is_tensor(edge_inverse_map):
            raise AttributeError("Batch missing edge_inverse_map; enable data.precompute_edge_inverse_map in collator.")
        edge_inverse_map = self._ensure_tensor(
            edge_inverse_map, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        if edge_inverse_map.numel() != edge_index.size(1):
            raise ValueError("edge_inverse_map length mismatch with edge_index.")
        dummy_mask = self._build_dummy_mask(answer_ptr=answer_ptr)
        node_batch = self._build_node_batch(node_ptr=node_ptr, device=self.device)
        node_is_cvt = self._resolve_node_is_cvt(node_global_ids, num_nodes_total=num_nodes_total, device=self.device)
        question_emb, node_embeddings, edge_embeddings = self._extract_embeddings(
            batch,
            edge_index=edge_index,
            node_is_cvt=node_is_cvt,
        )
        node_embeddings_fwd = node_embeddings
        node_embeddings_bwd = node_embeddings
        if self._cvt_enabled:
            node_embeddings_fwd = self.cvt_init_fwd(
                node_embeddings=node_embeddings_fwd,
                relation_embeddings=edge_embeddings,
                edge_index=edge_index,
                node_is_cvt=node_is_cvt,
            )
            if build_bwd:
                node_embeddings_bwd = self.cvt_init_bwd(
                    node_embeddings=node_embeddings_bwd,
                    relation_embeddings=edge_embeddings,
                    edge_index=edge_index,
                    node_is_cvt=node_is_cvt,
                )
        if edge_embeddings.size(0) != edge_index.size(1):
            raise ValueError("edge_embeddings length must match edge_index.")
        perm = self._edge_reorder_perm(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            node_ptr=node_ptr,
            num_edges_before=edge_index.size(1),
        )
        if perm is not None:
            edge_index = edge_index.index_select(1, perm)
            edge_batch = edge_batch.index_select(0, perm)
            edge_relations = edge_relations.index_select(0, perm)
            edge_embeddings = edge_embeddings.index_select(0, perm)
        edge_inverse_map = self._reorder_edge_inverse_map(edge_inverse_map=edge_inverse_map, perm=perm)
        node_tokens_fwd = self.backbone_fwd.project_node_embeddings(node_embeddings_fwd)
        node_tokens_bwd = node_tokens_fwd if not build_bwd else self.backbone_bwd.project_node_embeddings(node_embeddings_bwd)
        relation_tokens_fwd = self.backbone_fwd.project_relation_embeddings(edge_embeddings)
        relation_tokens_bwd = relation_tokens_fwd if not build_bwd else self.backbone_bwd.project_relation_embeddings(edge_embeddings)
        node_tokens_fwd = self.backbone_fwd.encode_graph(
            node_tokens=node_tokens_fwd,
            relation_tokens=relation_tokens_fwd,
            edge_index=edge_index,
            num_nodes=num_nodes_total,
        )
        if build_bwd:
            node_tokens_bwd = self.backbone_bwd.encode_graph(
                node_tokens=node_tokens_bwd,
                relation_tokens=relation_tokens_bwd,
                edge_index=edge_index,
                num_nodes=num_nodes_total,
            )
        question_tokens_fwd_base = self._resolve_context_tokens(
            self.backbone_fwd.project_question_embeddings(question_emb)
        )
        question_tokens_bwd_base = (
            question_tokens_fwd_base
            if not build_bwd
            else self._resolve_context_tokens(self.backbone_bwd.project_question_embeddings(question_emb))
        )
        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError("edge_ptr length mismatch with batch graph count.")
        start_temperature = self._resolve_start_temperature()
        start_nodes_fwd, start_tokens_fwd, start_tokens_bwd = self._select_start_nodes(
            question_tokens=question_tokens_fwd_base,
            node_tokens_fwd=node_tokens_fwd,
            node_tokens_bwd=node_tokens_bwd,
            local_indices=q_local_indices,
            ptr=q_ptr,
            allow_empty=False,
            name="q_local_indices",
            temperature=start_temperature,
        )
        answer_tokens_bwd = None
        context_tokens_bwd = None
        if build_bwd:
            pooling = self._resolve_answer_pooling()
            answer_tokens_bwd = self._pool_answer_tokens(
                node_tokens=node_tokens_bwd,
                answer_indices=a_local_indices,
                answer_ptr=a_ptr,
                num_graphs=num_graphs,
                pooling=pooling,
            )
        context_tokens_fwd = self._build_forward_context(
            question_tokens=question_tokens_fwd_base,
            start_tokens=start_tokens_fwd,
        )
        if build_bwd:
            context_tokens_bwd = self._build_backward_context(
                question_tokens=question_tokens_bwd_base,
                start_tokens=start_tokens_bwd,
                answer_tokens=answer_tokens_bwd if answer_tokens_bwd is not None else start_tokens_bwd,
            )
        inverse_map = self._relation_inverse_map
        inverse_mask = self._relation_inverse_mask
        if inverse_map is None or inverse_mask is None:
            raise RuntimeError("relation inverse assets are required but not initialized.")
        inverse_map = inverse_map.to(device=edge_relations.device, dtype=torch.long)
        inverse_mask = inverse_mask.to(device=edge_relations.device, dtype=torch.bool)
        edge_is_inverse = self._build_edge_inverse_mask(edge_relations=edge_relations, inverse_mask=inverse_mask)
        self_loop_mask = edge_relations == _SELF_RELATION_ID
        edge_mask_fwd = self._build_edge_direction_mask(
            edge_is_inverse=edge_is_inverse, self_loop_mask=self_loop_mask, forward=True
        )
        edge_mask_bwd = self._build_edge_direction_mask(
            edge_is_inverse=edge_is_inverse, self_loop_mask=self_loop_mask, forward=False
        )
        edge_ids_by_head_fwd, edge_ptr_by_head_fwd = build_edge_head_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_tail_fwd, edge_ptr_by_tail_fwd = build_edge_tail_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_head_bwd, edge_ptr_by_head_bwd = build_edge_head_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_tail_bwd, edge_ptr_by_tail_bwd = build_edge_tail_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        self._validate_edge_csr(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            edge_ids_by_head=edge_ids_by_head_fwd,
            edge_ptr_by_head=edge_ptr_by_head_fwd,
            edge_ids_by_tail=edge_ids_by_tail_fwd,
            edge_ptr_by_tail=edge_ptr_by_tail_fwd,
            num_nodes_total=num_nodes_total,
        )
        self._validate_edge_csr(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            edge_ids_by_head=edge_ids_by_head_bwd,
            edge_ptr_by_head=edge_ptr_by_head_bwd,
            edge_ids_by_tail=edge_ids_by_tail_bwd,
            edge_ptr_by_tail=edge_ptr_by_tail_bwd,
            num_nodes_total=num_nodes_total,
        )
        strict_inverse = self._resolve_strict_inverse()
        self._validate_edge_inverse_map(
            edge_inverse_map=edge_inverse_map,
            edge_relations=edge_relations,
            strict=strict_inverse,
        )
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")
        answer_entity_ids = getattr(batch, "answer_entity_ids", None)
        if not torch.is_tensor(answer_entity_ids):
            raise AttributeError("Batch missing answer_entity_ids required for DualFlow.")
        answer_entity_ids = self._ensure_tensor(
            answer_entity_ids, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        prepared_fwd = _PreparedBatch(
            num_graphs=num_graphs,
            num_nodes_total=num_nodes_total,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            question_emb_raw=question_emb,
            edge_embeddings_raw=edge_embeddings,
            node_embeddings=node_embeddings,
            node_tokens=node_tokens_fwd,
            relation_tokens=relation_tokens_fwd,
            context_tokens=context_tokens_fwd,
            node_batch=node_batch,
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            q_ptr=q_ptr,
            a_ptr=a_ptr,
            dummy_mask=dummy_mask,
            node_global_ids=node_global_ids,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
            sample_ids=sample_ids,
            start_nodes_fwd=start_nodes_fwd,
            start_tokens_fwd=start_tokens_fwd,
            start_tokens_bwd=start_tokens_bwd,
            edge_ids_by_head_fwd=edge_ids_by_head_fwd,
            edge_ptr_by_head_fwd=edge_ptr_by_head_fwd,
            edge_ids_by_tail_fwd=edge_ids_by_tail_fwd,
            edge_ptr_by_tail_fwd=edge_ptr_by_tail_fwd,
            edge_ids_by_head_bwd=edge_ids_by_head_bwd,
            edge_ptr_by_head_bwd=edge_ptr_by_head_bwd,
            edge_ids_by_tail_bwd=edge_ids_by_tail_bwd,
            edge_ptr_by_tail_bwd=edge_ptr_by_tail_bwd,
            edge_inverse_map=edge_inverse_map,
        )
        if not build_bwd:
            return prepared_fwd, None
        prepared_bwd = replace(
            prepared_fwd,
            node_tokens=node_tokens_bwd,
            relation_tokens=relation_tokens_bwd,
            context_tokens=context_tokens_bwd if context_tokens_bwd is not None else prepared_fwd.context_tokens,
        )
        return prepared_fwd, prepared_bwd

    @staticmethod
    def _build_edge_inverse_mask(*, edge_relations: torch.Tensor, inverse_mask: torch.Tensor) -> torch.Tensor:
        edge_relations = edge_relations.view(-1)
        mask = torch.zeros_like(edge_relations, dtype=torch.bool)
        valid = edge_relations >= 0
        if valid.any():
            mask[valid] = inverse_mask.index_select(0, edge_relations[valid])
        return mask

    @staticmethod
    def _build_edge_direction_mask(
        *,
        edge_is_inverse: torch.Tensor,
        self_loop_mask: torch.Tensor,
        forward: bool,
    ) -> torch.Tensor:
        base = ~edge_is_inverse if forward else edge_is_inverse
        return base | self_loop_mask

    @staticmethod
    def _validate_edge_csr(
        *,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        edge_ids_by_tail: torch.Tensor,
        edge_ptr_by_tail: torch.Tensor,
        num_nodes_total: int,
    ) -> None:
        if edge_index.numel() == 0:
            return
        num_edges = edge_index.size(1)
        mask = edge_mask.to(dtype=torch.bool)
        heads = edge_index[0]
        tails = edge_index[1]
        if edge_ids_by_head.numel() > 0:
            torch._assert((edge_ids_by_head >= 0).all(), "edge_ids_by_head contains negative values.")
            torch._assert((edge_ids_by_head < num_edges).all(), "edge_ids_by_head out of range.")
        if edge_ids_by_tail.numel() > 0:
            torch._assert((edge_ids_by_tail >= 0).all(), "edge_ids_by_tail contains negative values.")
            torch._assert((edge_ids_by_tail < num_edges).all(), "edge_ids_by_tail out of range.")
        expected = mask.sum()
        head_count = torch.tensor(edge_ids_by_head.numel(), device=edge_mask.device, dtype=expected.dtype)
        tail_count = torch.tensor(edge_ids_by_tail.numel(), device=edge_mask.device, dtype=expected.dtype)
        torch._assert(head_count == expected, "edge_ids_by_head length mismatch with edge_mask.")
        torch._assert(tail_count == expected, "edge_ids_by_tail length mismatch with edge_mask.")
        counts_head = torch.bincount(heads[mask], minlength=num_nodes_total)
        ptr_counts_head = edge_ptr_by_head[1:] - edge_ptr_by_head[:-1]
        torch._assert(torch.equal(counts_head, ptr_counts_head), "edge_ptr_by_head mismatch with edge_mask.")
        counts_tail = torch.bincount(tails[mask], minlength=num_nodes_total)
        ptr_counts_tail = edge_ptr_by_tail[1:] - edge_ptr_by_tail[:-1]
        torch._assert(torch.equal(counts_tail, ptr_counts_tail), "edge_ptr_by_tail mismatch with edge_mask.")

    @staticmethod
    def _validate_edge_inverse_map(
        *,
        edge_inverse_map: torch.Tensor,
        edge_relations: torch.Tensor,
        strict: bool,
    ) -> None:
        if not strict:
            return
        if edge_inverse_map.numel() == 0:
            return
        edge_inverse_map = edge_inverse_map.view(-1)
        edge_relations = edge_relations.view(-1)
        missing = (edge_relations >= 0) & (edge_inverse_map < 0)
        torch._assert(~missing.any(), "Missing inverse edges for relation pairs.")
        valid = edge_inverse_map >= 0
        inv_safe = edge_inverse_map[valid]
        idx = torch.arange(edge_inverse_map.numel(), device=edge_inverse_map.device, dtype=edge_inverse_map.dtype)[valid]
        back = edge_inverse_map.index_select(0, inv_safe)
        if not torch.equal(back, idx):
            raise ValueError("Edge inverse map is not symmetric.")


__all__ = ["DualFlowBatchMixin"]
