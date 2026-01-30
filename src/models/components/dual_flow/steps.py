from __future__ import annotations

from typing import Any

import torch

from src.metrics.common import extract_sample_ids
from src.utils import log_metric

from .constants import _ONE, _SCHED_INTERVAL_EPOCH, _SCHED_INTERVAL_STEP, _ZERO


class DualFlowStepsMixin:
    def forward(self, batch: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("DualFlowModule.forward is not supported; use training_step/eval.")

    def training_step(self, batch: Any, batch_idx: int):
        self._ensure_runtime_initialized()
        optimizer = self.optimizers()
        accum = float(self._accumulate_grad_batches())
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_training_loss(batch)
        metrics.update(self._collect_logit_scale_metrics())
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("train"))
        self.manual_backward(loss / accum)
        if self._should_step_optimizer(batch_idx):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
            if interval == _SCHED_INTERVAL_STEP:
                self._step_scheduler()
        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None:
            ptr = getattr(batch, "ptr", None)
            if ptr is None:
                raise AttributeError("Batch missing num_graphs/ptr required for logging batch_size.")
            ptr = torch.as_tensor(ptr)
            batch_size = int(ptr.numel() - _ONE)
        batch_size = int(batch_size)
        for name, value in metrics.items():
            log_metric(self, f"train/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        log_metric(self, "train/loss", loss.detach(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("val"))
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"val/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if name.startswith(("hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"val/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("test"))
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"test/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if name.startswith(("hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"test/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._ensure_runtime_initialized()
        _ = batch_idx, dataloader_idx
        prepared_fwd, _ = self._prepare_batch(batch, build_bwd=False)
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            return []
        valid_mask = ~prepared_fwd.dummy_mask
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        beam_size = self._resolve_beam_size()
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")

        predict_mode = str(self.runtime_cfg.get("predict_mode", "full")).strip().lower()
        lite_mode = predict_mode in {"lite", "light", "summary", "fast"}
        beams = None
        beam_state = None
        if lite_mode:
            beam_state = self._beam_search_state(
                prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target
            )
        else:
            beams = self._beam_search(prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target)
        rollouts_per_graph: list[list[dict[str, Any]]] = [[] for _ in range(num_graphs)]
        edge_index_cpu = prepared_fwd.edge_index.detach().cpu()
        edge_rel_cpu = prepared_fwd.edge_relations.detach().cpu()
        node_global_cpu = prepared_fwd.node_global_ids.detach().cpu()
        node_is_target_cpu = node_is_target.detach().cpu()
        edge_index_np = edge_index_cpu.numpy()
        edge_rel_np = edge_rel_cpu.numpy()
        node_global_np = node_global_cpu.numpy()
        node_is_target_np = node_is_target_cpu.numpy()
        if lite_mode and beam_state is not None:
            beam_nodes_np = beam_state.beam_nodes.detach().cpu().numpy()
            beam_scores_np = beam_state.beam_scores.detach().cpu().numpy()
            beam_paths_np = beam_state.beam_paths.detach().cpu().numpy()
            beam_lengths_np = beam_state.beam_lengths.detach().cpu().numpy()
            for graph_idx in range(num_graphs):
                for beam_idx in range(beam_state.beam_size):
                    stop_node = int(beam_nodes_np[graph_idx, beam_idx])
                    if stop_node < _ZERO:
                        continue
                    score = float(beam_scores_np[graph_idx, beam_idx])
                    length = int(beam_lengths_np[graph_idx, beam_idx])
                    if length <= _ZERO:
                        path = []
                    else:
                        path = beam_paths_np[graph_idx, beam_idx, :length].tolist()
                    success = bool(node_is_target_np[stop_node]) if stop_node >= _ZERO else False
                    rollouts_per_graph[graph_idx].append(
                        {
                            "rollout_index": beam_idx,
                            "score": score,
                            "path_edge_ids": path,
                            "stop_node_id": stop_node,
                            "reach_success": success,
                        }
                    )
        else:
            for graph_idx in range(num_graphs):
                beam = beams[graph_idx]
                for beam_idx, (stop_node, score, path) in enumerate(beam):
                    edges_list: list[dict[str, Any]] = []
                    for edge_id in path:
                        head = int(edge_index_np[_ZERO, edge_id])
                        tail = int(edge_index_np[_ONE, edge_id])
                        rel = int(edge_rel_np[edge_id])
                        head_ent = int(node_global_np[head])
                        tail_ent = int(node_global_np[tail])
                        edges_list.append(
                            {
                                "src_entity_id": head_ent,
                                "dst_entity_id": tail_ent,
                                "head_entity_id": head_ent,
                                "tail_entity_id": tail_ent,
                                "relation_id": rel,
                            }
                        )
                    stop_entity = int(node_global_np[stop_node]) if stop_node >= _ZERO else None
                    success = bool(node_is_target_np[stop_node]) if stop_node >= _ZERO else False
                    rollouts_per_graph[graph_idx].append(
                        {
                            "rollout_index": beam_idx,
                            "score": float(score),
                            "edges": edges_list,
                            "stop_node_entity_id": stop_entity,
                            "reach_success": success,
                        }
                    )

        node_ptr_cpu = prepared_fwd.node_ptr.detach().cpu()
        q_ptr_cpu = prepared_fwd.q_ptr.detach().cpu()
        a_ptr_cpu = prepared_fwd.answer_ptr.detach().cpu()
        q_local_cpu = prepared_fwd.q_local_indices.detach().cpu()
        answer_ids_cpu = prepared_fwd.answer_entity_ids.detach().cpu()
        node_ptr_np = node_ptr_cpu.numpy()
        q_ptr_np = q_ptr_cpu.numpy()
        a_ptr_np = a_ptr_cpu.numpy()
        answer_ids_np = answer_ids_cpu.numpy()
        records: list[dict[str, Any]] = []
        for graph_idx in range(num_graphs):
            node_start = int(node_ptr_np[graph_idx])
            node_end = int(node_ptr_np[graph_idx + _ONE])
            q_start = int(q_ptr_np[graph_idx])
            q_end = int(q_ptr_np[graph_idx + _ONE])
            start_indices = q_local_cpu[q_start:q_end].to(dtype=torch.long)
            start_entity_ids: list[int]
            if start_indices.numel() == _ZERO:
                start_entity_ids = []
            else:
                start_indices_np = start_indices.numpy()
                if (start_indices_np < _ZERO).any():
                    raise ValueError(f"q_local_indices contain negative values for sample_id={sample_ids[graph_idx]!r}.")
                if (start_indices_np >= num_nodes_total).any():
                    raise ValueError(f"q_local_indices out of range for sample_id={sample_ids[graph_idx]!r}.")
                in_graph = (start_indices_np >= node_start) & (start_indices_np < node_end)
                if not in_graph.all():
                    raise ValueError(f"q_local_indices mismatch node_ptr for sample_id={sample_ids[graph_idx]!r}.")
                start_entity_ids = node_global_np[start_indices_np].tolist()
            a_start = int(a_ptr_np[graph_idx])
            a_end = int(a_ptr_np[graph_idx + _ONE])
            answer_ids = answer_ids_np[a_start:a_end].tolist() if a_end > a_start else []
            record = {
                "sample_id": sample_ids[graph_idx],
                "start_entity_ids": start_entity_ids,
                "answer_entity_ids": answer_ids,
                "rollouts": rollouts_per_graph[graph_idx],
            }
            question_text = getattr(batch, "question", None)
            if isinstance(question_text, (list, tuple)) and graph_idx < len(question_text):
                record["question"] = question_text[graph_idx]
            elif isinstance(question_text, str):
                record["question"] = question_text
            records.append(record)
        return records

    def _accumulate_grad_batches(self) -> int:
        manual = self.training_cfg.get("accumulate_grad_batches", None)
        if manual is not None:
            return max(int(manual), _ONE)
        if self.trainer is None:
            return _ONE
        return max(int(getattr(self.trainer, "accumulate_grad_batches", _ONE) or _ONE), _ONE)

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        if self.trainer is None:
            return False
        total = getattr(self.trainer, "num_training_batches", None)
        if total is None:
            return False
        return (batch_idx + _ONE) >= int(total)

    def _should_zero_grad(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        return batch_idx % accum == _ZERO

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        if self._is_last_train_batch(batch_idx):
            return True
        return (batch_idx + _ONE) % accum == _ZERO


__all__ = ["DualFlowStepsMixin"]
