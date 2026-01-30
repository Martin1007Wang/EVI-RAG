from __future__ import annotations

from typing import Any

import torch

from .constants import _NEG_ONE, _ONE, _PB_MODE_TOPO_SEMANTIC, _TERMINAL_HIT, _TWO, _ZERO
class DualFlowEvalMixin:
    @torch.no_grad()
    def _compute_eval_metrics(self, batch: Any) -> tuple[dict[str, torch.Tensor], int]:
        build_bwd = not self._is_static_pb()
        prepared_fwd, prepared_bwd = self._prepare_batch(batch, build_bwd=build_bwd)
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            return {}, _ZERO
        valid_mask = ~prepared_fwd.dummy_mask
        valid_count_attr = getattr(batch, "num_valid_graphs", None)
        if valid_count_attr is None:
            raise AttributeError("Batch missing num_valid_graphs; collator must precompute answer counts.")
        valid_count = int(valid_count_attr)
        if valid_count <= _ZERO:
            return {}, _ZERO
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target_all = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        pb_cfg = None
        pb_distances = None
        if self._is_static_pb():
            pb_cfg = self._resolve_pb_cfg()
            if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC:
                precomputed = getattr(batch, "dist_to_start", None)
                if torch.is_tensor(precomputed):
                    precomputed = precomputed.to(device=prepared_fwd.edge_index.device, dtype=torch.long).view(-1)
                    if precomputed.numel() != num_nodes_total:
                        raise ValueError("dist_to_start length mismatch with num_nodes_total.")
                    pb_distances = precomputed
                else:
                    pb_distances = self._compute_distance_to_starts(
                        prepared=prepared_fwd,
                        max_hops=int(pb_cfg["max_hops"]),
                    )
        beam_size = self._resolve_beam_size()
        beam_state = self._beam_search_state(
            prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target_all
        )
        beam_nodes = beam_state.beam_nodes
        beam_lengths = beam_state.beam_lengths
        if beam_nodes.numel() == _ZERO:
            return {}, _ZERO
        beam_valid = beam_nodes >= _ZERO
        beam_nodes_safe = beam_nodes.clamp(min=_ZERO)
        beam_hits = node_is_target_all.index_select(0, beam_nodes_safe.view(-1)).view(num_graphs, -1)
        beam_hits = beam_hits & beam_valid
        hit_hits = beam_hits.any(dim=1).to(dtype=torch.float32)
        sorted_nodes, _ = torch.sort(beam_nodes, dim=1)
        sorted_valid = sorted_nodes >= _ZERO
        unique_mask = torch.ones_like(sorted_nodes, dtype=torch.bool)
        if sorted_nodes.size(1) > _ONE:
            unique_mask[:, _ONE:] = sorted_nodes[:, _ONE:] != sorted_nodes[:, :-_ONE]
        unique_mask = unique_mask & sorted_valid
        unique_nodes_safe = sorted_nodes.clamp(min=_ZERO)
        unique_hits = node_is_target_all.index_select(0, unique_nodes_safe.view(-1)).view(num_graphs, -1)
        unique_hit_counts = (unique_mask & unique_hits).sum(dim=1).to(dtype=torch.float32)
        unique_pred_counts = unique_mask.sum(dim=1).to(dtype=torch.float32)
        answer_counts = (prepared_fwd.a_ptr[_ONE:] - prepared_fwd.a_ptr[:-_ONE]).clamp(min=_ZERO).to(dtype=torch.float32)
        precision_scores = unique_hit_counts / unique_pred_counts.clamp(min=_ONE)
        recall_scores = unique_hit_counts / answer_counts.clamp(min=_ONE)
        denom = recall_scores + precision_scores
        f1_scores = torch.where(denom > float(_ZERO), (float(_TWO) * recall_scores * precision_scores / denom), torch.zeros_like(denom))
        beam_valid_counts = beam_valid.sum(dim=1).to(dtype=torch.float32)
        diversity_scores = unique_pred_counts / beam_valid_counts.clamp(min=_ONE)
        length = beam_lengths[:, _ZERO].to(dtype=torch.float32)
        metrics = {
            "hit@beam": hit_hits,
            "recall@beam": recall_scores,
            "precision@beam": precision_scores,
            "f1@beam": f1_scores,
            "diversity@beam": diversity_scores,
        }
        metrics["length_mean"] = length
        eval_temperature = float(_ONE)
        rollout_fwd = self._rollout_policy(
            policy=self.policy_fwd,
            prepared=prepared_fwd,
            graph_mask=valid_mask,
            start_nodes=prepared_fwd.start_nodes_fwd,
            node_is_target=node_is_target_all,
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            record_actions=True,
            record_log_pf=False,
            temperature=eval_temperature,
            context_tokens=prepared_fwd.context_tokens,
        )
        fwd_actions = rollout_fwd.actions
        if fwd_actions is None:
            fwd_actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=self.device, dtype=torch.long)
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=fwd_actions,
            graph_mask=valid_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target_all,
            sampling_temperature=eval_temperature,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & valid_mask
        metrics.update(db_metrics)
        metrics["rollout_success_rate"] = success.to(dtype=torch.float32).mean()
        metrics = self._reduce_eval_metrics(metrics, valid_mask=valid_mask)
        return metrics, valid_count

    @staticmethod
    def _reduce_eval_metrics(
        metrics: dict[str, torch.Tensor],
        *,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not metrics:
            return {}
        if valid_mask.numel() <= _ZERO:
            return {}
        valid_mask = valid_mask.to(dtype=torch.bool)
        reduced: dict[str, torch.Tensor] = {}
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                reduced[name] = value
                continue
            if value.numel() == _ONE:
                reduced[name] = value.reshape(())
                continue
            if value.dim() != _ONE or value.size(0) != valid_mask.numel():
                raise ValueError(f"Eval metric {name} must be [num_graphs]; got {tuple(value.shape)}.")
            selected = value.to(dtype=torch.float32)[valid_mask]
            if selected.numel() == _ZERO:
                continue
            reduced[name] = selected.mean()
        return reduced


__all__ = ["DualFlowEvalMixin"]
