from __future__ import annotations

import torch
from torch import nn


class ActionSampler(nn.Module):
    """Sample source-normalized move/STOP actions from ragged logits."""

    _MIN_TEMPERATURE = 1.0e-4
    _NEAR_ZERO_TEMPERATURE = 1.0e-3

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _sanitize_dense_logits(
        *,
        logits: torch.Tensor,
        invalid_logits_policy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_finite = torch.isfinite(logits).any(dim=1)
        has_nan = torch.isnan(logits).any(dim=1)
        has_pos_inf = torch.isposinf(logits).any(dim=1)
        invalid_rows = has_nan | has_pos_inf | (~has_finite)
        if not bool(invalid_rows.any().item()):
            return logits, invalid_rows
        if invalid_logits_policy == "raise":
            raise ValueError(
                "Non-finite logits detected in trajectory ActionSampler. "
                f"invalid_rows={int(invalid_rows.sum().item())}."
            )
        safe_logits = logits.clone()
        stop_idx = int(logits.size(1)) - 1
        safe_logits[invalid_rows] = float("-inf")
        safe_logits[invalid_rows, stop_idx] = 0.0
        return safe_logits, invalid_rows

    @staticmethod
    def _build_dense_logits(
        *,
        edge_logits: torch.Tensor,
        out_degrees: torch.Tensor,
        stop_logits: torch.Tensor,
        invalid_logits_policy: str,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        total_agents = int(out_degrees.numel())
        max_deg = int(out_degrees.max().item()) if total_agents > 0 else 0
        dense_edges = torch.full(
            (total_agents, max_deg),
            fill_value=float("-inf"),
            device=edge_logits.device,
            dtype=edge_logits.dtype,
        )
        if max_deg > 0:
            edge_slots = torch.arange(max_deg, device=edge_logits.device).unsqueeze(0)
            slot_mask = edge_slots < out_degrees.unsqueeze(1)
            dense_edges[slot_mask] = edge_logits
        dense_logits = torch.cat([dense_edges, stop_logits.view(-1, 1)], dim=1)
        dense_logits, invalid_rows = ActionSampler._sanitize_dense_logits(
            logits=dense_logits,
            invalid_logits_policy=invalid_logits_policy,
        )
        return dense_logits, max_deg, invalid_rows

    def forward(
        self,
        policy_output: dict[str, torch.Tensor],
        *,
        is_training: bool,
        sampling_temperature: float,
        invalid_logits_policy: str = "raise",
    ) -> dict[str, torch.Tensor]:
        if is_training and abs(float(sampling_temperature) - 1.0) > 1.0e-6:
            raise ValueError(
                "On-policy trajectory sampling requires sampling_temperature == 1.0."
            )
        edge_logits = policy_output["edge_logits"]
        out_degrees = policy_output["out_degrees"].view(-1)
        stop_logits = policy_output["stop_logits"].view(-1)
        edge_ids = policy_output["edge_ids"]
        target_nodes = policy_output["target_nodes"]
        temperature = max(float(sampling_temperature), self._MIN_TEMPERATURE)
        dense_logits, max_deg, invalid_rows = self._build_dense_logits(
            edge_logits=edge_logits,
            out_degrees=out_degrees,
            stop_logits=stop_logits,
            invalid_logits_policy=invalid_logits_policy,
        )
        scaled_logits = dense_logits / temperature
        log_partition = torch.logsumexp(scaled_logits, dim=1)
        stop_idx = int(scaled_logits.size(1)) - 1
        stop_log_prob = scaled_logits[:, stop_idx] - log_partition
        if temperature <= self._NEAR_ZERO_TEMPERATURE:
            action_idx = scaled_logits.argmax(dim=1)
        else:
            eps = torch.finfo(scaled_logits.dtype).tiny
            uniform = torch.rand_like(scaled_logits).clamp(min=eps, max=1.0 - eps)
            gumbel = -torch.log(-torch.log(uniform))
            action_idx = (scaled_logits + gumbel).argmax(dim=1)
        log_prob = (
            torch.log_softmax(scaled_logits, dim=1)
            .gather(1, action_idx.unsqueeze(1))
            .squeeze(1)
        )
        is_stop = action_idx == max_deg
        if int(edge_ids.numel()) == 0:
            chosen_edge_ids = torch.full_like(action_idx, -1)
            chosen_target_nodes = torch.zeros_like(action_idx)
        else:
            base_offsets = out_degrees.cumsum(0) - out_degrees
            safe_idx = (base_offsets + action_idx).clamp(max=int(edge_ids.numel()) - 1)
            chosen_edge_ids = edge_ids.index_select(0, safe_idx)
            chosen_target_nodes = target_nodes.index_select(0, safe_idx)
            chosen_edge_ids = torch.where(
                is_stop, torch.full_like(chosen_edge_ids, -1), chosen_edge_ids
            )
            chosen_target_nodes = torch.where(
                is_stop,
                torch.zeros_like(chosen_target_nodes),
                chosen_target_nodes,
            )
        if bool(invalid_rows.any().item()):
            zeros = torch.zeros_like(log_prob)
            log_prob = torch.where(invalid_rows, zeros, log_prob)
            log_partition = torch.where(invalid_rows, zeros, log_partition)
            stop_log_prob = torch.where(invalid_rows, zeros, stop_log_prob)
            is_stop = torch.where(invalid_rows, torch.ones_like(is_stop), is_stop)
            chosen_edge_ids = torch.where(
                invalid_rows,
                torch.full_like(chosen_edge_ids, -1),
                chosen_edge_ids,
            )
            chosen_target_nodes = torch.where(
                invalid_rows,
                torch.zeros_like(chosen_target_nodes),
                chosen_target_nodes,
            )
        return {
            "is_stop": is_stop,
            "chosen_edge_ids": chosen_edge_ids,
            "chosen_target_nodes": chosen_target_nodes,
            "log_prob": log_prob,
            "log_partition": log_partition,
            "stop_log_prob": stop_log_prob,
        }


__all__ = ["ActionSampler"]
