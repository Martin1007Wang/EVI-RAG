from __future__ import annotations

import torch
from torch import nn

from src.weaver.rollout.schema import RolloutBatch

from .schema import LossOutput


class BudgetedDAGDetailedBalanceLoss(nn.Module):
    """Loss reader for BDB traces produced by the policy during rollout."""

    requires_bdb_trace = True

    def __init__(
        self,
        *,
        child_flow_target: str = "detach_current",
        backward_kernel: str = "uniform_boundary",
        edge_mode: str = "full",
        child_chunk_size: int = 2048,
        state_sources: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.child_flow_target = str(child_flow_target)
        self.backward_kernel = str(backward_kernel)
        self.edge_mode = str(edge_mode)
        self.child_chunk_size = int(child_chunk_size)
        self.state_sources = dict(
            {"rollout": True, "oracle_prefix": False, "counterfactual": False}
            if state_sources is None
            else state_sources
        )
        # REMOVED: separate L_stop/L_edge/L_base weights — see methodology.md §3.9
        if self.child_flow_target != "detach_current":
            raise ValueError("BDB only supports child_flow_target='detach_current'.")
        if self.backward_kernel != "uniform_boundary":
            raise ValueError("BDB only supports backward_kernel='uniform_boundary'.")
        if self.edge_mode != "full":
            raise ValueError("BDB only supports edge_mode='full'.")
        if self.child_chunk_size < 1:
            raise ValueError("child_chunk_size must be >= 1.")
        if not bool(self.state_sources.get("rollout", False)):
            raise ValueError("BDB requires state_sources.rollout=true.")
        unsupported_sources = [
            name
            for name, enabled in self.state_sources.items()
            if name != "rollout" and bool(enabled)
        ]
        if unsupported_sources:
            raise ValueError(
                "BDB v1 only supports rollout state sources; unsupported enabled "
                f"sources={unsupported_sources}."
            )

    def forward(self, rollout: RolloutBatch) -> LossOutput:
        traces = rollout.traces
        required = (
            traces.bdb_stop_loss,
            traces.bdb_edge_loss,
            traces.bdb_base_loss,
            traces.bdb_stop_valid_mask,
            traces.bdb_edge_valid_mask,
            traces.bdb_base_valid_mask,
        )
        if any(value is None for value in required):
            raise ValueError("BDB loss requires bdb rollout traces.")

        assert traces.bdb_stop_loss is not None
        assert traces.bdb_edge_loss is not None
        assert traces.bdb_base_loss is not None
        assert traces.bdb_stop_valid_mask is not None
        assert traces.bdb_edge_valid_mask is not None
        assert traces.bdb_base_valid_mask is not None

        stop_loss = traces.bdb_stop_loss.float()
        edge_loss = traces.bdb_edge_loss.float()
        base_loss = traces.bdb_base_loss.float()
        stop_valid = traces.bdb_stop_valid_mask.bool()
        edge_valid = traces.bdb_edge_valid_mask.bool()
        base_valid = traces.bdb_base_valid_mask.bool()

        per_state_loss = torch.where(base_valid, base_loss, stop_loss + edge_loss)
        any_valid = stop_valid | edge_valid | base_valid
        loss = _masked_mean(per_state_loss, any_valid, per_state_loss)
        if bool((per_state_loss[any_valid] < -1.0e-8).any()):
            raise AssertionError("L_BDB must be non-negative.")
        loss_stop = _masked_mean(stop_loss, stop_valid, stop_loss)
        loss_edge = _masked_mean(edge_loss, edge_valid, edge_loss)
        loss_base = _masked_mean(base_loss, base_valid, base_loss)

        metrics = {
            "loss/total": loss.detach(),
            "loss/bdb": loss.detach(),
            "bdb/loss_total": loss.detach(),
            "bdb/loss_delta0_mean": loss_stop.detach(),
            "bdb/loss_edge_residual_mean": loss_edge.detach(),
            "bdb/loss_forced_terminal_mean": loss_base.detach(),
            "bdb/delta_stop_mean": _masked_mean(
                traces.bdb_delta_stop,
                stop_valid,
                stop_loss,
            ).detach(),
            "bdb/delta_edge_mean": _masked_mean(
                traces.bdb_delta_edge,
                edge_valid,
                stop_loss,
            ).detach(),
            "bdb/delta_base_mean": _masked_mean(
                traces.bdb_delta_base,
                base_valid,
                stop_loss,
            ).detach(),
            "bdb/base_state_rate": base_valid.float().mean().detach(),
            "bdb/mean_frontier_size": _masked_mean(
                traces.bdb_frontier_size,
                any_valid,
                stop_loss,
            ).detach(),
            "bdb/mean_parent_count": _masked_mean(
                traces.bdb_parent_count,
                edge_valid,
                stop_loss,
            ).detach(),
            "reward/log_reward_mean": _masked_mean(
                traces.bdb_log_reward,
                any_valid,
                stop_loss,
            ).detach(),
            "flow/log_flow_mean": _masked_mean(
                traces.bdb_log_flow,
                any_valid,
                stop_loss,
            ).detach(),
        }
        return LossOutput(loss=loss, metrics=metrics, per_trajectory_loss=None)


def _masked_mean(
    values: torch.Tensor | None,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if values is None or not bool(mask.any()):
        return reference.sum() * 0.0
    return values.float()[mask].mean()


__all__ = ["BudgetedDAGDetailedBalanceLoss"]
