from __future__ import annotations

import torch

from src.models.configs.environment import StopConfig
from src.models.environment.context import GraphEnvContext
from src.models.environment.ops import build_node_membership_mask


class DualFlowRewardEngine:
    def __init__(self, *, stop_cfg: StopConfig) -> None:
        self.stop_cfg = stop_cfg

    def reward_hyperparameters(
        self, *, reward_beta: float
    ) -> tuple[float, float, float]:
        epsilon = float(self.stop_cfg.reward_epsilon)
        reward_base = float(self.stop_cfg.reward_base)
        if epsilon <= 0:
            raise ValueError("stop.reward_epsilon must be > 0.")
        if reward_base <= 0:
            raise ValueError("stop.reward_base must be > 0.")
        return epsilon, reward_base, float(reward_beta)

    @staticmethod
    def _flatten_done_mask(
        *,
        terminal_done_mask: torch.Tensor | None,
        stop_nodes_abs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if terminal_done_mask is None:
            return None
        if tuple(terminal_done_mask.shape) != tuple(stop_nodes_abs.shape):
            raise ValueError(
                "terminal_done_mask shape mismatch with stop_nodes_abs: "
                f"done={tuple(terminal_done_mask.shape)} stop_nodes={tuple(stop_nodes_abs.shape)}."
            )
        return terminal_done_mask.to(device=device, dtype=torch.bool).view(-1)

    @staticmethod
    def _expand_graph_ids(
        *,
        stop_nodes_abs: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if stop_nodes_abs.dim() == 1:
            if int(stop_nodes_abs.numel()) != num_graphs:
                raise ValueError(
                    "1D stop_nodes_abs must have length equal to num_graphs: "
                    f"stop_nodes={int(stop_nodes_abs.numel())}, num_graphs={num_graphs}."
                )
            return torch.arange(
                num_graphs, device=stop_nodes_abs.device, dtype=torch.long
            )
        if stop_nodes_abs.dim() == 2:
            if int(stop_nodes_abs.size(0)) != num_graphs:
                raise ValueError(
                    "2D stop_nodes_abs leading dimension must equal num_graphs: "
                    f"stop_nodes={tuple(stop_nodes_abs.shape)}, num_graphs={num_graphs}."
                )
            num_rollouts = int(stop_nodes_abs.size(1))
            graph_ids = torch.arange(
                num_graphs, device=stop_nodes_abs.device, dtype=torch.long
            )
            return graph_ids.unsqueeze(1).expand(num_graphs, num_rollouts).reshape(-1)
        raise ValueError(
            f"stop_nodes_abs must be 1D or 2D, got shape={tuple(stop_nodes_abs.shape)}."
        )

    def compute_hit_mask(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (target_local_indices is None) != (target_ptr is None):
            raise ValueError(
                "target_local_indices and target_ptr must be both provided or both None."
            )
        if target_local_indices is None or target_ptr is None:
            target_local_indices = context.a_local_indices
            target_ptr = context.a_ptr
            target_field_name = "a_local_indices"
        node_is_target = build_node_membership_mask(
            local_indices=target_local_indices,
            ptr=target_ptr,
            node_ptr=context.node_ptr,
            num_nodes_total=context.num_nodes_total,
            device=stop_nodes_abs.device,
            field_name=target_field_name,
        )
        flat_stop = stop_nodes_abs.view(-1)
        done_mask_flat = self._flatten_done_mask(
            terminal_done_mask=terminal_done_mask,
            stop_nodes_abs=stop_nodes_abs,
            device=stop_nodes_abs.device,
        )
        valid = flat_stop >= 0
        if done_mask_flat is not None:
            valid = valid & done_mask_flat
        safe_nodes = flat_stop.clamp(min=0)
        hits_flat = torch.zeros_like(valid, dtype=torch.bool)
        if node_is_target.numel() > 0 and safe_nodes.numel() > 0:
            hits_flat = node_is_target.index_select(0, safe_nodes)
        return (hits_flat & valid).view_as(stop_nodes_abs)

    def build_node_membership_mask(
        self,
        *,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        num_nodes_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_node_membership_mask(
            local_indices=local_indices,
            ptr=ptr,
            node_ptr=node_ptr,
            num_nodes_total=num_nodes_total,
            device=device,
            field_name="local_indices",
        )

    def compute_hits_and_raw_rewards(
        self,
        *,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        epsilon: float,
        reward_base: float,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_stop = stop_nodes_abs.view(-1)

        hits = self.compute_hit_mask(
            stop_nodes_abs,
            context,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
            terminal_done_mask=terminal_done_mask,
        )
        hits_flat = hits.view(-1)
        rewards_flat = torch.full_like(
            flat_stop, fill_value=epsilon, dtype=torch.float32
        )

        if bool(hits_flat.any().item()):
            rewards_flat[hits_flat] = reward_base
        return hits, rewards_flat.view_as(stop_nodes_abs)

    @staticmethod
    def prepare_reward_metric_views(
        *,
        hits: torch.Tensor,
        rewards_raw: torch.Tensor,
        rewards_scaled: torch.Tensor,
        stop_nodes_abs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hits.dim() == 1:
            return (
                hits.unsqueeze(1),
                rewards_raw.unsqueeze(1),
                rewards_scaled.unsqueeze(1),
                stop_nodes_abs.unsqueeze(1),
            )
        return hits, rewards_raw, rewards_scaled, stop_nodes_abs

    def compute_coverage_per_graph(
        self,
        *,
        hits_for_metrics: torch.Tensor,
        stop_nodes_for_metrics: torch.Tensor,
        context: GraphEnvContext,
        target_local_indices: torch.Tensor,
        target_ptr: torch.Tensor,
        target_field_name: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, num_rollouts = hits_for_metrics.shape
        target_counts = (target_ptr[1:] - target_ptr[:-1]).clamp(min=0)
        target_counts_on_device = target_counts.to(device=device)
        coverage_per_graph = torch.zeros(
            (num_graphs,), device=device, dtype=torch.float32
        )
        max_targets = (
            int(target_counts.max().item()) if target_counts.numel() > 0 else 0
        )
        if max_targets <= 0:
            return coverage_per_graph, target_counts_on_device

        target_offsets = (
            context.node_ptr[:-1]
            .to(device=device)
            .repeat_interleave(target_counts_on_device)
        )
        target_abs = target_local_indices.to(device=device) + target_offsets
        target_ranks = torch.arange(
            target_abs.numel(), device=device, dtype=torch.long
        ) - target_ptr[:-1].to(device=device).repeat_interleave(target_counts_on_device)
        if bool((target_abs < 0).any().item()) or bool(
            (target_abs >= context.num_nodes_total).any().item()
        ):
            raise ValueError(
                f"{target_field_name} contains out-of-range node indices during reward coverage."
            )
        target_rank_by_node = torch.full(
            (context.num_nodes_total,), fill_value=-1, device=device, dtype=torch.long
        )
        target_rank_by_node.scatter_(0, target_abs, target_ranks)

        graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
        rollout_graph_ids = (
            graph_ids.unsqueeze(1).expand(num_graphs, num_rollouts).reshape(-1)
        )
        stop_flat = stop_nodes_for_metrics.reshape(-1).clamp(min=0)
        hit_flat = hits_for_metrics.reshape(-1)
        stop_target_ranks = target_rank_by_node.index_select(0, stop_flat)
        covered_mask = hit_flat & (stop_target_ranks >= 0)
        if bool(covered_mask.any().item()):
            covered = torch.zeros(
                (num_graphs, max_targets), device=device, dtype=torch.bool
            )
            covered[
                rollout_graph_ids[covered_mask], stop_target_ranks[covered_mask]
            ] = True
            coverage_per_graph = covered.sum(dim=1).to(
                dtype=torch.float32
            ) / target_counts_on_device.to(device=device, dtype=torch.float32).clamp(
                min=1.0
            )
        return coverage_per_graph, target_counts_on_device

    @staticmethod
    def summarize_reward_statistics(
        *,
        hits_for_metrics: torch.Tensor,
        rewards_for_metrics: torch.Tensor,
        rewards_scaled_for_metrics: torch.Tensor,
        coverage_per_graph: torch.Tensor,
        target_counts_on_device: torch.Tensor,
        dummy_mask: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        valid_graph_mask = ~dummy_mask
        valid_graph_with_targets = valid_graph_mask & (target_counts_on_device > 0)
        if bool(valid_graph_mask.any().item()):
            hit_mean = hits_for_metrics[valid_graph_mask, 0].float().mean()
            hit_beam = hits_for_metrics[valid_graph_mask].any(dim=1).float().mean()
            reward_mean = rewards_for_metrics[valid_graph_mask].mean()
            reward_mean_scaled = rewards_scaled_for_metrics[valid_graph_mask].mean()
        else:
            hit_mean = template.new_zeros(())
            hit_beam = template.new_zeros(())
            reward_mean = template.new_zeros(())
            reward_mean_scaled = template.new_zeros(())
        if bool(valid_graph_with_targets.any().item()):
            coverage_mean = coverage_per_graph[valid_graph_with_targets].mean()
        else:
            coverage_mean = template.new_zeros(())
        return {
            "hit@1": hit_mean,
            "hit@beam": hit_beam,
            "coverage@beam": coverage_mean,
            "reward_mean": reward_mean,
            "reward_mean_scaled": reward_mean_scaled,
        }, valid_graph_mask

    def compute_rewards(
        self,
        *,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        reward_beta: float,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (target_local_indices is None) != (target_ptr is None):
            raise ValueError(
                "target_local_indices and target_ptr must be both provided or both None."
            )
        if target_local_indices is None or target_ptr is None:
            target_local_indices = context.a_local_indices
            target_ptr = context.a_ptr
            target_field_name = "a_local_indices"
        epsilon, reward_base, reward_beta_value = self.reward_hyperparameters(
            reward_beta=reward_beta
        )
        hits, rewards_raw = self.compute_hits_and_raw_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            epsilon=epsilon,
            reward_base=reward_base,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
            terminal_done_mask=terminal_done_mask,
        )
        rewards_scaled = (
            rewards_raw.pow(reward_beta_value)
            if reward_beta_value != 1.0
            else rewards_raw
        )
        (
            hits_for_metrics,
            rewards_for_metrics,
            rewards_scaled_for_metrics,
            stop_nodes_for_metrics,
        ) = self.prepare_reward_metric_views(
            hits=hits,
            rewards_raw=rewards_raw,
            rewards_scaled=rewards_scaled,
            stop_nodes_abs=stop_nodes_abs,
        )
        coverage_per_graph, answer_counts_on_device = self.compute_coverage_per_graph(
            hits_for_metrics=hits_for_metrics,
            stop_nodes_for_metrics=stop_nodes_for_metrics,
            context=context,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
            device=stop_nodes_abs.device,
        )
        summary, valid_graph_mask = self.summarize_reward_statistics(
            hits_for_metrics=hits_for_metrics,
            rewards_for_metrics=rewards_for_metrics,
            rewards_scaled_for_metrics=rewards_scaled_for_metrics,
            coverage_per_graph=coverage_per_graph,
            target_counts_on_device=answer_counts_on_device,
            dummy_mask=context.dummy_mask,
            template=rewards_raw,
        )

        metrics = {
            "reward/hit@1": summary["hit@1"].detach(),
            "reward/hit@beam": summary["hit@beam"].detach(),
            "reward/coverage@beam": summary["coverage@beam"].detach(),
            "reward/reward_mean": summary["reward_mean"].detach(),
            "reward/valid_graph_ratio": valid_graph_mask.float().mean().detach(),
        }
        if reward_beta_value != 1.0:
            metrics["reward/reward_mean_scaled"] = summary[
                "reward_mean_scaled"
            ].detach()
            metrics["reward/reward_beta"] = torch.tensor(
                reward_beta_value,
                device=stop_nodes_abs.device,
                dtype=torch.float32,
            )
        return rewards_raw, metrics


__all__ = ["DualFlowRewardEngine"]
