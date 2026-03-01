from __future__ import annotations

import torch

from src.models.configs.environment import StopConfig
from src.models.environment.contracts import GraphEnvContext
from src.models.environment.masks import build_node_membership_mask as build_graph_node_membership_mask


class DualFlowRewardEngine:
    def __init__(self, *, stop_cfg: StopConfig) -> None:
        self.stop_cfg = stop_cfg

    def reward_hyperparameters(self, *, reward_beta: float) -> tuple[float, float, float, float, float]:
        epsilon = float(self.stop_cfg.reward_epsilon)
        reward_base = float(self.stop_cfg.reward_base)
        degree_penalty_alpha = float(self.stop_cfg.degree_penalty_alpha)
        degree_penalty_min_degree = float(self.stop_cfg.degree_penalty_min_degree)
        if degree_penalty_min_degree <= 0:
            raise ValueError("stop.degree_penalty_min_degree must be > 0.")
        return epsilon, reward_base, degree_penalty_alpha, degree_penalty_min_degree, float(reward_beta)

    def compute_hit_mask(self, stop_nodes_abs: torch.Tensor, context: GraphEnvContext) -> torch.Tensor:
        node_is_target = build_graph_node_membership_mask(
            local_indices=context.a_local_indices,
            ptr=context.a_ptr,
            node_ptr=context.node_ptr,
            num_nodes_total=context.num_nodes_total,
            device=stop_nodes_abs.device,
            field_name="a_local_indices",
        )
        flat_stop = stop_nodes_abs.view(-1)
        valid = flat_stop >= 0
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
        return build_graph_node_membership_mask(
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
        degree_penalty_alpha: float,
        degree_penalty_min_degree: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_stop = stop_nodes_abs.view(-1)
        safe_nodes = flat_stop.clamp(min=0)
        hits = self.compute_hit_mask(stop_nodes_abs, context)
        hits_flat = hits.view(-1)
        rewards_flat = torch.full_like(flat_stop, fill_value=epsilon, dtype=torch.float32)
        if bool(hits_flat.any().item()):
            crow = context.adj_t_bwd.crow_indices()
            hit_nodes = safe_nodes[hits_flat]
            in_degree = (crow[hit_nodes + 1] - crow[hit_nodes]).to(dtype=torch.float32)
            if degree_penalty_alpha > 0:
                denom = in_degree.clamp(min=degree_penalty_min_degree).pow(degree_penalty_alpha)
                hit_rewards = torch.full_like(in_degree, fill_value=reward_base) / denom
            else:
                hit_rewards = torch.full_like(in_degree, fill_value=reward_base)
            rewards_flat[hits_flat] = hit_rewards
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
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, num_rollouts = hits_for_metrics.shape
        answer_counts = (context.a_ptr[1:] - context.a_ptr[:-1]).clamp(min=0)
        answer_counts_on_device = answer_counts.to(device=device)
        coverage_per_graph = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        max_answers = int(answer_counts.max().item()) if answer_counts.numel() > 0 else 0
        if max_answers <= 0:
            return coverage_per_graph, answer_counts_on_device

        a_offsets = context.node_ptr[:-1].to(device=device).repeat_interleave(answer_counts_on_device)
        a_abs = context.a_local_indices.to(device=device) + a_offsets
        a_ranks = torch.arange(a_abs.numel(), device=device, dtype=torch.long) - context.a_ptr[:-1].to(
            device=device
        ).repeat_interleave(answer_counts_on_device)
        answer_rank_by_node = torch.full((context.num_nodes_total,), fill_value=-1, device=device, dtype=torch.long)
        answer_rank_by_node.scatter_(0, a_abs, a_ranks)

        graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
        rollout_graph_ids = graph_ids.unsqueeze(1).expand(num_graphs, num_rollouts).reshape(-1)
        stop_flat = stop_nodes_for_metrics.reshape(-1).clamp(min=0)
        hit_flat = hits_for_metrics.reshape(-1)
        stop_answer_ranks = answer_rank_by_node.index_select(0, stop_flat)
        covered_mask = hit_flat & (stop_answer_ranks >= 0)
        if bool(covered_mask.any().item()):
            covered = torch.zeros((num_graphs, max_answers), device=device, dtype=torch.bool)
            covered[rollout_graph_ids[covered_mask], stop_answer_ranks[covered_mask]] = True
            coverage_per_graph = covered.sum(dim=1).to(dtype=torch.float32) / answer_counts_on_device.to(
                device=device, dtype=torch.float32
            ).clamp(min=1.0)
        return coverage_per_graph, answer_counts_on_device

    @staticmethod
    def summarize_reward_statistics(
        *,
        hits_for_metrics: torch.Tensor,
        rewards_for_metrics: torch.Tensor,
        rewards_scaled_for_metrics: torch.Tensor,
        coverage_per_graph: torch.Tensor,
        answer_counts_on_device: torch.Tensor,
        dummy_mask: torch.Tensor,
        template: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        valid_graph_mask = ~dummy_mask
        valid_graph_with_answers = valid_graph_mask & (answer_counts_on_device > 0)
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
        if bool(valid_graph_with_answers.any().item()):
            coverage_mean = coverage_per_graph[valid_graph_with_answers].mean()
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        epsilon, reward_base, degree_penalty_alpha, degree_penalty_min_degree, reward_beta_value = (
            self.reward_hyperparameters(reward_beta=reward_beta)
        )
        hits, rewards_raw = self.compute_hits_and_raw_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            epsilon=epsilon,
            reward_base=reward_base,
            degree_penalty_alpha=degree_penalty_alpha,
            degree_penalty_min_degree=degree_penalty_min_degree,
        )
        rewards_scaled = rewards_raw.pow(reward_beta_value) if reward_beta_value != 1.0 else rewards_raw
        hits_for_metrics, rewards_for_metrics, rewards_scaled_for_metrics, stop_nodes_for_metrics = (
            self.prepare_reward_metric_views(
                hits=hits,
                rewards_raw=rewards_raw,
                rewards_scaled=rewards_scaled,
                stop_nodes_abs=stop_nodes_abs,
            )
        )
        coverage_per_graph, answer_counts_on_device = self.compute_coverage_per_graph(
            hits_for_metrics=hits_for_metrics,
            stop_nodes_for_metrics=stop_nodes_for_metrics,
            context=context,
            device=stop_nodes_abs.device,
        )
        summary, valid_graph_mask = self.summarize_reward_statistics(
            hits_for_metrics=hits_for_metrics,
            rewards_for_metrics=rewards_for_metrics,
            rewards_scaled_for_metrics=rewards_scaled_for_metrics,
            coverage_per_graph=coverage_per_graph,
            answer_counts_on_device=answer_counts_on_device,
            dummy_mask=context.dummy_mask,
            template=rewards_raw,
        )

        metrics = {
            "reward/hit@1": summary["hit@1"].detach(),
            "reward/hit@beam": summary["hit@beam"].detach(),
            "reward/coverage@beam": summary["coverage@beam"].detach(),
            "reward/reward_mean": summary["reward_mean"].detach(),
            "reward/reward_mean_raw": summary["reward_mean"].detach(),
            "reward/reward_mean_scaled": summary["reward_mean_scaled"].detach(),
            "reward/reward_beta": torch.tensor(
                reward_beta_value,
                device=stop_nodes_abs.device,
                dtype=torch.float32,
            ),
            "reward/degree_penalty_alpha": torch.tensor(
                degree_penalty_alpha,
                device=stop_nodes_abs.device,
                dtype=torch.float32,
            ),
            "reward/valid_graph_ratio": valid_graph_mask.float().mean().detach(),
        }
        return rewards_raw, metrics


__all__ = ["DualFlowRewardEngine"]
