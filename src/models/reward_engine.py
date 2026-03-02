from __future__ import annotations

import torch

from src.models.configs.environment import StopConfig
from src.models.environment.contracts import GraphEnvContext
from src.models.environment.masks import build_node_membership_mask as build_graph_node_membership_mask


class DualFlowRewardEngine:
    def __init__(self, *, stop_cfg: StopConfig) -> None:
        self.stop_cfg = stop_cfg

    def reward_hyperparameters(self, *, reward_beta: float) -> tuple[float, float, float, float, float, float]:
        epsilon = float(self.stop_cfg.reward_epsilon)
        reward_base = float(self.stop_cfg.reward_base)
        distance_reward_gamma = float(self.stop_cfg.distance_reward_gamma)
        degree_penalty_alpha = float(self.stop_cfg.degree_penalty_alpha)
        degree_penalty_min_degree = float(self.stop_cfg.degree_penalty_min_degree)
        if epsilon <= 0:
            raise ValueError("stop.reward_epsilon must be > 0.")
        if reward_base <= 0:
            raise ValueError("stop.reward_base must be > 0.")
        if not (0.0 < distance_reward_gamma <= 1.0):
            raise ValueError("stop.distance_reward_gamma must be in (0, 1].")
        if degree_penalty_min_degree <= 0:
            raise ValueError("stop.degree_penalty_min_degree must be > 0.")
        return (
            epsilon,
            reward_base,
            distance_reward_gamma,
            degree_penalty_alpha,
            degree_penalty_min_degree,
            float(reward_beta),
        )

    def compute_hit_mask(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> torch.Tensor:
        if (target_local_indices is None) != (target_ptr is None):
            raise ValueError("target_local_indices and target_ptr must be both provided or both None.")
        if target_local_indices is None or target_ptr is None:
            target_local_indices = context.a_local_indices
            target_ptr = context.a_ptr
            target_field_name = "a_local_indices"
        node_is_target = build_graph_node_membership_mask(
            local_indices=target_local_indices,
            ptr=target_ptr,
            node_ptr=context.node_ptr,
            num_nodes_total=context.num_nodes_total,
            device=stop_nodes_abs.device,
            field_name=target_field_name,
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

    @staticmethod
    def _segment_neighbor_nodes_from_csr(
        *,
        crow: torch.Tensor,
        col: torch.Tensor,
        source_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if int(source_nodes.numel()) == 0:
            return source_nodes.new_empty((0,))
        start_ptr = crow.index_select(0, source_nodes)
        end_ptr = crow.index_select(0, source_nodes + 1)
        degrees = end_ptr - start_ptr
        total_edges = int(degrees.sum().item())
        if total_edges <= 0:
            return source_nodes.new_empty((0,))
        base = start_ptr.repeat_interleave(degrees)
        segment_starts = degrees.cumsum(0) - degrees
        offsets = torch.arange(total_edges, device=source_nodes.device, dtype=torch.long)
        offsets = offsets - segment_starts.repeat_interleave(degrees)
        gather_idx = base + offsets
        return col.index_select(0, gather_idx)

    def _compute_graph_distances_to_targets(
        self,
        *,
        context: GraphEnvContext,
        target_local_indices: torch.Tensor,
        target_ptr: torch.Tensor,
        target_field_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        num_graphs = int(context.num_graphs)
        node_ptr = context.node_ptr.to(device=device, dtype=torch.long)
        target_ptr = target_ptr.to(device=device, dtype=torch.long)
        target_local_indices = target_local_indices.to(device=device, dtype=torch.long)
        target_counts = (target_ptr[1:] - target_ptr[:-1]).clamp(min=0)
        if int(target_counts.numel()) != num_graphs:
            raise ValueError(f"{target_field_name}_ptr shape mismatch with num_graphs in reward distance shaping.")
        if int(target_counts.sum().item()) != int(target_local_indices.numel()):
            raise ValueError(f"{target_field_name}_ptr mismatch with {target_field_name} length in reward shaping.")

        crow = context.adj_t_bwd.crow_indices().to(device=device, dtype=torch.long)
        col = context.adj_t_bwd.col_indices().to(device=device, dtype=torch.long)
        min_hops = torch.full((context.num_nodes_total,), fill_value=-1, device=device, dtype=torch.long)
        for graph_idx in range(num_graphs):
            node_start = int(node_ptr[graph_idx].item())
            node_end = int(node_ptr[graph_idx + 1].item())
            if node_end <= node_start:
                continue
            local_count = node_end - node_start
            local_dist = torch.full((local_count,), fill_value=-1, device=device, dtype=torch.long)
            t_start = int(target_ptr[graph_idx].item())
            t_end = int(target_ptr[graph_idx + 1].item())
            if t_end <= t_start:
                min_hops[node_start:node_end] = local_dist
                continue
            target_abs = target_local_indices[t_start:t_end] + node_start
            if bool((target_abs < node_start).any().item()) or bool((target_abs >= node_end).any().item()):
                raise ValueError(f"{target_field_name} contains out-of-range local node ids in reward shaping.")
            frontier = torch.unique(target_abs, sorted=False)
            local_dist[frontier - node_start] = 0
            hop = 1
            while int(frontier.numel()) > 0:
                neighbors = self._segment_neighbor_nodes_from_csr(
                    crow=crow,
                    col=col,
                    source_nodes=frontier,
                )
                if int(neighbors.numel()) == 0:
                    break
                in_graph = (neighbors >= node_start) & (neighbors < node_end)
                if not bool(in_graph.any().item()):
                    break
                neighbors = neighbors[in_graph]
                local_neighbors = neighbors - node_start
                unseen = local_dist.index_select(0, local_neighbors) < 0
                if not bool(unseen.any().item()):
                    break
                next_frontier = torch.unique(neighbors[unseen], sorted=False)
                local_dist[next_frontier - node_start] = hop
                frontier = next_frontier
                hop += 1
            min_hops[node_start:node_end] = local_dist
        return min_hops

    def compute_hits_and_raw_rewards(
        self,
        *,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        epsilon: float,
        reward_base: float,
        distance_reward_gamma: float,
        degree_penalty_alpha: float,
        degree_penalty_min_degree: float,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_stop = stop_nodes_abs.view(-1)
        safe_nodes = flat_stop.clamp(min=0)
        hits = self.compute_hit_mask(
            stop_nodes_abs,
            context,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
        )
        hits_flat = hits.view(-1)
        rewards_flat = torch.full_like(flat_stop, fill_value=epsilon, dtype=torch.float32)
        miss_mask = (~hits_flat) & (flat_stop >= 0)
        if bool(miss_mask.any().item()):
            if target_local_indices is None or target_ptr is None:
                raise ValueError("target_local_indices/target_ptr are required for distance-shaped miss rewards.")
            min_hops_to_target = self._compute_graph_distances_to_targets(
                context=context,
                target_local_indices=target_local_indices,
                target_ptr=target_ptr,
                target_field_name=target_field_name,
                device=stop_nodes_abs.device,
            )
            miss_nodes = safe_nodes[miss_mask]
            miss_hops = min_hops_to_target.index_select(0, miss_nodes)
            reachable_miss = miss_hops >= 0
            miss_rewards = torch.full(
                (int(miss_nodes.numel()),),
                fill_value=epsilon,
                device=stop_nodes_abs.device,
                dtype=torch.float32,
            )
            if bool(reachable_miss.any().item()):
                hops = miss_hops[reachable_miss].to(dtype=torch.float32)
                gamma_base = torch.full_like(hops, fill_value=distance_reward_gamma)
                dense_rewards = reward_base * torch.pow(gamma_base, hops)
                miss_rewards[reachable_miss] = dense_rewards.clamp(min=epsilon)
            rewards_flat[miss_mask] = miss_rewards
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
        target_local_indices: torch.Tensor,
        target_ptr: torch.Tensor,
        target_field_name: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, num_rollouts = hits_for_metrics.shape
        target_counts = (target_ptr[1:] - target_ptr[:-1]).clamp(min=0)
        target_counts_on_device = target_counts.to(device=device)
        coverage_per_graph = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        max_targets = int(target_counts.max().item()) if target_counts.numel() > 0 else 0
        if max_targets <= 0:
            return coverage_per_graph, target_counts_on_device

        target_offsets = context.node_ptr[:-1].to(device=device).repeat_interleave(target_counts_on_device)
        target_abs = target_local_indices.to(device=device) + target_offsets
        target_ranks = torch.arange(target_abs.numel(), device=device, dtype=torch.long) - target_ptr[:-1].to(
            device=device
        ).repeat_interleave(target_counts_on_device)
        if bool((target_abs < 0).any().item()) or bool((target_abs >= context.num_nodes_total).any().item()):
            raise ValueError(f"{target_field_name} contains out-of-range node indices during reward coverage.")
        target_rank_by_node = torch.full((context.num_nodes_total,), fill_value=-1, device=device, dtype=torch.long)
        target_rank_by_node.scatter_(0, target_abs, target_ranks)

        graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
        rollout_graph_ids = graph_ids.unsqueeze(1).expand(num_graphs, num_rollouts).reshape(-1)
        stop_flat = stop_nodes_for_metrics.reshape(-1).clamp(min=0)
        hit_flat = hits_for_metrics.reshape(-1)
        stop_target_ranks = target_rank_by_node.index_select(0, stop_flat)
        covered_mask = hit_flat & (stop_target_ranks >= 0)
        if bool(covered_mask.any().item()):
            covered = torch.zeros((num_graphs, max_targets), device=device, dtype=torch.bool)
            covered[rollout_graph_ids[covered_mask], stop_target_ranks[covered_mask]] = True
            coverage_per_graph = covered.sum(dim=1).to(dtype=torch.float32) / target_counts_on_device.to(
                device=device, dtype=torch.float32
            ).clamp(min=1.0)
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if (target_local_indices is None) != (target_ptr is None):
            raise ValueError("target_local_indices and target_ptr must be both provided or both None.")
        if target_local_indices is None or target_ptr is None:
            target_local_indices = context.a_local_indices
            target_ptr = context.a_ptr
            target_field_name = "a_local_indices"
        epsilon, reward_base, distance_reward_gamma, degree_penalty_alpha, degree_penalty_min_degree, reward_beta_value = (
            self.reward_hyperparameters(reward_beta=reward_beta)
        )
        hits, rewards_raw = self.compute_hits_and_raw_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            epsilon=epsilon,
            reward_base=reward_base,
            distance_reward_gamma=distance_reward_gamma,
            degree_penalty_alpha=degree_penalty_alpha,
            degree_penalty_min_degree=degree_penalty_min_degree,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
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
            "reward/distance_reward_gamma": torch.tensor(
                distance_reward_gamma,
                device=stop_nodes_abs.device,
                dtype=torch.float32,
            ),
            "reward/valid_graph_ratio": valid_graph_mask.float().mean().detach(),
        }
        return rewards_raw, metrics


__all__ = ["DualFlowRewardEngine"]
