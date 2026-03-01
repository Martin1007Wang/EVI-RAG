from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import torch

from src.models.configs.training import ReplayBufferConfig
from src.models.environment.contracts import GraphEnvContext
from src.utils.replay_oracle import OracleTrajectory, enumerate_oracle_trajectories


@dataclass(frozen=True)
class ReplayBatch:
    use_offline_mask: torch.Tensor
    start_local_indices: torch.Tensor
    edge_ids: torch.Tensor
    path_lengths: torch.Tensor
    graph_has_oracle: torch.Tensor


@dataclass(frozen=True)
class ReplayPairBucket:
    pair_key: tuple[int, int]
    traj_indices: tuple[int, ...]
    traj_probs: tuple[float, ...]


@dataclass(frozen=True)
class ReplayGraphBank:
    trajectories: tuple[OracleTrajectory, ...]
    pair_buckets: tuple[ReplayPairBucket, ...]


class HighEnergyReplayBuffer:
    """Offline oracle trajectory bank + online hybrid sampler.

    Sampling is pair-balanced by design: first sample (start, target) pair uniformly,
    then sample a trajectory within the pair by energy-based probabilities.
    """

    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self._bank: dict[str, ReplayGraphBank] = {}

    def build_and_sample(
        self,
        *,
        context: GraphEnvContext,
        num_rollouts: int,
        max_steps: int,
        alpha: float,
        stop_min_steps: int,
        device: torch.device,
    ) -> ReplayBatch:
        self._ensure_oracle_bank(context=context)
        num_graphs = int(context.num_graphs)
        use_offline_mask = torch.zeros((num_graphs, num_rollouts), dtype=torch.bool, device=device)
        start_local_indices = torch.zeros((num_graphs, num_rollouts), dtype=torch.long, device=device)
        edge_ids = torch.full((num_graphs, num_rollouts, max_steps), -1, dtype=torch.long, device=device)
        path_lengths = torch.zeros((num_graphs, num_rollouts), dtype=torch.long, device=device)
        graph_has_oracle = torch.zeros((num_graphs,), dtype=torch.bool, device=device)
        alpha_clamped = min(max(float(alpha), 0.0), 1.0)
        min_len = max(int(stop_min_steps), 1)

        for graph_idx in range(num_graphs):
            fallback_start = self._fallback_start_local(context=context, graph_idx=graph_idx)
            start_local_indices[graph_idx] = fallback_start
            sample_id = context.sample_ids[graph_idx]
            graph_bank = self._bank.get(sample_id)
            if graph_bank is None:
                continue
            if len(graph_bank.trajectories) == 0 or len(graph_bank.pair_buckets) == 0:
                continue
            graph_has_oracle[graph_idx] = True
            edge_abs = torch.where(context.edge_batch == graph_idx)[0]
            if int(edge_abs.numel()) == 0:
                continue
            for rollout_idx in range(num_rollouts):
                draw = torch.rand((), device=device)
                if float(draw.item()) >= alpha_clamped:
                    continue
                pair_idx = int(torch.randint(len(graph_bank.pair_buckets), (1,), device=device).item())
                pair_bucket = graph_bank.pair_buckets[pair_idx]
                traj_pick = self._sample_in_pair(pair_bucket=pair_bucket, device=device)
                traj_id = int(pair_bucket.traj_indices[traj_pick])
                traj = graph_bank.trajectories[traj_id]
                raw_len = min(len(traj.edge_local_ids), max_steps)
                if raw_len < min_len:
                    continue
                local_edges = torch.tensor(traj.edge_local_ids[:raw_len], device=device, dtype=torch.long)
                if bool((local_edges < 0).any().item()) or bool((local_edges >= edge_abs.numel()).any().item()):
                    continue
                abs_edges = edge_abs.index_select(0, local_edges)
                use_offline_mask[graph_idx, rollout_idx] = True
                start_local_indices[graph_idx, rollout_idx] = int(traj.start_local)
                edge_ids[graph_idx, rollout_idx, :raw_len] = abs_edges
                path_lengths[graph_idx, rollout_idx] = raw_len

        return ReplayBatch(
            use_offline_mask=use_offline_mask,
            start_local_indices=start_local_indices,
            edge_ids=edge_ids,
            path_lengths=path_lengths,
            graph_has_oracle=graph_has_oracle,
        )

    def _ensure_oracle_bank(self, *, context: GraphEnvContext) -> None:
        num_graphs = int(context.num_graphs)
        for graph_idx in range(num_graphs):
            sample_id = context.sample_ids[graph_idx]
            if sample_id in self._bank:
                continue
            trajectories, has_precomputed = self._build_graph_trajectories_from_precomputed(
                context=context,
                graph_idx=graph_idx,
            )
            if not has_precomputed:
                trajectories = self._build_graph_trajectories_online(context=context, graph_idx=graph_idx)
            annotated = self._annotate_trajectories(context=context, graph_idx=graph_idx, trajectories=trajectories)
            deduplicated = self._deduplicate_trajectories(annotated)
            self._bank[sample_id] = self._build_graph_bank(deduplicated)

    def _build_graph_trajectories_from_precomputed(
        self,
        *,
        context: GraphEnvContext,
        graph_idx: int,
    ) -> tuple[list[OracleTrajectory], bool]:
        if context.replay_start_local is None:
            return [], False
        if context.replay_path_lengths is None:
            return [], False
        if context.replay_edge_local_ids is None:
            return [], False
        if context.replay_path_ptr is None:
            return [], False
        if context.replay_edge_ptr is None:
            return [], False

        path_start = int(context.replay_path_ptr[graph_idx].item())
        path_end = int(context.replay_path_ptr[graph_idx + 1].item())
        edge_start = int(context.replay_edge_ptr[graph_idx].item())
        edge_end = int(context.replay_edge_ptr[graph_idx + 1].item())
        if path_end < path_start:
            raise ValueError("Replay path ptr must be non-decreasing.")
        if edge_end < edge_start:
            raise ValueError("Replay edge ptr must be non-decreasing.")

        starts = context.replay_start_local[path_start:path_end]
        lengths = context.replay_path_lengths[path_start:path_end]
        flat_edges = context.replay_edge_local_ids[edge_start:edge_end]
        if int(starts.numel()) != int(lengths.numel()):
            raise ValueError("Replay precomputed start/path length mismatch.")
        if int(lengths.sum().item()) != int(flat_edges.numel()):
            raise ValueError("Replay precomputed edge span mismatch with path lengths.")

        trajectories: list[OracleTrajectory] = []
        cursor = 0
        for idx in range(int(starts.numel())):
            path_len = int(lengths[idx].item())
            if path_len < 0:
                raise ValueError("Replay precomputed path length must be non-negative.")
            edge_slice = flat_edges[cursor : cursor + path_len]
            cursor += path_len
            if path_len == 0:
                continue
            trajectories.append(
                OracleTrajectory(
                    start_local=int(starts[idx].item()),
                    edge_local_ids=tuple(int(eid) for eid in edge_slice.tolist()),
                )
            )
        if cursor != int(flat_edges.numel()):
            raise ValueError("Replay precomputed edge decode cursor mismatch.")

        promoted = self._promote_trajectories_with_super_source(
            context=context,
            graph_idx=graph_idx,
            trajectories=trajectories,
        )
        return promoted, True

    def _build_graph_trajectories_online(self, *, context: GraphEnvContext, graph_idx: int) -> list[OracleTrajectory]:
        node_start = int(context.node_ptr[graph_idx].item())
        node_end = int(context.node_ptr[graph_idx + 1].item())
        num_nodes = node_end - node_start
        if num_nodes <= 0:
            return []

        edge_abs = torch.where(context.edge_batch == graph_idx)[0]
        if int(edge_abs.numel()) == 0:
            return []
        edge_heads = context.edge_index[0].index_select(0, edge_abs) - node_start
        edge_tails = context.edge_index[1].index_select(0, edge_abs) - node_start
        if bool((edge_heads < 0).any().item()) or bool((edge_heads >= num_nodes).any().item()):
            raise ValueError("Edge head local index out of range during oracle trajectory extraction.")
        if bool((edge_tails < 0).any().item()) or bool((edge_tails >= num_nodes).any().item()):
            raise ValueError("Edge tail local index out of range during oracle trajectory extraction.")

        starts = self._graph_start_nodes(context=context, graph_idx=graph_idx)
        targets = self._graph_target_nodes(context=context, graph_idx=graph_idx)
        if len(starts) == 0 or len(targets) == 0:
            return []

        max_per_pair = int(self.config.max_paths_per_pair)
        max_per_graph = int(self.config.max_paths_per_graph)
        max_shortest = int(self.config.max_shortest_paths_per_pair)
        max_dfs = int(self.config.max_dfs_paths_per_pair)
        max_depth = int(self.config.max_depth)
        allow_cycles = bool(self.config.allow_cycles)
        max_node_visits = int(self.config.max_node_visits)
        if max_depth <= 0:
            return []
        if max_per_pair <= 0:
            return []
        if max_per_graph <= 0:
            return []
        if max_node_visits <= 0:
            raise ValueError("replay_cfg.max_node_visits must be a positive integer.")

        return enumerate_oracle_trajectories(
            num_nodes=num_nodes,
            edge_src=edge_heads.tolist(),
            edge_dst=edge_tails.tolist(),
            start_nodes=starts,
            target_nodes=targets,
            max_paths_per_pair=max_per_pair,
            max_paths_per_graph=max_per_graph,
            max_shortest_paths_per_pair=max_shortest,
            max_dfs_paths_per_pair=max_dfs,
            max_depth=max_depth,
            allow_cycles=allow_cycles,
            max_node_visits=max_node_visits,
        )

    @staticmethod
    def _graph_target_nodes(*, context: GraphEnvContext, graph_idx: int) -> list[int]:
        start = int(context.a_ptr[graph_idx].item())
        end = int(context.a_ptr[graph_idx + 1].item())
        return [int(value) for value in context.a_local_indices[start:end].tolist()]

    @staticmethod
    def _graph_start_nodes(*, context: GraphEnvContext, graph_idx: int) -> list[int]:
        if context.start_local_indices is not None:
            return [int(context.start_local_indices[graph_idx].item())]
        start = int(context.q_ptr[graph_idx].item())
        end = int(context.q_ptr[graph_idx + 1].item())
        return [int(value) for value in context.q_local_indices[start:end].tolist()]

    @staticmethod
    def _graph_q_nodes(*, context: GraphEnvContext, graph_idx: int) -> set[int]:
        start = int(context.q_ptr[graph_idx].item())
        end = int(context.q_ptr[graph_idx + 1].item())
        return {int(value) for value in context.q_local_indices[start:end].tolist()}

    def _promote_trajectories_with_super_source(
        self,
        *,
        context: GraphEnvContext,
        graph_idx: int,
        trajectories: list[OracleTrajectory],
    ) -> list[OracleTrajectory]:
        if context.start_local_indices is None:
            return trajectories
        if len(trajectories) == 0:
            return trajectories

        super_local = int(context.start_local_indices[graph_idx].item())
        q_nodes = self._graph_q_nodes(context=context, graph_idx=graph_idx)
        super_edge_by_q = self._build_super_source_edge_lookup(
            context=context,
            graph_idx=graph_idx,
            super_local=super_local,
        )

        promoted: list[OracleTrajectory] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for traj in trajectories:
            if len(traj.edge_local_ids) == 0:
                continue
            start_local = int(traj.start_local)
            if start_local == super_local:
                promoted_traj = traj
            elif start_local in q_nodes:
                super_edge = super_edge_by_q.get(start_local)
                if super_edge is None:
                    continue
                promoted_traj = OracleTrajectory(
                    start_local=super_local,
                    edge_local_ids=(int(super_edge), *traj.edge_local_ids),
                    target_local=traj.target_local,
                    shortest_gap=traj.shortest_gap,
                    revisit_count=traj.revisit_count,
                )
            else:
                continue
            key = (promoted_traj.start_local, promoted_traj.edge_local_ids)
            if key in seen:
                continue
            seen.add(key)
            promoted.append(promoted_traj)
        return promoted

    def _build_super_source_edge_lookup(
        self,
        *,
        context: GraphEnvContext,
        graph_idx: int,
        super_local: int,
    ) -> dict[int, int]:
        node_start = int(context.node_ptr[graph_idx].item())
        edge_abs = torch.where(context.edge_batch == graph_idx)[0]
        if int(edge_abs.numel()) == 0:
            return {}
        edge_heads = context.edge_index[0].index_select(0, edge_abs) - node_start
        edge_tails = context.edge_index[1].index_select(0, edge_abs) - node_start
        lookup: dict[int, int] = {}
        for local_eid in range(int(edge_abs.numel())):
            head = int(edge_heads[local_eid].item())
            if head != super_local:
                continue
            tail = int(edge_tails[local_eid].item())
            if tail not in lookup:
                lookup[tail] = local_eid
        return lookup

    @staticmethod
    def _fallback_start_local(*, context: GraphEnvContext, graph_idx: int) -> int:
        starts = HighEnergyReplayBuffer._graph_start_nodes(context=context, graph_idx=graph_idx)
        if len(starts) == 0:
            raise ValueError("Replay fallback start node is empty; q/start set must be non-empty.")
        return int(starts[0])

    @staticmethod
    def _sample_in_pair(*, pair_bucket: ReplayPairBucket, device: torch.device) -> int:
        num_paths = len(pair_bucket.traj_indices)
        if num_paths <= 0:
            raise ValueError("Replay pair bucket must contain at least one trajectory.")
        if num_paths == 1:
            return 0
        probs = torch.as_tensor(pair_bucket.traj_probs, device=device, dtype=torch.float32)
        if probs.numel() != num_paths:
            raise ValueError(
                "Replay pair probability size mismatch. "
                f"got={int(probs.numel())}, expected={num_paths}."
            )
        probs = probs.clamp(min=0)
        prob_sum = probs.sum()
        if not torch.isfinite(prob_sum) or float(prob_sum.item()) <= 0.0:
            probs = torch.full_like(probs, fill_value=1.0 / float(num_paths))
        else:
            probs = probs / prob_sum
        pick = torch.multinomial(probs, num_samples=1, replacement=True)
        return int(pick.item())

    def _build_graph_bank(self, trajectories: list[OracleTrajectory]) -> ReplayGraphBank:
        if len(trajectories) == 0:
            return ReplayGraphBank(trajectories=tuple(), pair_buckets=tuple())
        pair_to_indices: dict[tuple[int, int], list[int]] = {}
        kept: list[OracleTrajectory] = []
        for traj in trajectories:
            if traj.target_local is None:
                continue
            pair_key = (int(traj.start_local), int(traj.target_local))
            pair_to_indices.setdefault(pair_key, []).append(len(kept))
            kept.append(traj)
        if len(kept) == 0:
            return ReplayGraphBank(trajectories=tuple(), pair_buckets=tuple())

        pair_buckets: list[ReplayPairBucket] = []
        for pair_key, traj_indices in pair_to_indices.items():
            probs = self._compute_pair_path_probs(trajectories=kept, traj_indices=traj_indices)
            pair_buckets.append(
                ReplayPairBucket(
                    pair_key=pair_key,
                    traj_indices=tuple(int(idx) for idx in traj_indices),
                    traj_probs=tuple(float(prob) for prob in probs),
                )
            )
        return ReplayGraphBank(
            trajectories=tuple(kept),
            pair_buckets=tuple(pair_buckets),
        )

    def _compute_pair_path_probs(
        self,
        *,
        trajectories: list[OracleTrajectory],
        traj_indices: list[int],
    ) -> list[float]:
        num_paths = len(traj_indices)
        if num_paths <= 0:
            return []
        if num_paths == 1:
            return [1.0]
        gap_weight = float(self.config.shortest_gap_weight)
        revisit_weight = float(self.config.revisit_penalty_weight)
        temperature = float(self.config.path_sampling_temperature)
        if temperature <= 0.0:
            raise ValueError("replay_cfg.path_sampling_temperature must be > 0.")

        energies = []
        for traj_idx in traj_indices:
            traj = trajectories[traj_idx]
            shortest_gap = max(int(traj.shortest_gap), 0)
            revisit_count = max(int(traj.revisit_count), 0)
            energy = gap_weight * float(shortest_gap) + revisit_weight * float(revisit_count)
            energies.append(energy)
        score = torch.as_tensor(energies, dtype=torch.float32)
        score = -score / float(temperature)
        probs = torch.softmax(score, dim=0)
        return [float(v) for v in probs.tolist()]

    @staticmethod
    def _deduplicate_trajectories(trajectories: list[OracleTrajectory]) -> list[OracleTrajectory]:
        if len(trajectories) == 0:
            return []
        best_by_key: dict[tuple[int, tuple[int, ...]], OracleTrajectory] = {}
        for traj in trajectories:
            key = (int(traj.start_local), tuple(int(eid) for eid in traj.edge_local_ids))
            incumbent = best_by_key.get(key)
            if incumbent is None:
                best_by_key[key] = traj
                continue
            incumbent_rank = (
                max(int(incumbent.shortest_gap), 0),
                max(int(incumbent.revisit_count), 0),
                len(incumbent.edge_local_ids),
            )
            candidate_rank = (
                max(int(traj.shortest_gap), 0),
                max(int(traj.revisit_count), 0),
                len(traj.edge_local_ids),
            )
            if candidate_rank < incumbent_rank:
                best_by_key[key] = traj
        return list(best_by_key.values())

    def _annotate_trajectories(
        self,
        *,
        context: GraphEnvContext,
        graph_idx: int,
        trajectories: list[OracleTrajectory],
    ) -> list[OracleTrajectory]:
        if len(trajectories) == 0:
            return []
        node_start = int(context.node_ptr[graph_idx].item())
        node_end = int(context.node_ptr[graph_idx + 1].item())
        num_nodes = node_end - node_start
        if num_nodes <= 0:
            return []
        edge_abs = torch.where(context.edge_batch == graph_idx)[0]
        if int(edge_abs.numel()) == 0:
            return []
        edge_heads = (context.edge_index[0].index_select(0, edge_abs) - node_start).tolist()
        edge_tails = (context.edge_index[1].index_select(0, edge_abs) - node_start).tolist()
        reverse_neighbors = self._build_reverse_neighbors(num_nodes=num_nodes, edge_heads=edge_heads, edge_tails=edge_tails)
        dist_cache: dict[int, list[int]] = {}
        annotated: list[OracleTrajectory] = []
        for traj in trajectories:
            if len(traj.edge_local_ids) == 0:
                continue
            start_local = int(traj.start_local)
            if start_local < 0 or start_local >= num_nodes:
                continue
            current = start_local
            visit_counts: dict[int, int] = {start_local: 1}
            revisit_count = 0
            valid_path = True
            for edge_local in traj.edge_local_ids:
                edge_id = int(edge_local)
                if edge_id < 0 or edge_id >= len(edge_tails):
                    valid_path = False
                    break
                if int(edge_heads[edge_id]) != current:
                    valid_path = False
                    break
                next_node = int(edge_tails[edge_id])
                current = next_node
                next_visits = visit_counts.get(next_node, 0) + 1
                visit_counts[next_node] = next_visits
                if next_visits > 1:
                    revisit_count += 1
            if not valid_path:
                continue
            target_local = current
            dist_to_target = dist_cache.get(target_local)
            if dist_to_target is None:
                dist_to_target = self._bfs_distance_to_target(
                    num_nodes=num_nodes,
                    reverse_neighbors=reverse_neighbors,
                    target_local=target_local,
                )
                dist_cache[target_local] = dist_to_target
            shortest_len = int(dist_to_target[start_local])
            if shortest_len < 0:
                continue
            shortest_gap = max(len(traj.edge_local_ids) - shortest_len, 0)
            annotated.append(
                replace(
                    traj,
                    target_local=target_local,
                    shortest_gap=shortest_gap,
                    revisit_count=revisit_count,
                )
            )
        return annotated

    @staticmethod
    def _build_reverse_neighbors(
        *,
        num_nodes: int,
        edge_heads: list[int],
        edge_tails: list[int],
    ) -> list[list[int]]:
        reverse_neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
        for head, tail in zip(edge_heads, edge_tails):
            if head < 0 or head >= num_nodes or tail < 0 or tail >= num_nodes:
                continue
            reverse_neighbors[tail].append(head)
        return reverse_neighbors

    @staticmethod
    def _bfs_distance_to_target(
        *,
        num_nodes: int,
        reverse_neighbors: list[list[int]],
        target_local: int,
    ) -> list[int]:
        dist = [-1 for _ in range(num_nodes)]
        if target_local < 0 or target_local >= num_nodes:
            return dist
        dist[target_local] = 0
        queue: deque[int] = deque([target_local])
        while len(queue) > 0:
            node = queue.popleft()
            next_dist = dist[node] + 1
            for prev in reverse_neighbors[node]:
                if dist[prev] != -1:
                    continue
                dist[prev] = next_dist
                queue.append(prev)
        return dist


__all__ = ["HighEnergyReplayBuffer", "ReplayBatch", "OracleTrajectory"]
