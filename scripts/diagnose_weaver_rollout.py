from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import rootutils
except ModuleNotFoundError:
    rootutils = None
else:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from lightning import seed_everything  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.schema import RetrievalBatch  # noqa: E402
from src.training.factory import build_datamodule, build_model  # noqa: E402
from src.training.resources import setup_datamodule  # noqa: E402
from src.weaver.legacy.coverage_proposal import (  # noqa: E402
    CandidateEdges,
    MinimalSufficiencyTeacher,
)
from src.weaver.loss import SubTrajectoryBalanceLoss  # noqa: E402
from src.weaver.policy import Policy, frontier_logit_summary  # noqa: E402
from src.weaver.reward import (
    RewardModel,
    TerminalRewardOutput,
    target_ids,
)  # noqa: E402
from src.weaver.rollout.executor import (
    budget_exhausted_mask,
    has_candidate,
)  # noqa: E402
from src.weaver.rollout.runner import concat_rollout_batches  # noqa: E402
from src.weaver.rollout.sampling import option_action_probs  # noqa: E402
from src.weaver.rollout.schema import RolloutBatch  # noqa: E402
from src.weaver.state import State  # noqa: E402
from src.weaver.state_ops import frontier_edges  # noqa: E402

UNREACHABLE = 1_000_000_000


@dataclass
class MeanAgg:
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, key: str, value: float | int | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().item())
        self.values[key].append(float(value))

    def extend(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.add(key, value)

    def mean(self) -> dict[str, float]:
        return {
            key: (sum(vals) / float(len(vals)) if vals else 0.0)
            for key, vals in sorted(self.values.items())
        }


@dataclass(frozen=True)
class GraphMeta:
    graph_id: int
    node_lo: int
    node_hi: int
    edge_lo: int
    edge_hi: int
    target_lo: int
    target_hi: int


def _as_device_batch(batch: RetrievalBatch, device: torch.device) -> RetrievalBatch:
    return batch.to(device)  # type: ignore[return-value]


def _graph_metas(batch: RetrievalBatch) -> list[GraphMeta]:
    ptr = batch.ptr.detach().cpu().long()
    edge_ptr = batch.edge_ptr.detach().cpu().long()
    target_ids = batch.reachable_target_node_ids.detach().cpu().long()
    node_batch = batch.batch.detach().cpu().long()
    if target_ids.numel() > 0:
        target_graph = node_batch.index_select(0, target_ids)
        target_counts = torch.bincount(target_graph, minlength=int(batch.num_graphs))
    else:
        target_counts = torch.zeros(int(batch.num_graphs), dtype=torch.long)

    metas: list[GraphMeta] = []
    target_offset = 0
    for graph_id in range(int(batch.num_graphs)):
        target_count = int(target_counts[graph_id].item())
        metas.append(
            GraphMeta(
                graph_id=graph_id,
                node_lo=int(ptr[graph_id].item()),
                node_hi=int(ptr[graph_id + 1].item()),
                edge_lo=int(edge_ptr[graph_id].item()),
                edge_hi=int(edge_ptr[graph_id + 1].item()),
                target_lo=target_offset,
                target_hi=target_offset + target_count,
            )
        )
        target_offset += target_count
    return metas


def _target_distances_for_graph(batch: RetrievalBatch, meta: GraphMeta) -> torch.Tensor:
    node_count = meta.node_hi - meta.node_lo
    target_count = meta.target_hi - meta.target_lo
    if target_count <= 0:
        return torch.empty(
            (0, node_count), dtype=torch.long, device=batch.edge_index.device
        )

    offset = 0
    for prev in _graph_metas(batch)[: meta.graph_id]:
        offset += (prev.target_hi - prev.target_lo) * (prev.node_hi - prev.node_lo)
    size = target_count * node_count
    return batch.target_node_distances_flat[offset : offset + size].view(
        target_count, node_count
    )


def _edge_path_mask_for_graph(batch: RetrievalBatch, meta: GraphMeta) -> torch.Tensor:
    edge_count = meta.edge_hi - meta.edge_lo
    target_count = meta.target_hi - meta.target_lo
    if target_count <= 0:
        return torch.empty(
            (0, edge_count), dtype=torch.bool, device=batch.edge_index.device
        )

    offset = 0
    for prev in _graph_metas(batch)[: meta.graph_id]:
        offset += (prev.target_hi - prev.target_lo) * (prev.edge_hi - prev.edge_lo)
    size = target_count * edge_count
    return batch.target_shortest_path_edge_mask_flat[offset : offset + size].view(
        target_count, edge_count
    )


def _local_nodes(ids: torch.Tensor, meta: GraphMeta) -> torch.Tensor:
    ids = ids.long()
    mask = (ids >= meta.node_lo) & (ids < meta.node_hi)
    return ids[mask] - meta.node_lo


def _rate(numer: float, denom: float) -> float:
    return float(numer / denom) if denom > 0.0 else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    return float(value)


def _empty_one_step_reward_oracle_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_stop_log_reward": 0.0,
        f"{prefix}_best_child_log_reward": 0.0,
        f"{prefix}_best_child_minus_stop_log_reward": 0.0,
        f"{prefix}_best_child_support": 0.0,
        f"{prefix}_answer_edge_rank_by_policy": 0.0,
        f"{prefix}_policy_top1_child_support": 0.0,
        f"{prefix}_policy_top5_child_support": 0.0,
        f"{prefix}_candidate_count": 0.0,
    }


def reachability_metrics(batch: RetrievalBatch) -> dict[str, float]:
    metas = _graph_metas(batch)
    reachable_graphs = 0
    has_anchor = 0
    num_targets: list[float] = []
    min_depths: list[int] = []
    undirected_min_depths: list[int] = []

    for meta in metas:
        anchors = _local_nodes(batch.anchor_node_ids, meta)
        if anchors.numel() > 0:
            has_anchor += 1
        dists = _target_distances_for_graph(batch, meta)
        target_count = int(dists.size(0))
        num_targets.append(float(target_count))
        if target_count > 0:
            reachable_graphs += 1
        if target_count == 0 or anchors.numel() == 0:
            continue

        anchor_dists = dists.index_select(1, anchors.to(dists.device))
        directed = anchor_dists.min(dim=1).values
        min_depths.extend(
            [int(x) for x in directed.detach().cpu().tolist() if int(x) < UNREACHABLE]
        )
        undirected_min_depths.extend(_undirected_target_depths(batch, meta, anchors))

    graph_count = float(len(metas))
    target_count = float(len(min_depths))
    metrics = {
        "oracle/has_anchor": _rate(float(has_anchor), graph_count),
        "oracle/num_targets_mean": _mean(num_targets),
        "oracle/reachable_target_rate": _rate(float(reachable_graphs), graph_count),
        "oracle/no_reachable_target_rate": 1.0
        - _rate(float(reachable_graphs), graph_count),
    }
    for depth in range(4):
        metrics[f"oracle/target_at_depth_{depth}_rate"] = _rate(
            float(sum(1 for value in min_depths if value <= depth)),
            target_count,
        )
        metrics[f"oracle/undirected_target_at_depth_{depth}_rate"] = _rate(
            float(sum(1 for value in undirected_min_depths if value <= depth)),
            float(len(undirected_min_depths)),
        )
    return metrics


def _undirected_target_depths(
    batch: RetrievalBatch,
    meta: GraphMeta,
    anchors: torch.Tensor,
) -> list[int]:
    node_count = meta.node_hi - meta.node_lo
    targets = (
        _local_nodes(batch.reachable_target_node_ids, meta).detach().cpu().tolist()
    )
    if not targets:
        return []
    graph_edges = (
        batch.edge_index[:, meta.edge_lo : meta.edge_hi].detach().cpu() - meta.node_lo
    )
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in graph_edges.t().tolist():
        if 0 <= src < node_count and 0 <= dst < node_count:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
    dist = [UNREACHABLE] * node_count
    queue = [int(x) for x in anchors.detach().cpu().tolist()]
    for anchor in queue:
        dist[anchor] = 0
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for nxt in adjacency[node]:
            if dist[nxt] != UNREACHABLE:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return [dist[target] for target in targets if dist[target] < UNREACHABLE]


def valid_progress_mask(
    *,
    batch: RetrievalBatch,
    state: State,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> torch.Tensor:
    device = candidate_edge_ids.device
    valid = torch.zeros(candidate_edge_ids.numel(), dtype=torch.bool, device=device)
    if candidate_edge_ids.numel() == 0:
        return valid

    metas = _graph_metas(batch)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)

    for graph_id in range(int(batch.num_graphs)):
        candidate_pos = (
            (candidate_batch_ids == graph_id).nonzero(as_tuple=False).view(-1)
        )
        if candidate_pos.numel() == 0:
            continue
        meta = metas[graph_id]
        target_ids = batch.reachable_target_node_ids[meta.target_lo : meta.target_hi]
        if target_ids.numel() == 0:
            continue
        target_local = set(_local_nodes(target_ids, meta).detach().cpu().tolist())
        uncovered = ~active_nodes.index_select(
            0, target_ids.to(device=device, dtype=torch.long)
        )
        if not bool(uncovered.any()):
            continue

        dists = _target_distances_for_graph(batch, meta).to(
            device=device, dtype=torch.long
        )
        dists = dists[uncovered]
        path_mask = _edge_path_mask_for_graph(batch, meta).to(
            device=device, dtype=torch.bool
        )
        path_mask = path_mask[uncovered]

        edges = candidate_edge_ids.index_select(0, candidate_pos)
        local_edges = edges - meta.edge_lo
        src = edge_index[0].index_select(0, edges) - meta.node_lo
        dst = edge_index[1].index_select(0, edges) - meta.node_lo
        global_src = src + meta.node_lo
        global_dst = dst + meta.node_lo
        src_active = active_nodes.index_select(0, global_src)
        dst_active = active_nodes.index_select(0, global_dst)
        exactly_one_active = src_active ^ dst_active

        for local_idx, pos in enumerate(candidate_pos.tolist()):
            if not bool(exactly_one_active[local_idx].item()):
                continue
            new_local = (
                int(dst[local_idx].item())
                if bool(src_active[local_idx].item())
                else int(src[local_idx].item())
            )
            if new_local in target_local:
                valid[pos] = True
                continue
            edge_local = int(local_edges[local_idx].item())
            if edge_local < 0 or edge_local >= path_mask.size(1):
                continue
            if not bool(path_mask[:, edge_local].any().item()):
                continue
            src_local = int(src[local_idx].item())
            dst_local = int(dst[local_idx].item())
            if (
                src_local < 0
                or dst_local < 0
                or src_local >= dists.size(1)
                or dst_local >= dists.size(1)
            ):
                continue
            if bool(src_active[local_idx].item()):
                active_dist = dists[:, src_local]
                new_dist = dists[:, dst_local]
            else:
                active_dist = dists[:, dst_local]
                new_dist = dists[:, src_local]
            if bool((active_dist > new_dist).logical_and(new_dist >= 0).any().item()):
                valid[pos] = True
    return valid


@torch.no_grad()
def ranking_metrics_for_state(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    state: State,
    reward_model: RewardModel,
    prefix: str,
) -> dict[str, float]:
    context = policy.prepare_rollout_context(batch)
    out = policy(
        batch,
        state,
        rollout_context=context,
        return_edge_breakdown=True,
        edge_logit_mode="semantic",
    )
    logits = out.edge_score_breakdown.semantic_logits
    valid = valid_progress_mask(
        batch=batch,
        state=state,
        candidate_edge_ids=out.candidate_edge_ids,
        candidate_batch_ids=out.candidate_batch_ids,
    )
    del reward_model
    return _rank_metrics(
        logits=logits.detach(),
        candidate_batch=out.candidate_batch_ids.detach(),
        valid=valid.detach(),
        num_graphs=int(batch.num_graphs),
        prefix=prefix,
    )


@torch.no_grad()
def root_one_step_reward_oracle_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
) -> dict[str, float]:
    """
    Compare root stop reward with all one-step frontier children.

    Metrics are graph-averaged over graphs that have at least one root frontier
    candidate. The rank metric uses the best reward-improving child per graph.
    """
    state = State.create_initial(batch, expand_budget=expand_budget)
    return one_step_reward_oracle_metrics(
        policy=policy,
        batch=batch,
        state=state,
        reward_model=reward_model,
        prefix="oracle/root",
    )


@torch.no_grad()
def one_step_reward_oracle_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    state: State,
    reward_model: RewardModel,
    prefix: str,
) -> dict[str, float]:
    context = policy.prepare_rollout_context(batch)
    current = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    out = policy(
        batch,
        state,
        rollout_context=context,
        return_edge_breakdown=True,
        stop_log_reward=current.log_reward,
    )
    logits = out.edge_logits.detach()
    candidate_edge_ids = out.candidate_edge_ids.detach()
    candidate_batch_ids = out.candidate_batch_ids.detach()

    metrics = _empty_one_step_reward_oracle_metrics(prefix)
    if candidate_edge_ids.numel() == 0:
        return metrics
    child_rewards = _evaluate_child_rewards(
        batch=batch,
        state=state,
        reward_model=reward_model,
        candidate_edge_ids=candidate_edge_ids,
    )

    stop_values: list[float] = []
    best_child_values: list[float] = []
    best_child_gains: list[float] = []
    best_child_supports: list[float] = []
    best_child_ranks: list[float] = []
    best_child_edge_probs: list[float] = []
    best_child_8sample_hits: list[float] = []
    top1_supports: list[float] = []
    top5_supports: list[float] = []
    candidate_counts: list[float] = []

    for graph_id in range(int(batch.num_graphs)):
        mask = candidate_batch_ids.eq(graph_id)
        if not bool(mask.any()):
            continue

        pos = mask.nonzero(as_tuple=False).view(-1)
        graph_child_log_reward = child_rewards.log_reward.index_select(0, pos)
        graph_child_support = child_rewards.answer_support.index_select(0, pos)
        graph_logits = logits.index_select(0, pos)

        best_reward_pos = graph_child_log_reward.argmax()
        best_reward = graph_child_log_reward[best_reward_pos]
        best_support = graph_child_support[best_reward_pos]
        rank = int((graph_logits > graph_logits[best_reward_pos]).sum().item()) + 1
        graph_edge_prob = torch.softmax(graph_logits.float(), dim=0)
        best_child_prob = float(graph_edge_prob[best_reward_pos].item())

        top_k = min(5, int(pos.numel()))
        _, logit_order = torch.topk(graph_logits, k=top_k)
        top1_supports.append(float(graph_child_support[logit_order[0]].item()))
        top5_supports.append(
            float(graph_child_support.index_select(0, logit_order).max().item())
        )

        stop_value = float(current.log_reward[graph_id].item())
        stop_values.append(stop_value)
        best_child_values.append(float(best_reward.item()))
        best_child_gains.append(float(best_reward.item()) - stop_value)
        best_child_supports.append(float(best_support.item()))
        best_child_ranks.append(float(rank))
        best_child_edge_probs.append(best_child_prob)
        best_child_8sample_hits.append(1.0 - (1.0 - best_child_prob) ** 8)
        candidate_counts.append(float(pos.numel()))

    metrics.update(
        {
            f"{prefix}_stop_log_reward": _mean(stop_values),
            f"{prefix}_best_child_log_reward": _mean(best_child_values),
            f"{prefix}_best_child_minus_stop_log_reward": _mean(best_child_gains),
            f"{prefix}_best_child_support": _mean(best_child_supports),
            f"{prefix}_answer_edge_rank_by_policy": _mean(best_child_ranks),
            f"{prefix}_best_child_edge_prob": _mean(best_child_edge_probs),
            f"{prefix}_best_child_8sample_hit_rate": _mean(best_child_8sample_hits),
            f"{prefix}_policy_top1_child_support": _mean(top1_supports),
            f"{prefix}_policy_top5_child_support": _mean(top5_supports),
            f"{prefix}_candidate_count": _mean(candidate_counts),
        }
    )
    return metrics


@dataclass(frozen=True)
class ChildRewardBatch:
    log_reward: torch.Tensor
    answer_support: torch.Tensor


@torch.no_grad()
def _evaluate_child_rewards(
    *,
    batch: RetrievalBatch,
    state: State,
    reward_model: RewardModel,
    candidate_edge_ids: torch.Tensor,
) -> ChildRewardBatch:
    candidate_edge_ids = candidate_edge_ids.to(
        device=batch.edge_index.device, dtype=torch.long
    ).view(-1)
    if candidate_edge_ids.numel() == 0:
        empty = candidate_edge_ids.new_empty((0,), dtype=torch.float32)
        return ChildRewardBatch(log_reward=empty, answer_support=empty)

    current = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    child_support = current.supported_answer_recall.index_select(
        0, batch.edge_batch.index_select(0, candidate_edge_ids).long()
    ).to(dtype=torch.float32)

    edge_index = batch.edge_index.to(device=batch.edge_index.device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=batch.edge_index.device, dtype=torch.long)
    node_batch = batch.batch.to(device=batch.edge_index.device, dtype=torch.long)
    target_mask = torch.zeros(
        int(batch.num_nodes_total), dtype=torch.bool, device=batch.edge_index.device
    )
    ids = (
        target_ids(batch).to(device=batch.edge_index.device, dtype=torch.long).view(-1)
    )
    if ids.numel() > 0:
        target_mask[ids] = True

    metas = _graph_metas(batch)
    target_count_by_graph = torch.zeros(
        int(batch.num_graphs), dtype=torch.float32, device=batch.edge_index.device
    )
    if ids.numel() > 0:
        target_graph = node_batch.index_select(0, ids)
        target_count_by_graph = torch.bincount(
            target_graph,
            minlength=int(batch.num_graphs),
        ).to(dtype=torch.float32)

    for idx, edge_id_tensor in enumerate(candidate_edge_ids):
        edge_id = int(edge_id_tensor.item())
        graph_id = int(edge_batch[edge_id].item())
        meta = metas[graph_id]
        if target_count_by_graph[graph_id] <= 0.0:
            continue

        active_edges = state.active_edges[meta.edge_lo : meta.edge_hi].detach().clone()
        active_edges[edge_id - meta.edge_lo] = True
        local_edges = active_edges.nonzero(as_tuple=False).view(-1) + meta.edge_lo

        anchors = _local_nodes(batch.anchor_node_ids, meta) + meta.node_lo
        active_anchors = anchors[
            state.active_nodes.index_select(0, anchors.to(state.active_nodes.device))
        ]
        if active_anchors.numel() == 0:
            continue

        reached = _connected_nodes_from_anchors_for_edges(
            edge_index=edge_index,
            edge_ids=local_edges,
            anchors=active_anchors.to(device=edge_index.device, dtype=torch.long),
        )
        if not reached:
            continue

        graph_targets = (
            (target_mask & node_batch.eq(graph_id)).nonzero(as_tuple=False).view(-1)
        )
        hits = sum(int(node_id) in reached for node_id in graph_targets.tolist())
        child_support[idx] = float(hits) / float(max(1, int(graph_targets.numel())))

    graph_ids = edge_batch.index_select(0, candidate_edge_ids).long()
    expanded_count = current.expanded_edge_count.index_select(0, graph_ids).to(
        dtype=torch.float32
    )
    newly_expanded = (~state.root_edges.index_select(0, candidate_edge_ids)).to(
        dtype=torch.float32
    ) * (~state.active_edges.index_select(0, candidate_edge_ids)).to(
        dtype=torch.float32
    )
    expanded_count = expanded_count + newly_expanded
    base_log_reward = (child_support + float(reward_model.utility_epsilon)).log()
    log_reward = (
        base_log_reward - float(reward_model.edge_cost) * expanded_count
    ).clamp_min(float(reward_model.log_reward_clip_min))

    return ChildRewardBatch(
        log_reward=log_reward.to(dtype=torch.float32),
        answer_support=child_support.to(dtype=torch.float32),
    )


def _connected_nodes_from_anchors_for_edges(
    *,
    edge_index: torch.Tensor,
    edge_ids: torch.Tensor,
    anchors: torch.Tensor,
) -> set[int]:
    visited = {int(node_id) for node_id in anchors.tolist()}
    frontier = list(visited)
    adjacency: dict[int, list[int]] = {}
    if edge_ids.numel() > 0:
        src = edge_index[0].index_select(0, edge_ids).tolist()
        dst = edge_index[1].index_select(0, edge_ids).tolist()
        for left, right in zip(src, dst):
            left_id = int(left)
            right_id = int(right)
            adjacency.setdefault(left_id, []).append(right_id)
            adjacency.setdefault(right_id, []).append(left_id)
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)
    return visited


@torch.no_grad()
def utility_gain_mask(
    *,
    batch: RetrievalBatch,
    state: State,
    reward_model: RewardModel,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    min_gain: float = 1.0e-8,
) -> torch.Tensor:
    device = candidate_edge_ids.device
    valid = torch.zeros(candidate_edge_ids.numel(), dtype=torch.bool, device=device)
    if candidate_edge_ids.numel() == 0:
        return valid

    current = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    for pos in range(int(candidate_edge_ids.numel())):
        graph_id = int(candidate_batch_ids[pos].item())
        edge_id = candidate_edge_ids[pos].view(1)
        next_nodes = state.active_nodes.detach().clone()
        next_edges = state.active_edges.detach().clone()
        src = edge_index[0].index_select(0, edge_id)
        dst = edge_index[1].index_select(0, edge_id)
        next_edges[edge_id] = True
        next_nodes[src] = True
        next_nodes[dst] = True
        nxt = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=next_nodes,
            active_edges=next_edges,
            state=state,
        )
        gain = float((nxt.utility[graph_id] - current.utility[graph_id]).item())
        if gain > float(min_gain):
            valid[pos] = True
    return valid


def _rank_metrics(
    *,
    logits: torch.Tensor,
    candidate_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
    prefix: str,
) -> dict[str, float]:
    exists = 0
    ranks: list[float] = []
    best_valid_probs: list[float] = []
    valid_prob_masses: list[float] = []
    valid_8sample_hits: list[float] = []
    candidate_counts: list[float] = []
    for graph_id in range(num_graphs):
        mask = candidate_batch.eq(graph_id)
        if not bool(mask.any()):
            continue
        graph_valid = valid[mask]
        if not bool(graph_valid.any()):
            continue
        exists += 1
        graph_logits = logits[mask]
        valid_logits = graph_logits[graph_valid]
        best_valid_logit = valid_logits.max()
        rank = int((graph_logits > best_valid_logit).sum().item()) + 1
        graph_edge_prob = torch.softmax(graph_logits.float(), dim=0)
        valid_mass = float(graph_edge_prob[graph_valid].sum().item())
        best_valid_prob = float(
            graph_edge_prob[graph_logits.eq(best_valid_logit)].max().item()
        )
        ranks.append(float(rank))
        best_valid_probs.append(best_valid_prob)
        valid_prob_masses.append(valid_mass)
        valid_8sample_hits.append(1.0 - (1.0 - valid_mass) ** 8)
        candidate_counts.append(float(graph_logits.numel()))

    denom = float(num_graphs)
    return {
        f"{prefix}_valid_edge_exists_rate": _rate(float(exists), denom),
        f"{prefix}_candidate_count_mean": _mean(candidate_counts),
        f"{prefix}_best_valid_rank_mean": _mean(ranks),
        f"{prefix}_best_valid_rank_median": _median(ranks),
        f"{prefix}_best_valid_prob_mean": _mean(best_valid_probs),
        f"{prefix}_valid_edge_prob_mass_mean": _mean(valid_prob_masses),
        f"{prefix}_valid_edge_8sample_hit_rate": _mean(valid_8sample_hits),
        f"{prefix}_valid_edge_top1_rate": _rate(
            float(sum(r <= 1 for r in ranks)), float(len(ranks))
        ),
        f"{prefix}_valid_edge_top3_rate": _rate(
            float(sum(r <= 3 for r in ranks)), float(len(ranks))
        ),
        f"{prefix}_valid_edge_top5_rate": _rate(
            float(sum(r <= 5 for r in ranks)), float(len(ranks))
        ),
        f"{prefix}_valid_edge_top10_rate": _rate(
            float(sum(r <= 10 for r in ranks)), float(len(ranks))
        ),
        f"{prefix}_valid_edge_mrr": _mean([1.0 / r for r in ranks]),
    }


@torch.no_grad()
def oracle_depth1_state(
    *,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
) -> State:
    state = State.create_initial(batch, expand_budget=expand_budget)
    context = CandidateEdgesForState.from_state(batch=batch, state=state)
    reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    teacher = MinimalSufficiencyTeacher(gain_margin=0.02)
    gains, valid = teacher.score_expands(
        retrieval_batch=batch,
        state=state,
        candidates=context.candidates,
        reward_model=reward_model,
        current_reward=reward,
        budget_per_graph=torch.ones(
            int(batch.num_graphs), device=batch.edge_index.device
        ),
        num_graphs=int(batch.num_graphs),
    )
    chosen: list[int] = []
    for graph_id in range(int(batch.num_graphs)):
        mask = context.candidates.batch_index.eq(graph_id) & valid
        if not bool(mask.any()):
            continue
        local = mask.nonzero(as_tuple=False).view(-1)
        best = local[gains.index_select(0, local).argmax()]
        if gains[best] > 0.0:
            chosen.append(int(context.candidates.edge_ids[best].item()))
    if chosen:
        state.apply_expansion(
            chosen_edges=torch.tensor(
                chosen, dtype=torch.long, device=batch.edge_index.device
            ),
            edge_index=batch.edge_index,
        )
    return state


@dataclass(frozen=True)
class CandidateEdgesForState:
    candidates: CandidateEdges

    @classmethod
    def from_state(
        cls, *, batch: RetrievalBatch, state: State
    ) -> "CandidateEdgesForState":
        edge_ids, edge_batch = frontier_edges(
            batch=batch,
            state=state,
            device=batch.edge_index.device,
        )
        return cls(
            candidates=CandidateEdges(
                edge_ids=edge_ids,
                expand_logits=torch.zeros(edge_ids.numel(), device=edge_ids.device),
                batch_index=edge_batch,
            )
        )


@torch.no_grad()
def policy_rollout_metrics(rollouts: list[RolloutBatch]) -> dict[str, float]:
    if not rollouts:
        return {}
    rollout = concat_rollout_batches(rollouts)
    stats = rollout.stats
    traces = rollout.traces
    f1 = stats.terminal_answer_f1.float()
    utility = stats.terminal_utility.float()
    log_reward = stats.terminal_log_reward.float()
    expanded = stats.terminal_expanded_edge_count.float()
    min_gap = getattr(
        stats, "terminal_minimality_gap", torch.zeros_like(log_reward)
    ).float()
    stop_mask = traces.stop_mask.bool()
    continue_mask = traces.continue_mask.bool()
    stop_now_f1 = traces.stop_now_answer_f1.float()
    stop_now_valid = traces.stop_now_valid_mask.bool()
    lengths = stats.trajectory_length.long()
    horizon = stop_mask.size(1)
    rows = torch.arange(lengths.numel(), device=lengths.device)
    terminal_idx = lengths.clamp(1, horizon) - 1
    terminal_model_stop = stop_mask[rows, terminal_idx]
    budget_exhausted = (
        traces.budget_exhausted_mask[rows, terminal_idx].bool()
        if traces.budget_exhausted_mask is not None
        else terminal_idx.ge(horizon - 1)
    )

    hit_state = stop_now_valid & stop_now_f1.gt(0.0)
    policy_valid = traces.policy_action_valid_mask.bool()
    valid_steps = torch.arange(horizon, device=lengths.device).unsqueeze(
        0
    ) < lengths.clamp_min(0).unsqueeze(1)
    stop_depth = terminal_idx[terminal_model_stop].to(dtype=torch.float32)
    continue_depth0 = (
        continue_mask[:, 0] if horizon > 0 else continue_mask.new_zeros((0,))
    )
    first_hit_depths: list[float] = []
    extra_after_hit: list[float] = []
    continue_after_first_hit = 0
    hit_depth_1 = 0
    hit_depth_2 = 0
    hit_rows = 0
    for row in range(hit_state.size(0)):
        hits = hit_state[row].nonzero(as_tuple=False).view(-1)
        if hits.numel() == 0:
            continue
        hit_rows += 1
        first = int(hits[0].item())
        first_hit_depths.append(float(first))
        hit_depth_1 += int(first <= 1)
        hit_depth_2 += int(first <= 2)
        after = torch.arange(horizon, device=continue_mask.device) >= first
        extra = int((continue_mask[row] & after).sum().item())
        extra_after_hit.append(float(extra))
        continue_after_first_hit += int(extra > 0)

    metrics = {
        "policy/nonzero_f1_rate": _safe_float(f1.gt(0.0).float().mean()),
        "policy/answer_f1_mean": _safe_float(f1.mean()),
        "policy/support_mean": _safe_float(stats.terminal_utility.float().mean()),
        "policy/utility_mean": _safe_float(utility.mean()),
        "policy/log_reward_mean": _safe_float(log_reward.mean()),
        "policy/expanded_edge_count_mean": _safe_float(expanded.mean()),
        "policy/minimality_gap_mean": _safe_float(min_gap.mean()),
        "policy/budget_exhausted_stop_rate": _safe_float(
            budget_exhausted.float().mean()
        ),
        "policy/model_stop_rate": _safe_float(terminal_model_stop.float().mean()),
        "policy/forced_stop_rate": _safe_float(
            (terminal_model_stop & budget_exhausted).float().mean()
        ),
        "policy/first_hit_depth_mean": _mean(first_hit_depths),
        "policy/hit_at_depth_1_rate": _rate(float(hit_depth_1), float(hit_rows)),
        "policy/hit_at_depth_2_rate": _rate(float(hit_depth_2), float(hit_rows)),
        "policy/continue_after_first_hit_rate": _rate(
            float(continue_after_first_hit), float(hit_rows)
        ),
        "policy/extra_edges_after_first_hit_mean": _mean(extra_after_hit),
        "train/policy/target_stop_prob_mean": (
            _safe_float(traces.target_stop_prob[policy_valid].float().mean())
            if bool(policy_valid.any())
            else 0.0
        ),
        "train/policy/target_continue_prob_mean": (
            _safe_float(traces.target_continue_prob[policy_valid].float().mean())
            if bool(policy_valid.any())
            else 0.0
        ),
        "train/rollout/budget_exhausted_ratio": _safe_float(
            budget_exhausted.float().mean()
        ),
        "train/rollout/continue_depth_0_rate": _safe_float(
            continue_depth0.float().mean()
        ),
        "train/rollout/continue_rate": (
            _safe_float(continue_mask[valid_steps].float().mean())
            if bool(valid_steps.any())
            else 0.0
        ),
    }
    for depth in range(horizon):
        metrics[f"train/rollout/stop_depth_hist_{depth}"] = _rate(
            float(stop_depth.eq(float(depth)).sum().item()),
            float(terminal_model_stop.sum().item()),
        )
    return metrics


@torch.no_grad()
def reward_sanity_metrics(rollouts: list[RolloutBatch]) -> dict[str, float]:
    if not rollouts:
        return {}
    rollout = concat_rollout_batches(rollouts)
    traces = rollout.traces
    stats = rollout.stats
    zero = torch.zeros_like(stats.terminal_log_reward.float())
    minimality_gap = getattr(stats, "terminal_minimality_gap", zero).float()
    minimality_penalty = getattr(
        stats,
        "terminal_minimality_penalty",
        (
            stats.terminal_complexity_penalty
            if stats.terminal_complexity_penalty is not None
            else zero
        ),
    ).float()
    minimal_edge_count = getattr(stats, "terminal_minimal_edge_count", zero).float()
    expanded_edge_count = (
        stats.terminal_expanded_edge_count.float()
        if stats.terminal_expanded_edge_count is not None
        else zero
    )
    first_hit_rewards: list[float] = []
    final_rewards: list[float] = []
    final_better = 0
    for row in range(traces.stop_now_answer_f1.size(0)):
        hit = (
            (
                traces.stop_now_valid_mask[row].bool()
                & traces.stop_now_answer_f1[row].gt(0.0)
            )
            .nonzero(as_tuple=False)
            .view(-1)
        )
        if hit.numel() == 0:
            continue
        idx = int(hit[0].item())
        first_value = float(traces.stop_now_log_reward[row, idx].item())
        final_value = float(stats.terminal_log_reward[row].item())
        first_hit_rewards.append(first_value)
        final_rewards.append(final_value)
        final_better += int(final_value > first_value)
    return {
        "reward/first_hit_log_reward_mean": _mean(first_hit_rewards),
        "reward/final_log_reward_mean": _mean(final_rewards),
        "reward/final_minus_first_hit_log_reward_mean": _mean(
            [b - a for a, b in zip(first_hit_rewards, final_rewards)]
        ),
        "reward/final_better_than_first_hit_rate": _rate(
            float(final_better), float(len(first_hit_rewards))
        ),
        "reward/minimality_gap_at_final_mean": _safe_float(minimality_gap.mean()),
        "reward/minimality_penalty_at_final_mean": _safe_float(
            minimality_penalty.mean()
        ),
        "reward/expanded_edge_count_at_final_mean": _safe_float(
            expanded_edge_count.mean()
        ),
        "reward/minimal_edge_count_at_final_mean": _safe_float(
            minimal_edge_count.mean()
        ),
    }


@torch.no_grad()
def oracle_rollout_metrics(
    *,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
) -> dict[str, float]:
    state = State.create_initial(batch, expand_budget=expand_budget)
    teacher = MinimalSufficiencyTeacher(gain_margin=0.02)
    active = torch.ones(
        int(batch.num_graphs), dtype=torch.bool, device=batch.edge_index.device
    )
    stop_depths: list[int] = []

    for depth in range(expand_budget + 1):
        reward = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )
        if depth == expand_budget:
            stop_depths.extend([depth] * int(active.sum().item()))
            break

        context = CandidateEdgesForState.from_state(batch=batch, state=state)
        budget = torch.full(
            (int(batch.num_graphs),),
            expand_budget - depth,
            device=batch.edge_index.device,
        )
        gains, valid = teacher.score_expands(
            retrieval_batch=batch,
            state=state,
            candidates=context.candidates,
            reward_model=reward_model,
            current_reward=reward,
            budget_per_graph=budget,
            num_graphs=int(batch.num_graphs),
        )
        chosen: list[int] = []
        for graph_id in active.nonzero(as_tuple=False).view(-1).tolist():
            graph_id = int(graph_id)
            mask = context.candidates.batch_index.eq(graph_id) & valid
            if not bool(mask.any()):
                active[graph_id] = False
                stop_depths.append(depth)
                continue
            pos = mask.nonzero(as_tuple=False).view(-1)
            best = pos[gains.index_select(0, pos).argmax()]
            best_gain = float(gains[best].item())
            if float(reward.utility[graph_id].item()) > 0.0 and best_gain <= 0.02:
                active[graph_id] = False
                stop_depths.append(depth)
                continue
            chosen.append(int(context.candidates.edge_ids[best].item()))
        if chosen:
            state.apply_expansion(
                chosen_edges=torch.tensor(
                    chosen, dtype=torch.long, device=batch.edge_index.device
                ),
                edge_index=batch.edge_index,
            )
        if not bool(active.any()):
            break

    final_reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    hist = {
        f"oracle/stop_depth_{depth}_rate": _rate(
            float(stop_depths.count(depth)), float(len(stop_depths))
        )
        for depth in range(expand_budget + 1)
    }
    hist.update(
        {
            "oracle/nonzero_f1_rate": _safe_float(
                final_reward.answer_f1.gt(0.0).float().mean()
            ),
            "oracle/answer_f1_mean": _safe_float(final_reward.answer_f1.float().mean()),
            "oracle/support_mean": _safe_float(
                final_reward.supported_answer_recall.float().mean()
            ),
            "oracle/log_reward_mean": _safe_float(
                final_reward.log_reward.float().mean()
            ),
            "oracle/expanded_edge_count_mean": _safe_float(
                final_reward.expanded_edge_count.float().mean()
            ),
            "oracle/minimality_gap_mean": 0.0,
        }
    )
    return hist


@torch.no_grad()
def oracle_path_probability_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    num_rollouts: int,
) -> dict[str, float]:
    """
    Follow the reward teacher greedily, then score that exact path under policy.

    This answers a different question from rank diagnostics: even if every chosen
    edge is ranked well locally, the sampled trajectory probability is the
    product of option probabilities and conditional edge probabilities.
    """
    state = State.create_initial(batch, expand_budget=expand_budget)
    context = policy.prepare_rollout_context(batch)
    teacher = MinimalSufficiencyTeacher(gain_margin=0.02)
    device = batch.edge_index.device
    num_graphs = int(batch.num_graphs)

    active = torch.ones(num_graphs, dtype=torch.bool, device=device)
    log_path_prob = torch.zeros(num_graphs, dtype=torch.float32, device=device)
    selected_edge_count = torch.zeros(num_graphs, dtype=torch.float32, device=device)

    chosen_edge_probs: list[float] = []
    chosen_continue_probs: list[float] = []
    chosen_edge_ranks: list[float] = []
    terminal_stop_probs: list[float] = []
    agg = MeanAgg()

    for depth in range(expand_budget + 1):
        if not bool(active.any()):
            break

        current = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )
        out = policy(
            batch,
            state,
            rollout_context=context,
            return_edge_breakdown=True,
            stop_log_reward=current.log_reward,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=num_graphs,
        )
        has_edge = has_candidate(
            candidate_batch_ids=out.candidate_batch_ids,
            num_graphs=num_graphs,
            device=device,
        )
        exhausted = budget_exhausted_mask(
            remaining_budget,
            num_graphs=num_graphs,
            device=device,
        )
        can_expand = active & has_edge & ~exhausted
        stop_prob, continue_prob, edge_prob = option_action_probs(
            stop_logits=out.stop_logits,
            expand_logits=out.expand_logits,
            edge_logits=out.edge_logits,
            candidate_batch_ids=out.candidate_batch_ids,
            can_expand=can_expand,
            batch_size=num_graphs,
        )

        if depth >= expand_budget:
            stopping = active.clone()
            if bool(stopping.any()):
                p_stop = stop_prob[stopping].float().clamp_min(1.0e-12)
                log_path_prob[stopping] += p_stop.log()
                terminal_stop_probs.extend(p_stop.detach().cpu().tolist())
            active[stopping] = False
            break

        candidate_state = CandidateEdgesForState.from_state(batch=batch, state=state)
        gains, valid = teacher.score_expands(
            retrieval_batch=batch,
            state=state,
            candidates=candidate_state.candidates,
            reward_model=reward_model,
            current_reward=current,
            budget_per_graph=remaining_budget,
            num_graphs=num_graphs,
        )

        chosen_edges: list[int] = []
        for graph_id in active.nonzero(as_tuple=False).view(-1).tolist():
            graph_id = int(graph_id)
            if not bool(can_expand[graph_id].item()):
                p_stop = stop_prob[graph_id].float().clamp_min(1.0e-12)
                log_path_prob[graph_id] += p_stop.log()
                terminal_stop_probs.append(float(p_stop.item()))
                active[graph_id] = False
                continue

            mask = candidate_state.candidates.batch_index.eq(graph_id) & valid
            if not bool(mask.any()):
                p_stop = stop_prob[graph_id].float().clamp_min(1.0e-12)
                log_path_prob[graph_id] += p_stop.log()
                terminal_stop_probs.append(float(p_stop.item()))
                active[graph_id] = False
                continue

            pos = mask.nonzero(as_tuple=False).view(-1)
            best = pos[gains.index_select(0, pos).argmax()]
            best_gain = float(gains[best].item())
            if float(current.utility[graph_id].item()) > 0.0 and best_gain <= 0.02:
                p_stop = stop_prob[graph_id].float().clamp_min(1.0e-12)
                log_path_prob[graph_id] += p_stop.log()
                terminal_stop_probs.append(float(p_stop.item()))
                active[graph_id] = False
                continue

            edge_id = int(candidate_state.candidates.edge_ids[best].item())
            edge_pos = (
                (
                    out.candidate_batch_ids.eq(graph_id)
                    & out.candidate_edge_ids.eq(edge_id)
                )
                .nonzero(as_tuple=False)
                .view(-1)
            )
            if edge_pos.numel() == 0:
                p_stop = stop_prob[graph_id].float().clamp_min(1.0e-12)
                log_path_prob[graph_id] += p_stop.log()
                terminal_stop_probs.append(float(p_stop.item()))
                active[graph_id] = False
                continue

            edge_pos = edge_pos[0]
            p_continue = continue_prob[graph_id].float().clamp_min(1.0e-12)
            p_edge = edge_prob[edge_pos].float().clamp_min(1.0e-12)
            log_path_prob[graph_id] += p_continue.log() + p_edge.log()
            selected_edge_count[graph_id] += 1.0
            chosen_edges.append(edge_id)

            graph_pos = (
                out.candidate_batch_ids.eq(graph_id).nonzero(as_tuple=False).view(-1)
            )
            graph_logits = out.edge_logits.index_select(0, graph_pos)
            local_pos = graph_pos.eq(edge_pos).nonzero(as_tuple=False).view(-1)
            rank = 0.0
            if local_pos.numel() > 0:
                chosen_logit = graph_logits[int(local_pos[0].item())]
                rank = 1.0 + float((graph_logits > chosen_logit).sum().item())

            edge_prob_value = float(p_edge.item())
            continue_prob_value = float(p_continue.item())
            chosen_edge_probs.append(edge_prob_value)
            chosen_continue_probs.append(continue_prob_value)
            chosen_edge_ranks.append(rank)
            agg.add(f"oracle_path/chosen_edge_prob_depth_{depth}", edge_prob_value)
            agg.add(f"oracle_path/continue_prob_depth_{depth}", continue_prob_value)
            agg.add(f"oracle_path/chosen_edge_rank_depth_{depth}", rank)

        if chosen_edges:
            state.apply_expansion(
                chosen_edges=torch.tensor(
                    chosen_edges, dtype=torch.long, device=device
                ),
                edge_index=batch.edge_index,
            )

    final_reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    path_prob = log_path_prob.exp().clamp(0.0, 1.0)
    expected_hit = 1.0 - (1.0 - path_prob).clamp(0.0, 1.0) ** int(num_rollouts)
    nonzero = final_reward.answer_f1.gt(0.0)

    metrics = {
        "oracle_path/nonzero_f1_rate": _safe_float(nonzero.float().mean()),
        "oracle_path/answer_f1_mean": _safe_float(
            final_reward.answer_f1.float().mean()
        ),
        "oracle_path/exact_path_prob_mean": _safe_float(path_prob.mean()),
        "oracle_path/exact_path_prob_when_nonzero_mean": (
            _safe_float(path_prob[nonzero].mean()) if bool(nonzero.any()) else 0.0
        ),
        f"oracle_path/expected_hit_rate_{int(num_rollouts)}_mean": _safe_float(
            expected_hit.mean()
        ),
        f"oracle_path/expected_hit_rate_{int(num_rollouts)}_when_nonzero_mean": (
            _safe_float(expected_hit[nonzero].mean()) if bool(nonzero.any()) else 0.0
        ),
        "oracle_path/selected_edge_count_mean": _safe_float(selected_edge_count.mean()),
        "oracle_path/chosen_edge_prob_mean": _mean(chosen_edge_probs),
        "oracle_path/continue_prob_mean": _mean(chosen_continue_probs),
        "oracle_path/chosen_edge_rank_mean": _mean(chosen_edge_ranks),
        "oracle_path/terminal_stop_prob_mean": _mean(terminal_stop_probs),
    }
    metrics.update(agg.mean())
    return metrics


@torch.no_grad()
def stop_improvement_oracle_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    max_depth: int | None = None,
) -> dict[str, float]:
    """
    Greedily visit rollout states and ask whether any one-step child improves logR.

    y_stop(s)=1[max_e logR(s+e) <= logR(s)] is only logged here; it is not used
    as a loss or intervention.
    """
    state = State.create_initial(batch, expand_budget=expand_budget)
    context = policy.prepare_rollout_context(batch)
    active = torch.ones(
        int(batch.num_graphs), dtype=torch.bool, device=batch.edge_index.device
    )
    agg = MeanAgg()
    limit = expand_budget if max_depth is None else min(expand_budget, int(max_depth))

    for depth in range(limit + 1):
        if not bool(active.any()):
            break
        current = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )
        out = policy(
            batch,
            state,
            rollout_context=context,
            return_edge_breakdown=True,
            stop_log_reward=current.log_reward,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=int(batch.num_graphs),
        )
        has_edge = has_candidate(
            candidate_batch_ids=out.candidate_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        exhausted = budget_exhausted_mask(
            remaining_budget,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        can_expand = active & has_edge & ~exhausted
        stop_prob, continue_prob, _ = option_action_probs(
            stop_logits=out.stop_logits,
            expand_logits=out.expand_logits,
            edge_logits=out.edge_logits,
            candidate_batch_ids=out.candidate_batch_ids,
            can_expand=can_expand,
            batch_size=int(batch.num_graphs),
        )

        child_rewards = _evaluate_child_rewards(
            batch=batch,
            state=state,
            reward_model=reward_model,
            candidate_edge_ids=out.candidate_edge_ids,
        )

        best_gains = torch.zeros(
            int(batch.num_graphs), dtype=torch.float32, device=batch.edge_index.device
        )
        best_edges: list[int] = []
        valid_graphs = active & can_expand
        for graph_id in valid_graphs.nonzero(as_tuple=False).view(-1).tolist():
            graph_id = int(graph_id)
            mask = out.candidate_batch_ids.eq(graph_id)
            if not bool(mask.any()):
                continue
            pos = mask.nonzero(as_tuple=False).view(-1)
            child_log_reward = child_rewards.log_reward.index_select(0, pos)
            best_pos = pos[child_log_reward.argmax()]
            best_gain = (
                child_rewards.log_reward[best_pos] - current.log_reward[graph_id]
            )
            best_gains[graph_id] = best_gain
            if float(best_gain.item()) > 0.0:
                best_edges.append(int(out.candidate_edge_ids[best_pos].item()))

        stop_better = valid_graphs & best_gains.le(0.0)
        continue_better = valid_graphs & best_gains.gt(0.0)
        valid_count = valid_graphs.float().sum().clamp_min(1.0)
        agg.add(
            "stop_oracle/stop_now_better_ratio",
            _safe_float(stop_better.float().sum() / valid_count),
        )
        agg.add(
            "stop_oracle/mean_best_continue_minus_stop_log_reward",
            best_gains[valid_graphs].mean() if bool(valid_graphs.any()) else 0.0,
        )
        if bool(stop_better.any()):
            agg.add(
                "stop_oracle/policy_stop_prob_when_stop_better",
                stop_prob[stop_better].mean(),
            )
        if bool(continue_better.any()):
            agg.add(
                "stop_oracle/policy_continue_prob_when_continue_better",
                continue_prob[continue_better].mean(),
            )
        agg.add(
            f"stop_oracle/stop_now_better_ratio_depth_{depth}",
            _safe_float(stop_better.float().sum() / valid_count),
        )

        if depth >= limit or not best_edges:
            break
        state.apply_expansion(
            chosen_edges=torch.tensor(
                best_edges, dtype=torch.long, device=batch.edge_index.device
            ),
            edge_index=batch.edge_index,
        )
        if valid_graphs.any():
            active &= continue_better

    metrics = {
        "stop_oracle/stop_now_better_ratio": 0.0,
        "stop_oracle/mean_best_continue_minus_stop_log_reward": 0.0,
        "stop_oracle/policy_stop_prob_when_stop_better": 0.0,
        "stop_oracle/policy_continue_prob_when_continue_better": 0.0,
    }
    metrics.update(agg.mean())
    return metrics


@torch.no_grad()
def prior_search_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    beam_sizes: list[int],
) -> dict[str, float]:
    greedy_rewards = _prior_beam_rewards(
        policy=policy,
        batch=batch,
        reward_model=reward_model,
        expand_budget=expand_budget,
        beam_size=1,
    )
    metrics = {
        "prior_greedy_nonzero_f1_rate": _safe_float(
            greedy_rewards.answer_f1.gt(0.0).float().mean()
        )
    }
    for beam_size in beam_sizes:
        reward = _prior_beam_rewards(
            policy=policy,
            batch=batch,
            reward_model=reward_model,
            expand_budget=expand_budget,
            beam_size=int(beam_size),
        )
        metrics[f"prior_beam{int(beam_size)}_nonzero_f1_rate"] = _safe_float(
            reward.answer_f1.gt(0.0).float().mean()
        )
    return metrics


@torch.no_grad()
def _prior_beam_rewards(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    beam_size: int,
) -> TerminalRewardOutput:
    base = State.create_initial(batch, expand_budget=expand_budget)
    best_nodes = base.active_nodes.clone()
    best_edges = base.active_edges.clone()
    best_reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=best_nodes,
        active_edges=best_edges,
        state=base,
    )
    context = policy.prepare_rollout_context(batch)

    for graph_id in range(int(batch.num_graphs)):
        beams: list[tuple[float, State]] = [(0.0, base.detach())]
        graph_best_reward = float(best_reward.log_reward[graph_id].item())
        graph_best_state = base.detach()
        for _depth in range(expand_budget):
            next_beams: list[tuple[float, State]] = []
            for score, state in beams:
                out = policy(
                    batch,
                    state,
                    rollout_context=context,
                    return_edge_breakdown=True,
                    edge_logit_mode="semantic",
                )
                prior = out.edge_score_breakdown.semantic_logits
                mask = out.candidate_batch_ids.eq(graph_id)
                if not bool(mask.any()):
                    continue
                pos = mask.nonzero(as_tuple=False).view(-1)
                vals = prior.index_select(0, pos)
                top_k = min(int(beam_size), int(vals.numel()))
                _, order = torch.topk(vals, k=top_k)
                for idx in order.tolist():
                    edge = int(out.candidate_edge_ids[pos[idx]].item())
                    child = state.detach()
                    child.apply_expansion(
                        chosen_edges=torch.tensor(
                            [edge], dtype=torch.long, device=batch.edge_index.device
                        ),
                        edge_index=batch.edge_index,
                    )
                    child_reward = reward_model.evaluate_terminal_state(
                        retrieval_batch=batch,
                        active_nodes=child.active_nodes,
                        active_edges=child.active_edges,
                        state=child,
                    )
                    child_value = float(child_reward.log_reward[graph_id].item())
                    if child_value > graph_best_reward:
                        graph_best_reward = child_value
                        graph_best_state = child.detach()
                    next_beams.append((score + float(vals[idx].item()), child))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0], reverse=True)
            beams = next_beams[: int(beam_size)]
        best_nodes[
            batch.ptr[graph_id] : batch.ptr[graph_id + 1]
        ] = graph_best_state.active_nodes[batch.ptr[graph_id] : batch.ptr[graph_id + 1]]
        best_edges[
            batch.edge_ptr[graph_id] : batch.edge_ptr[graph_id + 1]
        ] = graph_best_state.active_edges[
            batch.edge_ptr[graph_id] : batch.edge_ptr[graph_id + 1]
        ]

    final_state = State(
        root_edges=base.root_edges,
        active_nodes=best_nodes,
        active_edges=best_edges,
        expand_budget=expand_budget,
    )
    return reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=best_nodes,
        active_edges=best_edges,
        state=final_state,
    )


@torch.no_grad()
def edge_and_gate_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
) -> dict[str, float]:
    state = State.create_initial(batch, expand_budget=expand_budget)
    context = policy.prepare_rollout_context(batch)
    agg = MeanAgg()
    for depth in range(min(expand_budget + 1, 3)):
        current = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )
        out = policy(
            batch,
            state,
            rollout_context=context,
            return_edge_breakdown=True,
            stop_log_reward=current.log_reward,
        )
        breakdown = out.edge_score_breakdown
        prior = breakdown.semantic_logits
        residual = getattr(breakdown, "residual_logits", torch.zeros_like(prior))
        scale = getattr(breakdown, "residual_scale", prior.new_tensor(0.0))
        scale = scale.detach().to(prior.device, prior.dtype)
        scaled_residual = scale * residual
        final = breakdown.final_logits
        if prior.numel() > 0:
            agg.add("edge/prior_abs_mean", prior.abs().mean())
            agg.add("edge/residual_abs_mean", residual.abs().mean())
            agg.add("edge/residual_std", residual.float().std(unbiased=False))
            agg.add("edge/semantic_logit_std", prior.float().std(unbiased=False))
            agg.add("edge/residual_scaled_abs_mean", scaled_residual.abs().mean())
            agg.add(
                "edge/residual_to_prior_ratio",
                scaled_residual.abs().mean() / prior.abs().mean().clamp_min(1.0e-8),
            )
            agg.add(
                "edge/residual_to_prior_std_ratio",
                scaled_residual.float().std(unbiased=False)
                / prior.float().std(unbiased=False).clamp_min(1.0e-8),
            )
        agg.add("edge/logit_scale", policy.edge_scorer.logit_scale.detach())
        agg.add("edge/entity_weight", policy.edge_scorer.entity_weight.detach())
        scorer_residual_scale = getattr(
            policy.edge_scorer,
            "residual_scale",
            prior.new_tensor(0.0),
        )
        agg.add("edge/residual_scale", scorer_residual_scale.detach())

        valid = valid_progress_mask(
            batch=batch,
            state=state,
            candidate_edge_ids=out.candidate_edge_ids,
            candidate_batch_ids=out.candidate_batch_ids,
        )
        agg.extend(
            _prefix_rank_values(
                prior, final, out.candidate_batch_ids, valid, int(batch.num_graphs)
            )
        )

        summary = frontier_logit_summary(
            edge_logits=out.edge_logits,
            edge_batch=out.candidate_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        gap = out.expand_logits - out.stop_logits
        agg.add(
            f"gate/frontier_logmeanexp_depth_{depth}", summary.edge_logmeanexp.mean()
        )
        agg.add(f"gate/option_gap_depth_{depth}", gap.mean())
        if depth == 0:
            chosen = _top_edge_per_graph(
                out.candidate_edge_ids,
                out.candidate_batch_ids,
                final,
                int(batch.num_graphs),
            )
            if chosen:
                state.apply_expansion(
                    chosen_edges=torch.tensor(
                        chosen, dtype=torch.long, device=batch.edge_index.device
                    ),
                    edge_index=batch.edge_index,
                )
    return agg.mean()


def _prefix_rank_values(
    prior: torch.Tensor,
    final: torch.Tensor,
    candidate_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
) -> dict[str, float]:
    prior_by_graph = _best_valid_rank_by_graph(
        prior, candidate_batch, valid, num_graphs
    )
    final_by_graph = _best_valid_rank_by_graph(
        final, candidate_batch, valid, num_graphs
    )
    shared_graphs = sorted(set(prior_by_graph).intersection(final_by_graph))
    prior_ranks = [prior_by_graph[graph_id] for graph_id in shared_graphs]
    final_ranks = [final_by_graph[graph_id] for graph_id in shared_graphs]
    deltas = [
        final_by_graph[graph_id] - prior_by_graph[graph_id]
        for graph_id in shared_graphs
    ]
    return {
        "edge/valid_edge_prior_rank_mean": _mean(prior_ranks),
        "edge/valid_edge_final_rank_mean": _mean(final_ranks),
        "edge/valid_edge_rank_delta_mean": _mean(deltas),
        "edge/final_worse_than_prior_rate": _rate(
            float(sum(1 for value in deltas if value > 0.0)),
            float(len(deltas)),
        ),
    }


def _best_valid_rank_by_graph(
    logits: torch.Tensor,
    candidate_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
) -> dict[int, float]:
    ranks: dict[int, float] = {}
    for graph_id in range(num_graphs):
        mask = candidate_batch.eq(graph_id)
        if not bool(mask.any()):
            continue
        graph_valid = valid[mask]
        if not bool(graph_valid.any()):
            continue
        graph_logits = logits[mask]
        best_valid = graph_logits[graph_valid].max()
        rank = 1.0 + float((graph_logits > best_valid).sum().item())
        ranks[int(graph_id)] = rank
    return ranks


def _top_edge_per_graph(
    edge_ids: torch.Tensor,
    batch_ids: torch.Tensor,
    logits: torch.Tensor,
    num_graphs: int,
) -> list[int]:
    chosen: list[int] = []
    for graph_id in range(num_graphs):
        mask = batch_ids.eq(graph_id)
        if not bool(mask.any()):
            continue
        pos = mask.nonzero(as_tuple=False).view(-1)
        best = pos[logits.index_select(0, pos).argmax()]
        chosen.append(int(edge_ids[best].item()))
    return chosen


def stop_tb_metrics(
    *,
    rollouts: list[RolloutBatch],
    max_trajectory_len: int,
) -> dict[str, float]:
    if not rollouts:
        return {}
    rollout = concat_rollout_batches(rollouts)
    out0 = SubTrajectoryBalanceLoss(
        max_trajectory_len=max_trajectory_len, stop_tb_coef=0.0
    )(rollout)
    out5 = SubTrajectoryBalanceLoss(
        max_trajectory_len=max_trajectory_len, stop_tb_coef=0.05
    )(rollout)
    metrics = {
        "stop_tb_coef_0/loss_total": _safe_float(out0.metrics["loss/total"]),
        "stop_tb_coef_0/loss_subtb": _safe_float(out0.metrics["loss/subtb"]),
        "stop_tb_coef_0/loss_stop_tb": _safe_float(out0.metrics["loss/stop_tb"]),
        "stop_tb_coef_0_05/loss_total": _safe_float(out5.metrics["loss/total"]),
        "stop_tb_coef_0_05/loss_subtb": _safe_float(out5.metrics["loss/subtb"]),
        "stop_tb_coef_0_05/loss_stop_tb": _safe_float(out5.metrics["loss/stop_tb"]),
        "stop_tb/residual_abs_mean": _safe_float(
            out5.metrics["stop_tb/residual_abs_mean"]
        ),
        "stop_tb/valid_count_mean": _safe_float(
            out5.metrics["stop_tb/valid_count_mean"]
        ),
    }
    traces = rollout.traces
    hit = traces.stop_now_valid_mask.bool() & traces.stop_now_answer_f1.gt(0.0)
    residual = traces.state_log_flows + traces.stop_log_pf - traces.stop_now_log_reward
    after_hit = hit.cumsum(dim=1).gt(0) & traces.stop_tb_valid_mask.bool()
    metrics["stop_tb/residual_after_hit_abs_mean"] = (
        _safe_float(residual[after_hit].abs().mean()) if bool(after_hit.any()) else 0.0
    )
    metrics["policy/stop_prob_after_hit"] = (
        _safe_float(traces.target_stop_prob[after_hit].float().mean())
        if bool(after_hit.any())
        else 0.0
    )
    before_hit = (~hit.cumsum(dim=1).gt(0)) & traces.stop_tb_valid_mask.bool()
    metrics["policy/stop_prob_before_hit"] = (
        _safe_float(traces.target_stop_prob[before_hit].float().mean())
        if bool(before_hit.any())
        else 0.0
    )
    return metrics


def _resolve_ckpt(cfg: DictConfig) -> str | None:
    for key in ("diagnose_ckpt_path", "ckpt_path", "pretrained_ckpt_path"):
        value = cfg.get(key, None)
        if value not in (None, ""):
            return str(value)
    ckpt_cfg = cfg.get("ckpt", None)
    if ckpt_cfg is not None:
        for key in ("path", "pretrained"):
            value = ckpt_cfg.get(key, None)
            if value not in (None, ""):
                return str(value)
    return None


def _load_checkpoint_for_diagnostics(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> tuple[list[str], list[str]]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" not in message:
            raise
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def _format_table(metrics: dict[str, float], keys: list[str]) -> str:
    lines = ["| metric | value |", "|---|---:|"]
    for key in keys:
        value = metrics.get(key, 0.0)
        lines.append(f"| `{key}` | {_format_metric_value(value)} |")
    return "\n".join(lines)


def _format_metric_value(value: float) -> str:
    value = float(value)
    if value != 0.0 and abs(value) < 1.0e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def write_report(
    *,
    output_path: Path,
    ckpt_path: str | None,
    sample_count: int,
    metrics: dict[str, float],
) -> None:
    one_hop = metrics.get("oracle/target_at_depth_1_rate", 0.0) > 0.6
    prior_topk = metrics.get("prior/root_valid_edge_top10_rate", 0.0) > 0.6
    shallow_hit_continue = (
        metrics.get("policy/hit_at_depth_2_rate", 0.0) > 0.3
        and metrics.get("policy/continue_after_first_hit_rate", 0.0) > 0.3
    )
    minimality_prefers_first = (
        metrics.get("reward/final_minus_first_hit_log_reward_mean", 0.0) < 0.0
    )

    diagnosis: list[str] = [
        f"Validation answers one-hop reachable: {_yes_no(one_hop)}.",
        f"Semantic prior ranks valid progress edges in top-k: {_yes_no(prior_topk)}.",
        f"Current rollout hits shallow then continues: {_yes_no(shallow_hit_continue)}.",
        f"Minimality reward prefers first-hit over final: {_yes_no(minimality_prefers_first)}.",
    ]
    if metrics.get("oracle/reachable_target_rate", 0.0) < 0.5:
        next_change = "Check preprocessing/materialized graph reachability before changing policy."
    elif not prior_topk:
        next_change = "Add reward-aligned teacher warmup for edge selection before more GFlowNet tuning."
    elif shallow_hit_continue and minimality_prefers_first:
        next_change = "Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate."
    elif not minimality_prefers_first:
        next_change = "Fix reward minimality so first-hit sufficient states beat budget-full states."
    else:
        next_change = "Run the same diagnostic after a short teacher-warmup checkpoint to separate representation limits from sparse-training limits."

    lines = [
        "# Weaver Rollout Diagnostics",
        "",
        f"- checkpoint: `{ckpt_path or 'random initialization'}`",
        f"- validation samples: {sample_count}",
        "",
        "## Dataset Reachability",
        _format_table(
            metrics,
            [
                "oracle/has_anchor",
                "oracle/num_targets_mean",
                "oracle/reachable_target_rate",
                "oracle/target_at_depth_0_rate",
                "oracle/target_at_depth_1_rate",
                "oracle/target_at_depth_2_rate",
                "oracle/target_at_depth_3_rate",
                "oracle/no_reachable_target_rate",
                "oracle/undirected_target_at_depth_1_rate",
                "oracle/undirected_target_at_depth_2_rate",
            ],
        ),
        "",
        "## Prior Ranking",
        _format_table(
            metrics,
            [
                "prior/root_valid_edge_exists_rate",
                "prior/root_candidate_count_mean",
                "prior/root_best_valid_rank_mean",
                "prior/root_best_valid_rank_median",
                "prior/root_best_valid_prob_mean",
                "prior/root_valid_edge_prob_mass_mean",
                "prior/root_valid_edge_8sample_hit_rate",
                "prior/root_valid_edge_top1_rate",
                "prior/root_valid_edge_top3_rate",
                "prior/root_valid_edge_top5_rate",
                "prior/root_valid_edge_top10_rate",
                "prior/root_valid_edge_mrr",
                "prior/depth1_valid_edge_exists_rate",
                "prior/depth1_candidate_count_mean",
                "prior/depth1_best_valid_rank_mean",
                "prior/depth1_best_valid_prob_mean",
                "prior/depth1_valid_edge_prob_mass_mean",
                "prior/depth1_valid_edge_8sample_hit_rate",
                "prior/depth1_valid_edge_top10_rate",
            ],
        ),
        "",
        "## Root One-Step Reward Oracle",
        _format_table(
            metrics,
            [
                "oracle/root_stop_log_reward",
                "oracle/root_best_child_log_reward",
                "oracle/root_best_child_minus_stop_log_reward",
                "oracle/root_best_child_support",
                "oracle/root_answer_edge_rank_by_policy",
                "oracle/root_best_child_edge_prob",
                "oracle/root_best_child_8sample_hit_rate",
                "oracle/root_policy_top1_child_support",
                "oracle/root_policy_top5_child_support",
                "oracle/root_candidate_count",
            ],
        ),
        "",
        "## Sampling Sanity",
        _format_table(
            metrics,
            [
                "train/policy/target_stop_prob_mean",
                "train/policy/target_continue_prob_mean",
                "train/rollout/continue_depth_0_rate",
                "train/rollout/continue_rate",
                "train/rollout/budget_exhausted_ratio",
                "train/rollout/stop_depth_hist_0",
                "train/rollout/stop_depth_hist_1",
                "train/rollout/stop_depth_hist_2",
                "train/rollout/stop_depth_hist_3",
            ],
        ),
        "",
        "## Stop Improvement Oracle",
        _format_table(
            metrics,
            [
                "stop_oracle/stop_now_better_ratio",
                "stop_oracle/mean_best_continue_minus_stop_log_reward",
                "stop_oracle/policy_stop_prob_when_stop_better",
                "stop_oracle/policy_continue_prob_when_continue_better",
                "stop_oracle/stop_now_better_ratio_depth_0",
                "stop_oracle/stop_now_better_ratio_depth_1",
                "stop_oracle/stop_now_better_ratio_depth_2",
                "stop_oracle/stop_now_better_ratio_depth_3",
            ],
        ),
        "",
        "## Oracle Path Probability",
        _format_table(
            metrics,
            [
                "oracle_path/nonzero_f1_rate",
                "oracle_path/answer_f1_mean",
                "oracle_path/exact_path_prob_mean",
                "oracle_path/exact_path_prob_when_nonzero_mean",
                "oracle_path/expected_hit_rate_8_mean",
                "oracle_path/expected_hit_rate_8_when_nonzero_mean",
                "oracle_path/selected_edge_count_mean",
                "oracle_path/chosen_edge_rank_mean",
                "oracle_path/chosen_edge_prob_mean",
                "oracle_path/continue_prob_mean",
                "oracle_path/terminal_stop_prob_mean",
                "oracle_path/chosen_edge_prob_depth_0",
                "oracle_path/chosen_edge_prob_depth_1",
                "oracle_path/chosen_edge_prob_depth_2",
                "oracle_path/continue_prob_depth_0",
                "oracle_path/continue_prob_depth_1",
                "oracle_path/continue_prob_depth_2",
            ],
        ),
        "",
        "## Rollout Comparison",
        _format_table(
            metrics,
            [
                "policy/nonzero_f1_rate",
                "policy/answer_f1_mean",
                "policy/utility_mean",
                "policy/log_reward_mean",
                "policy/expanded_edge_count_mean",
                "policy/minimality_gap_mean",
                "policy/budget_exhausted_stop_rate",
                "policy/model_stop_rate",
                "policy/first_hit_depth_mean",
                "policy/hit_at_depth_1_rate",
                "policy/hit_at_depth_2_rate",
                "policy/continue_after_first_hit_rate",
                "policy/extra_edges_after_first_hit_mean",
                "prior_greedy_nonzero_f1_rate",
                "prior_beam4_nonzero_f1_rate",
                "prior_beam8_nonzero_f1_rate",
                "oracle/nonzero_f1_rate",
                "oracle/answer_f1_mean",
                "oracle/expanded_edge_count_mean",
                "oracle/minimality_gap_mean",
            ],
        ),
        "",
        "## Reward Sanity",
        _format_table(
            metrics,
            [
                "reward/first_hit_log_reward_mean",
                "reward/final_log_reward_mean",
                "reward/final_minus_first_hit_log_reward_mean",
                "reward/final_better_than_first_hit_rate",
                "reward/minimality_gap_at_final_mean",
                "reward/minimality_penalty_at_final_mean",
                "reward/expanded_edge_count_at_final_mean",
                "reward/minimal_edge_count_at_final_mean",
            ],
        ),
        "",
        "## StopTB Ablation",
        "This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.",
        _format_table(
            metrics,
            [
                "stop_tb_coef_0/loss_total",
                "stop_tb_coef_0/loss_subtb",
                "stop_tb_coef_0/loss_stop_tb",
                "stop_tb_coef_0_05/loss_total",
                "stop_tb_coef_0_05/loss_subtb",
                "stop_tb_coef_0_05/loss_stop_tb",
                "stop_tb/residual_abs_mean",
                "stop_tb/residual_after_hit_abs_mean",
                "stop_tb/valid_count_mean",
                "policy/stop_prob_after_hit",
                "policy/stop_prob_before_hit",
            ],
        ),
        "",
        "## Edge Prior/Residual",
        _format_table(
            metrics,
            [
                "edge/prior_abs_mean",
                "edge/residual_abs_mean",
                "edge/residual_std",
                "edge/semantic_logit_std",
                "edge/residual_scaled_abs_mean",
                "edge/residual_to_prior_ratio",
                "edge/residual_to_prior_std_ratio",
                "edge/logit_scale",
                "edge/entity_weight",
                "edge/residual_scale",
                "edge/valid_edge_prior_rank_mean",
                "edge/valid_edge_final_rank_mean",
                "edge/valid_edge_rank_delta_mean",
                "edge/final_worse_than_prior_rate",
                "gate/frontier_logmeanexp_depth_0",
                "gate/frontier_logmeanexp_depth_1",
                "gate/option_gap_depth_0",
                "gate/option_gap_depth_1",
                "gate/option_gap_depth_2",
            ],
        ),
        "",
        "## Main Diagnosis",
        "* " + "\n* ".join(diagnosis),
        "",
        "## Recommended Next Change",
        next_change,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(
    version_base=None, config_path="../configs", config_name="diagnose_weaver_rollout"
)
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed), workers=True)
    if bool(cfg.get("print_config", False)):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(datamodule, stage="fit")
    model = build_model(cfg, resources)
    ckpt_path = _resolve_ckpt(cfg)
    if ckpt_path is not None:
        missing, unexpected = _load_checkpoint_for_diagnostics(model, ckpt_path)
        print(
            f"Loaded checkpoint {ckpt_path!r}; missing={len(missing)}, unexpected={len(unexpected)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    split = str(cfg.get("split", "validation"))
    if split == "validation":
        loader = datamodule.val_dataloader()
    elif split == "test":
        loader = datamodule.test_dataloader()
    elif split == "train":
        loader = datamodule.train_dataloader()
    else:
        raise ValueError("split must be one of train, validation, test.")

    limit = int(cfg.get("limit", 128))
    beam_sizes = [int(x) for x in cfg.get("beam_sizes", [4, 8])]
    expand_budget = int(model.expand_budget)
    eval_rollouts = int(
        cfg.get("eval_num_rollout", model.rollout_runner.eval_num_rollout)
    )

    aggs = MeanAgg()
    policy_rollouts: list[RolloutBatch] = []
    sample_count = 0

    for batch in loader:
        batch = _as_device_batch(batch, device)
        batch_size = int(batch.num_graphs)
        if sample_count >= limit:
            break
        sample_count += batch_size

        aggs.extend(reachability_metrics(batch))
        root_state = State.create_initial(batch, expand_budget=expand_budget)
        aggs.extend(
            ranking_metrics_for_state(
                policy=model.policy,
                batch=batch,
                state=root_state,
                reward_model=model.reward_model,
                prefix="prior/root",
            )
        )
        aggs.extend(
            root_one_step_reward_oracle_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
            )
        )
        depth1 = oracle_depth1_state(
            batch=batch,
            reward_model=model.reward_model,
            expand_budget=expand_budget,
        )
        aggs.extend(
            ranking_metrics_for_state(
                policy=model.policy,
                batch=batch,
                state=depth1,
                reward_model=model.reward_model,
                prefix="prior/depth1",
            )
        )
        aggs.extend(
            prior_search_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
                beam_sizes=beam_sizes,
            )
        )
        aggs.extend(
            oracle_rollout_metrics(
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
            )
        )
        aggs.extend(
            oracle_path_probability_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
                num_rollouts=eval_rollouts,
            )
        )
        aggs.extend(
            stop_improvement_oracle_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
            )
        )
        aggs.extend(
            edge_and_gate_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
            )
        )
        batch_rollouts = model.rollout_runner.generate_eval_rollouts(
            policy=model.policy,
            reward_model=model.reward_model,
            batch=batch,
            temperature=float(model.temperature_schedule.eval_temperature),
            num_rollouts=eval_rollouts,
            collect_stop_counterfactual=True,
            collect_policy_diagnostics=True,
            validate_synchronous_depth=False,
        )
        policy_rollouts.extend(batch_rollouts)

    aggs.extend(policy_rollout_metrics(policy_rollouts))
    aggs.extend(reward_sanity_metrics(policy_rollouts))
    aggs.extend(
        stop_tb_metrics(rollouts=policy_rollouts, max_trajectory_len=expand_budget + 1)
    )

    original_cwd = Path(get_original_cwd())
    output_path = Path(str(cfg.get("output_path", "diagnostics_report.md")))
    if not output_path.is_absolute():
        output_path = original_cwd / output_path
    write_report(
        output_path=output_path,
        ckpt_path=ckpt_path,
        sample_count=min(sample_count, limit),
        metrics=aggs.mean(),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
