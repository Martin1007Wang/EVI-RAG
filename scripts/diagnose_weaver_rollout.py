from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.runtime import load_project_env

PROJECT_ROOT = load_project_env(__file__)

import hydra  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from lightning import seed_everything  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data.schema import RetrievalBatch  # noqa: E402
from src.training.factory import build_datamodule, build_model  # noqa: E402
from src.training.resources import setup_datamodule  # noqa: E402
from src.weaver.loss import BudgetedDAGDetailedBalanceLoss  # noqa: E402
from src.weaver.policy import Policy, frontier_logit_summary  # noqa: E402
from src.weaver.reward import (
    RewardModel,
    TerminalRewardOutput,
    target_ids,
)  # noqa: E402
from src.weaver.rollout.executor import (
    budget_exhausted_mask,
    has_frontier,
)  # noqa: E402
from src.weaver.rollout.runner import concat_rollout_batches  # noqa: E402
from src.weaver.rollout.sampling import action_probs  # noqa: E402
from src.weaver.rollout.schema import RolloutBatch  # noqa: E402
from src.weaver.state import State  # noqa: E402

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


def _prob_to_margin(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.detach().float().clamp(1.0e-12, 1.0 - 1.0e-7)
    return prob.log() - (1.0 - prob).log()


def _empty_one_step_reward_oracle_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_stop_log_reward": 0.0,
        f"{prefix}_best_child_log_reward": 0.0,
        f"{prefix}_best_child_minus_stop_log_reward": 0.0,
        f"{prefix}_best_child_support": 0.0,
        f"{prefix}_answer_edge_rank_by_policy": 0.0,
        f"{prefix}_policy_top1_child_support": 0.0,
        f"{prefix}_policy_top5_child_support": 0.0,
        f"{prefix}_frontier_edge_count": 0.0,
    }


def _empty_root_answer_edge_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/root_answer_edge_exists_rate": 0.0,
        f"{prefix}/root_answer_edge_count_mean": 0.0,
        f"{prefix}/root_frontier_edge_count_mean": 0.0,
        f"{prefix}/root_answer_edge_policy_best_rank_mean": 0.0,
        f"{prefix}/root_answer_edge_policy_top1_rate": 0.0,
        f"{prefix}/root_answer_edge_policy_top5_rate": 0.0,
        f"{prefix}/root_answer_edge_policy_top10_rate": 0.0,
        f"{prefix}/root_answer_edge_semantic_best_rank_mean": 0.0,
        f"{prefix}/root_answer_edge_semantic_top1_rate": 0.0,
        f"{prefix}/root_answer_edge_semantic_top5_rate": 0.0,
        f"{prefix}/root_answer_edge_semantic_top10_rate": 0.0,
        f"{prefix}/root_answer_edge_prob_mass": 0.0,
        f"{prefix}/root_answer_edge_sample_rate": 0.0,
    }


def _empty_oracle_1hop_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/oracle_1hop/answer_edge_exists_rate": 0.0,
        f"{prefix}/oracle_1hop/stop_logit": 0.0,
        f"{prefix}/oracle_1hop/continue_logprob": 0.0,
        f"{prefix}/oracle_1hop/stop_margin": 0.0,
        f"{prefix}/oracle_1hop/p_stop_after_answer_edge": 0.0,
        f"{prefix}/oracle_1hop/model_stop_rate_after_answer_edge": 0.0,
        f"{prefix}/oracle_1hop/f1_after_answer_edge": 0.0,
        f"{prefix}/oracle_1hop/support_after_answer_edge": 0.0,
        f"{prefix}/oracle_1hop/log_reward_after_answer_edge": 0.0,
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
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
) -> torch.Tensor:
    device = frontier_edge_ids.device
    valid = torch.zeros(frontier_edge_ids.numel(), dtype=torch.bool, device=device)
    if frontier_edge_ids.numel() == 0:
        return valid

    metas = _graph_metas(batch)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)

    for graph_id in range(int(batch.num_graphs)):
        frontier_pos = (
            (frontier_batch_ids == graph_id).nonzero(as_tuple=False).view(-1)
        )
        if frontier_pos.numel() == 0:
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

        edges = frontier_edge_ids.index_select(0, frontier_pos)
        local_edges = edges - meta.edge_lo
        src = edge_index[0].index_select(0, edges) - meta.node_lo
        dst = edge_index[1].index_select(0, edges) - meta.node_lo
        global_src = src + meta.node_lo
        global_dst = dst + meta.node_lo
        src_active = active_nodes.index_select(0, global_src)
        dst_active = active_nodes.index_select(0, global_dst)
        exactly_one_active = src_active ^ dst_active

        for local_idx, pos in enumerate(frontier_pos.tolist()):
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
        return_edge_diagnostics=True,
    )
    logits = out.edge_logits
    valid = valid_progress_mask(
        batch=batch,
        state=state,
        frontier_edge_ids=out.frontier_edge_ids,
        frontier_batch_ids=out.frontier_batch_ids,
    )
    del reward_model
    return _rank_metrics(
        logits=logits.detach(),
        frontier_batch=out.frontier_batch_ids.detach(),
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
    edge. The rank metric uses the best reward-improving child per graph.
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
        return_edge_diagnostics=True,
    )
    logits = out.edge_logits.detach()
    frontier_edge_ids = out.frontier_edge_ids.detach()
    frontier_batch_ids = out.frontier_batch_ids.detach()

    metrics = _empty_one_step_reward_oracle_metrics(prefix)
    if frontier_edge_ids.numel() == 0:
        return metrics
    child_rewards = _evaluate_child_rewards(
        batch=batch,
        state=state,
        reward_model=reward_model,
        frontier_edge_ids=frontier_edge_ids,
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
    frontier_edge_counts: list[float] = []

    for graph_id in range(int(batch.num_graphs)):
        mask = frontier_batch_ids.eq(graph_id)
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
        frontier_edge_counts.append(float(pos.numel()))

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
            f"{prefix}_frontier_edge_count": _mean(frontier_edge_counts),
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
    frontier_edge_ids: torch.Tensor,
) -> ChildRewardBatch:
    frontier_edge_ids = frontier_edge_ids.to(
        device=batch.edge_index.device, dtype=torch.long
    ).view(-1)
    if frontier_edge_ids.numel() == 0:
        empty = frontier_edge_ids.new_empty((0,), dtype=torch.float32)
        return ChildRewardBatch(log_reward=empty, answer_support=empty)

    edge_batch = batch.edge_batch.to(device=batch.edge_index.device, dtype=torch.long)
    log_reward = torch.empty(
        int(frontier_edge_ids.numel()),
        dtype=torch.float32,
        device=batch.edge_index.device,
    )
    child_support = torch.empty_like(log_reward)
    for idx, edge_id_tensor in enumerate(frontier_edge_ids):
        child_state = state.detach()
        child_state.apply_expansion(
            chosen_edges=edge_id_tensor.view(1),
            edge_index=batch.edge_index,
        )
        child = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            active_nodes=child_state.active_nodes,
            active_edges=child_state.active_edges,
            state=child_state,
            diagnostics="basic",
        )
        graph_id = int(edge_batch[int(edge_id_tensor.item())].item())
        log_reward[idx] = child.log_reward[graph_id].to(dtype=torch.float32)
        child_support[idx] = child.supported_answer_recall[graph_id].to(dtype=torch.float32)

    return ChildRewardBatch(
        log_reward=log_reward.to(dtype=torch.float32),
        answer_support=child_support.to(dtype=torch.float32),
    )


@torch.no_grad()
def root_answer_edge_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    policy_rollouts: list[RolloutBatch],
    expand_budget: int,
    prefix: str = "val",
) -> dict[str, float]:
    state = State.create_initial(batch, expand_budget=expand_budget)
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
        return_edge_diagnostics=True,
    )
    frontier_edge_ids = out.frontier_edge_ids.detach()
    frontier_batch_ids = out.frontier_batch_ids.detach()
    metrics = _empty_root_answer_edge_metrics(prefix)
    if frontier_edge_ids.numel() == 0:
        return metrics

    child_rewards = _evaluate_child_rewards(
        batch=batch,
        state=state,
        reward_model=reward_model,
        frontier_edge_ids=frontier_edge_ids,
    )
    answer_edge = child_rewards.answer_support.gt(0.0)
    edge_policy_logits = out.edge_logits.detach()
    edge_policy_diagnostics = out.edge_policy_diagnostics
    semantic_scores = (
        edge_policy_diagnostics.semantic_score.detach()
        if edge_policy_diagnostics is not None
        else edge_policy_logits.detach()
    )
    _, _, edge_prob = action_probs(
        stop_logits=out.stop_logits.detach(),
        edge_logits=edge_policy_logits,
        frontier_batch_ids=frontier_batch_ids,
        can_expand=has_frontier(
            frontier_batch_ids=frontier_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        ),
        batch_size=int(batch.num_graphs),
    )

    exists = 0
    answer_graphs: set[int] = set()
    answer_counts: list[float] = []
    frontier_edge_counts: list[float] = []
    policy_ranks: list[float] = []
    semantic_ranks: list[float] = []
    masses: list[float] = []

    for graph_id in range(int(batch.num_graphs)):
        graph_mask = frontier_batch_ids.eq(graph_id)
        if not bool(graph_mask.any()):
            continue
        frontier_edge_counts.append(float(graph_mask.sum().item()))
        graph_answer = answer_edge & graph_mask
        if not bool(graph_answer.any()):
            continue
        exists += 1
        answer_graphs.add(int(graph_id))
        answer_counts.append(float(graph_answer.sum().item()))
        policy_ranks.append(
            _best_mask_rank(
                logits=edge_policy_logits[graph_mask],
                mask=answer_edge[graph_mask],
            )
        )
        semantic_ranks.append(
            _best_mask_rank(
                logits=semantic_scores[graph_mask],
                mask=answer_edge[graph_mask],
            )
        )
        masses.append(float(edge_prob[graph_answer].sum().item()))

    selected_hits = 0.0
    selected_total = 0.0
    answer_edge_ids = set(int(x) for x in frontier_edge_ids[answer_edge].tolist())
    for rollout in policy_rollouts:
        if rollout.traces.selected_edge_ids.numel() == 0:
            continue
        selected = rollout.traces.selected_edge_ids[:, 0].detach().cpu().tolist()
        continued = rollout.traces.continue_mask[:, 0].detach().bool().cpu().tolist()
        for graph_id, (edge_id, did_continue) in enumerate(zip(selected, continued)):
            if int(graph_id) not in answer_graphs:
                continue
            selected_total += 1.0
            selected_hits += float(bool(did_continue) and int(edge_id) in answer_edge_ids)

    graph_count = float(int(batch.num_graphs))
    metrics.update(
        {
            f"{prefix}/root_answer_edge_exists_rate": _rate(
                float(exists), graph_count
            ),
            f"{prefix}/root_answer_edge_count_mean": _mean(answer_counts),
            f"{prefix}/root_frontier_edge_count_mean": _mean(frontier_edge_counts),
            f"{prefix}/root_answer_edge_policy_best_rank_mean": _mean(policy_ranks),
            f"{prefix}/root_answer_edge_policy_top1_rate": _rank_rate(
                policy_ranks, 1
            ),
            f"{prefix}/root_answer_edge_policy_top5_rate": _rank_rate(
                policy_ranks, 5
            ),
            f"{prefix}/root_answer_edge_policy_top10_rate": _rank_rate(
                policy_ranks, 10
            ),
            f"{prefix}/root_answer_edge_semantic_best_rank_mean": _mean(semantic_ranks),
            f"{prefix}/root_answer_edge_semantic_top1_rate": _rank_rate(
                semantic_ranks, 1
            ),
            f"{prefix}/root_answer_edge_semantic_top5_rate": _rank_rate(
                semantic_ranks, 5
            ),
            f"{prefix}/root_answer_edge_semantic_top10_rate": _rank_rate(
                semantic_ranks, 10
            ),
            f"{prefix}/root_answer_edge_prob_mass": _mean(masses),
            f"{prefix}/root_answer_edge_sample_rate": _rate(
                selected_hits, selected_total
            ),
        }
    )
    return metrics


def _best_mask_rank(*, logits: torch.Tensor, mask: torch.Tensor) -> float:
    if logits.numel() == 0 or not bool(mask.any()):
        return 0.0
    best = logits[mask].max()
    return 1.0 + float((logits > best).sum().item())


def _rank_rate(ranks: list[float], k: int) -> float:
    return _rate(float(sum(rank <= float(k) for rank in ranks)), float(len(ranks)))


@torch.no_grad()
def oracle_1hop_answer_stop_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    prefix: str = "val",
) -> dict[str, float]:
    state = State.create_initial(batch, expand_budget=expand_budget)
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
        return_edge_diagnostics=True,
    )
    frontier_edge_ids = out.frontier_edge_ids.detach()
    frontier_batch_ids = out.frontier_batch_ids.detach()
    metrics = _empty_oracle_1hop_metrics(prefix)
    if frontier_edge_ids.numel() == 0:
        return metrics

    child_rewards = _evaluate_child_rewards(
        batch=batch,
        state=state,
        reward_model=reward_model,
        frontier_edge_ids=frontier_edge_ids,
    )
    answer_edge = child_rewards.answer_support.gt(0.0)
    chosen_edges_by_graph: dict[int, int] = {}
    exists = 0
    for graph_id in range(int(batch.num_graphs)):
        graph_answer = frontier_batch_ids.eq(graph_id) & answer_edge
        if not bool(graph_answer.any()):
            continue
        exists += 1
        pos = graph_answer.nonzero(as_tuple=False).view(-1)
        best = pos[child_rewards.answer_support.index_select(0, pos).argmax()]
        chosen_edges_by_graph[int(graph_id)] = int(frontier_edge_ids[best].item())

    metrics[f"{prefix}/oracle_1hop/answer_edge_exists_rate"] = _rate(
        float(exists), float(int(batch.num_graphs))
    )
    if not chosen_edges_by_graph:
        return metrics

    oracle_state = state.detach()
    oracle_state.apply_expansion(
        chosen_edges=torch.tensor(
            list(chosen_edges_by_graph.values()),
            dtype=torch.long,
            device=batch.edge_index.device,
        ),
        edge_index=batch.edge_index,
    )
    oracle_reward = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=oracle_state.active_nodes,
        active_edges=oracle_state.active_edges,
        state=oracle_state,
    )
    oracle_out = policy(
        batch,
        oracle_state,
        rollout_context=context,
        return_edge_diagnostics=False,
    )
    remaining_budget = oracle_state.remaining_budget_per_graph(
        edge_batch=batch.edge_batch,
        num_graphs=int(batch.num_graphs),
    )
    can_expand = (
        has_frontier(
            frontier_batch_ids=oracle_out.frontier_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        & ~budget_exhausted_mask(
            remaining_budget,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
    )
    stop_prob, continue_prob, _ = action_probs(
        stop_logits=oracle_out.stop_logits,
        edge_logits=oracle_out.edge_logits,
        frontier_batch_ids=oracle_out.frontier_batch_ids,
        can_expand=can_expand,
        batch_size=int(batch.num_graphs),
    )

    graph_ids = torch.tensor(
        sorted(chosen_edges_by_graph),
        dtype=torch.long,
        device=batch.edge_index.device,
    )
    oracle_stop_logits = oracle_out.stop_logits.index_select(0, graph_ids).float()
    oracle_continue_logprob = oracle_out.log_p_continue.index_select(
        0, graph_ids
    ).float()
    oracle_margin = stop_prob.index_select(0, graph_ids).log() - oracle_continue_logprob
    metrics.update(
        {
            f"{prefix}/oracle_1hop/stop_logit": _safe_float(
                oracle_stop_logits.mean()
            ),
            f"{prefix}/oracle_1hop/continue_logprob": _safe_float(
                oracle_continue_logprob.mean()
            ),
            f"{prefix}/oracle_1hop/stop_margin": _safe_float(
                oracle_margin.mean()
            ),
            f"{prefix}/oracle_1hop/p_stop_after_answer_edge": _safe_float(
                stop_prob.index_select(0, graph_ids).mean()
            ),
            f"{prefix}/oracle_1hop/model_stop_rate_after_answer_edge": _safe_float(
                stop_prob.index_select(0, graph_ids)
                .ge(continue_prob.index_select(0, graph_ids))
                .float()
                .mean()
            ),
            f"{prefix}/oracle_1hop/f1_after_answer_edge": _safe_float(
                oracle_reward.answer_f1.index_select(0, graph_ids).float().mean()
            ),
            f"{prefix}/oracle_1hop/support_after_answer_edge": _safe_float(
                oracle_reward.supported_answer_recall.index_select(0, graph_ids)
                .float()
                .mean()
            ),
            f"{prefix}/oracle_1hop/log_reward_after_answer_edge": _safe_float(
                oracle_reward.log_reward.index_select(0, graph_ids).float().mean()
            ),
        }
    )
    return metrics


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
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    min_gain: float = 1.0e-8,
) -> torch.Tensor:
    device = frontier_edge_ids.device
    valid = torch.zeros(frontier_edge_ids.numel(), dtype=torch.bool, device=device)
    if frontier_edge_ids.numel() == 0:
        return valid

    current = reward_model.evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    for pos in range(int(frontier_edge_ids.numel())):
        graph_id = int(frontier_batch_ids[pos].item())
        edge_id = frontier_edge_ids[pos].view(1)
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
    frontier_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
    prefix: str,
) -> dict[str, float]:
    exists = 0
    ranks: list[float] = []
    best_valid_probs: list[float] = []
    valid_prob_masses: list[float] = []
    valid_8sample_hits: list[float] = []
    frontier_edge_counts: list[float] = []
    for graph_id in range(num_graphs):
        mask = frontier_batch.eq(graph_id)
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
        frontier_edge_counts.append(float(graph_logits.numel()))

    denom = float(num_graphs)
    return {
        f"{prefix}_valid_edge_exists_rate": _rate(float(exists), denom),
        f"{prefix}_frontier_edge_count_mean": _mean(frontier_edge_counts),
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
def depth_hit_stop_metrics(
    rollouts: list[RolloutBatch],
    *,
    prefix: str = "val",
) -> dict[str, float]:
    if not rollouts:
        return {}
    rollout = concat_rollout_batches(rollouts)
    traces = rollout.traces
    valid = traces.stop_now_valid_mask.bool()
    f1_now = traces.stop_now_answer_f1.float()
    hit_now = valid & f1_now.gt(0.0)
    miss_now = valid & ~hit_now
    policy_valid = traces.policy_action_valid_mask.bool()
    stop_prob = traces.target_stop_prob.float()
    stop_margin = _prob_to_margin(stop_prob)
    continue_mask = traces.continue_mask.bool()
    horizon = int(f1_now.size(1))

    metrics: dict[str, float] = {}
    for depth in range(horizon):
        valid_depth = valid[:, depth]
        hit_depth = hit_now[:, depth]
        miss_depth = miss_now[:, depth]
        policy_valid_depth = policy_valid[:, depth]
        hit_policy = hit_depth & policy_valid_depth
        miss_policy = miss_depth & policy_valid_depth

        metrics[f"{prefix}/depth_{depth}/hit_now_rate"] = _rate(
            float(hit_depth.sum().item()),
            float(valid_depth.sum().item()),
        )
        metrics[f"{prefix}/depth_{depth}/f1_now_mean"] = (
            _safe_float(f1_now[valid_depth, depth].mean())
            if bool(valid_depth.any())
            else 0.0
        )
        metrics[f"{prefix}/depth_{depth}/p_stop_when_hit"] = (
            _safe_float(stop_prob[hit_policy, depth].mean())
            if bool(hit_policy.any())
            else 0.0
        )
        metrics[f"{prefix}/depth_{depth}/p_stop_when_miss"] = (
            _safe_float(stop_prob[miss_policy, depth].mean())
            if bool(miss_policy.any())
            else 0.0
        )
        metrics[f"{prefix}/depth_{depth}/stop_margin_when_hit"] = (
            _safe_float(stop_margin[hit_policy, depth].mean())
            if bool(hit_policy.any())
            else 0.0
        )
        metrics[f"{prefix}/depth_{depth}/stop_margin_when_miss"] = (
            _safe_float(stop_margin[miss_policy, depth].mean())
            if bool(miss_policy.any())
            else 0.0
        )
        metrics[f"{prefix}/depth_{depth}/continue_after_hit_rate"] = (
            _safe_float(continue_mask[hit_policy, depth].float().mean())
            if bool(hit_policy.any())
            else 0.0
        )
    metrics[f"{prefix}/depth_1_hit/stop_margin"] = metrics.get(
        f"{prefix}/depth_1/stop_margin_when_hit", 0.0
    )
    metrics[f"{prefix}/depth_2_hit/stop_margin"] = metrics.get(
        f"{prefix}/depth_2/stop_margin_when_hit", 0.0
    )
    metrics[f"{prefix}/depth_1_miss/stop_margin"] = metrics.get(
        f"{prefix}/depth_1/stop_margin_when_miss", 0.0
    )
    metrics[f"{prefix}/depth_2_miss/stop_margin"] = metrics.get(
        f"{prefix}/depth_2/stop_margin_when_miss", 0.0
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
            return_edge_diagnostics=True,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=int(batch.num_graphs),
        )
        has_edge = has_frontier(
            frontier_batch_ids=out.frontier_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        exhausted = budget_exhausted_mask(
            remaining_budget,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        can_expand = active & has_edge & ~exhausted
        stop_prob, continue_prob, _ = action_probs(
            stop_logits=out.stop_logits,
            edge_logits=out.edge_logits,
            frontier_batch_ids=out.frontier_batch_ids,
            can_expand=can_expand,
            batch_size=int(batch.num_graphs),
        )

        child_rewards = _evaluate_child_rewards(
            batch=batch,
            state=state,
            reward_model=reward_model,
            frontier_edge_ids=out.frontier_edge_ids,
        )

        best_gains = torch.zeros(
            int(batch.num_graphs), dtype=torch.float32, device=batch.edge_index.device
        )
        best_edges: list[int] = []
        valid_graphs = active & can_expand
        for graph_id in valid_graphs.nonzero(as_tuple=False).view(-1).tolist():
            graph_id = int(graph_id)
            mask = out.frontier_batch_ids.eq(graph_id)
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
                best_edges.append(int(out.frontier_edge_ids[best_pos].item()))

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
def policy_search_metrics(
    *,
    policy: Policy,
    batch: RetrievalBatch,
    reward_model: RewardModel,
    expand_budget: int,
    beam_sizes: list[int],
) -> dict[str, float]:
    greedy_rewards = _policy_beam_rewards(
        policy=policy,
        batch=batch,
        reward_model=reward_model,
        expand_budget=expand_budget,
        beam_size=1,
    )
    metrics = {
        "policy_greedy_nonzero_f1_rate": _safe_float(
            greedy_rewards.answer_f1.gt(0.0).float().mean()
        )
    }
    for beam_size in beam_sizes:
        reward = _policy_beam_rewards(
            policy=policy,
            batch=batch,
            reward_model=reward_model,
            expand_budget=expand_budget,
            beam_size=int(beam_size),
        )
        metrics[f"policy_beam{int(beam_size)}_nonzero_f1_rate"] = _safe_float(
            reward.answer_f1.gt(0.0).float().mean()
        )
    return metrics


@torch.no_grad()
def _policy_beam_rewards(
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
                    return_edge_diagnostics=True,
                )
                mask = out.frontier_batch_ids.eq(graph_id)
                if not bool(mask.any()):
                    continue
                pos = mask.nonzero(as_tuple=False).view(-1)
                vals = out.edge_logits.index_select(0, pos)
                top_k = min(int(beam_size), int(vals.numel()))
                _, order = torch.topk(vals, k=top_k)
                for idx in order.tolist():
                    edge = int(out.frontier_edge_ids[pos[idx]].item())
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
            return_edge_diagnostics=True,
        )
        edge_policy_diagnostics = out.edge_policy_diagnostics
        final = edge_policy_diagnostics.final_logits
        semantic = edge_policy_diagnostics.semantic_score.detach()
        if final.numel() > 0:
            agg.add("edge/policy_logit_abs_mean", final.abs().mean())
            agg.add("edge/policy_logit_std", final.float().std(unbiased=False))
            agg.add("edge/semantic_score_abs_mean", semantic.abs().mean())
            agg.add("edge/semantic_score_std", semantic.float().std(unbiased=False))

        valid = valid_progress_mask(
            batch=batch,
            state=state,
            frontier_edge_ids=out.frontier_edge_ids,
            frontier_batch_ids=out.frontier_batch_ids,
        )
        agg.extend(
            _prefix_rank_values(
                semantic, final, out.frontier_batch_ids, valid, int(batch.num_graphs)
            )
        )

        summary = frontier_logit_summary(
            edge_logits=out.edge_logits,
            edge_batch=out.frontier_batch_ids,
            num_graphs=int(batch.num_graphs),
            device=batch.edge_index.device,
        )
        gap = out.log_p_continue - out.log_p_stop
        agg.add(
            f"gate/frontier_logmeanexp_depth_{depth}", summary.edge_logmeanexp.mean()
        )
        agg.add(f"gate/option_gap_depth_{depth}", gap.mean())
        if depth == 0:
            chosen = _top_edge_per_graph(
                out.frontier_edge_ids,
                out.frontier_batch_ids,
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
    semantic: torch.Tensor,
    final: torch.Tensor,
    frontier_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
) -> dict[str, float]:
    semantic_by_graph = _best_valid_rank_by_graph(
        semantic, frontier_batch, valid, num_graphs
    )
    final_by_graph = _best_valid_rank_by_graph(
        final, frontier_batch, valid, num_graphs
    )
    shared_graphs = sorted(set(semantic_by_graph).intersection(final_by_graph))
    semantic_ranks = [semantic_by_graph[graph_id] for graph_id in shared_graphs]
    final_ranks = [final_by_graph[graph_id] for graph_id in shared_graphs]
    deltas = [
        final_by_graph[graph_id] - semantic_by_graph[graph_id]
        for graph_id in shared_graphs
    ]
    return {
        "edge/valid_edge_semantic_rank_mean": _mean(semantic_ranks),
        "edge/valid_edge_final_rank_mean": _mean(final_ranks),
        "edge/valid_edge_final_minus_semantic_rank_mean": _mean(deltas),
        "edge/final_worse_than_semantic_rate": _rate(
            float(sum(1 for value in deltas if value > 0.0)),
            float(len(deltas)),
        ),
    }


def _best_valid_rank_by_graph(
    logits: torch.Tensor,
    frontier_batch: torch.Tensor,
    valid: torch.Tensor,
    num_graphs: int,
) -> dict[int, float]:
    ranks: dict[int, float] = {}
    for graph_id in range(num_graphs):
        mask = frontier_batch.eq(graph_id)
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


def bdb_metrics(
    *,
    rollouts: list[RolloutBatch],
) -> dict[str, float]:
    if not rollouts:
        return {}
    rollout = concat_rollout_batches(rollouts)
    out = BudgetedDAGDetailedBalanceLoss()(rollout)
    return {
        "bdb/loss_total": _safe_float(out.metrics["loss/total"]),
        "bdb/loss_delta0_mean": _safe_float(out.metrics["bdb/loss_delta0_mean"]),
        "bdb/loss_edge_residual_mean": _safe_float(
            out.metrics["bdb/loss_edge_residual_mean"]
        ),
        "bdb/loss_forced_terminal_mean": _safe_float(
            out.metrics["bdb/loss_forced_terminal_mean"]
        ),
        "flow/log_flow_mean": _safe_float(out.metrics["flow/log_flow_mean"]),
    }
    # REMOVED: trajectory-level SubTB diagnostics — see methodology.md §3.9


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


def _model_expand_budget(model: torch.nn.Module) -> int:
    value = getattr(model, "expand_budget", None)
    if value is not None:
        return int(value)

    rollout_runner = getattr(model, "rollout_runner", None)
    engine = getattr(rollout_runner, "engine", None)
    value = getattr(engine, "expand_budget", None)
    if value is not None:
        return int(value)

    raise AttributeError(
        "Cannot resolve expand_budget from model.expand_budget or "
        "model.rollout_runner.engine.expand_budget."
    )


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


def _bool_cfg(cfg: DictConfig, key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    return bool(value)


def _progress_enabled(cfg: DictConfig) -> bool:
    return _bool_cfg(cfg, "progress", True)


def write_report(
    *,
    output_path: Path,
    ckpt_path: str | None,
    sample_count: int,
    metrics: dict[str, float],
) -> None:
    one_hop = metrics.get("oracle/target_at_depth_1_rate", 0.0) > 0.6
    edge_topk = metrics.get("edge/root_valid_edge_top10_rate", 0.0) > 0.6
    shallow_hit_continue = (
        metrics.get("policy/hit_at_depth_2_rate", 0.0) > 0.3
        and metrics.get("policy/continue_after_first_hit_rate", 0.0) > 0.3
    )
    minimality_prefers_first = (
        metrics.get("reward/final_minus_first_hit_log_reward_mean", 0.0) < 0.0
    )

    diagnosis: list[str] = [
        f"Validation answers one-hop reachable: {_yes_no(one_hop)}.",
        f"Learned edge policy ranks valid progress edges in top-k: {_yes_no(edge_topk)}.",
        f"Current rollout hits shallow then continues: {_yes_no(shallow_hit_continue)}.",
        f"Minimality reward prefers first-hit over final: {_yes_no(minimality_prefers_first)}.",
    ]
    if metrics.get("oracle/reachable_target_rate", 0.0) < 0.5:
        next_change = "Check preprocessing/materialized graph reachability before changing policy."
    elif not edge_topk:
        next_change = "Inspect learned edge-head features and frontier labels before more GFlowNet tuning."
    elif shallow_hit_continue and minimality_prefers_first:
        next_change = "Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate."
    elif not minimality_prefers_first:
        next_change = "Fix reward minimality so first-hit sufficient states beat budget-full states."
    else:
        next_change = "Run the same diagnostic after a short core-training checkpoint to separate representation limits from sparse-training limits."

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
        "## Root Edge Policy Ranking",
        _format_table(
            metrics,
            [
                "edge/root_valid_edge_exists_rate",
                "edge/root_frontier_edge_count_mean",
                "edge/root_best_valid_rank_mean",
                "edge/root_best_valid_rank_median",
                "edge/root_best_valid_prob_mean",
                "edge/root_valid_edge_prob_mass_mean",
                "edge/root_valid_edge_8sample_hit_rate",
                "edge/root_valid_edge_top1_rate",
                "edge/root_valid_edge_top3_rate",
                "edge/root_valid_edge_top5_rate",
                "edge/root_valid_edge_top10_rate",
                "edge/root_valid_edge_mrr",
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
                "oracle/root_frontier_edge_count",
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
        "## Depth Hit vs Stop",
        _format_table(
            metrics,
            [
                "val/depth_0/hit_now_rate",
                "val/depth_0/f1_now_mean",
                "val/depth_0/p_stop_when_hit",
                "val/depth_0/p_stop_when_miss",
                "val/depth_0/stop_margin_when_hit",
                "val/depth_0/stop_margin_when_miss",
                "val/depth_0/continue_after_hit_rate",
                "val/depth_1/hit_now_rate",
                "val/depth_1/f1_now_mean",
                "val/depth_1/p_stop_when_hit",
                "val/depth_1/p_stop_when_miss",
                "val/depth_1/stop_margin_when_hit",
                "val/depth_1/stop_margin_when_miss",
                "val/depth_1/continue_after_hit_rate",
                "val/depth_2/hit_now_rate",
                "val/depth_2/f1_now_mean",
                "val/depth_2/p_stop_when_hit",
                "val/depth_2/p_stop_when_miss",
                "val/depth_2/stop_margin_when_hit",
                "val/depth_2/stop_margin_when_miss",
                "val/depth_2/continue_after_hit_rate",
                "val/depth_3/hit_now_rate",
                "val/depth_3/f1_now_mean",
                "val/depth_3/p_stop_when_hit",
                "val/depth_3/p_stop_when_miss",
                "val/depth_3/stop_margin_when_hit",
                "val/depth_3/stop_margin_when_miss",
                "val/depth_3/continue_after_hit_rate",
            ],
        ),
        "",
        "## Root Answer Edge",
        _format_table(
            metrics,
            [
                "val/root_answer_edge_exists_rate",
                "val/root_answer_edge_count_mean",
                "val/root_frontier_edge_count_mean",
                "val/root_answer_edge_policy_best_rank_mean",
                "val/root_answer_edge_policy_top1_rate",
                "val/root_answer_edge_policy_top5_rate",
                "val/root_answer_edge_policy_top10_rate",
                "val/root_answer_edge_semantic_best_rank_mean",
                "val/root_answer_edge_semantic_top1_rate",
                "val/root_answer_edge_semantic_top5_rate",
                "val/root_answer_edge_semantic_top10_rate",
                "val/root_answer_edge_prob_mass",
                "val/root_answer_edge_sample_rate",
            ],
        ),
        "",
        "## Oracle 1-Hop Stop",
        _format_table(
            metrics,
            [
                "val/oracle_1hop/answer_edge_exists_rate",
                "val/oracle_1hop/stop_logit",
                "val/oracle_1hop/continue_logprob",
                "val/oracle_1hop/stop_margin",
                "val/oracle_1hop/p_stop_after_answer_edge",
                "val/oracle_1hop/model_stop_rate_after_answer_edge",
                "val/oracle_1hop/f1_after_answer_edge",
                "val/oracle_1hop/support_after_answer_edge",
                "val/oracle_1hop/log_reward_after_answer_edge",
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
                "policy_greedy_nonzero_f1_rate",
                "policy_beam4_nonzero_f1_rate",
                "policy_beam8_nonzero_f1_rate",
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
        "## SubTB Balance",
        "This pass evaluates full subtrajectory balance on sampled trajectories.",
        _format_table(
            metrics,
            [
                "subtb/loss_total",
                "subtb/residual_abs_mean",
                "subtb/residual_square_mean",
                "subtb/subtrajectory_count_mean",
                "subtb/trajectory_length_mean",
                "subtb/log_pf_expand_sum_mean",
                "subtb/log_pb_sum_mean",
                "subtb/log_p_stop_terminal_mean",
                "subtb/terminal_log_reward_mean",
                "flow/state_log_flow_mean",
                "flow/state_log_flow_std",
            ],
        ),
        "",
        "## Edge Policy/Semantic Diagnostics",
        _format_table(
            metrics,
            [
                "edge/policy_logit_abs_mean",
                "edge/policy_logit_std",
                "edge/semantic_score_abs_mean",
                "edge/semantic_score_std",
                "edge/valid_edge_semantic_rank_mean",
                "edge/valid_edge_final_rank_mean",
                "edge/valid_edge_final_minus_semantic_rank_mean",
                "edge/final_worse_than_semantic_rate",
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
    expand_budget = _model_expand_budget(model)
    eval_rollouts = int(
        cfg.get("eval_num_rollout", model.rollout_runner.eval_num_rollout)
    )
    minimal_diagnostics = _bool_cfg(cfg, "minimal_diagnostics", False)
    progress = _progress_enabled(cfg)
    total_batches = (limit + max(1, int(getattr(loader, "batch_size", 1))) - 1) // max(
        1, int(getattr(loader, "batch_size", 1))
    )

    aggs = MeanAgg()
    policy_rollouts: list[RolloutBatch] = []
    sample_count = 0

    batch_iter = tqdm(
        loader,
        desc="diagnose validation",
        total=total_batches if total_batches > 0 else None,
        disable=not progress,
    )
    for batch in batch_iter:
        batch = _as_device_batch(batch, device)
        batch_size = int(batch.num_graphs)
        if sample_count >= limit:
            break
        sample_count += batch_size
        batch_iter.set_postfix(samples=min(sample_count, limit))

        def phase(name: str) -> None:
            if progress:
                batch_iter.set_description(f"diagnose {name}")

        phase("reachability")
        aggs.extend(reachability_metrics(batch))
        if not minimal_diagnostics:
            root_state = State.create_initial(batch, expand_budget=expand_budget)
            phase("edge-root")
            aggs.extend(
                ranking_metrics_for_state(
                    policy=model.policy,
                    batch=batch,
                    state=root_state,
                    reward_model=model.reward_model,
                    prefix="edge/root",
                )
            )
            phase("root-oracle")
            aggs.extend(
                root_one_step_reward_oracle_metrics(
                    policy=model.policy,
                    batch=batch,
                    reward_model=model.reward_model,
                    expand_budget=expand_budget,
                )
            )
            phase("policy-search")
            aggs.extend(
                policy_search_metrics(
                    policy=model.policy,
                    batch=batch,
                    reward_model=model.reward_model,
                    expand_budget=expand_budget,
                    beam_sizes=beam_sizes,
                )
            )
            phase("stop-oracle")
            aggs.extend(
                stop_improvement_oracle_metrics(
                    policy=model.policy,
                    batch=batch,
                    reward_model=model.reward_model,
                    expand_budget=expand_budget,
                )
            )
            phase("edge-gate")
            aggs.extend(
                edge_and_gate_metrics(
                    policy=model.policy,
                    batch=batch,
                    reward_model=model.reward_model,
                    expand_budget=expand_budget,
                )
            )
        phase("rollouts")
        batch_rollouts = model.rollout_runner.generate_eval_rollouts(
            policy=model.policy,
            reward_model=model.reward_model,
            batch=batch,
            temperature=float(model.temperature_schedule.eval_temperature),
            num_rollouts=eval_rollouts,
            collect_policy_diagnostics=True,
            validate_synchronous_depth=False,
        )
        phase("depth-hit-stop")
        aggs.extend(depth_hit_stop_metrics(batch_rollouts, prefix="val"))
        phase("root-answer-edge")
        aggs.extend(
            root_answer_edge_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                policy_rollouts=batch_rollouts,
                expand_budget=expand_budget,
                prefix="val",
            )
        )
        phase("oracle-1hop")
        aggs.extend(
            oracle_1hop_answer_stop_metrics(
                policy=model.policy,
                batch=batch,
                reward_model=model.reward_model,
                expand_budget=expand_budget,
                prefix="val",
            )
        )
        policy_rollouts.extend(batch_rollouts)

    if progress:
        batch_iter.set_description("diagnose aggregate")
    aggs.extend(policy_rollout_metrics(policy_rollouts))
    if not minimal_diagnostics:
        aggs.extend(reward_sanity_metrics(policy_rollouts))
        aggs.extend(
            bdb_metrics(rollouts=policy_rollouts)
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
