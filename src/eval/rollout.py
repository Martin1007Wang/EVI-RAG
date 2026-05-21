from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.eval.compactness import per_graph_counts
from src.eval.retrieval import compute_expected_node_retrieval_quality
from src.eval.retrieval import mean_over_valid_graphs, safe_divide, safe_f1
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask
from src.utils.scatter import scatter_sum
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor
from src.weaver.utility import TrueTerminalReward


@dataclass(frozen=True, slots=True)
class ReachableRecallScores:
    recall: torch.Tensor
    valid_graph_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RolloutEvalTensors:
    node_masks: torch.Tensor
    edge_masks: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    edge_count: torch.Tensor
    node_count: torch.Tensor
    trajectory_len: torch.Tensor
    policy_stop: torch.Tensor
    forced_stop: torch.Tensor
    model_score: torch.Tensor
    log_reward: torch.Tensor
    valid_graph_mask: torch.Tensor


def evaluate_rollout_samples(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    k_windows: Sequence[int] | None = None,
    best_of_k: int | None = None,
    context: GraphContext | None = None,
    features: EncodedFeatures | None = None,
    reward_model: TrueTerminalReward | None = None,
    target_context: TargetContext | None = None,
) -> dict[str, float]:
    ks = normalize_k_windows(
        k_windows if k_windows is not None else _legacy_windows(best_of_k),
        max_k=len(rollout_samples),
    )
    tensors = rollout_eval_tensors(
        rollout_samples=rollout_samples,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
        reward_model=reward_model,
        target_context=target_context,
    )

    metrics: dict[str, float] = {}
    metrics.update(sample_metrics(tensors))
    metrics.update(best_metrics(tensors, ks=ks, prefix="oracle_best", oracle=True))
    metrics.update(best_metrics(tensors, ks=ks, prefix="best", oracle=False))
    metrics.update(
        union_metrics(
            tensors,
            batch=batch,
            ks=ks,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
    )
    metrics.update(summary_metrics(metrics, ks=ks))
    metrics.update(stopped_reward_metrics(tensors))
    metrics.update(diversity_metrics(rollout_samples, tensors, batch=batch, ks=ks))
    metrics.update(stop_prefix_metrics(rollout_samples, batch=batch, ks=ks, valid_graph_mask=tensors.valid_graph_mask))
    metrics.update(
        reward_delta_metrics(
            rollout_samples,
            batch=batch,
            reward_model=reward_model,
            target_context=target_context,
        )
    )
    metrics.update(
        compute_expected_node_retrieval_quality(
            rollout_samples,
            batch,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
    )
    del context, features

    if ks:
        last_k = ks[-1]
        for k in ks:
            metrics[f"best_of_k_target_recall@{k}"] = metrics.get(
                f"oracle_best@{k}/target_recall",
                0.0,
            )
        metrics["eval/best_of_k_recall"] = metrics.get(
            f"oracle_best@{last_k}/target_recall",
            0.0,
        )
        metrics["eval/expected_recall"] = metrics.get("expected_target_recall", 0.0)
        metrics["eval/edge_count"] = metrics.get("sample@1/edge_count", 0.0)
        metrics["mean_edges"] = metrics.get("sample@1/edge_count", 0.0)
    return metrics


def rollout_eval_tensors(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    reward_model: TrueTerminalReward | None,
    target_context: TargetContext | None,
) -> RolloutEvalTensors:
    device = torch.device("cpu")
    node_masks, edge_masks = SubgraphReconstructor(batch, device=device).stack(rollout_samples)
    precision, recall, f1, valid_graph_mask = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )
    edge_count = per_graph_counts(
        edge_masks,
        batch.edge_batch.to(device=device, dtype=torch.long),
        num_graphs=int(batch.num_graphs),
    )
    node_count = per_graph_counts(
        node_masks,
        batch.batch.to(device=device, dtype=torch.long),
        num_graphs=int(batch.num_graphs),
    )
    trajectory_len = _stack_stop_step(rollout_samples).float() + 1.0
    policy_stop, forced_stop = stop_matrices(rollout_samples)
    model_score = _stack_trajectory_log_prob(rollout_samples)
    log_reward = log_reward_from_masks(
        node_masks=node_masks,
        edge_count=edge_count,
        batch=batch,
        reward_model=reward_model,
        target_context=target_context,
    )
    return RolloutEvalTensors(
        node_masks=node_masks,
        edge_masks=edge_masks,
        precision=precision,
        recall=recall,
        f1=f1,
        edge_count=edge_count,
        node_count=node_count,
        trajectory_len=trajectory_len,
        policy_stop=policy_stop,
        forced_stop=forced_stop,
        model_score=model_score,
        log_reward=log_reward,
        valid_graph_mask=valid_graph_mask,
    )


def retrieval_from_masks(
    *,
    node_masks: torch.Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = node_masks.device
    num_samples = int(node_masks.size(0))
    num_graphs = int(batch.num_graphs)
    if num_samples == 0:
        empty = torch.zeros((0, num_graphs), dtype=torch.float32, device=device)
        return empty, empty, empty, torch.zeros(num_graphs, dtype=torch.bool, device=device)

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )
    retrieved_nodes = node_masks
    if exclude_anchors_from_retrieved:
        retrieved_nodes = retrieved_nodes & ~anchor_node_mask(batch, device=device).unsqueeze(0)
    hit_nodes = retrieved_nodes & target_nodes.unsqueeze(0)
    expanded_index = _sample_item_graph_index(
        item_batch=node_batch,
        num_samples=num_samples,
        num_graphs=num_graphs,
    )
    hits = scatter_sum(
        hit_nodes.float().reshape(-1),
        expanded_index,
        dim=0,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)
    retrieved = scatter_sum(
        retrieved_nodes.float().reshape(-1),
        expanded_index,
        dim=0,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)
    gold = scatter_sum(
        target_nodes.float(),
        node_batch,
        dim=0,
        dim_size=num_graphs,
    )
    valid = gold.gt(0.0)
    precision = safe_divide(hits, retrieved)
    recall = safe_divide(hits, gold.unsqueeze(0).expand_as(hits))
    recall = torch.where(valid.unsqueeze(0), recall, torch.zeros_like(recall))
    return precision, recall, safe_f1(precision, recall), valid


def sample_metrics(tensors: RolloutEvalTensors) -> dict[str, float]:
    idx = torch.zeros(int(tensors.valid_graph_mask.numel()), dtype=torch.long)
    return {
        "sample@1/target_recall": _mean_selected(tensors.recall, idx, tensors.valid_graph_mask),
        "sample@1/target_precision": _mean_selected(tensors.precision, idx, tensors.valid_graph_mask),
        "sample@1/target_f1": _mean_selected(tensors.f1, idx, tensors.valid_graph_mask),
        "sample@1/edge_count": _mean_selected(tensors.edge_count, idx, tensors.valid_graph_mask),
        "sample@1/trajectory_len": _mean_selected(tensors.trajectory_len, idx, tensors.valid_graph_mask),
        "sample@1/policy_stop_rate": _mean_selected(tensors.policy_stop, idx, tensors.valid_graph_mask),
        "sample@1/forced_stop_rate": _mean_selected(tensors.forced_stop, idx, tensors.valid_graph_mask),
        "stop/sample@1/policy_stop_rate": _mean_selected(tensors.policy_stop, idx, tensors.valid_graph_mask),
        "stop/sample@1/forced_stop_rate": _mean_selected(tensors.forced_stop, idx, tensors.valid_graph_mask),
        "reward/sample@1/log_reward": _mean_selected(tensors.log_reward, idx, tensors.valid_graph_mask),
    }


def best_metrics(
    tensors: RolloutEvalTensors,
    *,
    ks: Sequence[int],
    prefix: str,
    oracle: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        idx = _oracle_indices(tensors, k) if oracle else _model_indices(tensors, k)
        metrics[f"{prefix}@{k}/target_recall"] = _mean_selected(tensors.recall, idx, tensors.valid_graph_mask)
        metrics[f"{prefix}@{k}/target_f1"] = _mean_selected(tensors.f1, idx, tensors.valid_graph_mask)
        metrics[f"{prefix}@{k}/edge_count"] = _mean_selected(tensors.edge_count, idx, tensors.valid_graph_mask)
        metrics[f"{prefix}@{k}/trajectory_len"] = _mean_selected(tensors.trajectory_len, idx, tensors.valid_graph_mask)
        metrics[f"{prefix}@{k}/effective_reward"] = _mean_selected(tensors.log_reward, idx, tensors.valid_graph_mask)
        metrics[f"stop/{prefix}@{k}/policy_stop_rate"] = _mean_selected(tensors.policy_stop, idx, tensors.valid_graph_mask)
        metrics[f"stop/{prefix}@{k}/forced_stop_rate"] = _mean_selected(tensors.forced_stop, idx, tensors.valid_graph_mask)
        metrics[f"reward/{prefix}@{k}/log_reward"] = _mean_selected(tensors.log_reward, idx, tensors.valid_graph_mask)
        if not oracle:
            oracle_idx = _oracle_indices(tensors, k)
            oracle_recall = _selected_values(tensors.recall, oracle_idx)
            model_recall = _selected_values(tensors.recall, idx)
            metrics[f"{prefix}@{k}/score_gap_to_oracle"] = mean_over_valid_graphs(
                oracle_recall - model_recall,
                tensors.valid_graph_mask,
            )
    return metrics


def union_metrics(
    tensors: RolloutEvalTensors,
    *,
    batch: RetrievalBatch,
    ks: Sequence[int],
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        node_masks = tensors.node_masks[:k].any(dim=0, keepdim=True)
        edge_masks = tensors.edge_masks[:k].any(dim=0, keepdim=True)
        precision, recall, f1, valid = retrieval_from_masks(
            node_masks=node_masks,
            batch=batch,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
        edge_count = per_graph_counts(edge_masks, batch.edge_batch.to(dtype=torch.long), num_graphs=int(batch.num_graphs)).squeeze(0)
        node_count = per_graph_counts(node_masks, batch.batch.to(dtype=torch.long), num_graphs=int(batch.num_graphs)).squeeze(0)
        denom = tensors.edge_count[:k].sum(dim=0)
        unique_ratio = safe_divide(edge_count, denom)
        metrics[f"union@{k}/target_recall"] = mean_over_valid_graphs(recall.squeeze(0), valid)
        metrics[f"union@{k}/target_precision"] = mean_over_valid_graphs(precision.squeeze(0), valid)
        metrics[f"union@{k}/target_f1"] = mean_over_valid_graphs(f1.squeeze(0), valid)
        metrics[f"union@{k}/edge_count"] = mean_over_valid_graphs(edge_count, valid)
        metrics[f"union@{k}/unique_node_count"] = mean_over_valid_graphs(node_count, valid)
        metrics[f"union@{k}/redundancy_ratio"] = mean_over_valid_graphs(1.0 - unique_ratio, valid)
        metrics[f"reward/union@{k}/effective_reward"] = mean_over_valid_graphs(
            log_reward_from_masks(
                node_masks=node_masks,
                edge_count=edge_count.unsqueeze(0),
                batch=batch,
                reward_model=None,
                target_context=None,
            ).squeeze(0),
            valid,
        )
    return metrics


def stopped_reward_metrics(tensors: RolloutEvalTensors) -> dict[str, float]:
    natural_stop = tensors.policy_stop.bool() & ~tensors.forced_stop.bool()
    valid = natural_stop & tensors.valid_graph_mask.view(1, -1)
    if not bool(valid.any()):
        return {"reward/mean_log_reward_of_stopped": 0.0}
    return {
        "reward/mean_log_reward_of_stopped": float(
            tensors.log_reward[valid].float().mean().item()
        )
    }


def summary_metrics(metrics: dict[str, float], *, ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in ("oracle_best", "best", "union"):
        values = [metrics[f"{prefix}@{k}/target_recall"] for k in ks if f"{prefix}@{k}/target_recall" in metrics]
        if values:
            out[f"{prefix}/recall_auc_logk"] = float(sum(values) / len(values))
        for prev, cur in zip(ks, ks[1:]):
            a = metrics.get(f"{prefix}@{prev}/target_recall")
            b = metrics.get(f"{prefix}@{cur}/target_recall")
            if a is not None and b is not None:
                out[f"{prefix}/recall_gain_{prev}_to_{cur}"] = float(b - a)
    return out


def diversity_metrics(
    rollouts: Sequence[RolloutResult],
    tensors: RolloutEvalTensors,
    *,
    batch: RetrievalBatch,
    ks: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    edge_batch = batch.edge_batch.to(device=tensors.edge_masks.device, dtype=torch.long)
    node_batch = batch.batch.to(device=tensors.node_masks.device, dtype=torch.long)
    for k in ks:
        prefix_edges = tensors.edge_masks[:k]
        prefix_nodes = tensors.node_masks[:k]
        edge_sum = tensors.edge_count[:k].sum(dim=0)
        edge_union = per_graph_counts(prefix_edges.any(dim=0, keepdim=True), edge_batch, num_graphs=int(batch.num_graphs)).squeeze(0)
        metrics[f"diversity/edge_jaccard_mean@{k}"] = mean_pairwise_jaccard_by_graph(prefix_edges, edge_batch, num_graphs=int(batch.num_graphs))
        metrics[f"diversity/node_jaccard_mean@{k}"] = mean_pairwise_jaccard_by_graph(prefix_nodes, node_batch, num_graphs=int(batch.num_graphs))
        metrics[f"diversity/unique_edge_ratio@{k}"] = mean_over_valid_graphs(safe_divide(edge_union, edge_sum), tensors.valid_graph_mask)
        metrics[f"diversity/unique_terminal_state_ratio@{k}"] = unique_terminal_state_ratio(rollouts[:k], tensors.valid_graph_mask)
    return metrics


def stop_prefix_metrics(
    rollouts: Sequence[RolloutResult],
    *,
    batch: RetrievalBatch,
    ks: Sequence[int],
    valid_graph_mask: torch.Tensor,
) -> dict[str, float]:
    stats = hit_stop_stats(rollouts, batch=batch)
    metrics: dict[str, float] = {}
    for k in ks:
        hit = stats["hit"][:k]
        continued = stats["continued"][:k]
        first_hit = stats["first_hit"][:k]
        delay = stats["delay"][:k]
        metrics[f"stop/hit_then_continue_rate@{k}"] = _mean_hit_values(continued.float(), hit, valid_graph_mask)
        metrics[f"stop/first_hit_depth_mean@{k}"] = _mean_hit_values(first_hit.float(), hit, valid_graph_mask)
        metrics[f"stop/stop_after_first_hit_delay@{k}"] = _mean_hit_values(delay.float(), hit, valid_graph_mask)
    return metrics


def reward_delta_metrics(
    rollouts: Sequence[RolloutResult],
    *,
    batch: RetrievalBatch,
    reward_model: TrueTerminalReward | None,
    target_context: TargetContext | None,
) -> dict[str, float]:
    del reward_model, target_context
    deltas = extra_edge_after_hit_deltas(rollouts, batch=batch)
    if deltas.numel() == 0:
        return {
            "reward/delta_extra_edge_after_hit_mean": 0.0,
            "reward/delta_extra_edge_after_hit_positive_rate": 0.0,
        }
    return {
        "reward/delta_extra_edge_after_hit_mean": float(deltas.mean().item()),
        "reward/delta_extra_edge_after_hit_positive_rate": float(deltas.gt(0.0).float().mean().item()),
    }


def reachable_recall_scores(
    *,
    node_masks: torch.Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> ReachableRecallScores:
    _, recall, _, valid = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )
    return ReachableRecallScores(recall=recall, valid_graph_mask=valid)


def best_of_k_reachable_recall(
    *,
    scores: ReachableRecallScores,
    best_of_k: int,
) -> float:
    if scores.recall.numel() == 0:
        return 0.0
    k = min(int(best_of_k), int(scores.recall.size(0)))
    if k <= 0:
        return 0.0
    return mean_over_valid_graphs(scores.recall[:k].max(dim=0).values, scores.valid_graph_mask)


def one_sample_reachable_recall(scores: ReachableRecallScores) -> float:
    return mean_over_valid_graphs(scores.recall[:1], scores.valid_graph_mask)


def budget_forced_stop_rate(
    *,
    rollout_samples: Sequence[RolloutResult],
    valid_graph_mask: torch.Tensor,
) -> float:
    _, forced = stop_matrices(rollout_samples)
    return mean_over_valid_graphs(forced, valid_graph_mask)


def stop_matrices(rollouts: Sequence[RolloutResult]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rollouts:
        return torch.zeros((0, 0)), torch.zeros((0, 0))
    policy: list[torch.Tensor] = []
    forced: list[torch.Tensor] = []
    for rollout in rollouts:
        forced_row = rollout.forced_terminal_mask.any(dim=1)
        terminal_row = rollout.terminal_mask.any(dim=1)
        policy.append((terminal_row & ~forced_row).to(device=torch.device("cpu"), dtype=torch.float32))
        forced.append(forced_row.to(device=torch.device("cpu"), dtype=torch.float32))
    return torch.stack(policy, dim=0), torch.stack(forced, dim=0)


def log_reward_from_masks(
    *,
    node_masks: torch.Tensor,
    edge_count: torch.Tensor,
    batch: RetrievalBatch,
    reward_model: TrueTerminalReward | None,
    target_context: TargetContext | None,
) -> torch.Tensor:
    del target_context
    if node_masks.numel() == 0:
        return torch.zeros((0, int(batch.num_graphs)), dtype=torch.float32)
    target_nodes = eval_target_node_mask(batch, device=node_masks.device, use_reachable_targets=True)
    node_batch = batch.batch.to(device=node_masks.device, dtype=torch.long)
    expanded_index = _sample_item_graph_index(
        item_batch=node_batch,
        num_samples=int(node_masks.size(0)),
        num_graphs=int(batch.num_graphs),
    )
    supported = scatter_sum(
        (node_masks & target_nodes.unsqueeze(0)).float().reshape(-1),
        expanded_index,
        dim=0,
        dim_size=int(node_masks.size(0)) * int(batch.num_graphs),
    ).view(int(node_masks.size(0)), int(batch.num_graphs))
    target_count = scatter_sum(
        target_nodes.float(),
        node_batch,
        dim=0,
        dim_size=int(batch.num_graphs),
    )
    recall = safe_divide(supported, target_count.unsqueeze(0).expand_as(supported))
    epsilon = float(getattr(reward_model, "epsilon", 1.0e-6))
    answer_weight = float(getattr(reward_model, "answer_weight", 1.0))
    edge_cost = float(getattr(reward_model, "edge_cost", 0.05))
    fail_cost = float(getattr(reward_model, "fail_cost", 1.0))
    raw_log_reward = torch.log(
        torch.tensor(epsilon, device=node_masks.device)
        + answer_weight * recall
    ) - edge_cost * edge_count - fail_cost * supported.eq(0).float()
    log_reward_scale = float(getattr(reward_model, "log_reward_scale", 1.0))
    return raw_log_reward / log_reward_scale


def hit_stop_stats(
    rollouts: Sequence[RolloutResult],
    *,
    batch: RetrievalBatch,
) -> dict[str, torch.Tensor]:
    k = len(rollouts)
    b = int(batch.num_graphs)
    hit = torch.zeros((k, b), dtype=torch.bool)
    continued = torch.zeros((k, b), dtype=torch.bool)
    first_hit = torch.zeros((k, b), dtype=torch.long)
    delay = torch.zeros((k, b), dtype=torch.long)
    targets = eval_target_node_mask(batch, device=torch.device("cpu"), use_reachable_targets=True)
    anchors = anchor_node_mask(batch, device=torch.device("cpu"))
    edge_index = batch.edge_index.to(device=torch.device("cpu"), dtype=torch.long)

    for rollout_id, rollout in enumerate(rollouts):
        selected = rollout.selected_edge_ids.to(device=torch.device("cpu"), dtype=torch.long)
        expand = rollout.expand_mask.to(device=torch.device("cpu"), dtype=torch.bool)
        terminal_step = rollout.stop_step.to(device=torch.device("cpu"), dtype=torch.long)
        for graph_id in range(b):
            active = anchors.clone()
            graph_nodes = batch.batch.to(device=torch.device("cpu")).eq(graph_id)
            active &= graph_nodes
            first_step = -1
            last_expand_after_hit = False
            for step in range(int(selected.size(1))):
                if bool(expand[graph_id, step]) and int(selected[graph_id, step]) >= 0:
                    edge_id = int(selected[graph_id, step].item())
                    active[edge_index[0, edge_id]] = True
                    active[edge_index[1, edge_id]] = True
                    if first_step >= 0:
                        last_expand_after_hit = True
                if first_step < 0 and bool((active & targets & graph_nodes).any()):
                    first_step = step
            if first_step >= 0:
                hit[rollout_id, graph_id] = True
                continued[rollout_id, graph_id] = last_expand_after_hit
                first_hit[rollout_id, graph_id] = first_step
                delay[rollout_id, graph_id] = max(0, int(terminal_step[graph_id].item()) - first_step)
    return {"hit": hit, "continued": continued, "first_hit": first_hit, "delay": delay}


def extra_edge_after_hit_deltas(
    rollouts: Sequence[RolloutResult],
    *,
    batch: RetrievalBatch,
) -> torch.Tensor:
    values: list[float] = []
    targets = eval_target_node_mask(batch, device=torch.device("cpu"), use_reachable_targets=True)
    anchors = anchor_node_mask(batch, device=torch.device("cpu"))
    edge_index = batch.edge_index.to(device=torch.device("cpu"), dtype=torch.long)
    for rollout in rollouts:
        selected = rollout.selected_edge_ids.to(device=torch.device("cpu"), dtype=torch.long)
        expand = rollout.expand_mask.to(device=torch.device("cpu"), dtype=torch.bool)
        for graph_id in range(int(batch.num_graphs)):
            graph_nodes = batch.batch.to(device=torch.device("cpu")).eq(graph_id)
            active = anchors & graph_nodes
            hit = bool((active & targets & graph_nodes).any())
            edge_count = 0.0
            for step in range(int(selected.size(1))):
                if not bool(expand[graph_id, step]) or int(selected[graph_id, step]) < 0:
                    continue
                before = _simple_log_reward(active, targets, graph_nodes, edge_count)
                edge_id = int(selected[graph_id, step].item())
                active = active.clone()
                active[edge_index[0, edge_id]] = True
                active[edge_index[1, edge_id]] = True
                edge_count += 1.0
                after = _simple_log_reward(active, targets, graph_nodes, edge_count)
                if hit:
                    values.append(after - before)
                hit = hit or bool((active & targets & graph_nodes).any())
    return torch.tensor(values, dtype=torch.float32)


def mean_pairwise_jaccard_by_graph(
    masks: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    num_graphs: int,
) -> float:
    if int(masks.size(0)) <= 1:
        return 0.0
    pair_mask = torch.triu(torch.ones((masks.size(0), masks.size(0)), dtype=torch.bool), diagonal=1)
    values: list[torch.Tensor] = []
    for graph_id in range(int(num_graphs)):
        item_mask = batch_index.eq(graph_id)
        if not bool(item_mask.any()):
            continue
        graph_masks = masks[:, item_mask].float()
        intersection = graph_masks @ graph_masks.T
        sizes = graph_masks.sum(dim=1)
        union = sizes.unsqueeze(0) + sizes.unsqueeze(1) - intersection
        jaccard = torch.where(union.gt(0.0), intersection / union.clamp_min(1.0e-8), torch.ones_like(union))
        values.append(jaccard[pair_mask])
    return 0.0 if not values else float(torch.cat(values).mean().item())


def unique_terminal_state_ratio(
    rollouts: Sequence[RolloutResult],
    valid_graph_mask: torch.Tensor,
) -> float:
    if not rollouts:
        return 0.0
    rates: list[float] = []
    for graph_id in range(int(valid_graph_mask.numel())):
        if not bool(valid_graph_mask[graph_id]):
            continue
        unique = set()
        for rollout in rollouts:
            edge_ids = rollout.selected_edge_ids[graph_id]
            expand = rollout.expand_mask[graph_id]
            unique.add(tuple(sorted(int(x) for x in edge_ids[expand & edge_ids.ge(0)].tolist())))
        rates.append(float(len(unique)) / float(len(rollouts)))
    return 0.0 if not rates else float(sum(rates) / len(rates))


def normalize_k_windows(ks: Sequence[int], *, max_k: int) -> tuple[int, ...]:
    if max_k <= 0:
        return tuple(sorted({int(k) for k in ks if int(k) >= 1}))
    out = tuple(sorted({int(k) for k in ks if 1 <= int(k) <= int(max_k)}))
    return out or (1,)


def _legacy_windows(best_of_k: int | None) -> tuple[int, ...]:
    if best_of_k is None:
        return (1, 2, 4, 8, 16)
    k = int(best_of_k)
    values = [1]
    while values[-1] < k:
        values.append(values[-1] * 2)
    return tuple(x for x in values if x <= k)


def _oracle_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    out = torch.zeros(int(tensors.valid_graph_mask.numel()), dtype=torch.long)
    for graph_id in range(int(tensors.valid_graph_mask.numel())):
        best = 0
        best_key = (-1.0, -1.0, float("-inf"), float("-inf"), 0.0)
        for row in range(min(int(k), int(tensors.recall.size(0)))):
            key = (
                float(tensors.recall[row, graph_id]),
                float(tensors.f1[row, graph_id]),
                -float(tensors.edge_count[row, graph_id]),
                -float(tensors.trajectory_len[row, graph_id]),
                -float(row),
            )
            if key > best_key:
                best = row
                best_key = key
        out[graph_id] = best
    return out


def _model_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    if tensors.model_score.numel() == 0:
        return torch.zeros(int(tensors.valid_graph_mask.numel()), dtype=torch.long)
    return torch.argmax(tensors.model_score[:k], dim=0).to(dtype=torch.long)


def _selected_values(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros_like(idx, dtype=torch.float32)
    return values.gather(0, idx.view(1, -1)).squeeze(0)


def _mean_selected(values: torch.Tensor, idx: torch.Tensor, valid_graph_mask: torch.Tensor) -> float:
    return mean_over_valid_graphs(_selected_values(values, idx), valid_graph_mask)


def _stack_rollout_vector(
    rollouts: Sequence[RolloutResult],
    name: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not rollouts:
        return torch.zeros((0, 0), dtype=dtype)
    return torch.stack(
        [getattr(rollout, name).detach().to(device=torch.device("cpu"), dtype=dtype) for rollout in rollouts],
        dim=0,
    )


def _stack_trajectory_log_prob(rollouts: Sequence[RolloutResult]) -> torch.Tensor:
    if not rollouts:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.stack(
        [
            rollout.policy_trajectory_log_prob.detach().to(
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for rollout in rollouts
        ],
        dim=0,
    )


def _stack_stop_step(rollouts: Sequence[RolloutResult]) -> torch.Tensor:
    if not rollouts:
        return torch.zeros((0, 0), dtype=torch.long)
    return torch.stack(
        [
            rollout.stop_step.detach().to(device=torch.device("cpu"), dtype=torch.long)
            for rollout in rollouts
        ],
        dim=0,
    )


def _sample_item_graph_index(
    *,
    item_batch: torch.Tensor,
    num_samples: int,
    num_graphs: int,
) -> torch.Tensor:
    offsets = torch.arange(int(num_samples), device=item_batch.device).unsqueeze(1) * int(num_graphs)
    return (item_batch.unsqueeze(0) + offsets).reshape(-1)


def _mean_hit_values(
    values: torch.Tensor,
    hit_mask: torch.Tensor,
    valid_graph_mask: torch.Tensor,
) -> float:
    valid = hit_mask & valid_graph_mask.unsqueeze(0)
    if not bool(valid.any()):
        return 0.0
    return float(values[valid].float().mean().item())


def _simple_log_reward(
    active: torch.Tensor,
    targets: torch.Tensor,
    graph_nodes: torch.Tensor,
    edge_count: float,
) -> float:
    target_count = float((targets & graph_nodes).sum().item())
    supported = float((active & targets & graph_nodes).sum().item())
    recall = supported / max(target_count, 1.0)
    return float(torch.log(torch.tensor(1.0e-6 + recall)).item() - 0.05 * edge_count - (1.0 if supported == 0.0 else 0.0))


__all__ = [
    "ReachableRecallScores",
    "best_of_k_reachable_recall",
    "budget_forced_stop_rate",
    "evaluate_rollout_samples",
    "one_sample_reachable_recall",
    "reachable_recall_scores",
]
