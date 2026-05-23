from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.eval.compactness import per_graph_counts
from src.eval.retrieval import mean_over_valid_graphs, safe_divide, safe_f1
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask
from src.utils.scatter import scatter_sum
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import State
from src.weaver.utility import TrueTerminalReward


@dataclass(frozen=True, slots=True)
class ReachableRecallScores:
    recall: torch.Tensor
    valid_graph_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class TerminalPolicyScores:
    state_flow: torch.Tensor
    terminal_flow: torch.Tensor


@dataclass(frozen=True, slots=True)
class RolloutEvalTensors:
    node_masks: torch.Tensor
    edge_masks: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    hit: torch.Tensor
    edge_count: torch.Tensor
    trajectory_len: torch.Tensor
    policy_stop: torch.Tensor
    no_frontier_stop: torch.Tensor
    budget_truncated: torch.Tensor
    traj_prob_score: torch.Tensor
    state_flow_score: torch.Tensor
    terminal_flow_score: torch.Tensor
    log_reward: torch.Tensor
    valid_graph_mask: torch.Tensor


SelectorFn = Callable[[RolloutEvalTensors, int], torch.Tensor]
MAX_LOGGED_K = 8


def evaluate_rollout_samples(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    k_windows: Sequence[int],
    enable_calibration_metrics: bool,
    enable_terminal_diagnostics: bool,
    context: GraphContext | None = None,
    features: EncodedFeatures | None = None,
    reward_model: TrueTerminalReward | None = None,
    target_context: TargetContext | None = None,
    policy: ForwardPolicy | None = None,
) -> dict[str, float]:
    if context is None or features is None or reward_model is None or target_context is None or policy is None:
        raise ValueError("rollout evaluation requires context, features, reward_model, target_context, and policy.")

    ks = normalize_k_windows(k_windows, max_k=len(rollout_samples))
    tensors = rollout_eval_tensors(
        rollout_samples=rollout_samples,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
        context=context,
        features=features,
        reward_model=reward_model,
        target_context=target_context,
        policy=policy,
    )

    metrics: dict[str, float] = {}
    metrics.update(sample_metrics(tensors))
    metrics.update(candidate_best_metrics(tensors, ks=ks, name="oracle_best", selector=_oracle_indices))
    metrics.update(candidate_best_metrics(tensors, ks=ks, name="reward_best", selector=_reward_indices))
    metrics.update(
        union_metrics(
            tensors,
            batch=batch,
            ks=ks,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
    )
    metrics.update(selector_metrics(tensors, ks=ks))
    if enable_calibration_metrics:
        metrics.update(calibration_metrics(tensors))
    if enable_terminal_diagnostics:
        metrics.update(terminal_metrics(rollout_samples, tensors, batch=batch))
    return metrics


def rollout_eval_tensors(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    context: GraphContext,
    features: EncodedFeatures,
    reward_model: TrueTerminalReward,
    target_context: TargetContext,
    policy: ForwardPolicy,
) -> RolloutEvalTensors:
    device = torch.device("cpu")
    node_masks, edge_masks = stacked_terminal_masks(
        rollout_samples,
        num_nodes=int(context.num_nodes),
        num_edges=int(context.num_edges),
        device=device,
    )
    _, recall, f1, valid_graph_mask = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )
    answer_count = answer_count_from_masks(
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
    trajectory_len = _stack_terminal_step(rollout_samples).float() + 1.0
    policy_stop, no_frontier_stop, budget_truncated = terminal_matrices(rollout_samples)
    traj_prob_score = _stack_trajectory_log_prob(rollout_samples)

    terminal_state = stacked_terminal_state(rollout_samples, context=context)
    log_reward = log_reward_from_state(
        state=terminal_state,
        num_samples=int(edge_masks.size(0)),
        num_graphs=int(batch.num_graphs),
        reward_model=reward_model,
        context=context,
        target_context=target_context,
        device=device,
    )
    terminal_scores = terminal_policy_scores(
        state=terminal_state,
        num_samples=int(edge_masks.size(0)),
        num_graphs=int(batch.num_graphs),
        context=context,
        features=features,
        policy=policy,
        expand_budget=_rollout_expand_budget(rollout_samples),
        device=device,
    )

    return RolloutEvalTensors(
        node_masks=node_masks,
        edge_masks=edge_masks,
        recall=recall,
        f1=f1,
        hit=answer_count.gt(0.0),
        edge_count=edge_count,
        trajectory_len=trajectory_len,
        policy_stop=policy_stop,
        no_frontier_stop=no_frontier_stop,
        budget_truncated=budget_truncated,
        traj_prob_score=traj_prob_score,
        state_flow_score=terminal_scores.state_flow,
        terminal_flow_score=terminal_scores.terminal_flow,
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


def answer_count_from_masks(
    *,
    node_masks: torch.Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> torch.Tensor:
    device = node_masks.device
    num_samples = int(node_masks.size(0))
    num_graphs = int(batch.num_graphs)
    if num_samples == 0:
        return torch.zeros((0, num_graphs), dtype=torch.float32, device=device)
    targets = eval_target_node_mask(batch, device=device, use_reachable_targets=use_reachable_targets)
    nodes = node_masks
    if exclude_anchors_from_retrieved:
        nodes = nodes & ~anchor_node_mask(batch, device=device).unsqueeze(0)
    index = _sample_item_graph_index(
        item_batch=batch.batch.to(device=device, dtype=torch.long),
        num_samples=num_samples,
        num_graphs=num_graphs,
    )
    return scatter_sum(
        (nodes & targets.unsqueeze(0)).float().reshape(-1),
        index,
        dim=0,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)


def sample_metrics(tensors: RolloutEvalTensors) -> dict[str, float]:
    valid = tensors.valid_graph_mask
    return {
        "sample/mean_recall": mean_over_valid_graphs(tensors.recall, valid),
        "sample/mean_edges": mean_over_valid_graphs(tensors.edge_count, valid),
        "sample/mean_log_reward": mean_over_valid_graphs(tensors.log_reward, valid),
    }


def candidate_best_metrics(
    tensors: RolloutEvalTensors,
    *,
    ks: Sequence[int],
    name: str,
    selector: SelectorFn,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        idx = selector(tensors, k)
        prefix = f"candidate_{name}@{k}"
        metrics[f"{prefix}/recall"] = _mean_selected(tensors.recall, idx, tensors.valid_graph_mask)
    return metrics


def selector_metrics(tensors: RolloutEvalTensors, *, ks: Sequence[int]) -> dict[str, float]:
    selectors: tuple[tuple[str, SelectorFn], ...] = (
        ("traj_prob", _traj_prob_indices),
        ("state_flow", _state_flow_indices),
        ("terminal_flow", _terminal_flow_indices),
    )
    metrics: dict[str, float] = {}
    for k in ks:
        for name, selector in selectors:
            idx = selector(tensors, k)
            prefix = f"selector_{name}@{k}"
            metrics[f"{prefix}/recall"] = _mean_selected(tensors.recall, idx, tensors.valid_graph_mask)
            metrics[f"{prefix}/f1"] = _mean_selected(tensors.f1, idx, tensors.valid_graph_mask)
            oracle_idx = _oracle_indices(tensors, k)
            oracle_recall = _selected_values(tensors.recall, oracle_idx)
            selected_recall = _selected_values(tensors.recall, idx)
            metrics[f"{prefix}/oracle_gap"] = mean_over_valid_graphs(
                oracle_recall - selected_recall,
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
    edge_batch = batch.edge_batch.to(device=tensors.edge_masks.device, dtype=torch.long)
    for k in ks:
        node_masks = tensors.node_masks[:k].any(dim=0, keepdim=True)
        edge_masks = tensors.edge_masks[:k].any(dim=0, keepdim=True)
        _, recall, _, valid = retrieval_from_masks(
            node_masks=node_masks,
            batch=batch,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
        edge_count = per_graph_counts(edge_masks, edge_batch, num_graphs=int(batch.num_graphs)).squeeze(0)
        denom = tensors.edge_count[:k].sum(dim=0)
        unique_ratio = safe_divide(edge_count, denom)

        prefix = f"candidate_union@{k}"
        metrics[f"{prefix}/recall"] = mean_over_valid_graphs(recall.squeeze(0), valid)
        metrics[f"{prefix}/edges"] = mean_over_valid_graphs(edge_count, valid)
        metrics[f"{prefix}/redundancy"] = mean_over_valid_graphs(1.0 - unique_ratio, valid)
    return metrics


def calibration_metrics(tensors: RolloutEvalTensors) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, score in (
        ("terminal_flow", tensors.terminal_flow_score),
        ("state_flow", tensors.state_flow_score),
        ("traj_prob", tensors.traj_prob_score),
    ):
        mean, valid_rate = per_graph_spearman(score, tensors.log_reward, tensors.valid_graph_mask)
        metrics[f"calibration/{name}_reward_spearman"] = mean
        metrics[f"calibration/{name}_reward_spearman_valid_rate"] = valid_rate

    auc, valid_rate = per_graph_auc(tensors.terminal_flow_score, tensors.hit, tensors.valid_graph_mask)
    metrics["calibration/terminal_flow_hit_auc"] = auc
    metrics["calibration/terminal_flow_hit_auc_valid_rate"] = valid_rate
    return metrics


def terminal_metrics(
    rollouts: Sequence[RolloutResult],
    tensors: RolloutEvalTensors,
    *,
    batch: RetrievalBatch,
) -> dict[str, float]:
    stats = hit_terminal_stats(rollouts, batch=batch)
    hit = stats["hit"]
    continued = stats["continued"]
    valid = tensors.valid_graph_mask
    return {
        "terminal/policy_stop_rate": mean_over_valid_graphs(tensors.policy_stop, valid),
        "terminal/structural_stop_rate": mean_over_valid_graphs(tensors.no_frontier_stop, valid),
        "terminal/budget_truncate_rate": mean_over_valid_graphs(tensors.budget_truncated, valid),
        "terminal/policy_terminal_rate": mean_over_valid_graphs(tensors.policy_stop, valid),
        "terminal/forced_terminal_rate": mean_over_valid_graphs(
            tensors.no_frontier_stop + tensors.budget_truncated,
            valid,
        ),
        "terminal/hit_then_continue_rate": _mean_hit_values(continued.float(), hit, valid),
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


def one_sample_reachable_recall(scores: ReachableRecallScores) -> float:
    return mean_over_valid_graphs(scores.recall[:1], scores.valid_graph_mask)


def budget_forced_terminal_rate(
    *,
    rollout_samples: Sequence[RolloutResult],
    valid_graph_mask: torch.Tensor,
) -> float:
    _, _, truncated = terminal_matrices(rollout_samples)
    return mean_over_valid_graphs(truncated, valid_graph_mask)


def terminal_matrices(rollouts: Sequence[RolloutResult]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rollouts:
        return torch.zeros((0, 0)), torch.zeros((0, 0)), torch.zeros((0, 0))
    policy: list[torch.Tensor] = []
    structural: list[torch.Tensor] = []
    truncated: list[torch.Tensor] = []
    for rollout in rollouts:
        policy.append(rollout.policy_stop.to(device=torch.device("cpu"), dtype=torch.float32))
        structural.append(rollout.no_frontier_stop.to(device=torch.device("cpu"), dtype=torch.float32))
        truncated.append(rollout.budget_truncated.to(device=torch.device("cpu"), dtype=torch.float32))
    return torch.stack(policy, dim=0), torch.stack(structural, dim=0), torch.stack(truncated, dim=0)


def stacked_terminal_state(
    rollouts: Sequence[RolloutResult],
    *,
    context: GraphContext,
) -> State:
    states = [rollout.terminal_state for rollout in rollouts]
    if not states:
        return State.initial(
            graph=context,
            graph_ids=torch.empty(0, dtype=torch.long, device=context.device),
            expand_budget=0,
        )
    if any(state is None for state in states):
        raise ValueError("rollout terminal_state must be present for evaluation.")
    return State.concat(states)


def stacked_terminal_masks(
    rollouts: Sequence[RolloutResult],
    *,
    num_nodes: int,
    num_edges: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rollouts:
        return (
            torch.zeros((0, int(num_nodes)), dtype=torch.bool, device=device),
            torch.zeros((0, int(num_edges)), dtype=torch.bool, device=device),
        )
    states = [rollout.terminal_state for rollout in rollouts]
    if any(state is None for state in states):
        raise ValueError("rollout terminal_state must be present for evaluation.")
    node_masks = torch.stack(
        [
            state.active_node_mask.any(dim=0).to(device=device, dtype=torch.bool)
            for state in states
        ],
        dim=0,
    )
    edge_masks = torch.stack(
        [
            state.selected_edge_mask.any(dim=0).to(device=device, dtype=torch.bool)
            for state in states
        ],
        dim=0,
    )
    return node_masks, edge_masks


def log_reward_from_state(
    *,
    state: State,
    num_samples: int,
    num_graphs: int,
    reward_model: TrueTerminalReward,
    context: GraphContext,
    target_context: TargetContext,
    device: torch.device,
) -> torch.Tensor:
    reward_out = reward_model(
        state=state,
        graph_context=context,
        target_context=target_context,
    )
    return reward_out.log_reward.detach().to(device=device, dtype=torch.float32).view(int(num_samples), int(num_graphs))


def terminal_policy_scores(
    *,
    state: State,
    num_samples: int,
    num_graphs: int,
    context: GraphContext,
    features: EncodedFeatures,
    policy: ForwardPolicy,
    expand_budget: int | None,
    device: torch.device,
) -> TerminalPolicyScores:
    frontier = state.frontier(context, expand_budget=expand_budget)
    with torch.no_grad():
        out = policy(
            features=features,
            state=state,
            context=context,
            frontier=frontier,
        )
    state_flow = out.state_log_flow.detach().to(device=device, dtype=torch.float32).view(int(num_samples), int(num_graphs))
    terminal_flow = out.terminal_log_flow.detach().to(device=device, dtype=torch.float32).view(int(num_samples), int(num_graphs))
    return TerminalPolicyScores(state_flow=state_flow, terminal_flow=terminal_flow)


def hit_terminal_stats(
    rollouts: Sequence[RolloutResult],
    *,
    batch: RetrievalBatch,
) -> dict[str, torch.Tensor]:
    k = len(rollouts)
    b = int(batch.num_graphs)
    hit = torch.zeros((k, b), dtype=torch.bool)
    continued = torch.zeros((k, b), dtype=torch.bool)
    targets = eval_target_node_mask(batch, device=torch.device("cpu"), use_reachable_targets=True)
    anchors = anchor_node_mask(batch, device=torch.device("cpu"))
    edge_index = batch.edge_index.to(device=torch.device("cpu"), dtype=torch.long)

    for rollout_id, rollout in enumerate(rollouts):
        selected = rollout.selected_edge_ids.to(device=torch.device("cpu"), dtype=torch.long)
        expand = rollout.expand_mask.to(device=torch.device("cpu"), dtype=torch.bool)
        for graph_id in range(b):
            active = anchors.clone()
            graph_nodes = batch.batch.to(device=torch.device("cpu")).eq(graph_id)
            active &= graph_nodes
            seen_hit = bool((active & targets & graph_nodes).any())
            expanded_after_hit = False
            for step in range(int(selected.size(1))):
                if bool(expand[graph_id, step]) and int(selected[graph_id, step]) >= 0:
                    if seen_hit:
                        expanded_after_hit = True
                    edge_id = int(selected[graph_id, step].item())
                    active[edge_index[0, edge_id]] = True
                    active[edge_index[1, edge_id]] = True
                seen_hit = seen_hit or bool((active & targets & graph_nodes).any())
            hit[rollout_id, graph_id] = seen_hit
            continued[rollout_id, graph_id] = expanded_after_hit
    return {"hit": hit, "continued": continued}


def per_graph_spearman(scores: torch.Tensor, targets: torch.Tensor, valid_graph_mask: torch.Tensor) -> tuple[float, float]:
    values: list[float] = []
    valid_total = int(valid_graph_mask.sum().item())
    for graph_id in range(int(valid_graph_mask.numel())):
        if not bool(valid_graph_mask[graph_id]):
            continue
        score = [float(x) for x in scores[:, graph_id].tolist()]
        target = [float(x) for x in targets[:, graph_id].tolist()]
        value = spearman(score, target)
        if value is not None:
            values.append(value)
    if not values:
        return 0.0, 0.0
    return float(sum(values) / len(values)), float(len(values)) / float(max(valid_total, 1))


def per_graph_auc(scores: torch.Tensor, labels: torch.Tensor, valid_graph_mask: torch.Tensor) -> tuple[float, float]:
    values: list[float] = []
    valid_total = int(valid_graph_mask.sum().item())
    for graph_id in range(int(valid_graph_mask.numel())):
        if not bool(valid_graph_mask[graph_id]):
            continue
        value = auc_score(
            [float(x) for x in scores[:, graph_id].tolist()],
            [bool(x) for x in labels[:, graph_id].tolist()],
        )
        if value is not None:
            values.append(value)
    if not values:
        return 0.5, 0.0
    return float(sum(values) / len(values)), float(len(values)) / float(max(valid_total, 1))


def spearman(scores: Sequence[float], targets: Sequence[float]) -> float | None:
    if len(scores) < 2 or _all_equal(scores) or _all_equal(targets):
        return None
    score_rank = average_ranks(scores)
    target_rank = average_ranks(targets)
    return pearson(score_rank, target_rank)


def auc_score(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    total = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                total += 1.0
            elif pos == neg:
                total += 0.5
    return total / float(len(positives) * len(negatives))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (float(pos) + float(end - 1)) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    x_mean = sum(xs) / float(len(xs))
    y_mean = sum(ys) / float(len(ys))
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = sum(x * x for x in x_centered) ** 0.5
    y_norm = sum(y * y for y in y_centered) ** 0.5
    denom = x_norm * y_norm
    if denom <= 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / denom


def normalize_k_windows(ks: Sequence[int], *, max_k: int) -> tuple[int, ...]:
    max_logged_k = min(int(max_k), MAX_LOGGED_K) if max_k > 0 else MAX_LOGGED_K
    if max_k <= 0:
        return tuple(sorted({int(k) for k in ks if 1 <= int(k) <= max_logged_k}))
    out = tuple(sorted({int(k) for k in ks if 1 <= int(k) <= max_logged_k}))
    return out or (1,)


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


def _score_indices(score: torch.Tensor, k: int) -> torch.Tensor:
    if score.numel() == 0:
        return torch.zeros((int(score.size(1)) if score.ndim == 2 else 0,), dtype=torch.long)
    return torch.argmax(score[:k], dim=0).to(dtype=torch.long)


def _reward_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    return _score_indices(tensors.log_reward, k)


def _traj_prob_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    return _score_indices(tensors.traj_prob_score, k)


def _state_flow_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    return _score_indices(tensors.state_flow_score, k)


def _terminal_flow_indices(tensors: RolloutEvalTensors, k: int) -> torch.Tensor:
    return _score_indices(tensors.terminal_flow_score, k)


def _selected_values(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros_like(idx, dtype=torch.float32)
    return values.gather(0, idx.view(1, -1)).squeeze(0)


def _mean_selected(values: torch.Tensor, idx: torch.Tensor, valid_graph_mask: torch.Tensor) -> float:
    return mean_over_valid_graphs(_selected_values(values, idx), valid_graph_mask)


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


def _stack_terminal_step(rollouts: Sequence[RolloutResult]) -> torch.Tensor:
    if not rollouts:
        return torch.zeros((0, 0), dtype=torch.long)
    return torch.stack(
        [
            rollout.terminal_step.detach().to(device=torch.device("cpu"), dtype=torch.long)
            for rollout in rollouts
        ],
        dim=0,
    )


def _rollout_expand_budget(rollouts: Sequence[RolloutResult]) -> int | None:
    if not rollouts:
        return None
    return max(int(rollout.expand_budget) for rollout in rollouts)


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


def _all_equal(values: Sequence[float]) -> bool:
    return all(value == values[0] for value in values)


__all__ = [
    "ReachableRecallScores",
    "budget_forced_terminal_rate",
    "evaluate_rollout_samples",
    "one_sample_reachable_recall",
    "reachable_recall_scores",
]
