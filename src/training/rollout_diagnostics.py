from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.weaver.reward import target_ids
from src.weaver.rollout.schema import RolloutBatch
from src.weaver.state import State


def compute_stop_and_teacher_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
    *,
    batch: RetrievalBatch,
) -> dict[str, float]:
    metrics = compute_stop_counterfactual_diagnostics(rollouts)
    metrics.update(compute_teacher_edge_diagnostics(rollouts=rollouts, batch=batch))
    return metrics


def compute_policy_behavior_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
    *,
    max_depth: int = 4,
) -> dict[str, float]:
    valid = _cat_trace(
        rollouts, lambda rollout: rollout.traces.policy_action_valid_mask
    ).bool()
    stop_prob = _cat_trace(
        rollouts, lambda rollout: rollout.traces.target_stop_prob
    ).float()
    continue_prob = _cat_trace(
        rollouts, lambda rollout: rollout.traces.target_continue_prob
    ).float()

    entropy = _cat_trace(
        rollouts, lambda rollout: rollout.traces.edge_action_entropy
    ).float()
    entropy_valid = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.edge_action_entropy_valid_mask,
    ).float()

    metrics = {
        "policy/target_stop_prob_mean": _masked_mean(stop_prob, valid),
        "policy/target_continue_prob_mean": _masked_mean(continue_prob, valid),
        "policy/edge_action_entropy_mean": _safe_ratio(
            float(entropy.sum().item()),
            float(entropy_valid.sum().item()),
        ),
        "policy/edge_action_entropy_valid_rate": _safe_ratio(
            float(entropy_valid.sum().item()),
            float(valid.sum().item()),
        ),
    }

    if valid.numel() > 0:
        horizon = int(valid.size(1))
        for depth in range(min(max_depth, horizon)):
            depth_valid = valid[:, depth]
            metrics[f"policy/target_stop_prob_depth_{depth}"] = _masked_mean(
                stop_prob[:, depth],
                depth_valid,
            )
            metrics[f"policy/target_continue_prob_depth_{depth}"] = _masked_mean(
                continue_prob[:, depth],
                depth_valid,
            )
    else:
        for depth in range(max_depth):
            metrics[f"policy/target_stop_prob_depth_{depth}"] = 0.0
            metrics[f"policy/target_continue_prob_depth_{depth}"] = 0.0

    return metrics


@torch.no_grad()
def compute_root_answer_edge_ranking_diagnostics(
    policy: object,
    *,
    batch: RetrievalBatch,
    expand_budget: int = 3,
) -> dict[str, float]:
    """
    Diagnose whether answer-touching edges are present and ranked at the root.

    Answer edge convention:
        E_A = {(u, r, v): u in A or v in A}

    Rank is computed only for graphs whose root frontier contains an answer
    edge, using the best answer-edge logit per graph:
        rank = 1 + |{e' in C(s_0): logit(e') > max_answer_logit}|
    """
    default = _root_answer_edge_rank_default_metrics()

    num_graphs = int(batch.num_graphs)
    if num_graphs <= 0:
        return default

    state = State.create_initial(batch, expand_budget=int(expand_budget))
    rollout_context = policy.prepare_rollout_context(batch)  # type: ignore[attr-defined]
    step_out = policy(  # type: ignore[operator]
        batch,
        state,
        rollout_context=rollout_context,
        return_edge_breakdown=True,
    )

    edge_ids = step_out.candidate_edge_ids.view(-1)
    edge_batch = step_out.candidate_batch_ids.view(-1)
    final_logits = step_out.edge_logits.view(-1)
    breakdown = step_out.edge_score_breakdown

    if edge_ids.numel() == 0:
        return default
    if breakdown is None:
        raise RuntimeError(
            "Root answer-edge rank diagnostics require Policy.forward(... ) to "
            "return edge_score_breakdown."
        )

    edge_index = batch.edge_index.to(device=edge_ids.device, dtype=torch.long)
    src, dst = edge_index.index_select(1, edge_ids)

    target = torch.zeros(
        int(batch.num_nodes_total),
        dtype=torch.bool,
        device=edge_ids.device,
    )
    ids = target_ids(batch).to(device=edge_ids.device, dtype=torch.long).view(-1)
    if ids.numel() > 0:
        target[ids] = True

    answer_edge = target.index_select(0, src) | target.index_select(0, dst)
    answer_edge_counts = scatter_sum(
        answer_edge.to(dtype=final_logits.dtype),
        edge_batch,
        dim=0,
        dim_size=num_graphs,
    )
    candidate_counts = scatter_sum(
        torch.ones_like(final_logits),
        edge_batch,
        dim=0,
        dim_size=num_graphs,
    )
    prior_logits = breakdown.semantic_logits.view(-1).to(
        device=final_logits.device,
        dtype=final_logits.dtype,
    )
    residual_logits = breakdown.residual_scale.to(
        device=final_logits.device, dtype=final_logits.dtype
    ) * breakdown.residual_logits.view(-1).to(
        device=final_logits.device,
        dtype=final_logits.dtype,
    )
    prior_ranks, has_answer = _best_answer_edge_ranks(
        logits=prior_logits,
        answer_edge=answer_edge,
        edge_batch=edge_batch,
        num_graphs=num_graphs,
    )
    final_ranks, _ = _best_answer_edge_ranks(
        logits=final_logits,
        answer_edge=answer_edge,
        edge_batch=edge_batch,
        num_graphs=num_graphs,
    )
    rank_delta = final_ranks - prior_ranks
    base_logit_std = _tensor_std(prior_logits)
    residual_logit_std = _tensor_std(residual_logits)

    metrics = {
        "edge/base_logit_std": base_logit_std,
        "edge/residual_logit_std": residual_logit_std,
        "edge/residual_to_base_std_ratio": _safe_ratio(
            residual_logit_std,
            base_logit_std,
        ),
        "edge/prior_rank_vs_final_rank_kendall": _tensor_mean(
            _kendall_tau_by_graph(
                prior_logits=prior_logits,
                final_logits=final_logits,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
            )
        ),
        "edge/answer_edge_prior_rank": _tensor_mean(prior_ranks),
        "edge/answer_edge_final_rank": _tensor_mean(final_ranks),
        "root/frontier_answer_edge_rate": _tensor_mean(
            has_answer.to(dtype=torch.float32)
        ),
        "root/frontier_answer_edge_count_mean": _tensor_mean(answer_edge_counts),
        "root/frontier_candidate_count_mean": _tensor_mean(candidate_counts),
        "root/prior_answer_edge_best_rank_mean": _tensor_mean(prior_ranks),
        "root/prior_answer_edge_best_rank_median": _tensor_median(prior_ranks),
        "root/prior_answer_edge_top1_rate": _tensor_mean(
            prior_ranks.le(1.0).to(dtype=torch.float32)
        ),
        "root/prior_answer_edge_top5_rate": _tensor_mean(
            prior_ranks.le(5.0).to(dtype=torch.float32)
        ),
        "root/prior_answer_edge_mrr": _tensor_mean(1.0 / prior_ranks.clamp_min(1.0)),
        "root/policy_answer_edge_best_rank_mean": _tensor_mean(final_ranks),
        "root/policy_answer_edge_best_rank_median": _tensor_median(final_ranks),
        "root/policy_answer_edge_top1_rate": _tensor_mean(
            final_ranks.le(1.0).to(dtype=torch.float32)
        ),
        "root/policy_answer_edge_top5_rate": _tensor_mean(
            final_ranks.le(5.0).to(dtype=torch.float32)
        ),
        "root/policy_answer_edge_mrr": _tensor_mean(1.0 / final_ranks.clamp_min(1.0)),
        "root/answer_edge_rank_delta_mean": _tensor_mean(rank_delta),
        "root/final_worse_than_prior_rate": _tensor_mean(
            rank_delta.gt(0.0).to(dtype=torch.float32)
        ),
        "root/answer_edge_q_rel_mean": _masked_mean_1d(
            breakdown.query_relation_score, answer_edge
        ),
        "root/answer_edge_q_new_mean": _masked_mean_1d(
            breakdown.query_new_node_score, answer_edge
        ),
        "root/answer_edge_q_candidate_mean": _masked_mean_1d(
            breakdown.semantic_score,
            answer_edge,
        ),
        "root/answer_edge_new_text_rate": _masked_mean_1d(
            breakdown.new_text_mask,
            answer_edge,
        ),
        "root/answer_edge_logit_mean": _masked_mean_1d(
            breakdown.final_logits,
            answer_edge,
        ),
        "root/nonanswer_edge_q_rel_mean": _masked_mean_1d(
            breakdown.query_relation_score,
            ~answer_edge,
        ),
        "root/nonanswer_edge_q_new_mean": _masked_mean_1d(
            breakdown.query_new_node_score,
            ~answer_edge,
        ),
        "root/nonanswer_edge_q_candidate_mean": _masked_mean_1d(
            breakdown.semantic_score,
            ~answer_edge,
        ),
        "root/nonanswer_edge_new_text_rate": _masked_mean_1d(
            breakdown.new_text_mask,
            ~answer_edge,
        ),
        "root/nonanswer_edge_logit_mean": _masked_mean_1d(
            breakdown.final_logits,
            ~answer_edge,
        ),
    }
    default.update(metrics)
    return default


def _kendall_tau_by_graph(
    *,
    prior_logits: torch.Tensor,
    final_logits: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    prior_logits = prior_logits.view(-1)
    final_logits = final_logits.to(
        device=prior_logits.device, dtype=prior_logits.dtype
    ).view(-1)
    edge_batch = edge_batch.to(device=prior_logits.device, dtype=torch.long).view(-1)

    if (
        prior_logits.numel() == 0
        or final_logits.numel() != prior_logits.numel()
        or edge_batch.numel() != prior_logits.numel()
    ):
        return prior_logits.new_zeros((0,))

    values: list[torch.Tensor] = []
    prior_cpu = prior_logits.detach().float().cpu()
    final_cpu = final_logits.detach().float().cpu()
    edge_batch_cpu = edge_batch.detach().cpu()
    for graph_id in range(int(num_graphs)):
        mask = edge_batch_cpu.eq(graph_id)
        if int(mask.sum().item()) < 2:
            continue

        tau = _kendall_tau_1d(prior_cpu[mask], final_cpu[mask])
        if tau is None:
            continue

        values.append(prior_logits.new_tensor(float(tau)))

    if not values:
        return prior_logits.new_zeros((0,))
    return torch.stack(values, dim=0).to(dtype=prior_logits.dtype)


def _kendall_tau_1d(
    prior: torch.Tensor,
    final: torch.Tensor,
) -> float | None:
    pairs = [
        (float(prior_value), float(final_value))
        for prior_value, final_value in zip(
            prior.view(-1).tolist(), final.view(-1).tolist()
        )
    ]
    if len(pairs) < 2:
        return None

    final_values = sorted({final_value for _, final_value in pairs})
    final_rank = {value: index + 1 for index, value in enumerate(final_values)}
    tree = [0] * (len(final_values) + 2)

    def add(index: int, value: int) -> None:
        while index < len(tree):
            tree[index] += int(value)
            index += index & -index

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    pairs.sort(key=lambda item: (item[0], item[1]))
    numerator = 0.0
    denominator = 0.0
    seen = 0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1

        for _, final_value in pairs[start:end]:
            rank = final_rank[final_value]
            less = prefix_sum(rank - 1)
            greater = seen - prefix_sum(rank)
            numerator += float(less - greater)
            denominator += float(less + greater)

        for _, final_value in pairs[start:end]:
            add(final_rank[final_value], 1)
            seen += 1

        start = end

    if denominator <= 0.0:
        return None
    return numerator / denominator


def _best_answer_edge_ranks(
    *,
    logits: torch.Tensor,
    answer_edge: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_answer_logit, _ = scatter_max(
        logits.masked_fill(~answer_edge, -float("inf")),
        edge_batch,
        dim=0,
        dim_size=int(num_graphs),
    )
    has_answer = torch.isfinite(best_answer_logit)

    if not bool(has_answer.any()):
        return logits.new_zeros((0,)), has_answer

    threshold = best_answer_logit.index_select(0, edge_batch)
    better = logits > threshold
    ranks = 1.0 + scatter_sum(
        better.to(dtype=logits.dtype),
        edge_batch,
        dim=0,
        dim_size=int(num_graphs),
    )
    return ranks[has_answer], has_answer


def compute_terminal_reward_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
    *,
    batch: RetrievalBatch | None = None,
) -> dict[str, float]:
    del batch
    f1_values = _cat_stat(rollouts, lambda rollout: rollout.stats.terminal_answer_f1)
    log_rewards = _cat_stat(rollouts, lambda rollout: rollout.stats.terminal_log_reward)
    complexity_penalties = _cat_optional_stat(
        rollouts,
        lambda rollout: getattr(rollout.stats, "terminal_complexity_penalty", None),
    )
    base_log_rewards = _cat_optional_stat(
        rollouts,
        lambda rollout: getattr(rollout.stats, "terminal_base_log_reward", None),
    )
    utilities = _cat_optional_stat(
        rollouts,
        lambda rollout: getattr(rollout.stats, "terminal_utility", None),
    )
    expanded_edge_counts = _cat_optional_stat(
        rollouts,
        lambda rollout: getattr(rollout.stats, "terminal_expanded_edge_count", None),
    )
    answer_degree_excess = _cat_optional_stat(
        rollouts,
        lambda rollout: getattr(rollout.stats, "terminal_answer_degree_excess", None),
    )

    return {
        "reward/log_reward_mean": _tensor_mean(log_rewards),
        "reward/terminal_answer_f1_mean": _tensor_mean(f1_values),
        "reward/nonzero_f1_rate": _tensor_mean(
            f1_values.gt(0.0).to(dtype=torch.float32)
        ),
        "reward/zero_f1_rate": _tensor_mean(f1_values.eq(0.0).to(dtype=torch.float32)),
        "reward/positive_answer_f1_mean": _masked_mean_1d(
            f1_values,
            f1_values.gt(0.0),
        ),
        "reward/terminal_log_reward_mean": _tensor_mean(log_rewards),
        "reward/log_reward_std": _tensor_std(log_rewards),
        "reward/log_reward_p90": _tensor_quantile(log_rewards, 0.90),
        "reward/log_reward_max": _tensor_max(log_rewards),
        "reward/base_log_reward_mean": _tensor_mean(base_log_rewards),
        "reward/complexity_penalty_mean": _tensor_mean(complexity_penalties),
        "reward/utility_mean": _tensor_mean(utilities),
        "reward/supported_answer_recall_mean": _tensor_mean(utilities),
        "reward/expanded_edge_count_mean": _tensor_mean(expanded_edge_counts),
        "reward/answer_degree_excess_mean": _tensor_mean(answer_degree_excess),
    }


def compute_stop_behavior_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
    *,
    max_depth: int = 4,
) -> dict[str, float]:
    lengths = _cat_stat(
        rollouts, lambda rollout: rollout.stats.trajectory_length
    ).long()
    stop_mask = _cat_trace(rollouts, lambda rollout: rollout.traces.stop_mask).bool()
    continue_mask = _cat_trace(
        rollouts, lambda rollout: rollout.traces.continue_mask
    ).bool()
    policy_valid = _cat_trace(
        rollouts, lambda rollout: rollout.traces.policy_action_valid_mask
    ).bool()
    budget_exhausted = _cat_budget_exhausted(rollouts)

    if lengths.numel() == 0 or stop_mask.numel() == 0:
        metrics = {
            "rollout/stop_depth_mean": 0.0,
            "rollout/model_stop_rate": 0.0,
            "rollout/forced_stop_rate": 0.0,
            "rollout/early_stop_rate": 0.0,
            "rollout/budget_exhausted_stop_rate": 0.0,
            "rollout/no_frontier_stop_rate": 0.0,
            "rollout/continue_rate": 0.0,
            "rollout/terminal_length_mean": 0.0,
        }
        for depth in range(max_depth):
            metrics[f"rollout/stop_depth_hist_{depth}"] = 0.0
        return metrics

    horizon = int(stop_mask.size(1))
    valid = lengths.gt(0)
    step = torch.arange(horizon, device=stop_mask.device).unsqueeze(0)
    valid_steps = step.lt(lengths.clamp_min(0).unsqueeze(1))
    terminal_index = lengths.clamp(1, horizon) - 1

    row = torch.arange(lengths.numel(), device=stop_mask.device)
    terminal_stop = valid & stop_mask[row, terminal_index]
    terminal_policy_valid = terminal_stop & policy_valid[row, terminal_index]
    terminal_budget_exhausted = terminal_stop & budget_exhausted[row, terminal_index]
    terminal_forced = terminal_stop & ~terminal_policy_valid
    terminal_no_frontier = terminal_forced & ~terminal_budget_exhausted
    early_stop = terminal_stop & ~terminal_budget_exhausted
    stop_depth = terminal_index[terminal_stop].to(dtype=torch.float32)

    metrics = {
        "rollout/stop_depth_mean": _tensor_mean(stop_depth),
        "rollout/model_stop_rate": _safe_ratio(
            float(terminal_policy_valid.sum().item()),
            float(terminal_stop.sum().item()),
        ),
        "rollout/forced_stop_rate": _safe_ratio(
            float(terminal_forced.sum().item()),
            float(terminal_stop.sum().item()),
        ),
        "rollout/early_stop_rate": _safe_ratio(
            float(early_stop.sum().item()),
            float(terminal_stop.sum().item()),
        ),
        "rollout/budget_exhausted_stop_rate": _safe_ratio(
            float(terminal_budget_exhausted.sum().item()),
            float(terminal_stop.sum().item()),
        ),
        "rollout/no_frontier_stop_rate": _safe_ratio(
            float(terminal_no_frontier.sum().item()),
            float(terminal_stop.sum().item()),
        ),
        "rollout/continue_rate": _safe_ratio(
            float((continue_mask & valid_steps).sum().item()),
            float(valid_steps.sum().item()),
        ),
        "rollout/terminal_length_mean": _tensor_mean(
            lengths[valid].to(dtype=torch.float32)
        ),
    }

    for depth in range(max_depth):
        metrics[f"rollout/stop_depth_hist_{depth}"] = _safe_ratio(
            float(stop_depth.eq(float(depth)).sum().item()),
            float(terminal_stop.sum().item()),
        )

    return metrics


def compute_after_hit_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
) -> dict[str, float]:
    stop_now_valid = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.stop_now_valid_mask,
    ).bool()
    stop_now_f1 = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.stop_now_answer_f1,
    ).float()
    continue_mask = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.continue_mask,
    ).bool()
    stop_prob = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.target_stop_prob,
    ).float()
    policy_valid = _cat_trace(
        rollouts,
        lambda rollout: rollout.traces.policy_action_valid_mask,
    ).bool()

    if stop_now_valid.numel() == 0:
        return {
            "rollout/continue_after_first_hit_rate": 0.0,
            "rollout/extra_edges_after_first_hit": 0.0,
            "policy/stop_prob_after_hit": 0.0,
            "policy/stop_prob_before_hit": 0.0,
        }

    hit_state = stop_now_valid & stop_now_f1.gt(0.0)
    hit_seen = hit_state.to(dtype=torch.long).cumsum(dim=1).gt(0)
    after_hit = stop_now_valid & hit_seen
    before_hit = stop_now_valid & ~hit_seen

    extra_edges = (continue_mask & after_hit).to(dtype=torch.float32).sum(dim=1)

    return {
        "rollout/continue_after_first_hit_rate": _safe_ratio(
            float((continue_mask & after_hit).sum().item()),
            float(after_hit.sum().item()),
        ),
        "rollout/extra_edges_after_first_hit": _tensor_mean(extra_edges),
        "policy/stop_prob_after_hit": _masked_mean(stop_prob, policy_valid & after_hit),
        "policy/stop_prob_before_hit": _masked_mean(
            stop_prob, policy_valid & before_hit
        ),
    }


def compute_debug_rollout_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
) -> dict[str, float]:
    log_pf = _cat_trace(rollouts, lambda rollout: rollout.traces.log_pf).float()
    log_pb = _cat_trace(rollouts, lambda rollout: rollout.traces.log_pb).float()
    rewards = _cat_stat(
        rollouts, lambda rollout: rollout.stats.terminal_log_reward
    ).float()
    lengths = _cat_stat(
        rollouts, lambda rollout: rollout.stats.trajectory_length
    ).long()
    stop_mask = _cat_trace(rollouts, lambda rollout: rollout.traces.stop_mask).bool()

    unfinished = 0.0
    if lengths.numel() > 0 and stop_mask.numel() > 0:
        horizon = int(stop_mask.size(1))
        row = torch.arange(lengths.numel(), device=stop_mask.device)
        terminal_index = lengths.clamp(1, horizon) - 1
        valid = lengths.gt(0)
        terminal_stop = torch.zeros_like(valid)
        terminal_stop[valid] = stop_mask[row[valid], terminal_index[valid]]
        unfinished = float((~valid | ~terminal_stop).sum().item())

    return {
        "debug/nonfinite_log_pf_count": _nonfinite_count(log_pf),
        "debug/nonfinite_log_pb_count": _nonfinite_count(log_pb),
        "debug/nonfinite_reward_count": _nonfinite_count(rewards),
        "debug/unfinished_rollout_count": unfinished,
    }


def compute_eval_rollout_diagnostics(
    rollouts: Sequence[RolloutBatch],
    *,
    batch: RetrievalBatch,
    include_stop_counterfactual: bool = False,
) -> dict[str, float]:
    del batch
    rollout_tuple = tuple(rollouts)
    metrics = _strip_metric_prefix(
        compute_stop_behavior_diagnostics(rollout_tuple),
        prefix="rollout/",
    )
    metrics.update(
        _strip_metric_prefix(
            compute_policy_behavior_diagnostics(rollout_tuple),
            prefix="policy/",
        )
    )
    if include_stop_counterfactual:
        metrics.update(
            _strip_metric_prefix(
                compute_stop_counterfactual_diagnostics(rollout_tuple),
                prefix="stop_counterfactual/",
            )
        )
    return metrics


def collect_training_rollout_diagnostics(
    rollouts: tuple[RolloutBatch, ...],
    *,
    batch: RetrievalBatch,
    debug: bool = False,
) -> dict[str, float]:
    metrics = {
        **compute_terminal_reward_diagnostics(rollouts, batch=batch),
        **compute_stop_behavior_diagnostics(rollouts),
        **compute_stop_and_teacher_diagnostics(rollouts, batch=batch),
        **compute_policy_behavior_diagnostics(rollouts),
        **compute_after_hit_diagnostics(rollouts),
    }

    from src.eval.compactness import compute_compactness_expectations
    from src.eval.diversity import compute_exploration_diversity

    compactness = compute_compactness_expectations(rollouts, batch)
    metrics.update(
        {f"compactness/{name}": value for name, value in compactness.items()}
    )

    diversity = compute_exploration_diversity(rollouts, batch)
    metrics.update(
        {
            "diversity/pairwise_edge_jaccard_distance": diversity[
                "pairwise_edge_jaccard_distance"
            ],
            "diversity/unique_terminal_subgraph_rate": diversity[
                "unique_terminal_subgraph_rate"
            ],
            "diversity/unique_selected_edge_set_rate": diversity[
                "unique_selected_edge_set_rate"
            ],
        }
    )

    if debug:
        metrics.update(compute_debug_rollout_diagnostics(rollouts))

    return metrics


def compute_stop_counterfactual_diagnostics(
    rollouts: Sequence[RolloutBatch],
) -> dict[str, float]:
    diffs: list[torch.Tensor] = []
    stop_now_f1: list[torch.Tensor] = []

    for rollout in rollouts:
        valid = (
            rollout.traces.stop_now_valid_mask.bool()
            & rollout.traces.continue_mask.bool()
        )
        if not bool(valid.any()):
            continue
        final_log_reward = rollout.stats.terminal_log_reward.to(
            device=rollout.traces.stop_now_log_reward.device,
            dtype=torch.float32,
        )
        diff = (
            final_log_reward.unsqueeze(1) - rollout.traces.stop_now_log_reward.float()
        )
        diffs.append(diff[valid])
        stop_now_f1.append(rollout.traces.stop_now_answer_f1.float()[valid])

    if not diffs:
        return {
            "stop_counterfactual/stop_now_better_rate": 0.0,
            "stop_counterfactual/continue_better_rate": 0.0,
            "stop_counterfactual/mean_continue_minus_stop_log_reward": 0.0,
            "stop_counterfactual/stop_now_answer_f1_mean": 0.0,
        }

    values = torch.cat(diffs, dim=0)
    return {
        "stop_counterfactual/stop_now_better_rate": _tensor_mean(
            values.lt(0.0).to(dtype=torch.float32)
        ),
        "stop_counterfactual/continue_better_rate": _tensor_mean(
            values.gt(0.0).to(dtype=torch.float32)
        ),
        "stop_counterfactual/mean_continue_minus_stop_log_reward": _tensor_mean(values),
        "stop_counterfactual/stop_now_answer_f1_mean": _tensor_mean(
            torch.cat(stop_now_f1, dim=0)
        ),
    }


def compute_teacher_edge_diagnostics(
    *,
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
) -> dict[str, float]:
    num_selected_edges = 0.0
    on_shortest_path = 0.0
    reduces_distance = 0.0
    trajectory_count = 0.0
    any_shortest_path_hit = 0.0
    any_progress_hit = 0.0

    teacher_meta = _build_teacher_batch_meta(batch)
    if teacher_meta is None:
        return {
            "teacher_edge/selected_edge_on_target_shortest_path_rate": 0.0,
            "teacher_edge/selected_edge_reduces_target_distance_rate": 0.0,
            "teacher_edge/trajectory_any_shortest_path_hit_rate": 0.0,
            "teacher_edge/trajectory_any_progress_rate": 0.0,
        }

    horizon = rollouts[0].traces.selected_edge_ids.size(1) if rollouts else 0

    for rollout in rollouts:
        state = State.create_initial(batch, expand_budget=max(0, horizon - 1))
        considered_graphs = teacher_meta.graph_has_target
        trajectory_count += float(considered_graphs.sum().item())
        graph_shortest_path_hit = torch.zeros(
            teacher_meta.num_graphs,
            dtype=torch.bool,
            device=batch.edge_index.device,
        )
        graph_progress_hit = torch.zeros_like(graph_shortest_path_hit)

        for step_id in range(horizon):
            selected_edge_ids = rollout.traces.selected_edge_ids[:, step_id]
            continue_mask = rollout.traces.continue_mask[:, step_id]
            valid_steps = continue_mask & selected_edge_ids.ge(0)
            graph_ids = valid_steps.nonzero(as_tuple=False).view(-1)

            if graph_ids.numel() > 0:
                chosen_edges = selected_edge_ids.index_select(0, graph_ids)
                for local_index, graph_id_tensor in enumerate(graph_ids):
                    graph_id = int(graph_id_tensor.item())
                    graph_meta = teacher_meta.graphs[graph_id]
                    if graph_meta.target_count == 0:
                        continue

                    edge_id = int(chosen_edges[local_index].item())
                    edge_hit, progress_hit = _selected_edge_teacher_hits(
                        graph_id=graph_id,
                        edge_id=edge_id,
                        state=state,
                        meta=teacher_meta,
                    )
                    num_selected_edges += 1.0
                    on_shortest_path += float(edge_hit)
                    reduces_distance += float(progress_hit)
                    graph_shortest_path_hit[graph_id] |= edge_hit
                    graph_progress_hit[graph_id] |= progress_hit

                state.apply_expansion(
                    chosen_edges=chosen_edges,
                    edge_index=batch.edge_index,
                )

        considered_graphs = teacher_meta.graph_has_target
        any_shortest_path_hit += float(
            graph_shortest_path_hit[considered_graphs].float().sum().item()
        )
        any_progress_hit += float(
            graph_progress_hit[considered_graphs].float().sum().item()
        )

    return {
        "teacher_edge/selected_edge_on_target_shortest_path_rate": _safe_ratio(
            on_shortest_path,
            num_selected_edges,
        ),
        "teacher_edge/selected_edge_reduces_target_distance_rate": _safe_ratio(
            reduces_distance,
            num_selected_edges,
        ),
        "teacher_edge/trajectory_any_shortest_path_hit_rate": _safe_ratio(
            any_shortest_path_hit,
            trajectory_count,
        ),
        "teacher_edge/trajectory_any_progress_rate": _safe_ratio(
            any_progress_hit,
            trajectory_count,
        ),
    }


class _TeacherGraphMeta:
    def __init__(
        self,
        *,
        node_offset: int,
        edge_offset: int,
        local_edge_index: torch.Tensor,
        target_node_ids: torch.Tensor,
        node_to_target_distances: torch.Tensor,
        edge_on_target_path: torch.Tensor,
    ) -> None:
        self.node_offset = int(node_offset)
        self.edge_offset = int(edge_offset)
        self.local_edge_index = local_edge_index
        self.target_node_ids = target_node_ids
        self.node_to_target_distances = node_to_target_distances
        self.edge_on_target_path = edge_on_target_path

    @property
    def target_count(self) -> int:
        return int(self.target_node_ids.numel())


class _TeacherBatchMeta:
    def __init__(
        self, *, graphs: tuple[_TeacherGraphMeta, ...], graph_has_target: torch.Tensor
    ) -> None:
        self.graphs = graphs
        self.graph_has_target = graph_has_target

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)


def _build_teacher_batch_meta(batch: RetrievalBatch) -> _TeacherBatchMeta | None:
    reachable_target_ids = getattr(batch, "reachable_target_node_ids", None)
    if not isinstance(reachable_target_ids, torch.Tensor):
        return None

    num_graphs = int(batch.num_graphs)
    device = batch.edge_index.device
    ptr = batch.ptr.to(device=device, dtype=torch.long)
    edge_ptr = batch.edge_ptr.to(device=device, dtype=torch.long)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    node_batch = batch.batch.to(device=device, dtype=torch.long)

    reachable_target_ids = reachable_target_ids.to(device=device, dtype=torch.long)
    target_batch = (
        node_batch.index_select(0, reachable_target_ids)
        if reachable_target_ids.numel() > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )
    target_count_per_graph = torch.bincount(target_batch, minlength=num_graphs)

    graphs: list[_TeacherGraphMeta] = []
    node_flat_offset = 0
    edge_flat_offset = 0
    target_offset = 0
    graph_has_target = torch.zeros(num_graphs, dtype=torch.bool, device=device)

    for graph_id in range(num_graphs):
        node_lo = int(ptr[graph_id].item())
        node_hi = int(ptr[graph_id + 1].item())
        edge_lo = int(edge_ptr[graph_id].item())
        edge_hi = int(edge_ptr[graph_id + 1].item())
        node_count = node_hi - node_lo
        edge_count = edge_hi - edge_lo
        target_count = int(target_count_per_graph[graph_id].item())

        next_node_flat_offset = node_flat_offset + target_count * node_count
        next_edge_flat_offset = edge_flat_offset + target_count * edge_count
        next_target_offset = target_offset + target_count

        graph_target_ids = reachable_target_ids[target_offset:next_target_offset]
        if target_count > 0:
            graph_has_target[graph_id] = True

        local_edge_index = _to_local_edge_index(
            edge_index=edge_index,
            graph_id=graph_id,
            node_lo=node_lo,
            node_hi=node_hi,
            edge_lo=edge_lo,
            edge_hi=edge_hi,
        )
        _validate_target_ids_for_graph(
            target_node_ids=graph_target_ids,
            graph_id=graph_id,
            node_lo=node_lo,
            node_hi=node_hi,
        )

        graphs.append(
            _TeacherGraphMeta(
                node_offset=node_lo,
                edge_offset=edge_lo,
                local_edge_index=local_edge_index,
                target_node_ids=graph_target_ids,
                node_to_target_distances=batch.target_node_distances_flat[
                    node_flat_offset:next_node_flat_offset
                ]
                .to(device=device, dtype=torch.long)
                .view(target_count, node_count),
                edge_on_target_path=batch.target_shortest_path_edge_mask_flat[
                    edge_flat_offset:next_edge_flat_offset
                ]
                .to(device=device, dtype=torch.bool)
                .view(target_count, edge_count),
            )
        )

        node_flat_offset = next_node_flat_offset
        edge_flat_offset = next_edge_flat_offset
        target_offset = next_target_offset

    _validate_teacher_flat_offsets(
        batch=batch,
        node_flat_offset=node_flat_offset,
        edge_flat_offset=edge_flat_offset,
        target_offset=target_offset,
        reachable_target_count=int(reachable_target_ids.numel()),
    )

    return _TeacherBatchMeta(graphs=tuple(graphs), graph_has_target=graph_has_target)


def _to_local_edge_index(
    *,
    edge_index: torch.Tensor,
    graph_id: int,
    node_lo: int,
    node_hi: int,
    edge_lo: int,
    edge_hi: int,
) -> torch.Tensor:
    graph_edge_index = edge_index[:, edge_lo:edge_hi]
    if graph_edge_index.numel() == 0:
        return graph_edge_index.contiguous()

    min_node = int(graph_edge_index.min().item())
    max_node = int(graph_edge_index.max().item())
    if min_node < node_lo or max_node >= node_hi:
        raise RuntimeError(
            "Batched edge_index endpoints do not match graph node range: "
            f"graph_id={graph_id}, node_range=[{node_lo}, {node_hi}), "
            f"edge_range=[{edge_lo}, {edge_hi}), endpoint_range=[{min_node}, {max_node}]."
        )

    return (graph_edge_index - node_lo).contiguous()


def _validate_target_ids_for_graph(
    *,
    target_node_ids: torch.Tensor,
    graph_id: int,
    node_lo: int,
    node_hi: int,
) -> None:
    if target_node_ids.numel() == 0:
        return

    min_node = int(target_node_ids.min().item())
    max_node = int(target_node_ids.max().item())
    if min_node < node_lo or max_node >= node_hi:
        raise RuntimeError(
            "reachable_target_node_ids do not match graph node range: "
            f"graph_id={graph_id}, node_range=[{node_lo}, {node_hi}), "
            f"target_range=[{min_node}, {max_node}]."
        )


def _validate_teacher_flat_offsets(
    *,
    batch: RetrievalBatch,
    node_flat_offset: int,
    edge_flat_offset: int,
    target_offset: int,
    reachable_target_count: int,
) -> None:
    expected_node_values = int(batch.target_node_distances_flat.numel())
    expected_edge_values = int(batch.target_shortest_path_edge_mask_flat.numel())

    if node_flat_offset != expected_node_values:
        raise RuntimeError(
            "target_node_distances_flat length does not match reachable targets and "
            f"node counts: consumed={node_flat_offset}, actual={expected_node_values}."
        )
    if edge_flat_offset != expected_edge_values:
        raise RuntimeError(
            "target_shortest_path_edge_mask_flat length does not match reachable "
            f"targets and edge counts: consumed={edge_flat_offset}, actual={expected_edge_values}."
        )
    if target_offset != reachable_target_count:
        raise RuntimeError(
            "reachable_target_node_ids count mismatch while building teacher metadata: "
            f"consumed={target_offset}, actual={reachable_target_count}."
        )


def _selected_edge_teacher_hits(
    *,
    graph_id: int,
    edge_id: int,
    state: State,
    meta: _TeacherBatchMeta,
) -> tuple[bool, bool]:
    graph_meta = meta.graphs[graph_id]
    local_edge_id = edge_id - graph_meta.edge_offset
    if local_edge_id < 0 or local_edge_id >= graph_meta.local_edge_index.size(1):
        return False, False

    local_src = int(graph_meta.local_edge_index[0, local_edge_id].item())
    local_dst = int(graph_meta.local_edge_index[1, local_edge_id].item())
    global_src = graph_meta.node_offset + local_src
    global_dst = graph_meta.node_offset + local_dst

    src_active = bool(state.active_nodes[global_src])
    dst_active = bool(state.active_nodes[global_dst])
    if src_active == dst_active:
        return False, False

    uncovered_targets = ~state.active_nodes.index_select(0, graph_meta.target_node_ids)
    if not bool(uncovered_targets.any()):
        return False, False

    distances = graph_meta.node_to_target_distances[uncovered_targets]
    path_edges = graph_meta.edge_on_target_path[uncovered_targets]

    edge_hit = bool(path_edges[:, local_edge_id].any().item())
    src_dist = distances[:, local_src]
    dst_dist = distances[:, local_dst]
    if src_active:
        active_dist = src_dist
        new_dist = dst_dist
    else:
        active_dist = dst_dist
        new_dist = src_dist
    progress_hit = bool((active_dist.gt(new_dist) & new_dist.ge(0)).any().item())
    return edge_hit, progress_hit


def _cat_stat(
    rollouts: Sequence[RolloutBatch],
    selector,
) -> torch.Tensor:
    if not rollouts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(
        [selector(rollout).detach().to(dtype=torch.float32) for rollout in rollouts],
        dim=0,
    )


def _cat_optional_stat(
    rollouts: Sequence[RolloutBatch],
    selector,
) -> torch.Tensor:
    values = [
        selected.detach().to(dtype=torch.float32)
        for rollout in rollouts
        for selected in (selector(rollout),)
        if selected is not None
    ]
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(values, dim=0)


def _cat_trace(
    rollouts: Sequence[RolloutBatch],
    selector,
) -> torch.Tensor:
    if not rollouts:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat([selector(rollout).detach() for rollout in rollouts], dim=0)


def _cat_budget_exhausted(rollouts: Sequence[RolloutBatch]) -> torch.Tensor:
    if not rollouts:
        return torch.empty((0, 0), dtype=torch.bool)

    values: list[torch.Tensor] = []
    for rollout in rollouts:
        if rollout.traces.budget_exhausted_mask is not None:
            values.append(rollout.traces.budget_exhausted_mask.detach().bool())
            continue

        lengths = rollout.stats.trajectory_length.detach().long()
        stop_mask = rollout.traces.stop_mask.detach().bool()
        fallback = torch.zeros_like(stop_mask)
        if stop_mask.numel() > 0:
            horizon = int(stop_mask.size(1))
            terminal_index = lengths.clamp(1, horizon) - 1
            row = torch.arange(lengths.numel(), device=stop_mask.device)
            fallback[row, terminal_index] = lengths.ge(horizon)
        values.append(fallback)

    return torch.cat(values, dim=0)


def _strip_metric_prefix(metrics: dict[str, float], *, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in metrics.items():
        if name.startswith(prefix):
            out[name[len(prefix) :]] = value
        else:
            out[name] = value
    return out


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or mask.numel() == 0 or not bool(mask.any()):
        return 0.0
    return float(values[mask].mean().item())


def _tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _tensor_median(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().median().item())


def _tensor_std(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().std(unbiased=False).item())


def _tensor_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), float(q)).item())


def _tensor_max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().max().item())


def _masked_mean_1d(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values.view(-1)
    mask = mask.to(device=values.device, dtype=torch.bool).view(-1)
    if values.numel() != mask.numel():
        raise RuntimeError(
            "Root answer-edge diagnostics received mismatched candidate tensors: "
            f"values={tuple(values.shape)}, mask={tuple(mask.shape)}."
        )
    if values.numel() == 0 or not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().item())


def _root_answer_edge_rank_default_metrics() -> dict[str, float]:
    return {
        "edge/base_logit_std": 0.0,
        "edge/residual_logit_std": 0.0,
        "edge/residual_to_base_std_ratio": 0.0,
        "edge/prior_rank_vs_final_rank_kendall": 0.0,
        "edge/answer_edge_prior_rank": 0.0,
        "edge/answer_edge_final_rank": 0.0,
        "root/frontier_answer_edge_rate": 0.0,
        "root/frontier_answer_edge_count_mean": 0.0,
        "root/frontier_candidate_count_mean": 0.0,
        "root/prior_answer_edge_best_rank_mean": 0.0,
        "root/prior_answer_edge_best_rank_median": 0.0,
        "root/prior_answer_edge_top1_rate": 0.0,
        "root/prior_answer_edge_top5_rate": 0.0,
        "root/prior_answer_edge_mrr": 0.0,
        "root/policy_answer_edge_best_rank_mean": 0.0,
        "root/policy_answer_edge_best_rank_median": 0.0,
        "root/policy_answer_edge_top1_rate": 0.0,
        "root/policy_answer_edge_top5_rate": 0.0,
        "root/policy_answer_edge_mrr": 0.0,
        "root/answer_edge_rank_delta_mean": 0.0,
        "root/final_worse_than_prior_rate": 0.0,
        "root/answer_edge_q_rel_mean": 0.0,
        "root/answer_edge_q_new_mean": 0.0,
        "root/answer_edge_q_candidate_mean": 0.0,
        "root/answer_edge_new_text_rate": 0.0,
        "root/answer_edge_logit_mean": 0.0,
        "root/nonanswer_edge_q_rel_mean": 0.0,
        "root/nonanswer_edge_q_new_mean": 0.0,
        "root/nonanswer_edge_q_candidate_mean": 0.0,
        "root/nonanswer_edge_new_text_rate": 0.0,
        "root/nonanswer_edge_logit_mean": 0.0,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _nonfinite_count(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float((~torch.isfinite(values)).sum().item())


__all__ = [
    "collect_training_rollout_diagnostics",
    "compute_after_hit_diagnostics",
    "compute_debug_rollout_diagnostics",
    "compute_eval_rollout_diagnostics",
    "compute_policy_behavior_diagnostics",
    "compute_root_answer_edge_ranking_diagnostics",
    "compute_stop_and_teacher_diagnostics",
    "compute_stop_behavior_diagnostics",
    "compute_stop_counterfactual_diagnostics",
    "compute_teacher_edge_diagnostics",
    "compute_terminal_reward_diagnostics",
]
