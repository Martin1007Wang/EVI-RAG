from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import OmegaConf

from src.graph.segments import segment_logsumexp
from src.training.factory import build_datamodule, build_model, setup_datamodule
from src.weaver.context import GraphContext, RewardContext
from src.weaver.reward import Reward
from src.weaver.rollout.replay import replay_trajectories, transitions_from_rollouts
from src.weaver.rollout.subgraph import SubgraphReconstructor
from src.weaver.state import State


@dataclass(frozen=True)
class SampleChoice:
    idx: int
    sample_id: str
    num_nodes: int
    num_edges: int
    replay_path: tuple[int, ...]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ.setdefault("DATA_DIR", args.data_dir)

    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_dir).resolve()),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                f"paths.data_dir={args.data_dir}",
                "logger=none",
                "trainer=cpu",
                "datamodule.batch_size=1",
                "datamodule.eval_batch_size=1",
                "datamodule.num_workers=0",
                "datamodule.eval_num_workers=0",
                "datamodule.train_shuffle=false",
                f"model.runner.train_num_rollouts={args.train_rollouts}",
                f"model.runner.eval_num_rollouts={args.eval_rollouts}",
            ],
        )

    if args.hidden_dim is not None:
        OmegaConf.update(cfg, "model.feature_encoder.model_dim", args.hidden_dim)
        OmegaConf.update(cfg, "model.policy.state_encoder.hidden_dim", args.hidden_dim)
        OmegaConf.update(
            cfg,
            "model.policy.state_encoder.edge_encoder.hidden_dim",
            args.hidden_dim,
        )

    dm = build_datamodule(cfg)
    resources = setup_datamodule(dm, stage="fit")
    model = build_model(cfg, resources).to(args.device)
    model.train_temperature = args.temperature
    model.eval_temperature = args.temperature

    choice, batch = choose_sample(
        dm=dm,
        budget=int(model.runner.engine.expand_budget),
        max_scan=args.max_scan,
        max_edges=args.max_edges,
        device=args.device,
    )
    print(
        "sample "
        f"idx={choice.idx} id={choice.sample_id} "
        f"nodes={choice.num_nodes} edges={choice.num_edges} "
        f"replay_path={list(choice.replay_path)}"
    )

    optimizer = model.build_optimizer()

    print("== before ==")
    print_rollout_summary(model=model, batch=batch, num_rollouts=args.eval_rollouts)
    print_trace(model=model, batch=batch, title="greedy_trace_before")
    print_path_trace(
        model=model,
        batch=batch,
        edge_ids=choice.replay_path,
        title="replay_path_trace_before",
    )

    for epoch in range(args.epochs):
        metrics = train_one_step(model=model, batch=batch, optimizer=optimizer)
        print(
            "train "
            f"epoch={epoch + 1} "
            f"loss={metrics['loss']:.4f} "
            f"forced_stop={metrics['rollout/forced_stop_rate']:.4f} "
            f"hit_then_continue={metrics['rollout/hit_then_continue_rate']:.4f} "
            f"m_after_hit={metrics['policy/stop_expand_margin_after_hit']:.4f} "
            f"delta_after_hit={metrics['reward/delta_after_hit_mean']:.4f} "
            f"state_minus_action_lse={metrics['flow/state_minus_action_lse_mean']:.4f} "
            f"db_stop_residual={metrics['loss/db_stop_residual_mean']:.4f}"
        )

    print("== after ==")
    print_rollout_summary(model=model, batch=batch, num_rollouts=args.eval_rollouts)
    print_trace(model=model, batch=batch, title="greedy_trace_after")
    print_path_trace(
        model=model,
        batch=batch,
        edge_ids=choice.replay_path,
        title="replay_path_trace_after",
    )


def train_one_step(
    *,
    model,
    batch,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    context = GraphContext.from_batch(batch)
    reward_context = RewardContext.from_batch(
        batch=batch,
        graph_context=context,
        expand_budget=model.runner.engine.expand_budget,
    )

    model.train()
    with torch.no_grad():
        rollout_features = model.feature_encoder(batch)
        rollout_batch = model.runner.train_rollouts(
            policy=model.policy,
            batch=batch,
            context=context,
            features=rollout_features,
            temperature=model.train_temperature,
        )

    if rollout_batch.transitions is None:
        raise RuntimeError("No transitions from single-sample rollout.")

    features = model.feature_encoder(batch)
    output = model.loss_model(
        policy=model.policy,
        reward_model=model.reward_model,
        transitions=rollout_batch.transitions,
        context=context,
        reward_context=reward_context,
        features=features,
    )

    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    optimizer.step()

    metrics = {key: float(value.detach().cpu()) for key, value in output.metrics.items()}
    metrics["loss"] = float(output.loss.detach().cpu())
    return metrics


def choose_sample(
    *,
    dm,
    budget: int,
    max_scan: int,
    max_edges: int,
    device: str,
) -> tuple[SampleChoice, object]:
    dataset = dm.train_dataset
    if dataset is None:
        raise RuntimeError("train dataset was not initialized")

    limit = min(int(max_scan), len(dataset))
    for idx in range(limit):
        data = dataset[idx]
        batch = dm.collator([data]).to(device)
        context = GraphContext.from_batch(batch)
        if context.num_edges > int(max_edges):
            continue
        trajectories = replay_trajectories(
            batch=batch,
            context=context,
            budget=budget,
            max_trajectories=1,
        )
        if not trajectories:
            continue
        choice = SampleChoice(
            idx=idx,
            sample_id=str(data.sample_id),
            num_nodes=int(context.num_nodes),
            num_edges=int(context.num_edges),
            replay_path=trajectories[0].edge_ids,
        )
        return choice, batch

    raise RuntimeError(
        f"No sample with replay path and <= {max_edges} edges found in first {limit} rows."
    )


@torch.no_grad()
def print_rollout_summary(
    *,
    model,
    batch,
    num_rollouts: int,
) -> None:
    context = GraphContext.from_batch(batch)
    reward_context = RewardContext.from_batch(
        batch=batch,
        graph_context=context,
        expand_budget=model.runner.engine.expand_budget,
    )
    features = model.feature_encoder(batch)
    rollouts = model.runner.eval_rollouts(
        policy=model.policy,
        context=context,
        features=features,
        temperature=model.eval_temperature,
        num_rollouts=int(num_rollouts),
    )

    reconstructor = SubgraphReconstructor(batch, device=context.device)
    node_masks, _ = reconstructor.stack(rollouts)
    target_hits = (node_masks & reward_context.target_mask).sum(dim=1)
    target_count = int(reward_context.target_count_by_graph.sum().item())
    recall = target_hits.float() / max(1, target_count)

    transitions = transitions_from_rollouts(
        rollouts=rollouts,
        budget=model.runner.engine.expand_budget,
        context=context,
    )
    hit_then_continue = 0.0
    if transitions is not None and transitions.num_transitions > 0:
        reward = model.reward_model(
            state=transitions.parent_state,
            context=reward_context,
        )
        hit = reward.supported_count.gt(0)
        continue_edge = transitions.action_edge_ids.ge(0)
        denom = hit.float().sum()
        if bool(denom.gt(0)):
            hit_then_continue = float((hit & continue_edge).float().sum() / denom)

    forced_stop_rate = sum(
        float(rollout.forced_terminal_mask.any(dim=1).float().mean().item())
        for rollout in rollouts
    ) / max(1, len(rollouts))

    print(
        "rollout_summary "
        f"num={len(rollouts)} "
        f"target_recall_mean={float(recall.mean()):.4f} "
        f"target_recall_best={float(recall.max()):.4f} "
        f"hit_rate={float(target_hits.gt(0).float().mean()):.4f} "
        f"hit_then_continue={hit_then_continue:.4f} "
        f"forced_stop_rate={forced_stop_rate:.4f}"
    )


@torch.no_grad()
def print_trace(
    *,
    model,
    batch,
    title: str,
) -> None:
    context = GraphContext.from_batch(batch)
    reward_context = RewardContext.from_batch(
        batch=batch,
        graph_context=context,
        expand_budget=model.runner.engine.expand_budget,
    )
    features = model.feature_encoder(batch)
    state = State.initial(
        context=context,
        budget=model.runner.engine.expand_budget,
        rollouts_per_graph=1,
    )

    print(title)
    for depth in range(model.runner.engine.expand_budget + 1):
        frontier = state.frontier(context)
        policy_out = model.policy(
            features=features,
            state=state,
            context=context,
            frontier=frontier,
        )
        reward = model.reward_model(state=state, context=reward_context)
        row = torch.tensor([0], dtype=torch.long, device=context.device)
        stop_edge = torch.tensor([-1], dtype=torch.long, device=context.device)
        stop_log_prob = policy_out.gather_log_prob(row_ids=row, edge_ids=stop_edge)[0]
        stop_logit = policy_out.stop_logit[0]
        edge_lse = segment_logsumexp(
            values=policy_out.edge_logit.float(),
            segment_ids=policy_out.edge_row_ids,
            num_segments=state.num_rows,
        )[0]
        margin = stop_logit.float()
        action_lp = policy_out.action_log_prob()
        candidate_edge_ids = torch.cat([stop_edge, policy_out.edge_ids], dim=0)
        chosen_pos = int(torch.argmax(action_lp).item())
        chosen_edge = int(candidate_edge_ids[chosen_pos].item())
        chosen = "STOP" if chosen_edge < 0 else "EDGE"
        delta_mean, delta_positive_rate = reward_delta_for_frontier(
            model.reward_model,
            state,
            context,
            reward_context,
            frontier.edge_ids,
            reward.log_reward[0],
        )

        print(
            "trace "
            f"depth={depth} "
            f"selected_edge_count={int(state.edge_mask.sum().item())} "
            f"supported_count={int(reward.supported_count[0].item())} "
            f"reward={float(reward.log_reward[0]):.4f} "
            f"state_log_flow={float(policy_out.state_log_flow[0]):.4f} "
            f"stop_logit={float(stop_logit):.4f} "
            f"edge_logit_lse={float(edge_lse):.4f} "
            f"margin={float(margin):.4f} "
            f"stop_log_prob={float(stop_log_prob):.4f} "
            f"chosen_action={chosen} "
            f"chosen_edge_id={chosen_edge} "
            f"frontier_size={frontier.num_actions} "
            f"delta_R_mean={delta_mean:.4f} "
            f"delta_R_positive_rate={delta_positive_rate:.4f}"
        )

        if chosen_edge < 0:
            break
        state.apply_edges_(
            context=context,
            row_ids=row,
            edge_ids=torch.tensor([chosen_edge], dtype=torch.long, device=context.device),
        )


@torch.no_grad()
def print_path_trace(
    *,
    model,
    batch,
    edge_ids: tuple[int, ...],
    title: str,
) -> None:
    context = GraphContext.from_batch(batch)
    reward_context = RewardContext.from_batch(
        batch=batch,
        graph_context=context,
        expand_budget=model.runner.engine.expand_budget,
    )
    features = model.feature_encoder(batch)
    state = State.initial(
        context=context,
        budget=model.runner.engine.expand_budget,
        rollouts_per_graph=1,
    )

    print(title)
    path = list(edge_ids)
    for depth in range(len(path) + 1):
        frontier = state.frontier(context)
        policy_out = model.policy(
            features=features,
            state=state,
            context=context,
            frontier=frontier,
        )
        reward = model.reward_model(state=state, context=reward_context)
        row = torch.tensor([0], dtype=torch.long, device=context.device)
        stop_edge = torch.tensor([-1], dtype=torch.long, device=context.device)
        stop_log_prob = policy_out.gather_log_prob(row_ids=row, edge_ids=stop_edge)[0]
        stop_logit = policy_out.stop_logit[0]
        edge_lse = segment_logsumexp(
            values=policy_out.edge_logit.float(),
            segment_ids=policy_out.edge_row_ids,
            num_segments=state.num_rows,
        )[0]
        margin = stop_logit.float()
        delta_mean, delta_positive_rate = reward_delta_for_frontier(
            model.reward_model,
            state,
            context,
            reward_context,
            frontier.edge_ids,
            reward.log_reward[0],
        )
        next_edge = path[depth] if depth < len(path) else -1
        next_edge_tensor = torch.tensor(
            [next_edge],
            dtype=torch.long,
            device=context.device,
        )
        next_log_prob = policy_out.gather_log_prob(row_ids=row, edge_ids=next_edge_tensor)[0]

        print(
            "path_trace "
            f"depth={depth} "
            f"selected_edge_count={int(state.edge_mask.sum().item())} "
            f"supported_count={int(reward.supported_count[0].item())} "
            f"reward={float(reward.log_reward[0]):.4f} "
            f"state_log_flow={float(policy_out.state_log_flow[0]):.4f} "
            f"stop_logit={float(stop_logit):.4f} "
            f"edge_logit_lse={float(edge_lse):.4f} "
            f"margin={float(margin):.4f} "
            f"stop_log_prob={float(stop_log_prob):.4f} "
            f"path_next_edge_id={next_edge} "
            f"path_next_log_prob={float(next_log_prob):.4f} "
            f"frontier_size={frontier.num_actions} "
            f"delta_R_mean={delta_mean:.4f} "
            f"delta_R_positive_rate={delta_positive_rate:.4f}"
        )

        if depth >= len(path):
            break
        state.apply_edges_(
            context=context,
            row_ids=row,
            edge_ids=next_edge_tensor,
        )


def reward_delta_for_frontier(
    reward_model: Reward,
    state: State,
    context: GraphContext,
    reward_context: RewardContext,
    edge_ids: torch.Tensor,
    parent_log_reward: torch.Tensor,
) -> tuple[float, float]:
    if edge_ids.numel() == 0:
        return 0.0, 0.0

    repeated = State(
        node_mask=state.node_mask.expand(edge_ids.numel(), -1).clone(),
        edge_mask=state.edge_mask.expand(edge_ids.numel(), -1).clone(),
        max_budget_by_row=state.max_budget_by_row.expand(edge_ids.numel()).clone(),
        row_to_graph=state.row_to_graph.expand(edge_ids.numel()).clone(),
    )
    repeated.apply_edges_(
        context=context,
        row_ids=torch.arange(edge_ids.numel(), dtype=torch.long, device=edge_ids.device),
        edge_ids=edge_ids,
    )
    child_reward = reward_model(state=repeated, context=reward_context).log_reward.float()
    delta = child_reward - parent_log_reward.float()
    return float(delta.mean().item()), float(delta.gt(0.0).float().mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-rollouts", type=int, default=8)
    parser.add_argument("--eval-rollouts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
