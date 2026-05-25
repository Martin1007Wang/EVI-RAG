from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch import nn

from src.training.factory import prepare_training_components
from src.training.checkpoint import load_checkpoint_weights
from src.graph.segments import segment_log_softmax, segment_logsumexp
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import (
    select_edge_relation_model,
    select_node_model,
    select_query_model,
)
from src.weaver.nn.state_encoder import SegmentTokenPool
from src.weaver.rollout.replay import (
    build_replay_target_views,
    initial_state_for_graph_ids,
    replay_trajectories_with_stats,
    training_from_trajectories,
)
from src.weaver.utility import Reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-trajectories-per-graph", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = _compose_cfg(args.data_dir)
    dm, resources = prepare_training_components(cfg, stage="fit")
    model = None
    policy_features = None
    if args.ckpt:
        from src.training.factory import build_model

        model = build_model(cfg, resources)
        legacy_kind = _legacy_checkpoint_kind(args.ckpt)
        if legacy_kind == "node_state":
            model.policy = LegacyForwardPolicy(
                state_encoder=LegacyNodeStateEncoder(
                    hidden_dim=model.policy.hidden_dim,
                    edge_encoder=model.policy.state_encoder.edge_encoder,
                )
            )
        elif legacy_kind == "edge_state":
            model.policy = LegacyEdgeStateForwardPolicy(
                state_encoder=model.policy.state_encoder,
            )
        missing, unexpected = load_checkpoint_weights(model, args.ckpt, strict=False)
        print(f"loaded_ckpt path={args.ckpt} missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

    dataset = {
        "train": dm.train_dataset,
        "validation": dm.val_dataset,
        "val": dm.val_dataset,
    }.get(args.split)
    if dataset is None:
        raise ValueError(f"Unsupported split for this probe: {args.split!r}")

    samples = [dataset[int(args.idx) + offset] for offset in range(int(args.batch_size))]
    batch = dm.collator(samples)
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    if model is not None:
        with torch.no_grad():
            policy_features = model.policy_feature_encoder(batch)

    trajectories, stats = replay_trajectories_with_stats(
        batch=batch,
        context=graph,
        budget=int(args.budget),
        max_trajectories_per_graph=int(args.max_trajectories_per_graph),
    )
    training = training_from_trajectories(
        trajectories=trajectories,
        graph=graph,
        budget=int(args.budget),
    )
    reward = Reward(edge_cost=0.05, fail_cost=1.0, reward_temperature=1.0)(
        state=training.terminals.state,
        graph_context=graph,
        target_context=target,
    )

    print(
        "batch "
        f"split={args.split} idx={args.idx} size={args.batch_size} "
        f"graphs={graph.num_graphs} nodes={graph.num_nodes} edges={graph.num_edges} "
        f"anchors={int(batch.anchor_node_ids.numel())} "
        f"targets={int(batch.target_node_ids.numel())} "
        f"reachable_targets={int(batch.reachable_target_node_ids.numel())}"
    )
    print(
        "replay_stats "
        f"eligible_graphs={stats.eligible_graphs} "
        f"covered_graphs={stats.covered_graphs} "
        f"generated_trajectories={stats.generated_trajectories} "
        f"skipped_by_reward={stats.skipped_by_reward}"
    )
    print("training " f"expansions={training.expansions.num_items} " f"terminals={training.terminals.num_items}")

    target_graph = graph.node_to_graph.index_select(0, batch.reachable_target_node_ids.long())
    target_views = build_replay_target_views(
        batch=batch,
        context=graph,
        targets=batch.reachable_target_node_ids.long(),
        target_graph=target_graph,
    )
    views_by_graph: dict[int, list[Any]] = {}
    for view in target_views:
        views_by_graph.setdefault(int(view.graph_id), []).append(view)

    for graph_id in range(int(graph.num_graphs)):
        _print_graph_summary(
            batch=batch,
            graph=graph,
            graph_id=graph_id,
            views=views_by_graph.get(graph_id, []),
        )

    for idx, trajectory in enumerate(trajectories[: int(args.max_trajectories_per_graph)]):
        terminal_row = training.terminals.meta.trajectory_ids.eq(idx).nonzero(as_tuple=False).flatten()
        if terminal_row.numel() != 1:
            continue
        term_idx = int(terminal_row[0].item())
        state = training.terminals.state.select_rows(torch.tensor([term_idx]))
        answer_count = int((state.active_node_mask & target.target_mask.view(1, -1)).sum().item())
        selected_edges = state.selected_edge_mask.nonzero(as_tuple=True)[1].tolist()
        print(
            "trajectory "
            f"id={idx} graph={trajectory.graph_id} edges={list(trajectory.edge_ids)} "
            f"selected_edges={selected_edges} terminal_answer_count={answer_count} "
            f"log_reward={float(reward.log_reward[term_idx].item()):.4f}"
        )
        _print_trajectory_steps(
            graph=graph,
            graph_id=trajectory.graph_id,
            edge_ids=trajectory.edge_ids,
            budget=args.budget,
            model=model,
            policy_features=policy_features,
            top_k=int(args.top_k),
        )


def _compose_cfg(data_dir: str):
    config_dir = str((Path.cwd() / "configs").resolve())
    overrides = [
        "experiment=train/webqsp",
        "logger=none",
        "trainer=cpu",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        "datamodule.train_shuffle=false",
        f"paths.data_dir={data_dir}",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="train", overrides=overrides)


def _print_graph_summary(*, batch, graph: GraphContext, graph_id: int, views: list[Any]) -> None:
    nodes = graph.node_to_graph.eq(graph_id).nonzero(as_tuple=False).flatten()
    edges = graph.edge_to_graph.eq(graph_id).nonzero(as_tuple=False).flatten()
    anchors = batch.anchor_node_ids[graph.node_to_graph.index_select(0, batch.anchor_node_ids.long()).eq(graph_id)]
    targets = batch.reachable_target_node_ids[graph.node_to_graph.index_select(0, batch.reachable_target_node_ids.long()).eq(graph_id)]
    dists = [view.anchor_target_distance for view in views]
    print(
        "graph "
        f"id={graph_id} nodes={int(nodes.numel())} edges={int(edges.numel())} "
        f"anchors={anchors.tolist()} reachable_targets={targets.tolist()} "
        f"anchor_target_distances={dists}"
    )


def _print_trajectory_steps(
    *,
    graph: GraphContext,
    graph_id: int,
    edge_ids: tuple[int, ...],
    budget: int,
    model: Any | None,
    policy_features: Any | None,
    top_k: int,
) -> None:
    current = initial_state_for_graph_ids(
        context=graph,
        graph_ids=torch.tensor([int(graph_id)], dtype=torch.long),
    )
    for step, edge_id in enumerate(edge_ids):
        frontier = current.frontier(graph, budget=int(budget))
        in_frontier = bool(frontier.edge_ids.eq(int(edge_id)).any())
        src = int(graph.edge_index[0, int(edge_id)].item())
        dst = int(graph.edge_index[1, int(edge_id)].item())
        active = current.active_node_mask[0].nonzero(as_tuple=False).flatten().tolist()
        frontier_edges = frontier.edge_ids.tolist()
        policy_text = ""
        if model is not None and policy_features is not None:
            with torch.no_grad():
                out = model.policy(
                    features=policy_features,
                    state=current,
                    context=graph,
                    frontier=frontier,
                )
            edge_positions = out.frontier_edge_ids.eq(int(edge_id)).nonzero(as_tuple=False).flatten()
            if edge_positions.numel() > 0:
                pos = int(edge_positions[0].item())
                chosen_log_prob = float(out.edge_log_prob[pos].item())
                chosen_prob = float(out.edge_log_prob[pos].exp().item())
                row_positions = out.frontier_row_ids.eq(0).nonzero(as_tuple=False).flatten()
                row_log_probs = out.edge_log_prob.index_select(0, row_positions).float()
                row_edge_ids = out.frontier_edge_ids.index_select(0, row_positions)
                rank = int(row_log_probs.gt(row_log_probs[pos]).sum().item()) + 1
                top_count = min(int(top_k), int(row_positions.numel()))
                top_vals, top_idx = torch.topk(row_log_probs, k=top_count)
                top_edges = row_edge_ids.index_select(0, top_idx).tolist()
                top_probs = top_vals.exp().tolist()
                terminal_prob = float(_terminal_log_prob(out)[0].exp().item())
                edge_mass = float(out.edge_prob_mass()[0].item())
                policy_text = (
                    f" rank={rank}/{int(row_positions.numel())}"
                    f" log_prob={chosen_log_prob:.4f} prob={chosen_prob:.6f}"
                    f" terminal_prob={terminal_prob:.6f} edge_mass={edge_mass:.6f}"
                    f" top_edges={top_edges} top_probs={[round(float(x), 6) for x in top_probs]}"
                )
        print(
            "  step "
            f"{step}: active_nodes={active} frontier_size={len(frontier_edges)} "
            f"chosen_edge={int(edge_id)} edge=({src}->{dst}) in_frontier={in_frontier} "
            f"frontier_preview={frontier_edges[:12]}{policy_text}"
        )
        current = current.expand(
            graph=graph,
            rows=torch.zeros(1, dtype=torch.long),
            edge_ids=torch.tensor([int(edge_id)], dtype=torch.long),
            budget=int(budget),
        )


def _legacy_checkpoint_kind(path: str) -> str:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    weight = state_dict.get("policy.state_encoder.fuse.0.weight")
    has_legacy_heads = any(key.startswith("policy.stop_head.") for key in state_dict)
    if not has_legacy_heads or not isinstance(weight, torch.Tensor):
        return ""
    if weight.size(1) == 3072:
        return "node_state"
    if weight.size(1) == 2048:
        return "edge_state"
    return ""


def _terminal_log_prob(out: Any) -> torch.Tensor:
    if hasattr(out, "terminal_log_prob"):
        return out.terminal_log_prob
    return out.stop_log_prob


class LegacyStateEncoding:
    def __init__(
        self,
        *,
        query_h: torch.Tensor,
        row_state_h: torch.Tensor,
        node_state_h: torch.Tensor,
        edge_state_h: torch.Tensor,
    ) -> None:
        self.query_h = query_h
        self.row_state_h = row_state_h
        self.node_state_h = node_state_h
        self.edge_state_h = edge_state_h


class LegacyNodeStateEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int, edge_encoder: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_encoder = edge_encoder
        self.node_pool = SegmentTokenPool(input_dim=hidden_dim, output_dim=hidden_dim)
        self.edge_pool = SegmentTokenPool(
            input_dim=self.edge_encoder.output_dim,
            output_dim=hidden_dim,
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, *, features: Any, state: Any, context: GraphContext) -> LegacyStateEncoding:
        num_rows = state.num_rows
        query_h = select_query_model(features, state.graph_ids)
        node_state_h = self.encode_active_nodes(
            features=features,
            state=state,
            num_rows=num_rows,
            like=query_h,
        )
        edge_state_h = self.encode_selected_edges(
            features=features,
            state=state,
            context=context,
            num_rows=num_rows,
            like=query_h,
        )
        row_state_h = self.fuse(torch.cat([query_h, node_state_h, edge_state_h], dim=-1))
        return LegacyStateEncoding(
            query_h=query_h,
            row_state_h=row_state_h,
            node_state_h=node_state_h,
            edge_state_h=edge_state_h,
        )

    def encode_edge_tokens(
        self,
        *,
        features: Any,
        src_node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_encoder(
            src_h=select_node_model(features, src_node_ids),
            rel_h=select_edge_relation_model(features, edge_ids),
            dst_h=select_node_model(features, dst_node_ids),
        )

    def encode_active_nodes(
        self,
        *,
        features: Any,
        state: Any,
        num_rows: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        row_ids, node_ids = state.active_node_mask.nonzero(as_tuple=True)
        if node_ids.numel() == 0:
            return like.new_zeros((num_rows, self.hidden_dim))
        return self.node_pool(
            tokens=select_node_model(features, node_ids),
            row_ids=row_ids,
            num_rows=num_rows,
        )

    def encode_selected_edges(
        self,
        *,
        features: Any,
        state: Any,
        context: GraphContext,
        num_rows: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        row_ids, edge_ids = state.selected_edge_mask.nonzero(as_tuple=True)
        if edge_ids.numel() == 0:
            return like.new_zeros((num_rows, self.hidden_dim))
        src_node_ids = context.edge_index[0].index_select(0, edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, edge_ids)
        edge_h = self.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=edge_ids,
            dst_node_ids=dst_node_ids,
        )
        return self.edge_pool(tokens=edge_h, row_ids=row_ids, num_rows=num_rows)


class LegacyPolicyOutput:
    def __init__(
        self,
        *,
        stop_logit: torch.Tensor,
        log_flow: torch.Tensor,
        edge_logit: torch.Tensor,
        frontier_row_ids: torch.Tensor,
        frontier_edge_ids: torch.Tensor,
        num_rows: int,
        num_edges: int,
    ) -> None:
        self.stop_logit = stop_logit
        self.log_flow = log_flow
        self.edge_logit = edge_logit
        self.frontier_row_ids = frontier_row_ids
        self.frontier_edge_ids = frontier_edge_ids
        self.num_rows = int(num_rows)
        self.num_edges = int(num_edges)

    @property
    def edge_log_cond_prob(self) -> torch.Tensor:
        if self.edge_logit.numel() == 0:
            return self.edge_logit.new_empty((0,)).float()
        return segment_log_softmax(
            self.edge_logit.float(),
            self.frontier_row_ids.to(device=self.edge_logit.device, dtype=torch.long),
            num_segments=int(self.num_rows),
        )

    @property
    def stop_log_prob(self) -> torch.Tensor:
        has_edge = torch.zeros(int(self.num_rows), dtype=torch.bool, device=self.stop_logit.device)
        if self.frontier_row_ids.numel() > 0:
            has_edge.index_fill_(0, self.frontier_row_ids.to(device=self.stop_logit.device), True)
        log_prob = F.logsigmoid(self.stop_logit.float())
        return torch.where(has_edge, log_prob, torch.zeros_like(log_prob))

    @property
    def continue_log_prob(self) -> torch.Tensor:
        has_edge = torch.zeros(int(self.num_rows), dtype=torch.bool, device=self.stop_logit.device)
        if self.frontier_row_ids.numel() > 0:
            has_edge.index_fill_(0, self.frontier_row_ids.to(device=self.stop_logit.device), True)
        log_prob = F.logsigmoid(-self.stop_logit.float())
        return torch.where(has_edge, log_prob, torch.full_like(log_prob, -torch.inf))

    @property
    def edge_log_prob(self) -> torch.Tensor:
        if self.edge_logit.numel() == 0:
            return self.edge_logit.new_empty((0,)).float()
        row_ids = self.frontier_row_ids.to(device=self.stop_logit.device, dtype=torch.long)
        return self.continue_log_prob.index_select(0, row_ids) + self.edge_log_cond_prob

    def edge_prob_mass(self) -> torch.Tensor:
        mass = self.stop_logit.new_zeros((int(self.num_rows),)).float()
        if self.edge_logit.numel() > 0:
            mass.scatter_add_(
                0,
                self.frontier_row_ids.to(device=mass.device, dtype=torch.long),
                self.edge_log_prob.exp(),
            )
        return mass


class LegacyForwardPolicy(nn.Module):
    def __init__(self, *, state_encoder: LegacyNodeStateEncoder) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        hidden_dim = state_encoder.hidden_dim
        edge_dim = state_encoder.edge_encoder.output_dim
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.flow_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def hidden_dim(self) -> int:
        return self.state_encoder.hidden_dim

    def forward(self, *, features: Any, state: Any, context: GraphContext, frontier: Any) -> LegacyPolicyOutput:
        encoding = self.state_encoder(features=features, state=state, context=context)
        state_type_h = self._state_type_features(state=state, frontier=frontier, like=encoding.query_h)
        stop_logit = self.stop_head(torch.cat([encoding.query_h, encoding.row_state_h, state_type_h], dim=-1)).squeeze(-1)
        log_flow = self.flow_head(torch.cat([encoding.query_h, encoding.row_state_h], dim=-1)).squeeze(-1)
        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=encoding.query_h.device)
            edge_logit = stop_logit.new_empty((0,))
            return LegacyPolicyOutput(
                stop_logit=stop_logit,
                log_flow=log_flow,
                edge_logit=edge_logit,
                frontier_row_ids=empty,
                frontier_edge_ids=empty,
                num_rows=state.num_rows,
                num_edges=state.num_edges,
            )
        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)
        src_active = state.active_node_mask[frontier.row_ids, src_node_ids].float()
        dst_active = state.active_node_mask[frontier.row_ids, dst_node_ids].float()
        edge_type_h = torch.stack([src_active, dst_active], dim=-1)
        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
        )
        edge_logit = self.edge_head(
            torch.cat(
                [
                    encoding.query_h.index_select(0, frontier.row_ids),
                    encoding.row_state_h.index_select(0, frontier.row_ids),
                    edge_h,
                    edge_type_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return LegacyPolicyOutput(
            stop_logit=stop_logit,
            log_flow=log_flow,
            edge_logit=edge_logit,
            frontier_row_ids=frontier.row_ids,
            frontier_edge_ids=frontier.edge_ids,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )

    def _state_type_features(self, *, state: Any, frontier: Any, like: torch.Tensor) -> torch.Tensor:
        has_frontier = torch.zeros((state.num_rows,), dtype=like.dtype, device=like.device)
        if frontier.row_ids.numel() > 0:
            has_frontier.index_fill_(0, frontier.row_ids.to(device=like.device, dtype=torch.long), 1.0)
        depth = state.depth.to(device=like.device, dtype=like.dtype).view(-1)
        return torch.stack([depth, has_frontier], dim=-1)


class LegacyEdgeStateForwardPolicy(nn.Module):
    def __init__(self, *, state_encoder: nn.Module) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        hidden_dim = state_encoder.hidden_dim
        edge_dim = state_encoder.edge_encoder.output_dim
        self.stop_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.flow_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def hidden_dim(self) -> int:
        return self.state_encoder.hidden_dim

    def forward(self, *, features: Any, state: Any, context: GraphContext, frontier: Any) -> LegacyPolicyOutput:
        encoding = self.state_encoder(features=features, state=state, context=context)
        stop_logit = self.stop_head(torch.cat([encoding.query_h, encoding.row_state_h], dim=-1)).squeeze(-1)
        log_flow = self.flow_head(torch.cat([encoding.query_h, encoding.row_state_h], dim=-1)).squeeze(-1)
        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=encoding.query_h.device)
            return LegacyPolicyOutput(
                stop_logit=stop_logit,
                log_flow=log_flow,
                edge_logit=stop_logit.new_empty((0,)),
                frontier_row_ids=empty,
                frontier_edge_ids=empty,
                num_rows=state.num_rows,
                num_edges=state.num_edges,
            )
        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)
        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
        )
        edge_logit = self.edge_head(
            torch.cat(
                [
                    encoding.query_h.index_select(0, frontier.row_ids),
                    encoding.row_state_h.index_select(0, frontier.row_ids),
                    edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return LegacyPolicyOutput(
            stop_logit=stop_logit,
            log_flow=log_flow,
            edge_logit=edge_logit,
            frontier_row_ids=frontier.row_ids,
            frontier_edge_ids=frontier.edge_ids,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )


if __name__ == "__main__":
    main()
