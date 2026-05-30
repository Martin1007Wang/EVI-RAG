from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rollout import evaluate_rollout_samples
from src.graph.oracle_replay import _bfs_reverse, enumerate_shortest_paths, outgoing_edges_by_src
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.rollout.replay import ReplaySource, _decode_replay_program, _rank_replay_edges_for_state, initial_replay_state_batch
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches


@dataclass(frozen=True, slots=True)
class Candidate:
    edge_tuple: tuple[int, ...]
    target_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StateRecord:
    sample_id: str
    sample_index: int
    state_source: str
    graph_id: int
    depth: int
    frontier_size: int
    positive_count_materialized: int
    top1_hit_materialized: float
    best_positive_rank_materialized: int | None
    positive_mass_materialized: float
    sampled_action_hit_materialized: float | None
    stop_prob: float


@dataclass(frozen=True, slots=True)
class TruncatedStateRecord:
    sample_id: str
    state_source: str
    depth: int
    frontier_size: int
    materialized_only_mass: float
    omitted_only_mass: float
    shared_mass: float
    neither_mass: float
    top1_category: str
    sampled_category: str | None


@dataclass(frozen=True, slots=True)
class TruncatedRolloutAction:
    sample_id: str
    rollout_row: int
    depth: int
    edge_id: int
    category: str


class ScalarStat:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        value = float(value)
        if math.isfinite(value):
            self.values.append(value)

    def mean(self) -> float | None:
        if not self.values:
            return None
        return float(sum(self.values) / len(self.values))

    def count(self) -> int:
        return len(self.values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose replay-buffer competence versus policy rollout competence.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--truncated-rollouts", type=int, default=512)
    parser.add_argument("--state-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = resolve_out_dir(args)
    cfg = load_run_config(run_dir=args.run_dir.resolve(), data_dir=str(args.data_dir), device=str(args.device))
    datamodule, resources = prepare_training_components(cfg, stage="fit")
    dataset = {
        "train": datamodule.train_dataset,
        "validation": datamodule.val_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }[args.split]
    if dataset is None:
        raise RuntimeError(f"split {args.split!r} is not initialized")

    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(args.ckpt.resolve()), strict=False)
    device = torch.device(str(args.device))
    model = model.to(device)
    model.eval()

    replay_stats = defaultdict(ScalarStat)
    policy_stats = defaultdict(ScalarStat)
    rollout_stats = defaultdict(ScalarStat)
    global_rows: list[dict[str, Any]] = []

    truncated_index, truncated_sample_id = find_truncated_sample(dataset)
    with torch.no_grad():
        for start in range(0, len(dataset), int(args.batch_size)):
            end = min(len(dataset), start + int(args.batch_size))
            samples = [dataset[idx] for idx in range(start, end)]
            batch = datamodule.collator(samples).to(device)
            graph = GraphContext.from_batch(batch)
            target = TargetContext.from_batch(batch=batch, graph_context=graph)
            replay = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)
            features = model.feature_encoder(batch)

            replay_rows = collect_replay_state_rows(
                model=model,
                batch=batch,
                graph=graph,
                target=target,
                replay=replay,
                features=features,
                sample_offset=start,
                state_chunk_size=int(args.state_chunk_size),
            )
            for row in replay_rows:
                global_rows.append(asdict(row))
                ingest_state_stats(replay_stats, row)

            trajectories = model.runner.eval_rollouts(
                policy=model.policy,
                context=graph,
                features=features,
                budget=int(model.budget),
                num_rollouts=int(args.num_rollouts),
            )
            for name, value in evaluate_rollout_samples(
                trajectories=trajectories,
                batch=batch,
                context=graph,
                exclude_anchors_from_retrieved=bool(model.evaluation.exclude_anchors_from_retrieved),
                use_reachable_targets=bool(model.evaluation.use_reachable_targets),
                k_windows=tuple(model.evaluation.k_windows),
                enable_terminal_diagnostics=True,
            ).items():
                rollout_stats[name].add(value)

            policy_rows = collect_policy_state_rows(
                model=model,
                batch=batch,
                graph=graph,
                target=target,
                replay=replay,
                features=features,
                trajectories=trajectories,
                sample_offset=start,
                state_chunk_size=int(args.state_chunk_size),
            )
            for row in policy_rows:
                global_rows.append(asdict(row))
                ingest_state_stats(policy_stats, row)

    truncated_summary = {}
    truncated_state_rows: list[dict[str, Any]] = []
    truncated_rollout_rows: list[dict[str, Any]] = []
    if truncated_index is not None and truncated_sample_id is not None:
        summary, state_rows, rollout_rows = diagnose_truncated_sample(
            model=model,
            datamodule=datamodule,
            dataset=dataset,
            sample_index=truncated_index,
            sample_id=truncated_sample_id,
            rollouts=int(args.truncated_rollouts),
            device=device,
            state_chunk_size=int(args.state_chunk_size),
        )
        truncated_summary = summary
        truncated_state_rows = [asdict(row) for row in state_rows]
        truncated_rollout_rows = [asdict(row) for row in rollout_rows]

    manifest = {
        "run_dir": str(args.run_dir.resolve()),
        "ckpt": str(args.ckpt.resolve()),
        "split": args.split,
        "data_dir": str(args.data_dir),
        "device": str(device),
        "seed": int(args.seed),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "truncated_sample_id": truncated_sample_id,
        "global_row_count": len(global_rows),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
    }
    global_summary = {
        "replay_states": summarize_stats(replay_stats),
        "policy_states": summarize_stats(policy_stats),
        "rollouts": summarize_stats(rollout_stats),
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "global_summary.json", global_summary)
    write_csv(out_dir / "global_state_rows.csv", global_rows)
    write_json(out_dir / "truncated_sample_summary.json", truncated_summary)
    write_csv(out_dir / "truncated_state_rows.csv", truncated_state_rows)
    write_csv(out_dir / "truncated_rollout_actions.csv", truncated_rollout_rows)
    write_report(
        out_dir / "report.md",
        manifest=manifest,
        global_summary=global_summary,
        truncated_summary=truncated_summary,
    )
    print(f"diagnostic_out={out_dir}")


def load_run_config(*, run_dir: Path, data_dir: str, device: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = str(data_dir)
        cfg.logger = None
        cfg.trainer.accelerator = "gpu" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.enable_checkpointing = False
    return cfg


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "outputs" / "diagnostics" / "replay_buffer_vs_policy" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()


def find_truncated_sample(dataset) -> tuple[int | None, str | None]:
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(sample.replay_program.path_truncated.item()) != 0:
            return idx, str(sample.sample_id)
    return None, None


def collect_replay_state_rows(*, model, batch, graph: GraphContext, target: TargetContext, replay: ReplayContext, features, sample_offset: int, state_chunk_size: int) -> list[StateRecord]:
    source = model.runner.replay_source
    if source is None:
        return []
    replay_out = source.collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=int(model.budget),
        ),
    )
    nonterminal = replay_out.nonterminal
    if nonterminal is None or int(nonterminal.num_transitions) == 0:
        return []
    rows = nonterminal.parent_state_ids.to(dtype=torch.long, device=nonterminal.device).view(-1)
    states = nonterminal.parent_state.take(rows)
    return score_states(
        model=model,
        batch=batch,
        graph=graph,
        target=target,
        replay=replay,
        features=features,
        states=states,
        state_source="replay_state",
        sample_offset=sample_offset,
        sampled_edge_ids=None,
        state_chunk_size=state_chunk_size,
    )


def collect_policy_state_rows(*, model, batch, graph: GraphContext, target: TargetContext, replay: ReplayContext, features, trajectories, sample_offset: int, state_chunk_size: int) -> list[StateRecord]:
    states, sampled = sampled_parent_prefix_states(trajectories=trajectories, graph_context=graph)
    if states.num_states == 0:
        return []
    return score_states(
        model=model,
        batch=batch,
        graph=graph,
        target=target,
        replay=replay,
        features=features,
        states=states,
        state_source="policy_state",
        sample_offset=sample_offset,
        sampled_edge_ids=sampled,
        state_chunk_size=state_chunk_size,
    )


def sampled_parent_prefix_states(*, trajectories, graph_context: GraphContext) -> tuple[StateBatch, torch.Tensor]:
    device = trajectories.device
    budget = int(trajectories.budget)
    state = StateBatch.initial(
        graph_ids=trajectories.graph_ids,
        budget=budget,
        graph_context=graph_context,
    )
    parents: list[StateBatch] = []
    sampled_edges: list[torch.Tensor] = []
    for step in range(budget):
        rows = trajectories.edge_count.gt(step).nonzero(as_tuple=False).flatten()
        if int(rows.numel()) == 0:
            continue
        parents.append(state.take(rows))
        sampled_edges.append(trajectories.edge_ids.index_select(0, rows)[:, step])
        state = state.advance(
            ExpansionBatch(
                state_ids=rows,
                edge_ids=trajectories.edge_ids.index_select(0, rows)[:, step],
            ),
            graph_context=graph_context,
        )
    if not parents:
        empty_state = StateBatch.initial(
            graph_ids=torch.empty(0, dtype=torch.long, device=device),
            budget=budget,
            graph_context=graph_context,
        )
        return empty_state, torch.empty(0, dtype=torch.long, device=device)
    return cat_state_batches(parents), torch.cat(sampled_edges, dim=0)


def score_states(*, model, batch, graph: GraphContext, target: TargetContext, replay: ReplayContext, features, states: StateBatch, state_source: str, sample_offset: int, sampled_edge_ids: torch.Tensor | None, state_chunk_size: int) -> list[StateRecord]:
    if int(states.num_states) == 0:
        return []
    decoded = _decode_replay_program(
        replay_context=replay,
        num_edges=int(graph.num_edges),
    )
    rows: list[StateRecord] = []
    for start_row in range(0, int(states.num_states), int(state_chunk_size)):
        end_row = min(int(states.num_states), start_row + int(state_chunk_size))
        chunk_state_ids = torch.arange(
            start_row,
            end_row,
            dtype=torch.long,
            device=states.device,
        )
        chunk_states = states.take(chunk_state_ids)
        chunk_sampled = None if sampled_edge_ids is None else sampled_edge_ids.index_select(0, chunk_state_ids)
        rows.extend(
            score_state_chunk(
                model=model,
                batch=batch,
                graph=graph,
                target=target,
                replay_program=decoded,
                features=features,
                states=chunk_states,
                state_source=state_source,
                sample_offset=sample_offset,
                sampled_edge_ids=chunk_sampled,
            )
        )
    return rows


def score_state_chunk(*, model, batch, graph: GraphContext, target: TargetContext, replay_program, features, states: StateBatch, state_source: str, sample_offset: int, sampled_edge_ids: torch.Tensor | None) -> list[StateRecord]:
    action_space = states.action_space(graph)
    policy_out = model.policy(
        state=states,
        features=features,
    )
    edge_prob = policy_out.edge_log_flow.detach() - policy_out.continue_log_flow.index_select(0, action_space.expand_state_ids).detach()
    edge_prob = edge_prob.exp().cpu()
    stop_prob = (policy_out.stop_log_flow.detach() - policy_out.state_log_flow.detach()).exp().cpu()
    rows: list[StateRecord] = []
    ptr = action_space.expand_ptr.detach().cpu()
    edge_ids = action_space.expand_edge_ids.detach().cpu()
    for state_id in range(int(states.num_states)):
        start = int(ptr[state_id].item())
        end = int(ptr[state_id + 1].item())
        legal = edge_ids[start:end].tolist()
        positive = materialized_positive_edges_for_state(
            state=states,
            state_id=state_id,
            legal_edges=legal,
            target=target,
            replay_program=replay_program,
        )
        prob = edge_prob[start:end]
        top1_hit = 0.0
        best_rank: int | None = None
        positive_mass = 0.0
        if legal and positive:
            best_idx = int(torch.argmax(prob).item())
            top1_hit = 1.0 if int(legal[best_idx]) in positive else 0.0
            positive_mask = torch.tensor([edge in positive for edge in legal], dtype=torch.bool)
            positive_mass = float(prob[positive_mask].sum().item())
            best_positive = prob[positive_mask].max()
            best_rank = int(prob.gt(best_positive).sum().item()) + 1
        sampled_hit = None
        if sampled_edge_ids is not None:
            sampled_hit = 1.0 if int(sampled_edge_ids[state_id].item()) in positive else 0.0
        graph_id = int(states.graph_ids[state_id].item())
        sample_index = int(sample_offset + graph_id)
        rows.append(
            StateRecord(
                sample_id=str(batch.sample_id[graph_id]),
                sample_index=sample_index,
                state_source=state_source,
                graph_id=graph_id,
                depth=int(states.edge_count[state_id].item()),
                frontier_size=len(legal),
                positive_count_materialized=len(positive),
                top1_hit_materialized=top1_hit,
                best_positive_rank_materialized=best_rank,
                positive_mass_materialized=positive_mass,
                sampled_action_hit_materialized=sampled_hit,
                stop_prob=float(stop_prob[state_id].item()),
            )
        )
    return rows


def materialized_positive_edges_for_state(*, state: StateBatch, state_id: int, legal_edges: list[int], target: TargetContext, replay_program) -> set[int]:
    if not legal_edges:
        return set()
    legal_tensor = torch.tensor(legal_edges, dtype=torch.long, device=state.device)
    ranked = _rank_replay_edges_for_state(
        frontier=state,
        state_id=int(state_id),
        legal_edge_ids=legal_tensor,
        target_context=target,
        replay_program=replay_program,
    )
    return {int(edge_id) for edge_id in ranked}


def replay_program_candidates(program_sample) -> list[Candidate]:
    candidates: list[Candidate] = []
    edge_ptr = program_sample.candidate_ptr.tolist()
    edge_ids = program_sample.candidate_edge_ids_local.tolist()
    target_ptr = program_sample.candidate_target_ptr.tolist()
    target_pos = program_sample.candidate_target_positions.tolist()
    for candidate_id in range(max(len(edge_ptr) - 1, 0)):
        edge_start = int(edge_ptr[candidate_id])
        edge_end = int(edge_ptr[candidate_id + 1])
        target_start = int(target_ptr[candidate_id])
        target_end = int(target_ptr[candidate_id + 1])
        candidates.append(
            Candidate(
                edge_tuple=tuple(int(edge) for edge in edge_ids[edge_start:edge_end]),
                target_positions=tuple(int(pos) for pos in target_pos[target_start:target_end]),
            )
        )
    return candidates


def exhaustive_shortest_path_candidates(*, edge_index: torch.Tensor, anchor_node_ids: torch.Tensor, reachable_target_node_ids: torch.Tensor, num_nodes: int) -> list[Candidate]:
    outgoing = outgoing_edges_by_src(edge_index=edge_index, num_nodes=int(num_nodes))
    target_pos_by_node = {int(node): pos for pos, node in enumerate(reachable_target_node_ids.tolist())}
    candidates: list[Candidate] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for target_pos, target_node_id in enumerate(reachable_target_node_ids.tolist()):
        distances = torch.tensor(
            _bfs_reverse(edge_index=edge_index, start=int(target_node_id), num_nodes=int(num_nodes)),
            dtype=torch.long,
        )
        anchor_order = sorted(
            (
                int(distances[int(anchor_id)].item()),
                int(anchor_id),
            )
            for anchor_id in anchor_node_ids.tolist()
            if int(distances[int(anchor_id)].item()) >= 0
        )
        for distance, anchor_id in anchor_order:
            if distance == 0:
                continue
            paths, _ = enumerate_shortest_paths(
                outgoing=outgoing,
                distances=distances,
                anchor=int(anchor_id),
                limit=10**9,
            )
            for edge_tuple, node_tuple in paths:
                covered_target_positions = tuple(
                    sorted(
                        {
                            int(target_pos_by_node[int(node_id)])
                            for node_id in node_tuple
                            if int(node_id) in target_pos_by_node
                        }
                    )
                )
                key = (tuple(int(edge) for edge in edge_tuple), covered_target_positions)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(Candidate(edge_tuple=key[0], target_positions=key[1]))
    return candidates


def prefix_sets_by_edge(candidates: list[Candidate]) -> dict[int, set[frozenset[int]]]:
    out: dict[int, set[frozenset[int]]] = defaultdict(set)
    for candidate in candidates:
        prefix: list[int] = []
        for edge_id in candidate.edge_tuple:
            prefix.append(int(edge_id))
            prefix_set = frozenset(prefix)
            for member in prefix:
                out[int(member)].add(prefix_set)
    return out


def state_edge_category(*, state_edges: frozenset[int], edge_id: int, materialized_prefixes: dict[int, set[frozenset[int]]], exhaustive_prefixes: dict[int, set[frozenset[int]]]) -> str:
    next_state = frozenset(set(state_edges) | {int(edge_id)})
    in_materialized = next_state in materialized_prefixes.get(int(edge_id), set())
    in_exhaustive = next_state in exhaustive_prefixes.get(int(edge_id), set())
    if in_materialized and in_exhaustive:
        return "shared"
    if in_materialized:
        return "materialized_only"
    if in_exhaustive:
        return "omitted_only"
    return "neither_shortest_path"


def diagnose_truncated_sample(*, model, datamodule, dataset, sample_index: int, sample_id: str, rollouts: int, device: torch.device, state_chunk_size: int) -> tuple[dict[str, Any], list[TruncatedStateRecord], list[TruncatedRolloutAction]]:
    sample = dataset[sample_index]
    batch = datamodule.collator([sample]).to(device)
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)
    features = model.feature_encoder(batch)

    materialized = replay_program_candidates(sample.replay_program)
    exhaustive = exhaustive_shortest_path_candidates(
        edge_index=sample.edge_index,
        anchor_node_ids=sample.anchor_node_ids,
        reachable_target_node_ids=sample.reachable_target_node_ids,
        num_nodes=int(sample.num_nodes),
    )
    materialized_set = {(candidate.edge_tuple, candidate.target_positions) for candidate in materialized}
    exhaustive_set = {(candidate.edge_tuple, candidate.target_positions) for candidate in exhaustive}
    omitted = [
        Candidate(edge_tuple=edge_tuple, target_positions=target_positions)
        for edge_tuple, target_positions in sorted(exhaustive_set - materialized_set)
    ]
    materialized_prefixes = prefix_sets_by_edge(materialized)
    exhaustive_prefixes = prefix_sets_by_edge(exhaustive)

    replay_rows = collect_truncated_state_rows(
        model=model,
        graph=graph,
        target=target,
        features=features,
        replay=replay,
        sample_id=sample_id,
        state_source="replay_state",
        states=replay_parent_states(model=model, graph=graph, target=target, replay=replay),
        sampled_edge_ids=None,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
        state_chunk_size=state_chunk_size,
    )

    trajectories = model.runner.eval_rollouts(
        policy=model.policy,
        context=graph,
        features=features,
        budget=int(model.budget),
        num_rollouts=int(rollouts),
    )
    policy_states, sampled = sampled_parent_prefix_states(trajectories=trajectories, graph_context=graph)
    policy_rows = collect_truncated_state_rows(
        model=model,
        graph=graph,
        target=target,
        features=features,
        replay=replay,
        sample_id=sample_id,
        state_source="policy_state",
        states=policy_states,
        sampled_edge_ids=sampled,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
        state_chunk_size=state_chunk_size,
    )
    rollout_rows = collect_truncated_rollout_actions(
        sample_id=sample_id,
        states=policy_states,
        sampled_edge_ids=sampled,
        materialized_prefixes=materialized_prefixes,
        exhaustive_prefixes=exhaustive_prefixes,
    )

    materialized_counter = Counter()
    exhaustive_counter = Counter()
    omitted_counter = Counter()
    for candidate in materialized:
        for pos in candidate.target_positions:
            materialized_counter[int(pos)] += 1
    for candidate in exhaustive:
        for pos in candidate.target_positions:
            exhaustive_counter[int(pos)] += 1
    for candidate in omitted:
        for pos in candidate.target_positions:
            omitted_counter[int(pos)] += 1

    summary = {
        "sample_id": sample_id,
        "sample_index": int(sample_index),
        "materialized_candidate_count": len(materialized),
        "exhaustive_candidate_count": len(exhaustive),
        "omitted_candidate_count": len(omitted),
        "per_target_materialized": dict(materialized_counter),
        "per_target_exhaustive": dict(exhaustive_counter),
        "per_target_omitted": dict(omitted_counter),
        "replay_state_count": len(replay_rows),
        "policy_state_count": len(policy_rows),
        "rollout_action_count": len(rollout_rows),
    }
    return summary, replay_rows + policy_rows, rollout_rows


def replay_parent_states(*, model, graph: GraphContext, target: TargetContext, replay: ReplayContext) -> StateBatch:
    source = model.runner.replay_source
    assert source is not None
    replay_out = source.collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=int(model.budget),
        ),
    )
    nonterminal = replay_out.nonterminal
    if nonterminal is None or int(nonterminal.num_transitions) == 0:
        return StateBatch.initial(
            graph_ids=torch.empty(0, dtype=torch.long, device=graph.device),
            budget=int(model.budget),
            graph_context=graph,
        )
    return nonterminal.parent_state.take(nonterminal.parent_state_ids.to(dtype=torch.long, device=nonterminal.device))


def collect_truncated_state_rows(*, model, graph: GraphContext, target: TargetContext, features, replay: ReplayContext, sample_id: str, state_source: str, states: StateBatch, sampled_edge_ids: torch.Tensor | None, materialized_prefixes: dict[int, set[frozenset[int]]], exhaustive_prefixes: dict[int, set[frozenset[int]]], state_chunk_size: int) -> list[TruncatedStateRecord]:
    if states.num_states == 0:
        return []
    rows: list[TruncatedStateRecord] = []
    for start_row in range(0, int(states.num_states), int(state_chunk_size)):
        end_row = min(int(states.num_states), start_row + int(state_chunk_size))
        chunk_state_ids = torch.arange(
            start_row,
            end_row,
            dtype=torch.long,
            device=states.device,
        )
        chunk_states = states.take(chunk_state_ids)
        chunk_sampled = None if sampled_edge_ids is None else sampled_edge_ids.index_select(0, chunk_state_ids)
        action_space = chunk_states.action_space(graph)
        policy_out = model.policy(state=chunk_states, features=features)
        edge_prob = (policy_out.edge_log_flow.detach() - policy_out.continue_log_flow.index_select(0, action_space.expand_state_ids).detach()).exp().cpu()
        ptr = action_space.expand_ptr.detach().cpu()
        legal_edges = action_space.expand_edge_ids.detach().cpu()
        for state_id in range(int(chunk_states.num_states)):
            start = int(ptr[state_id].item())
            end = int(ptr[state_id + 1].item())
            state_edges = frozenset(
                int(edge)
                for edge in chunk_states.edge_ids[state_id].tolist()
                if int(edge) >= 0
            )
            mass = {
                "materialized_only": 0.0,
                "omitted_only": 0.0,
                "shared": 0.0,
                "neither_shortest_path": 0.0,
            }
            top1_category = "none"
            sampled_category = None
            if end > start:
                probs = edge_prob[start:end]
                row_edges = legal_edges[start:end].tolist()
                categories = [
                    state_edge_category(
                        state_edges=state_edges,
                        edge_id=int(edge_id),
                        materialized_prefixes=materialized_prefixes,
                        exhaustive_prefixes=exhaustive_prefixes,
                    )
                    for edge_id in row_edges
                ]
                for prob, category in zip(probs.tolist(), categories, strict=True):
                    mass[category] += float(prob)
                top1_category = categories[int(torch.argmax(probs).item())]
                if chunk_sampled is not None:
                    sampled_category = state_edge_category(
                        state_edges=state_edges,
                        edge_id=int(chunk_sampled[state_id].item()),
                        materialized_prefixes=materialized_prefixes,
                        exhaustive_prefixes=exhaustive_prefixes,
                    )
            rows.append(
                TruncatedStateRecord(
                    sample_id=sample_id,
                    state_source=state_source,
                    depth=int(chunk_states.edge_count[state_id].item()),
                    frontier_size=max(0, end - start),
                    materialized_only_mass=mass["materialized_only"],
                    omitted_only_mass=mass["omitted_only"],
                    shared_mass=mass["shared"],
                    neither_mass=mass["neither_shortest_path"],
                    top1_category=top1_category,
                    sampled_category=sampled_category,
                )
            )
    return rows


def collect_truncated_rollout_actions(*, sample_id: str, states: StateBatch, sampled_edge_ids: torch.Tensor, materialized_prefixes: dict[int, set[frozenset[int]]], exhaustive_prefixes: dict[int, set[frozenset[int]]]) -> list[TruncatedRolloutAction]:
    rows: list[TruncatedRolloutAction] = []
    for state_id in range(int(states.num_states)):
        state_edges = frozenset(
            int(edge)
            for edge in states.edge_ids[state_id].tolist()
            if int(edge) >= 0
        )
        edge_id = int(sampled_edge_ids[state_id].item())
        rows.append(
            TruncatedRolloutAction(
                sample_id=sample_id,
                rollout_row=state_id,
                depth=int(states.edge_count[state_id].item()),
                edge_id=edge_id,
                category=state_edge_category(
                    state_edges=state_edges,
                    edge_id=edge_id,
                    materialized_prefixes=materialized_prefixes,
                    exhaustive_prefixes=exhaustive_prefixes,
                ),
            )
        )
    return rows


def ingest_state_stats(stats: dict[str, ScalarStat], row: StateRecord) -> None:
    stats["frontier_size"].add(row.frontier_size)
    stats["positive_count_materialized"].add(row.positive_count_materialized)
    stats["top1_hit_materialized"].add(row.top1_hit_materialized)
    stats["best_positive_rank_materialized"].add(row.best_positive_rank_materialized)
    stats["positive_mass_materialized"].add(row.positive_mass_materialized)
    stats["sampled_action_hit_materialized"].add(row.sampled_action_hit_materialized)
    stats["stop_prob"].add(row.stop_prob)


def summarize_stats(stats: dict[str, ScalarStat]) -> dict[str, Any]:
    return {
        name: {
            "mean": stat.mean(),
            "count": stat.count(),
        }
        for name, stat in sorted(stats.items())
    }


def write_report(path: Path, *, manifest: dict[str, Any], global_summary: dict[str, Any], truncated_summary: dict[str, Any]) -> None:
    replay_top1 = global_summary.get("replay_states", {}).get("top1_hit_materialized", {}).get("mean")
    policy_top1 = global_summary.get("policy_states", {}).get("top1_hit_materialized", {}).get("mean")
    replay_mass = global_summary.get("replay_states", {}).get("positive_mass_materialized", {}).get("mean")
    policy_mass = global_summary.get("policy_states", {}).get("positive_mass_materialized", {}).get("mean")
    text = f"""# Replay Buffer vs Policy Diagnostic

- checkpoint: `{manifest['ckpt']}`
- split: `{manifest['split']}`
- truncated sample: `{manifest['truncated_sample_id']}`

## Global Summary
- replay-state top1 hit on materialized positives: {replay_top1}
- policy-state top1 hit on materialized positives: {policy_top1}
- replay-state positive mass on materialized positives: {replay_mass}
- policy-state positive mass on materialized positives: {policy_mass}

## Truncation Case Study
- materialized candidate count: {truncated_summary.get('materialized_candidate_count')}
- exhaustive candidate count: {truncated_summary.get('exhaustive_candidate_count')}
- omitted candidate count: {truncated_summary.get('omitted_candidate_count')}

## Interpretation Guardrail
- The validation set contains only one truncated sample, so truncation-specific findings are a case study rather than a strong population-level claim.
"""
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
