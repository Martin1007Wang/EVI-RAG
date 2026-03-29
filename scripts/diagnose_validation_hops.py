from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.answer_metrics import AnswerPosteriorRecord, SupportWindowResult
from src.models.configs import SearchEvalConfig
from src.models.gflownet.path import (
    append_relation_and_node_tokens_inplace,
    initialize_path_token_ids,
)
from src.models.gflownet.transitions import apply_forward_constraints
from src.models.gflownet.prefix_state import SearchState


def _strip_instantiate_metadata(cfg_node: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(cfg_node, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected mapping config, got {type(container)!r}.")
    container.pop("contract", None)
    return OmegaConf.create(container)


def _load_checkpoint(model: Any, *, ckpt_path: str) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a `state_dict` mapping.")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = sorted(getattr(incompatible, "missing_keys", []))
    unexpected = sorted(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "checkpoint_load": {
                        "missing_keys": missing,
                        "unexpected_keys": unexpected,
                    }
                },
                ensure_ascii=True,
            )
        )


def _load_cfg(
    *,
    config_path: Path,
    output_dir: Path,
    split: str,
    eval_batch_size: int,
) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.output_dir = str(output_dir)
    cfg.data.eval_batch_size = int(eval_batch_size)
    cfg.data.eval_num_workers = 0
    cfg.data.eval_prefetch_factor = None
    cfg.data.eval_persistent_workers = False
    cfg.data.eval_multiprocessing_context = None
    cfg.data.eval_split = str(split)
    if cfg.get("run") is None:
        cfg.run = OmegaConf.create({})
    cfg.run.split = str(split)
    return cfg


def _instantiate_objects(cfg: DictConfig) -> tuple[Any, Any]:
    data_cfg = _strip_instantiate_metadata(cfg.data)
    model_cfg = _strip_instantiate_metadata(cfg.model)
    datamodule = hydra.utils.instantiate(data_cfg)
    model = hydra.utils.instantiate(model_cfg)
    return datamodule, model


def _prepare_batch(datamodule: Any, *, raw_batch: Any, device: torch.device) -> Any:
    batch = datamodule.on_before_batch_transfer(raw_batch, 0)
    return datamodule.transfer_batch_to_device(batch, device, 0)


def _float_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _float_median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _rank_from_values_desc(values: list[float], *, target_index: int) -> int:
    ordered = sorted(
        range(len(values)),
        key=lambda idx: (-values[idx], idx),
    )
    return ordered.index(int(target_index)) + 1


def _target_rank_from_tensor(
    values: torch.Tensor, *, target_ids: set[int]
) -> int | None:
    if int(values.numel()) == 0 or not target_ids:
        return None
    ordered = sorted(
        range(int(values.numel())),
        key=lambda idx: (-float(values[idx].item()), idx),
    )
    for rank, idx in enumerate(ordered, start=1):
        if idx in target_ids:
            return rank
    return None


def _answer_topk_hit(result: SupportWindowResult, *, k: int) -> bool:
    posterior = sorted(
        result.answer_posterior,
        key=lambda record: (-float(record.prob), int(record.answer_entity_id)),
    )
    top_ids = {int(record.answer_entity_id) for record in posterior[: int(k)]}
    gold_ids = {int(value) for value in result.gold_answer_entity_ids}
    return bool(top_ids & gold_ids)


def _build_graph_lists(
    batch: Any,
) -> tuple[list[list[tuple[int, int]]], list[list[int]]]:
    num_nodes = int(batch.num_nodes_total)
    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    edge_index = batch.edge_index.detach().to(dtype=torch.long).cpu()
    for edge_id in range(int(edge_index.size(1))):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        outgoing[src].append((edge_id, dst))
        incoming[dst].append(src)
    for neighbors in outgoing:
        neighbors.sort(key=lambda item: item[0])
    for neighbors in incoming:
        neighbors.sort()
    return outgoing, incoming


def _reverse_distances_to_answers(
    *,
    num_nodes: int,
    incoming: list[list[int]],
    answer_nodes: set[int],
) -> list[int | None]:
    distances: list[int | None] = [None] * int(num_nodes)
    frontier = list(sorted(answer_nodes))
    for node in frontier:
        distances[node] = 0
    cursor = 0
    while cursor < len(frontier):
        node = frontier[cursor]
        cursor += 1
        base_distance = distances[node]
        assert base_distance is not None
        for parent in incoming[node]:
            if distances[parent] is None:
                distances[parent] = base_distance + 1
                frontier.append(parent)
    return distances


def _resolve_shortest_path(batch: Any) -> dict[str, Any] | None:
    outgoing, incoming = _build_graph_lists(batch)
    start_nodes = sorted({int(value) for value in batch.q_local_indices.tolist()})
    answer_nodes = sorted({int(value) for value in batch.a_local_indices.tolist()})
    if not start_nodes or not answer_nodes:
        return None
    distances = _reverse_distances_to_answers(
        num_nodes=int(batch.num_nodes_total),
        incoming=incoming,
        answer_nodes=set(answer_nodes),
    )
    best_start: int | None = None
    best_hop: int | None = None
    for start in start_nodes:
        hop = distances[start]
        if hop is None:
            continue
        if (
            best_hop is None
            or hop < best_hop
            or (hop == best_hop and best_start is not None and start < best_start)
        ):
            best_hop = hop
            best_start = start
    if best_start is None or best_hop is None:
        return None
    path_nodes = [int(best_start)]
    path_edge_ids: list[int] = []
    current = int(best_start)
    remaining = int(best_hop)
    while remaining > 0:
        candidates = [
            (edge_id, dst)
            for edge_id, dst in outgoing[current]
            if distances[dst] is not None and distances[dst] == remaining - 1
        ]
        if not candidates:
            return None
        edge_id, dst = min(candidates, key=lambda item: (item[0], item[1]))
        path_edge_ids.append(int(edge_id))
        path_nodes.append(int(dst))
        current = int(dst)
        remaining -= 1
    return {
        "hop": int(best_hop),
        "start_node": int(best_start),
        "path_nodes": path_nodes,
        "path_edge_ids": path_edge_ids,
        "distances_to_answer": distances,
    }


def _single_agent_action_view(
    distribution: Any,
    move_log_probs: torch.Tensor,
) -> dict[str, Any]:
    mask = distribution.edge_agent_batch == 0
    is_stop_action = distribution.is_stop_action
    if is_stop_action is None:
        stop_mask = torch.zeros_like(distribution.edge_ids[mask], dtype=torch.bool)
    else:
        stop_mask = is_stop_action[mask].to(dtype=torch.bool)
    return {
        "edge_ids": distribution.edge_ids[mask],
        "target_nodes": distribution.target_nodes[mask],
        "log_probs": move_log_probs[mask],
        "stop_mask": stop_mask,
    }


def _edge_prob_mass(
    *,
    action_view: dict[str, Any],
    keep_edge_ids: set[int],
) -> float:
    if not keep_edge_ids:
        return 0.0
    edge_ids = action_view["edge_ids"]
    log_probs = action_view["log_probs"]
    stop_mask = action_view["stop_mask"]
    keep_positions = [
        idx
        for idx in range(int(edge_ids.numel()))
        if (not bool(stop_mask[idx].item()))
        and int(edge_ids[idx].item()) in keep_edge_ids
    ]
    if not keep_positions:
        return 0.0
    selected = log_probs.index_select(
        0, torch.tensor(keep_positions, device=log_probs.device, dtype=torch.long)
    )
    return float(selected.exp().sum().item())


def _action_rank(action_log_probs: torch.Tensor, *, selected_index: int) -> int:
    values = [float(value) for value in action_log_probs.detach().cpu().tolist()]
    return _rank_from_values_desc(values, target_index=int(selected_index))


def _compute_teacher_forced_shortest_path_stats(
    batch: Any, model: Any
) -> dict[str, Any]:
    graph_info = _resolve_shortest_path(batch)
    sample_id = str(batch.sample_ids[0])
    if graph_info is None:
        return {"sample_id": sample_id, "hop": None, "path_found": False}

    hop = int(graph_info["hop"])
    path_nodes = [int(node) for node in graph_info["path_nodes"]]
    path_edge_ids = [int(edge_id) for edge_id in graph_info["path_edge_ids"]]
    distances_to_answer = list(graph_info["distances_to_answer"])
    prepared_batch = model.policy.prepare_batch(batch)
    policy = model.policy
    base_policy = getattr(policy, "base_policy", policy)
    device = batch.edge_index.device
    max_steps = int(model.cfg.horizon_cfg.max_steps)

    root_distribution = policy.compute_root_action_distribution(prepared_batch)
    candidate_nodes = root_distribution.candidate_nodes_abs.to(dtype=torch.long)
    root_match = torch.nonzero(
        candidate_nodes == int(graph_info["start_node"]),
        as_tuple=False,
    ).view(-1)
    if int(root_match.numel()) != 1:
        raise ValueError(
            f"Expected exactly one root match for sample_id={sample_id}, got {root_match.tolist()}."
        )
    root_index = int(root_match.item())
    root_log_probs = root_distribution.log_probs.to(dtype=torch.float32)
    root_prob = float(root_log_probs[root_index].exp().item())
    root_rank = _rank_from_values_desc(
        [float(value) for value in root_log_probs.detach().cpu().tolist()],
        target_index=root_index,
    )

    flat_nodes = torch.arange(
        int(batch.num_nodes_total), device=device, dtype=torch.long
    )
    flat_num_steps = torch.zeros_like(flat_nodes)
    flat_done_mask = torch.zeros_like(flat_nodes, dtype=torch.bool)
    local_state_features = base_policy.build_local_state_features(
        prepared_batch,
        flat_nodes=flat_nodes,
        flat_num_steps=flat_num_steps,
        flat_done_mask=flat_done_mask,
    )
    graph_ids = prepared_batch.topology.graph_index_from_nodes(flat_nodes)
    node_scores = base_policy._compute_log_state_scores_from_flat_features(
        prepared_batch=prepared_batch,
        flat_state_features=local_state_features,
        graph_ids=graph_ids,
    )
    answer_rank_t0 = _target_rank_from_tensor(
        node_scores,
        target_ids={int(value) for value in batch.a_local_indices.tolist()},
    )
    bridge_rank_t0 = None
    bridge_score_gap = None
    if hop >= 2:
        bridge_node = int(path_nodes[1])
        bridge_rank_t0 = _target_rank_from_tensor(node_scores, target_ids={bridge_node})
        answer_scores = [
            float(node_scores[int(idx)].item())
            for idx in batch.a_local_indices.tolist()
        ]
        if answer_scores:
            bridge_score_gap = float(
                max(answer_scores) - float(node_scores[bridge_node].item())
            )

    current_nodes = torch.tensor(
        [[int(graph_info["start_node"])]],
        device=device,
        dtype=torch.long,
    )
    done_mask = torch.zeros_like(current_nodes, dtype=torch.bool)
    num_steps = torch.zeros_like(current_nodes, dtype=torch.long)
    path_token_ids = initialize_path_token_ids(
        start_nodes=current_nodes, max_steps=max_steps
    )
    control_state = policy.build_start_control_states(prepared_batch, current_nodes)
    step_records: list[dict[str, Any]] = []
    total_log_prob = float(root_log_probs[root_index].item())

    for step_idx, edge_id in enumerate(path_edge_ids, start=1):
        search_state = SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=current_nodes,
            done_mask=done_mask,
            num_steps=num_steps,
            path_token_ids=path_token_ids,
            control_state=control_state,
        )
        distribution = apply_forward_constraints(
            policy.compute_forward_distribution(prepared_batch, search_state),
            state=search_state,
            max_steps=max_steps,
        )
        move_log_probs, _, _ = policy.compute_move_log_probs(distribution)
        action_view = _single_agent_action_view(distribution, move_log_probs)
        selected_match = torch.nonzero(
            (~action_view["stop_mask"]) & (action_view["edge_ids"] == int(edge_id)),
            as_tuple=False,
        ).view(-1)
        if int(selected_match.numel()) != 1:
            raise ValueError(
                "Teacher-forced edge is missing under current policy state. "
                f"sample_id={sample_id} step={step_idx} edge_id={edge_id}."
            )
        selected_index = int(selected_match.item())
        selected_log_prob = float(action_view["log_probs"][selected_index].item())
        total_log_prob += selected_log_prob
        current_local = int(current_nodes.item())
        remaining_distance = distances_to_answer[current_local]
        if remaining_distance is None:
            raise ValueError(
                f"Current node has no answer distance for sample_id={sample_id} step={step_idx}."
            )
        shortest_consistent_edges = {
            int(candidate_edge_id.item())
            for candidate_edge_id, candidate_target, is_stop in zip(
                action_view["edge_ids"],
                action_view["target_nodes"],
                action_view["stop_mask"],
            )
            if (not bool(is_stop.item()))
            and distances_to_answer[int(candidate_target.item())] is not None
            and int(distances_to_answer[int(candidate_target.item())])
            == int(remaining_distance) - 1
        }
        chosen_target = int(action_view["target_nodes"][selected_index].item())
        chosen_relation = int(prepared_batch.topology.edge_type[edge_id].item())
        step_records.append(
            {
                "step": int(step_idx),
                "edge_id": int(edge_id),
                "prob": float(math.exp(selected_log_prob)),
                "rank": int(
                    _action_rank(
                        action_view["log_probs"], selected_index=selected_index
                    )
                ),
                "shortest_consistent_mass": _edge_prob_mass(
                    action_view=action_view,
                    keep_edge_ids=shortest_consistent_edges,
                ),
                "action_count": int(action_view["edge_ids"].numel()),
                "target_node": int(chosen_target),
            }
        )
        relation_tensor = torch.tensor(
            [[chosen_relation]], device=device, dtype=torch.long
        )
        target_tensor = torch.tensor([[chosen_target]], device=device, dtype=torch.long)
        path_token_ids = append_relation_and_node_tokens_inplace(
            path_token_ids=path_token_ids,
            num_steps=num_steps,
            relation_ids=relation_tensor,
            target_nodes=target_tensor,
        )
        control_state = policy.compute_next_control_states(
            prepared_batch,
            control_states=control_state,
            next_nodes=target_tensor,
            relation_ids=relation_tensor,
        )
        current_nodes = target_tensor
        num_steps = num_steps + 1

    terminal_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=current_nodes,
        done_mask=done_mask,
        num_steps=num_steps,
        path_token_ids=path_token_ids,
        control_state=control_state,
    )
    terminal_distribution = apply_forward_constraints(
        policy.compute_forward_distribution(prepared_batch, terminal_state),
        state=terminal_state,
        max_steps=max_steps,
    )
    terminal_log_probs, _, _ = policy.compute_move_log_probs(terminal_distribution)
    terminal_view = _single_agent_action_view(terminal_distribution, terminal_log_probs)
    stop_match = torch.nonzero(terminal_view["stop_mask"], as_tuple=False).view(-1)
    if int(stop_match.numel()) != 1:
        raise ValueError(
            f"Expected exactly one stop action for sample_id={sample_id}, got {stop_match.tolist()}."
        )
    stop_index = int(stop_match.item())
    stop_log_prob = float(terminal_view["log_probs"][stop_index].item())
    total_log_prob += stop_log_prob

    return {
        "sample_id": sample_id,
        "path_found": True,
        "hop": int(hop),
        "start_node": int(graph_info["start_node"]),
        "path_nodes": path_nodes,
        "path_edge_ids": path_edge_ids,
        "root_prob": root_prob,
        "root_rank": int(root_rank),
        "stop_prob": float(math.exp(stop_log_prob)),
        "stop_rank": int(
            _action_rank(terminal_view["log_probs"], selected_index=stop_index)
        ),
        "total_path_prob": float(math.exp(total_log_prob)),
        "total_path_log_prob": float(total_log_prob),
        "answer_rank_t0": answer_rank_t0,
        "bridge_rank_t0": bridge_rank_t0,
        "bridge_score_gap": bridge_score_gap,
        "step_records": step_records,
    }


def _summarize_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    stop_reasons = Counter(str(record["stop_reason"]) for record in records)
    return {
        "count": int(len(records)),
        "hit@1": _float_mean([float(record["hit@1"]) for record in records]),
        "hit@5": _float_mean([float(record["hit@5"]) for record in records]),
        "hit@10": _float_mean([float(record["hit@10"]) for record in records]),
        "gold_answer_mass_mean": _float_mean(
            [float(record["gold_answer_mass"]) for record in records]
        ),
        "gold_answer_mass_median": _float_median(
            [float(record["gold_answer_mass"]) for record in records]
        ),
        "covered_gold_answer_mass_mean": _float_mean(
            [float(record["covered_gold_answer_mass"]) for record in records]
        ),
        "remaining_mass_upper_mean": _float_mean(
            [float(record["remaining_mass_upper"]) for record in records]
        ),
        "probe_count_mean": _float_mean(
            [float(record["probe_count"]) for record in records]
        ),
        "coverage_certified_rate": _float_mean(
            [float(record["coverage_certified"]) for record in records]
        ),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
    }


def _summarize_teacher_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    step_ids = sorted(
        {
            int(step_record["step"])
            for record in records
            for step_record in record.get("step_records", [])
        }
    )
    summary = {
        "count": int(len(records)),
        "total_path_prob_mean": _float_mean(
            [float(record["total_path_prob"]) for record in records]
        ),
        "total_path_prob_median": _float_median(
            [float(record["total_path_prob"]) for record in records]
        ),
        "total_path_log_prob_mean": _float_mean(
            [float(record["total_path_log_prob"]) for record in records]
        ),
        "root_prob_mean": _float_mean(
            [float(record["root_prob"]) for record in records]
        ),
        "root_rank_mean": _float_mean(
            [float(record["root_rank"]) for record in records]
        ),
        "stop_prob_mean": _float_mean(
            [float(record["stop_prob"]) for record in records]
        ),
        "stop_rank_mean": _float_mean(
            [float(record["stop_rank"]) for record in records]
        ),
        "answer_rank_t0_mean": _float_mean(
            [
                float(record["answer_rank_t0"])
                for record in records
                if record["answer_rank_t0"] is not None
            ]
        ),
        "bridge_rank_t0_mean": _float_mean(
            [
                float(record["bridge_rank_t0"])
                for record in records
                if record["bridge_rank_t0"] is not None
            ]
        ),
        "bridge_score_gap_mean": _float_mean(
            [
                float(record["bridge_score_gap"])
                for record in records
                if record["bridge_score_gap"] is not None
            ]
        ),
        "steps": {},
    }
    for step_id in step_ids:
        step_records = [
            step_record
            for record in records
            for step_record in record.get("step_records", [])
            if int(step_record["step"]) == int(step_id)
        ]
        summary["steps"][str(step_id)] = {
            "count": int(len(step_records)),
            "prob_mean": _float_mean(
                [float(record["prob"]) for record in step_records]
            ),
            "rank_mean": _float_mean(
                [float(record["rank"]) for record in step_records]
            ),
            "shortest_consistent_mass_mean": _float_mean(
                [float(record["shortest_consistent_mass"]) for record in step_records]
            ),
            "action_count_mean": _float_mean(
                [float(record["action_count"]) for record in step_records]
            ),
        }
    return summary


def _prediction_record(result: SupportWindowResult) -> dict[str, Any]:
    return {
        "sample_id": str(result.sample_id),
        "hit@1": float(_answer_topk_hit(result, k=1)),
        "hit@5": float(_answer_topk_hit(result, k=5)),
        "hit@10": float(_answer_topk_hit(result, k=10)),
        "gold_answer_mass": float(result.gold_answer_mass),
        "covered_gold_answer_mass": float(result.covered_gold_answer_mass),
        "remaining_mass_upper": float(result.remaining_mass_upper),
        "probe_count": int(result.probe_count),
        "coverage_certified": bool(result.coverage_certified),
        "stop_reason": str(result.stop_reason),
        "unique_path_count": int(result.unique_path_count),
    }


def _run_prediction_pass(
    *,
    datamodule: Any,
    model: Any,
    device: torch.device,
    split: str,
    eval_cfg: SearchEvalConfig,
    sample_limit: int | None,
) -> dict[str, dict[str, Any]]:
    model.reconfigure_evaluation(eval_cfg=eval_cfg)
    model.eval()
    outputs: dict[str, dict[str, Any]] = {}
    datamodule.set_eval_split(split)
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()
    with torch.inference_mode():
        for batch_idx, raw_batch in enumerate(dataloader):
            if sample_limit is not None and batch_idx >= int(sample_limit):
                break
            batch = _prepare_batch(datamodule, raw_batch=raw_batch, device=device)
            batch_results = model.runtime_controller.predict_batch(batch=batch)
            for result in batch_results:
                outputs[str(result.sample_id)] = _prediction_record(result)
    return outputs


def _run_teacher_pass(
    *,
    datamodule: Any,
    model: Any,
    device: torch.device,
    split: str,
    sample_limit: int | None,
) -> dict[str, dict[str, Any]]:
    datamodule.set_eval_split(split)
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for batch_idx, raw_batch in enumerate(dataloader):
            if sample_limit is not None and batch_idx >= int(sample_limit):
                break
            batch = _prepare_batch(datamodule, raw_batch=raw_batch, device=device)
            diagnostics = _compute_teacher_forced_shortest_path_stats(
                batch=batch, model=model
            )
            outputs[str(diagnostics["sample_id"])] = diagnostics
    return outputs


def _build_sample_cases(
    *,
    teacher_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    flow_key = "flow_frontier_prune0_full"
    mc_key = "monte_carlo_256_rank_only"
    for sample_id, teacher in teacher_records.items():
        if not teacher.get("path_found"):
            continue
        hop = teacher.get("hop")
        if hop is None or int(hop) < 2:
            continue
        flow_record = prediction_records.get(flow_key, {}).get(sample_id)
        mc_record = prediction_records.get(mc_key, {}).get(sample_id)
        if flow_record is None or mc_record is None:
            continue
        answer_rank_t0 = teacher.get("answer_rank_t0")
        bridge_rank_t0 = teacher.get("bridge_rank_t0")
        if answer_rank_t0 is None or bridge_rank_t0 is None:
            continue
        cases.append(
            {
                "sample_id": sample_id,
                "hop": int(hop),
                "answer_rank_t0": int(answer_rank_t0),
                "bridge_rank_t0": int(bridge_rank_t0),
                "bridge_minus_answer_rank": int(bridge_rank_t0) - int(answer_rank_t0),
                "total_path_prob": float(teacher["total_path_prob"]),
                "step_records": teacher.get("step_records", []),
                "flow_frontier_prune0_hit10": float(flow_record["hit@10"]),
                "flow_frontier_prune0_gold_answer_mass": float(
                    flow_record["gold_answer_mass"]
                ),
                "monte_carlo_256_hit10": float(mc_record["hit@10"]),
                "monte_carlo_256_gold_answer_mass": float(
                    mc_record["gold_answer_mass"]
                ),
            }
        )
    cases.sort(
        key=lambda record: (
            -record["bridge_minus_answer_rank"],
            record["total_path_prob"],
            record["sample_id"],
        )
    )
    return cases[: int(limit)]


def _aggregate_by_hop(
    *,
    teacher_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    hop_counts = Counter()
    teacher_by_hop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in teacher_records.values():
        hop = record.get("hop")
        if hop is None:
            hop_counts["unreachable"] += 1
            continue
        hop_key = str(int(hop))
        hop_counts[hop_key] += 1
        if record.get("path_found"):
            teacher_by_hop[hop_key].append(record)

    predictions_by_experiment: dict[str, Any] = {}
    for experiment_name, records in prediction_records.items():
        by_hop: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample_id, record in records.items():
            teacher = teacher_records.get(sample_id)
            hop = None if teacher is None else teacher.get("hop")
            hop_key = "unreachable" if hop is None else str(int(hop))
            by_hop[hop_key].append(record)
        predictions_by_experiment[experiment_name] = {
            "overall": _summarize_prediction_records(list(records.values())),
            "by_hop": {
                hop_key: _summarize_prediction_records(hop_records)
                for hop_key, hop_records in sorted(
                    by_hop.items(),
                    key=lambda item: (item[0] == "unreachable", item[0]),
                )
            },
        }

    teacher_summary = {
        "overall": _summarize_teacher_records(
            [record for record in teacher_records.values() if record.get("path_found")]
        ),
        "by_hop": {
            hop_key: _summarize_teacher_records(records)
            for hop_key, records in sorted(
                teacher_by_hop.items(), key=lambda item: item[0]
            )
        },
    }
    return {
        "hop_distribution": dict(sorted(hop_counts.items(), key=lambda item: item[0])),
        "prediction_experiments": predictions_by_experiment,
        "teacher_forced_shortest_path": teacher_summary,
    }


def _build_eval_cfgs(base_eval_cfg: SearchEvalConfig) -> dict[str, SearchEvalConfig]:
    return {
        "monte_carlo_256_rank_only": replace(
            base_eval_cfg,
            report_profile="rank_only",
            answer_posterior_backend="monte_carlo",
            monte_carlo=replace(base_eval_cfg.monte_carlo, rollouts=256),
        ),
        "monte_carlo_4096_rank_only": replace(
            base_eval_cfg,
            report_profile="rank_only",
            answer_posterior_backend="monte_carlo",
            monte_carlo=replace(base_eval_cfg.monte_carlo, rollouts=4096),
        ),
        "flow_frontier_prune1e-3_full": replace(
            base_eval_cfg,
            report_profile="full",
            answer_posterior_backend="flow_frontier",
            flow_frontier=replace(
                base_eval_cfg.flow_frontier,
                prune_epsilon=1.0e-3,
            ),
        ),
        "flow_frontier_prune0_full": replace(
            base_eval_cfg,
            report_profile="full",
            answer_posterior_backend="flow_frontier",
            flow_frontier=replace(
                base_eval_cfg.flow_frontier,
                prune_epsilon=0.0,
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose validation hop behavior.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = _load_cfg(
        config_path=args.config_path,
        output_dir=args.output_path.parent,
        split=args.split,
        eval_batch_size=args.eval_batch_size,
    )
    datamodule, model = _instantiate_objects(cfg)
    datamodule.prepare_data()
    datamodule.setup("predict")
    _load_checkpoint(model, ckpt_path=str(args.ckpt_path))
    model = model.to(device)
    base_eval_cfg = model.cfg.eval_cfg
    eval_cfgs = _build_eval_cfgs(base_eval_cfg)

    teacher_records = _run_teacher_pass(
        datamodule=datamodule,
        model=model,
        device=device,
        split=args.split,
        sample_limit=args.sample_limit,
    )
    prediction_records = {
        experiment_name: _run_prediction_pass(
            datamodule=datamodule,
            model=model,
            device=device,
            split=args.split,
            eval_cfg=eval_cfg,
            sample_limit=args.sample_limit,
        )
        for experiment_name, eval_cfg in eval_cfgs.items()
    }
    aggregate = _aggregate_by_hop(
        teacher_records=teacher_records,
        prediction_records=prediction_records,
    )
    sample_cases = _build_sample_cases(
        teacher_records=teacher_records,
        prediction_records=prediction_records,
        limit=args.case_limit,
    )

    output = {
        "config_path": str(args.config_path),
        "ckpt_path": str(args.ckpt_path),
        "split": str(args.split),
        "device": str(device),
        "dataset_name": str(cfg.dataset.name),
        "sample_limit": args.sample_limit,
        "eval_cfgs": {
            name: {
                "report_profile": eval_cfg.report_profile,
                "answer_posterior_backend": eval_cfg.answer_posterior_backend,
                "flow_frontier": {
                    "prune_epsilon": float(eval_cfg.flow_frontier.prune_epsilon),
                    "max_expansions": int(eval_cfg.flow_frontier.max_expansions),
                    "max_frontier_size": int(eval_cfg.flow_frontier.max_frontier_size),
                },
                "monte_carlo": {
                    "rollouts": int(eval_cfg.monte_carlo.rollouts),
                    "confidence": float(eval_cfg.monte_carlo.confidence),
                },
            }
            for name, eval_cfg in eval_cfgs.items()
        },
        **aggregate,
        "sample_cases": sample_cases,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=True, indent=2))
    datamodule.teardown()


if __name__ == "__main__":
    main()
