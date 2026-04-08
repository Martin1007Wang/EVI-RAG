from __future__ import annotations

import json
import random
from pathlib import Path
import sys
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.collate import RetrievalCollator
from src.eval.hit_graph_reward import build_teacher_hit_graph, evaluate_hit_graph_reward
from src.models.rollout import RolloutState
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _select_dataset(datamodule: Any, split_name: str) -> Any:
    split_name = str(split_name)
    split_to_dataset = {
        str(
            datamodule.dataset_cfg.get("train_split", "train")
        ): datamodule.train_dataset,
        str(
            datamodule.dataset_cfg.get("val_split", "validation")
        ): datamodule.val_dataset,
        str(datamodule.dataset_cfg.get("eval_split", "test")): datamodule.test_dataset,
        str(
            datamodule.dataset_cfg.get(
                "predict_split", datamodule.dataset_cfg.get("eval_split", "test")
            )
        ): datamodule.predict_dataset,
    }
    dataset = split_to_dataset.get(split_name)
    if dataset is None:
        available = sorted(
            name for name, value in split_to_dataset.items() if value is not None
        )
        raise ValueError(
            f"Split {split_name!r} is unavailable. Available loaded splits: {available}."
        )
    return dataset


def _resolve_device(raw_device: str) -> torch.device:
    requested = str(raw_device).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _build_root_diagnostics(model: Any, batch: Any) -> dict[str, Any]:
    rollout_state = RolloutState.initialize(batch)
    step_output = model.policy(batch, rollout_state.snapshot())

    edge_src = batch.edge_index[0]
    edge_dst = batch.edge_index[1]
    valid_edges_mask = (
        rollout_state.active_nodes[edge_src] | rollout_state.active_nodes[edge_dst]
    ) & ~rollout_state.active_edges
    relation_prior = model.policy._build_relation_prior_logits(batch)
    node_query_scores = model.policy._build_node_query_scores(batch)
    type_features = model.policy._build_type_features(
        node_query_scores=node_query_scores,
        relation_prior_logits=relation_prior,
        valid_edges_mask=valid_edges_mask,
        active_nodes=rollout_state.active_nodes,
        active_edges=rollout_state.active_edges,
        edge_index=batch.edge_index,
        node_batch_index=batch.batch,
        edge_batch_index=batch.edge_batch,
        num_graphs=batch.num_graphs,
    )
    type_probs = torch.softmax(step_output.action_logits["type_logits"], dim=-1)

    top_k = min(10, int(valid_edges_mask.sum().item()))
    top_edges: list[dict[str, Any]] = []
    if top_k > 0:
        candidate_ids = torch.nonzero(valid_edges_mask, as_tuple=False).view(-1)
        candidate_scores = relation_prior.index_select(0, candidate_ids)
        order = torch.argsort(candidate_scores, descending=True)[:top_k]
        ranked_edge_ids = candidate_ids.index_select(0, order)
        ranked_scores = candidate_scores.index_select(0, order)
        for edge_id, score in zip(ranked_edge_ids.tolist(), ranked_scores.tolist()):
            top_edges.append({"edge_id": int(edge_id), "prior_score": float(score)})

    return {
        "num_valid_root_edges": int(valid_edges_mask.sum().item()),
        "type_logits": step_output.action_logits["type_logits"].detach().cpu().tolist(),
        "type_probs": type_probs.detach().cpu().tolist(),
        "type_features": type_features.detach().cpu().tolist(),
        "max_relation_prior": float(type_features[:, 0].max().item())
        if type_features.numel()
        else 0.0,
        "isfinite": {
            "relation_prior": bool(torch.isfinite(relation_prior).all().item()),
            "type_logits": bool(
                torch.isfinite(step_output.action_logits["type_logits"]).all().item()
            ),
            "expand_edge_logits": bool(
                torch.isfinite(step_output.action_logits["expand_edge_logits"])
                .all()
                .item()
            ),
            "subgraph_h": bool(torch.isfinite(step_output.subgraph_h).all().item()),
            "question_h": bool(torch.isfinite(step_output.question_h).all().item()),
        },
        "top_root_prior_edges": top_edges,
    }


def _teacher_root_rank(
    model: Any, batch: Any, teacher_state: RolloutState
) -> dict[str, Any]:
    relation_prior = model.policy._build_relation_prior_logits(batch)
    edge_src = batch.edge_index[0]
    edge_dst = batch.edge_index[1]
    root_edges = teacher_state.root_active_edges
    valid_root_edges = (
        batch.is_anchor_mask[edge_src] | batch.is_anchor_mask[edge_dst]
    ) & ~root_edges
    candidate_ids = torch.nonzero(valid_root_edges, as_tuple=False).view(-1)
    teacher_added_edges = torch.nonzero(
        teacher_state.active_edges & ~teacher_state.root_active_edges,
        as_tuple=False,
    ).view(-1)
    if candidate_ids.numel() == 0 or teacher_added_edges.numel() == 0:
        return {
            "teacher_edge_ids": teacher_added_edges.detach().cpu().tolist(),
            "teacher_root_rank": None,
            "teacher_root_score": None,
        }

    candidate_scores = relation_prior.index_select(0, candidate_ids)
    ordered = candidate_ids.index_select(
        0, torch.argsort(candidate_scores, descending=True)
    )
    teacher_set = set(teacher_added_edges.tolist())
    rank = next(
        (
            i
            for i, edge_id in enumerate(ordered.tolist(), start=1)
            if edge_id in teacher_set
        ),
        None,
    )
    return {
        "teacher_edge_ids": teacher_added_edges.detach().cpu().tolist(),
        "teacher_root_rank": rank,
        "teacher_root_score": float(relation_prior[teacher_added_edges[0]].item()),
    }


@torch.no_grad()
def _teacher_path_diagnostics(
    model: Any,
    batch: Any,
    *,
    path_mode: str,
    stop_on_first_hit: bool,
) -> list[dict[str, Any]]:
    teacher_status, teacher_state = build_teacher_hit_graph(
        batch,
        path_mode=path_mode,
        stop_on_first_hit=stop_on_first_hit,
    )
    if teacher_status not in {"ok", "root_hit"}:
        return []

    rollout_state = RolloutState.initialize(batch)
    positive_edge_ids = torch.nonzero(
        teacher_state.active_edges & ~teacher_state.root_active_edges,
        as_tuple=False,
    ).view(-1)
    diagnostics: list[dict[str, Any]] = []

    def _record_state(step_idx: int, chosen_edge_id: int | None = None) -> None:
        step_output = model.policy(batch, rollout_state.snapshot())
        type_probs = torch.softmax(step_output.action_logits["type_logits"], dim=-1)
        valid_edges_mask = (
            rollout_state.active_nodes[batch.edge_index[0]]
            | rollout_state.active_nodes[batch.edge_index[1]]
        ) & ~rollout_state.active_edges
        candidate_ids = torch.nonzero(valid_edges_mask, as_tuple=False).view(-1)
        edge_probs = None
        teacher_rank = None
        teacher_prob = None
        if candidate_ids.numel() > 0:
            candidate_logits = step_output.action_logits[
                "expand_edge_logits"
            ].index_select(0, candidate_ids)
            edge_probs = torch.softmax(candidate_logits, dim=0)
            if chosen_edge_id is not None:
                local_match = torch.nonzero(
                    candidate_ids == chosen_edge_id, as_tuple=False
                ).view(-1)
                if local_match.numel() == 1:
                    local_idx = int(local_match.item())
                    teacher_prob = float(edge_probs[local_idx].item())
                    ranking = torch.argsort(edge_probs, descending=True)
                    teacher_rank = int(
                        torch.nonzero(ranking == local_idx, as_tuple=False).item() + 1
                    )

        diagnostics.append(
            {
                "step": step_idx,
                "type_probs": type_probs.detach().cpu().tolist(),
                "active_nodes": int(rollout_state.active_nodes.sum().item()),
                "active_edges": int(rollout_state.active_edges.sum().item()),
                "chosen_teacher_edge": chosen_edge_id,
                "teacher_edge_rank": teacher_rank,
                "teacher_edge_prob": teacher_prob,
            }
        )

    _record_state(
        step_idx=0,
        chosen_edge_id=int(positive_edge_ids[0].item())
        if positive_edge_ids.numel()
        else None,
    )
    for step_idx, edge_id in enumerate(positive_edge_ids.tolist(), start=1):
        rollout_state.apply_expansion(
            chosen_edges=torch.tensor([edge_id], device=batch.edge_index.device),
            src=batch.edge_index[0],
            dst=batch.edge_index[1],
        )
        _record_state(step_idx=step_idx, chosen_edge_id=None)

    return diagnostics


def _run_single_training_step(
    model: Any, batch: Any, *, temperature: float, global_step: int
) -> dict[str, float]:
    rollouts = model.rollout_engine.run_exploration(
        policy=model.policy,
        base_graph=batch,
        reward_model=model.reward_model,
        num_rollouts=model.num_rollout,
        temperature=temperature,
        collect_terminal_state=False,
    )
    loss_outputs = [model.loss_fn(rollout) for rollout in rollouts]
    tb_loss = torch.stack([output.tb_loss for output in loss_outputs]).mean()
    curriculum = model._curriculum_weights(global_step)
    teacher_loss = tb_loss.new_zeros(())
    teacher_type_loss = tb_loss.new_zeros(())
    teacher_edge_loss = tb_loss.new_zeros(())
    teacher_states = tb_loss.new_zeros(())
    if curriculum["teacher_scale"] > 0.0:
        teacher_output = model.teacher_warmstart(policy=model.policy, batch=batch)
        teacher_type_loss = teacher_output.type_loss
        teacher_edge_loss = teacher_output.edge_loss
        teacher_loss = teacher_type_loss + teacher_edge_loss
        teacher_states = teacher_output.supervised_states
    total_loss = (
        tb_loss * curriculum["tb_weight"]
        + teacher_edge_loss * curriculum["edge_teacher_weight"]
        + teacher_type_loss * curriculum["stop_teacher_weight"]
    )
    residual_abs = torch.stack([output.residual_abs for output in loss_outputs]).mean()
    residual_variance = torch.stack(
        [output.residual_variance for output in loss_outputs]
    ).mean()
    log_z_mean = torch.stack([output.log_z_mean for output in loss_outputs]).mean()
    log_reward_mean = torch.stack(
        [rollout.terminal_log_rewards.mean() for rollout in rollouts]
    ).mean()
    trajectory_length_mean = torch.stack(
        [output.trajectory_length_mean for output in loss_outputs]
    ).mean()
    return {
        "loss_tensor": total_loss,
        "tb_loss": float(tb_loss.detach().cpu()),
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "teacher_type_loss": float(teacher_type_loss.detach().cpu()),
        "teacher_edge_loss": float(teacher_edge_loss.detach().cpu()),
        "teacher_scale": float(curriculum["teacher_scale"]),
        "tb_weight": float(curriculum["tb_weight"]),
        "teacher_stop_weight": float(curriculum["stop_teacher_weight"]),
        "teacher_edge_weight": float(curriculum["edge_teacher_weight"]),
        "teacher_states": float(teacher_states.detach().cpu()),
        "residual_abs": float(residual_abs.detach().cpu()),
        "residual_variance": float(residual_variance.detach().cpu()),
        "log_z_mean": float(log_z_mean.detach().cpu()),
        "log_reward_mean": float(log_reward_mean.detach().cpu()),
        "trajectory_length_mean": float(trajectory_length_mean.detach().cpu()),
    }


@torch.no_grad()
def _evaluate_single_graph(model: Any, batch: Any) -> dict[str, Any]:
    results = model.evaluate_subgraph_retrieval(batch)
    flat_results: dict[str, float] = {}
    for group_name, metrics in results.items():
        for metric_name, value in metrics.items():
            flat_results[f"{group_name}/{metric_name}"] = float(value)
    return flat_results


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage=None)
    if datamodule._shared_resources is None:
        raise RuntimeError("Data resources were not initialized by datamodule.setup().")

    dataset = _select_dataset(datamodule, str(cfg.split))
    dataset_index = int(cfg.dataset_index)
    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError(
            f"dataset_index={dataset_index} is out of range for split {cfg.split!r} with size {len(dataset)}."
        )

    collator = RetrievalCollator(datamodule._shared_resources)
    raw_sample = dataset[dataset_index]
    batch = collator([raw_sample])

    device = _resolve_device(str(cfg.device))
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)
    batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.overfit_lr),
        weight_decay=float(cfg.overfit_weight_decay),
    )

    teacher_status, teacher_state = build_teacher_hit_graph(
        batch,
        path_mode=str(cfg.path_mode),
        stop_on_first_hit=bool(cfg.stop_on_first_hit),
    )
    hit_graph = evaluate_hit_graph_reward(
        batch,
        reward_model=model.reward_model,
        path_mode=str(cfg.path_mode),
        stop_on_first_hit=bool(cfg.stop_on_first_hit),
    )
    root_before = _build_root_diagnostics(model, batch)
    eval_before = _evaluate_single_graph(model, batch)

    history: list[dict[str, Any]] = []
    overfit_steps = int(cfg.overfit_steps)
    eval_every = max(int(cfg.eval_every), 1)
    for step in range(overfit_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_stats = _run_single_training_step(
            model,
            batch,
            temperature=(
                1.0 if bool(model.strict_on_policy) else float(cfg.overfit_temperature)
            ),
            global_step=step,
        )
        loss = train_stats.pop("loss_tensor")
        loss.backward()
        grad_sq_sum = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_sq_sum += float(param.grad.detach().pow(2).sum().cpu())
        optimizer.step()

        record: dict[str, Any] = {
            "step": step + 1,
            **train_stats,
            "grad_norm": grad_sq_sum**0.5,
        }
        if step == 0 or (step + 1) % eval_every == 0 or step + 1 == overfit_steps:
            model.eval()
            record["eval"] = _evaluate_single_graph(model, batch)
            record["root"] = _build_root_diagnostics(model, batch)
        history.append(record)

    summary = {
        "sample_id": getattr(raw_sample, "sample_id", None),
        "split": str(cfg.split),
        "dataset_index": dataset_index,
        "device": str(device),
        "graph": {
            "num_nodes": int(batch.num_nodes),
            "num_edges": int(batch.edge_index.size(1)),
            "num_anchors": int(batch.is_anchor_mask.sum().item()),
            "num_targets": int(batch.is_target_mask.sum().item()),
        },
        "hit_graph": {
            "status": hit_graph.status,
            "log_reward": hit_graph.log_reward,
            "recall": hit_graph.recall,
            "added_edges": hit_graph.added_edges,
            **_teacher_root_rank(model, batch, teacher_state),
            "teacher_build_status": teacher_status,
        },
        "teacher_path": _teacher_path_diagnostics(
            model,
            batch,
            path_mode=str(cfg.path_mode),
            stop_on_first_hit=bool(cfg.stop_on_first_hit),
        ),
        "before": {
            "root": root_before,
            "eval": eval_before,
        },
        "after": history[-1] if history else None,
        "history": history,
    }

    datamodule.teardown(stage=None)
    return summary


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="debug_single_graph_fit.yaml",
)
def main(cfg: DictConfig) -> None:
    summary = run_experiment(cfg)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "single_graph_fit_debug.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Single-graph debug summary written to %s", output_path)
    log.info(
        "Sample %s summary: %s",
        summary.get("sample_id"),
        json.dumps(summary["after"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
