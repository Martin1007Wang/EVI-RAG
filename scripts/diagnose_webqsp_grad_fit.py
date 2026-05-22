from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.factory import build_datamodule, build_model
from src.training.optimization import build_optimizer
from src.weaver.module import (
    graph_batch_size,
    terminal_flow_branch_gradient_metrics,
)
from src.weaver.transition import SRC_REPLAY


@dataclass(frozen=True)
class ParamSnapshot:
    name: str
    tensor: torch.Tensor
    group_index: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--out-dir", default="outputs/webqsp_grad_diagnosis")
    parser.add_argument("--max-fit-params", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "steps.jsonl"
    report_path = out_dir / "report.md"

    set_seed(int(args.seed))
    cfg = compose_cfg(data_dir=str(args.data_dir))
    dm = build_datamodule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    if dm.train_dataset is None:
        raise RuntimeError("train_dataset was not initialized.")

    samples = [dm.train_dataset[int(index)] for index in args.indices]
    sample_ids = [str(getattr(sample, "sample_id", "")) for sample in samples]
    batch = dm.collator(samples)

    model = build_model(cfg, dm.model_resources)
    model.runner.replay_schedule = None
    model.train()

    optimizer = build_optimizer(
        modules=(model.policy_feature_encoder, model.policy),
        cfg=model.optimization.optimizer,
    )
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    param_to_name = {id(param): name for name, param in named_params}
    tracked_names = choose_tracked_names(named_params, max_count=int(args.max_fit_params))

    metadata = {
        "indices": args.indices,
        "sample_ids": sample_ids,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "data_dir": str(args.data_dir),
        "device": "cpu",
        "replay_schedule": None,
        "train_num_rollouts": int(model.runner.train_num_rollouts),
        "expand_budget": int(model.runner.engine.expand_budget),
        "optimizer": {
            "type": type(optimizer).__name__,
            "lr": float(model.optimization.optimizer.lr),
            "weight_decay": float(model.optimization.optimizer.weight_decay),
            "betas": list(model.optimization.optimizer.betas),
            "no_decay_on_bias_and_norm": bool(model.optimization.optimizer.no_decay_on_bias_and_norm),
        },
        "gradient_clip_val": None if model.gradient_clip_val is None else float(model.gradient_clip_val),
        "batch": batch_summary(batch),
        "tracked_params": tracked_names,
    }

    records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
        for epoch in range(1, int(args.epochs) + 1):
            set_seed(int(args.seed) + epoch - 1)
            optimizer.zero_grad(set_to_none=True)

            before = snapshot_params(named_params, optimizer, tracked_names)
            output = model.compute_step(batch=batch)
            branch_grad_metrics = terminal_flow_branch_gradient_metrics(
                stop_flow_head=model.policy.stop_flow_head,
                edge_advantage_head=model.policy.edge_advantage_head,
                terminal_loss=output.terminal_branch_loss,
                expansion_loss=output.expansion_branch_loss,
            )
            output.loss.backward()

            raw_grad_stats = gradient_stats(named_params)
            raw_total_grad_norm = total_grad_norm(named_params)
            raw_grads = {
                name: param.grad.detach().clone()
                for name, param in named_params
                if name in tracked_names and param.grad is not None
            }

            clip_return = None
            if model.gradient_clip_val is not None and float(model.gradient_clip_val) > 0.0:
                clip_return = torch.nn.utils.clip_grad_norm_(
                    [param for _, param in named_params],
                    max_norm=float(model.gradient_clip_val),
                    norm_type=2.0,
                )
            clipped_total_grad_norm = total_grad_norm(named_params)
            clipped_grads = {
                name: param.grad.detach().clone()
                for name, param in named_params
                if name in tracked_names and param.grad is not None
            }

            expected_after = adamw_expected_after(before, clipped_grads, optimizer, param_to_name)
            optimizer.step()
            fit = update_fit(before, expected_after, named_params, tracked_names)

            rollout_metrics = scalarize(output.metrics)
            record = {
                "type": "step",
                "epoch": epoch,
                "loss": float(output.loss.detach().item()),
                "expansion_branch_loss": float(output.expansion_branch_loss.detach().item()),
                "terminal_branch_loss": float(output.terminal_branch_loss.detach().item()),
                "batch_graphs": graph_batch_size(batch),
                "metrics": rollout_metrics,
                "branch_grad_metrics": scalarize(branch_grad_metrics),
                "raw_total_grad_norm": float(raw_total_grad_norm),
                "clip_return_norm": None if clip_return is None else float(clip_return.detach().item()),
                "clipped_total_grad_norm": float(clipped_total_grad_norm),
                "clip_scale_estimate": (
                    1.0
                    if raw_total_grad_norm <= 0.0 or model.gradient_clip_val is None
                    else min(1.0, float(model.gradient_clip_val) / (float(raw_total_grad_norm) + 1.0e-6))
                ),
                "grad_stats": raw_grad_stats,
                "tracked_raw_grads": tensor_summaries(raw_grads),
                "tracked_clipped_grads": tensor_summaries(clipped_grads),
                "adamw_fit": fit,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"epoch={epoch} loss={record['loss']:.6f} "
                f"raw_grad={record['raw_total_grad_norm']:.6f} "
                f"clip_grad={record['clipped_total_grad_norm']:.6f} "
                f"max_fit_err={fit['max_abs_error']:.3e}",
                flush=True,
            )

    report_path.write_text(render_report(metadata, records), encoding="utf-8")
    print(f"wrote {jsonl_path}")
    print(f"wrote {report_path}")


def compose_cfg(*, data_dir: str):
    config_dir = str((Path.cwd() / "configs").resolve())
    overrides = [
        "experiment=train/webqsp",
        "logger=none",
        "trainer=cpu",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        "datamodule.train_shuffle=false",
        "model.runner.replay_schedule=null",
        "model.optimization.scheduler=null",
        f"paths.data_dir={data_dir}",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="train", overrides=overrides)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def batch_summary(batch: Any) -> dict[str, Any]:
    return {
        "graphs": int(batch.num_graphs_total),
        "nodes": int(batch.num_nodes_total),
        "edges": int(batch.num_edges_total),
        "anchors": int(batch.anchor_node_ids.numel()),
        "targets": int(batch.target_node_ids.numel()),
        "reachable_targets": int(batch.reachable_target_node_ids.numel()),
    }


def choose_tracked_names(
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    max_count: int,
) -> list[str]:
    preferred = [
        "policy.stop_flow_head.2.weight",
        "policy.stop_flow_head.2.bias",
        "policy.edge_advantage_head.2.weight",
        "policy.edge_advantage_head.2.bias",
        "policy.stop_flow_head.0.weight",
        "policy.edge_advantage_head.0.weight",
        "policy.budget_embedding.weight",
        "policy.state_encoder.fuse.0.weight",
        "policy.state_encoder.edge_encoder.edge_project.0.weight",
        "policy_feature_encoder.query_project.0.weight",
        "policy_feature_encoder.node_project.0.weight",
        "policy_feature_encoder.relation_project.0.weight",
    ]
    existing = {name for name, _ in named_params}
    chosen = [name for name in preferred if name in existing]
    if len(chosen) < max_count:
        for name, _ in named_params:
            if name not in chosen:
                chosen.append(name)
            if len(chosen) >= max_count:
                break
    return chosen[:max_count]


def snapshot_params(
    named_params: list[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    tracked_names: list[str],
) -> dict[str, ParamSnapshot]:
    group_by_param: dict[int, tuple[int, dict[str, Any]]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for param in group["params"]:
            group_by_param[id(param)] = (group_index, group)

    out: dict[str, ParamSnapshot] = {}
    tracked = set(tracked_names)
    for name, param in named_params:
        if name not in tracked:
            continue
        group_index, group = group_by_param[id(param)]
        beta1, beta2 = group["betas"]
        out[name] = ParamSnapshot(
            name=name,
            tensor=param.detach().clone(),
            group_index=int(group_index),
            lr=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(group["eps"]),
        )
    return out


def adamw_expected_after(
    before: dict[str, ParamSnapshot],
    clipped_grads: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    param_to_name: dict[int, str],
) -> dict[str, torch.Tensor]:
    expected: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        eps = float(group["eps"])
        for param in group["params"]:
            name = param_to_name.get(id(param))
            if name is None or name not in before or name not in clipped_grads:
                continue
            grad = clipped_grads[name]
            state = optimizer.state[param]
            step = int(state.get("step", 0)) + 1
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None:
                exp_avg = torch.zeros_like(param)
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(param)
            next_exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
            next_exp_avg_sq = exp_avg_sq * beta2 + grad.square() * (1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = next_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            denom = denom.add(eps)
            decayed = before[name].tensor * (1.0 - lr * weight_decay)
            expected[name] = decayed.addcdiv(next_exp_avg, denom, value=-(lr / bias_correction1))
    return expected


def update_fit(
    before: dict[str, ParamSnapshot],
    expected_after: dict[str, torch.Tensor],
    named_params: list[tuple[str, torch.nn.Parameter]],
    tracked_names: list[str],
) -> dict[str, Any]:
    current = {name: param.detach() for name, param in named_params if name in tracked_names}
    per_param: dict[str, Any] = {}
    max_abs_error = 0.0
    max_rel_error = 0.0
    for name in tracked_names:
        if name not in before or name not in expected_after or name not in current:
            continue
        actual_delta = current[name] - before[name].tensor
        expected_delta = expected_after[name] - before[name].tensor
        err = current[name] - expected_after[name]
        denom = expected_delta.detach().abs().max().item()
        abs_err = float(err.detach().abs().max().item())
        rel_err = abs_err / max(denom, 1.0e-12)
        max_abs_error = max(max_abs_error, abs_err)
        max_rel_error = max(max_rel_error, rel_err)
        per_param[name] = {
            "actual_delta_norm": float(actual_delta.float().norm().item()),
            "expected_delta_norm": float(expected_delta.float().norm().item()),
            "delta_cosine": cosine(actual_delta, expected_delta),
            "max_abs_error": abs_err,
            "max_rel_error": float(rel_err),
        }
    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "per_param": per_param,
    }


def gradient_stats(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    modules: dict[str, dict[str, float]] = defaultdict(lambda: {"sq": 0.0, "abs_sum": 0.0, "numel": 0.0, "zero": 0.0})
    params: dict[str, Any] = {}
    for name, param in named_params:
        grad = param.grad
        if grad is None:
            params[name] = {"norm": 0.0, "mean_abs": 0.0, "zero_fraction": 1.0, "numel": int(param.numel())}
            continue
        g = grad.detach().float()
        numel = int(g.numel())
        norm = float(g.norm().item())
        mean_abs = float(g.abs().mean().item()) if numel else 0.0
        zero_fraction = float(g.eq(0).sum().item() / max(numel, 1))
        params[name] = {
            "norm": norm,
            "mean_abs": mean_abs,
            "zero_fraction": zero_fraction,
            "numel": numel,
        }
        module = module_key(name)
        modules[module]["sq"] += norm * norm
        modules[module]["abs_sum"] += mean_abs * numel
        modules[module]["numel"] += numel
        modules[module]["zero"] += zero_fraction * numel
    module_out = {}
    for name, values in modules.items():
        numel = max(values["numel"], 1.0)
        module_out[name] = {
            "norm": math.sqrt(values["sq"]),
            "mean_abs": values["abs_sum"] / numel,
            "zero_fraction": values["zero"] / numel,
            "numel": int(values["numel"]),
        }
    top_params = sorted(
        params.items(),
        key=lambda item: float(item[1]["norm"]),
        reverse=True,
    )[:20]
    return {
        "modules": module_out,
        "top_params_by_norm": dict(top_params),
    }


def module_key(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "policy" and len(parts) >= 2:
        return ".".join(parts[:2])
    if parts[0] == "policy_feature_encoder":
        return parts[0]
    return parts[0]


def total_grad_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    for _, param in named_params:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().float().norm().item())
        total += norm * norm
    return math.sqrt(total)


def tensor_summaries(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: tensor_summary(tensor) for name, tensor in values.items()}


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int]:
    t = tensor.detach().float().flatten()
    if t.numel() == 0:
        return {"numel": 0, "norm": 0.0, "mean": 0.0, "mean_abs": 0.0, "min": 0.0, "max": 0.0}
    return {
        "numel": int(t.numel()),
        "norm": float(t.norm().item()),
        "mean": float(t.mean().item()),
        "mean_abs": float(t.abs().mean().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    denom = float(af.norm().item() * bf.norm().item())
    if denom <= 0.0:
        return 0.0
    return float(torch.dot(af, bf).item() / denom)


def scalarize(values: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().float().item())
        else:
            out[key] = float(value)
    return out


def render_report(metadata: dict[str, Any], records: list[dict[str, Any]]) -> str:
    losses = [record["loss"] for record in records]
    raw_norms = [record["raw_total_grad_norm"] for record in records]
    clipped_norms = [record["clipped_total_grad_norm"] for record in records]
    fit_errors = [record["adamw_fit"]["max_abs_error"] for record in records]
    replay_segments = [record["metrics"].get("source_replay_num_segments", 0.0) for record in records]
    policy_segments = [record["metrics"].get("source_policy_num_segments", 0.0) for record in records]
    terminal_fracs = [record["metrics"].get("subtb_terminal_fraction", 0.0) for record in records]
    residual_abs = [record["metrics"].get("subtb_residual_abs_mean", 0.0) for record in records]

    lines = [
        "# WebQSP 4-Sample Gradient-Fit Diagnosis",
        "",
        "## Setup",
        f"- indices: {metadata['indices']}",
        f"- sample_ids: {metadata['sample_ids']}",
        f"- batch: {metadata['batch']}",
        f"- epochs/steps: {metadata['epochs']}",
        f"- replay: disabled by `model.runner.replay_schedule = None`; replay segments observed: {replay_segments}",
        f"- train_num_rollouts: {metadata['train_num_rollouts']}, expand_budget: {metadata['expand_budget']}",
        f"- optimizer: {metadata['optimizer']}",
        f"- gradient_clip_val: {metadata['gradient_clip_val']}",
        "",
        "## Step Summary",
        "| epoch | loss | residual_abs | policy_segments | replay_segments | terminal_frac | raw_grad_norm | clipped_grad_norm | adamw_max_abs_err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        epoch = record["epoch"]
        lines.append(
            f"| {epoch} | {record['loss']:.6f} | "
            f"{record['metrics'].get('subtb_residual_abs_mean', 0.0):.6f} | "
            f"{record['metrics'].get('source_policy_num_segments', 0.0):.0f} | "
            f"{record['metrics'].get('source_replay_num_segments', 0.0):.0f} | "
            f"{record['metrics'].get('subtb_terminal_fraction', 0.0):.3f} | "
            f"{record['raw_total_grad_norm']:.6f} | "
            f"{record['clipped_total_grad_norm']:.6f} | "
            f"{record['adamw_fit']['max_abs_error']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Experiment-Supported Conclusions",
            f"1. Parameter movement is consistent with the clipped AdamW gradient update. Across {len(records)} steps, the largest tracked-parameter prediction error was {max(fit_errors):.3e}.",
            f"2. Replay was not used in this run. `source_replay_num_segments` stayed at {sorted(set(replay_segments))}, while policy segments ranged from {min(policy_segments):.0f} to {max(policy_segments):.0f}.",
            f"3. Gradients were large enough to trigger clipping on every step if raw norm exceeded 1.0. Raw total gradient norm ranged {min(raw_norms):.6f} to {max(raw_norms):.6f}; clipped norm ranged {min(clipped_norms):.6f} to {max(clipped_norms):.6f}.",
            f"4. The optimization target is noisy under policy-only rollout. Loss values were {format_series(losses)}, and mean absolute SubTB residuals were {format_series(residual_abs)}.",
            f"5. The terminal/nonterminal mix changed with sampled trajectories. Terminal segment fraction ranged {min(terminal_fracs):.3f} to {max(terminal_fracs):.3f}, so each epoch optimizes a different empirical objective even on the same 4 questions.",
            "",
            "## Gradient Detail",
        ]
    )
    for record in records:
        lines.append(f"### Epoch {record['epoch']}")
        lines.append(f"- branch_grad_metrics: {record['branch_grad_metrics']}")
        lines.append(f"- module_grad_stats: {record['grad_stats']['modules']}")
        top = list(record["grad_stats"]["top_params_by_norm"].items())[:5]
        lines.append(f"- top5_param_grad_norms: {top}")
        tracked = {
            name: summary
            for name, summary in list(record["tracked_raw_grads"].items())[:6]
        }
        lines.append(f"- tracked_raw_grad_summaries: {tracked}")
        fit = {
            name: summary
            for name, summary in list(record["adamw_fit"]["per_param"].items())[:6]
        }
        lines.append(f"- tracked_adamw_fit: {fit}")

    lines.extend(
        [
            "",
            "## Raw Artifacts",
            f"- step-level JSONL: `{Path(metadata.get('out_dir', 'outputs/webqsp_grad_diagnosis')) / 'steps.jsonl'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def format_series(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


if __name__ == "__main__":
    main()
