from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.runs.common import compose_config  # noqa: E402
from src.utils.cuda_memory import (  # noqa: E402
    format_cuda_bytes,
    get_cuda_memory_records,
    profile_cuda_memory,
    reset_cuda_memory_records,
)
from src.utils.entrypoint_utils import _strip_instantiate_metadata  # noqa: E402
from src.utils.precision_utils import normalize_precision  # noqa: E402


@dataclass
class _FakeTrainerState:
    fn: str


@dataclass
class _FakeStrategy:
    root_device: torch.device


@dataclass
class _FakeTrainer:
    precision: Any
    state: _FakeTrainerState
    strategy: _FakeStrategy
    lightning_module: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile per-stage CUDA memory for one RankFlow train/val batch."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "val", "both"),
        default="both",
        help="Which path to profile.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to profile.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file to write the raw profiling result.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. Repeat for multiple overrides.",
    )
    return parser.parse_args()


def _compose_train_cfg(overrides: list[str]) -> DictConfig:
    base_overrides = [
        "extras.enforce_tags=false",
        "extras.print_config=false",
        "logger=none",
        "run.test=false",
        "data.num_workers=0",
        "data.eval_num_workers=0",
        "data.persistent_workers=false",
        "data.eval_persistent_workers=false",
    ]
    return compose_config(
        config_name="train.yaml",
        overrides=[*base_overrides, *overrides],
    )


def _instantiate_objects(cfg: DictConfig) -> tuple[Any, Any]:
    data_cfg = _strip_instantiate_metadata(cfg.data)
    model_cfg = _strip_instantiate_metadata(cfg.model)
    datamodule = hydra.utils.instantiate(data_cfg)
    model = hydra.utils.instantiate(model_cfg)
    return datamodule, model


def _attach_fake_trainer(
    *,
    datamodule: Any,
    model: Any,
    device: torch.device,
    precision: Any,
    fn_name: str,
) -> _FakeTrainer:
    trainer = _FakeTrainer(
        precision=precision,
        state=_FakeTrainerState(fn=fn_name),
        strategy=_FakeStrategy(root_device=device),
        lightning_module=model,
    )
    datamodule.trainer = trainer
    return trainer


@contextmanager
def _autocast_context(precision: object, device: torch.device) -> Iterator[None]:
    normalized = normalize_precision(precision)
    if device.type != "cuda":
        yield
        return
    if normalized == "bf16-mixed":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    if normalized == "16-mixed":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return
    with nullcontext():
        yield


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _materialize_runtime_batch(
    *,
    dataloader: Any,
    datamodule: Any,
    device: torch.device,
    stage_label: str,
) -> Any:
    iterator = iter(dataloader)
    with profile_cuda_memory(f"{stage_label}.next_batch", device=device):
        batch = next(iterator)
    with profile_cuda_memory(f"{stage_label}.before_transfer", device=device):
        batch = datamodule.on_before_batch_transfer(batch, 0)
    with profile_cuda_memory(f"{stage_label}.transfer_to_device", device=device):
        batch = datamodule.transfer_batch_to_device(batch, device, 0)
    with profile_cuda_memory(f"{stage_label}.after_transfer", device=device):
        batch = datamodule.on_after_batch_transfer(batch, 0)
    return batch


def _top_stage_summary(
    records: list[dict[str, Any]], *, limit: int = 20
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            int(record.get("peak_delta_allocated_bytes") or 0),
            int(record.get("delta_allocated_bytes") or 0),
        ),
        reverse=True,
    )
    summary: list[dict[str, Any]] = []
    for record in ranked[:limit]:
        summary.append(
            {
                "label": record.get("label"),
                "status": record.get("status"),
                "extra": record.get("extra"),
                "elapsed_s": round(float(record.get("elapsed_s") or 0.0), 6),
                "delta_allocated": format_cuda_bytes(
                    record.get("delta_allocated_bytes")
                ),
                "peak_delta_allocated": format_cuda_bytes(
                    record.get("peak_delta_allocated_bytes")
                ),
                "delta_reserved": format_cuda_bytes(record.get("delta_reserved_bytes")),
                "peak_delta_reserved": format_cuda_bytes(
                    record.get("peak_delta_reserved_bytes")
                ),
            }
        )
    return summary


def _profile_train_phase(
    *,
    cfg: DictConfig,
    datamodule: Any,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    _attach_fake_trainer(
        datamodule=datamodule,
        model=model,
        device=device,
        precision=cfg.trainer.precision,
        fn_name="fit",
    )
    datamodule.setup(stage="fit")
    reset_cuda_memory_records()
    batch = _materialize_runtime_batch(
        dataloader=datamodule.train_dataloader(),
        datamodule=datamodule,
        device=device,
        stage_label="train.batch",
    )
    sampler = model._require_subgraph_sampler()
    action_pruning_cfg = model._training_action_pruning_cfg()
    rollouts_per_graph = int(model.cfg.training_cfg["rollouts_per_graph"])
    sampling_temperature = model._resolve_sampling_temperature()
    proposal_bias_scale = model._resolve_proposal_bias_scale()
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        with _autocast_context(cfg.trainer.precision, device):
            with profile_cuda_memory(
                "train.policy.prepare_batch",
                device=device,
                extra=f"num_graphs={int(batch.num_graphs)}",
            ):
                prepared_batch = model.policy.prepare_batch(batch)
            with profile_cuda_memory(
                "train.sampler.sample",
                device=device,
                extra=(
                    f"num_graphs={int(batch.num_graphs)} "
                    f"rollouts_per_graph={rollouts_per_graph}"
                ),
            ):
                sample_batch = sampler.sample(
                    policy=model.policy,
                    prepared_batch=prepared_batch,
                    rollouts_per_graph=rollouts_per_graph,
                    temperature=sampling_temperature,
                    proposal_bias_scale=proposal_bias_scale,
                    action_pruning=action_pruning_cfg,
                )
            with profile_cuda_memory(
                "train.loss.compute",
                device=device,
                extra=(
                    f"num_graphs={int(sample_batch.num_graphs)} "
                    f"num_rollouts={int(sample_batch.num_rollouts)}"
                ),
            ):
                loss_output = model.loss_fn.compute(sample_batch)
                loss = loss_output.loss
        loss_value = float(loss.detach().cpu().item())
        with profile_cuda_memory(
            "train.backward",
            device=device,
            extra=f"loss={loss_value:.6f}",
        ):
            loss.backward()
        status = "ok"
        error = None
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        error = str(exc)
        loss_value = None
    records = get_cuda_memory_records()
    result = {
        "status": status,
        "error": error,
        "loss": loss_value,
        "num_graphs": int(batch.num_graphs),
        "rollouts_per_graph": rollouts_per_graph,
        "top_stages": _top_stage_summary(records),
        "records": records,
    }
    model.zero_grad(set_to_none=True)
    datamodule.teardown()
    _cleanup_cuda_memory()
    return result


def _profile_val_phase(
    *,
    cfg: DictConfig,
    datamodule: Any,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    _attach_fake_trainer(
        datamodule=datamodule,
        model=model,
        device=device,
        precision=cfg.trainer.precision,
        fn_name="validate",
    )
    datamodule.setup(stage="fit")
    reset_cuda_memory_records()
    batch = _materialize_runtime_batch(
        dataloader=datamodule.val_dataloader(),
        datamodule=datamodule,
        device=device,
        stage_label="val.batch",
    )
    model.eval()
    try:
        with torch.inference_mode():
            with _autocast_context(cfg.trainer.precision, device):
                with profile_cuda_memory(
                    "val.evaluate_batch_output",
                    device=device,
                    extra=f"num_graphs={int(batch.num_graphs)}",
                ):
                    outputs = model._evaluate_batch_output(batch=batch)
        status = "ok"
        error = None
        primary_metrics = {
            key: float(value) for key, value in dict(outputs.primary_metrics).items()
        }
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        error = str(exc)
        primary_metrics = {}
    records = get_cuda_memory_records()
    result = {
        "status": status,
        "error": error,
        "num_graphs": int(batch.num_graphs),
        "primary_metrics": primary_metrics,
        "top_stages": _top_stage_summary(records),
        "records": records,
    }
    datamodule.teardown()
    _cleanup_cuda_memory()
    return result


def main() -> None:
    args = parse_args()
    os.environ["RANKFLOW_PROFILE_CUDA_MEMORY"] = "1"
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This profiler only supports CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot profile GPU memory.")

    cfg = _compose_train_cfg(list(args.override))
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    datamodule, model = _instantiate_objects(cfg)
    model.to(device)

    result: dict[str, Any] = {
        "config": {
            "dataset": str((cfg.get("dataset") or {}).get("name") or ""),
            "experiment": str(
                (cfg.get("hydra") or {})
                .get("runtime", {})
                .get("choices", {})
                .get("experiment", "")
            ),
            "precision": str(cfg.trainer.precision),
            "device": str(device),
        }
    }
    if args.mode in {"train", "both"}:
        result["train"] = _profile_train_phase(
            cfg=cfg,
            datamodule=datamodule,
            model=model,
            device=device,
        )
    if args.mode in {"val", "both"}:
        result["val"] = _profile_val_phase(
            cfg=cfg,
            datamodule=datamodule,
            model=model,
            device=device,
        )

    text = json.dumps(result, indent=2, ensure_ascii=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
