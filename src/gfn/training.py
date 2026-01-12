from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import torch
from torch import nn
from torchmetrics import MeanMetric, Metric, MetricCollection

from src.utils import log_metric, setup_optimizer

if TYPE_CHECKING:  # pragma: no cover
    from lightning import LightningModule

_ZERO = 0
_ONE = 1
_HALF = 0.5
_NAN = float("nan")
_DEFAULT_SCHED_T_MAX = 10
_DEFAULT_SCHED_T0 = 10
_DEFAULT_SCHED_T_MULT = 1
_DEFAULT_SCHED_ETA_MIN = 0.0
_DEFAULT_SCHED_INTERVAL = "epoch"


class GFlowNetMetricLogger:
    def __init__(self, module: "LightningModule") -> None:
        self.module = module

    def init_metric_stores(self) -> None:
        self.module.train_metrics = MetricCollection({})
        self.module.val_metrics = MetricCollection({})
        self.module.test_metrics = MetricCollection({})

    def update_metrics(self, metrics: Dict[str, torch.Tensor], *, prefix: str, batch_size: int) -> None:
        store = self._get_metric_store(prefix)
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if not torch.is_floating_point(value):
                value = value.float()
            value = value.detach().to(device=self.module.device)
            metric = self._get_or_create_metric(store, name)
            if value.numel() == _ONE:
                metric.update(value, weight=batch_size)
            else:
                metric.update(value)

    def log_metric_store(self, *, prefix: str, batch_size: int) -> None:
        if prefix == "predict":
            return
        store = self._get_metric_store(prefix)
        sync_dist = bool(self.module.trainer and getattr(self.module.trainer, "num_devices", _ONE) > _ONE)
        is_train = prefix == "train"
        prog_bar_set = set(self.module._train_prog_bar if is_train else self.module._eval_prog_bar)
        log_on_step = self.module._log_on_step_train if is_train else False
        for name, metric in store.items():
            prog_bar = name in prog_bar_set or (is_train and name == "loss")
            metric_attribute = f"{prefix}_metrics.{name}" if isinstance(metric, Metric) else None
            log_metric(
                self.module,
                f"{prefix}/{name}",
                metric,
                sync_dist=sync_dist,
                prog_bar=prog_bar,
                on_step=log_on_step,
                on_epoch=True,
                batch_size=batch_size,
                metric_attribute=metric_attribute,
            )

    def _get_metric_store(self, prefix: str) -> MetricCollection:
        if prefix == "train":
            return self.module.train_metrics
        if prefix == "val":
            return self.module.val_metrics
        if prefix == "test":
            return self.module.test_metrics
        raise ValueError(f"Unknown metric prefix: {prefix}")

    def _get_or_create_metric(self, store: MetricCollection, name: str) -> MeanMetric:
        if name in store:
            return store[name]  # type: ignore[return-value]
        metric = MeanMetric().to(self.module.device)
        if hasattr(store, "add_metrics"):
            store.add_metrics({name: metric})
        else:
            store[name] = metric
        return metric


class GFlowNetGradClipper:
    def __init__(self, module: "LightningModule") -> None:
        self.module = module
        self._param_name_map: Optional[Dict[int, str]] = None

    def clip_gradients(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        clip_val = self._resolve_clip_val()
        algorithm = self._resolve_clip_algorithm()
        optimizer_ref = self._unwrap_optimizer(optimizer)
        params, device, dtype = self._collect_grad_params(optimizer_ref)
        if not params or device is None or dtype is None:
            return {}
        self._assert_finite_grads(params)
        if self.module._adaptive_grad_clip:
            clip_val = self._resolve_adaptive_clip_val(clip_val, device=device, dtype=dtype)
        norm_type = self._resolve_norm_type()
        return self._build_clip_metrics(
            optimizer_ref=optimizer,
            params=params,
            clip_val=clip_val,
            norm_type=norm_type,
            algorithm=algorithm,
            device=device,
            dtype=dtype,
        )

    def _build_clip_metrics(
        self,
        *,
        optimizer_ref: torch.optim.Optimizer,
        params: Sequence[torch.nn.Parameter],
        clip_val: float,
        norm_type: float,
        algorithm: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        flow_pre = self._compute_module_grad_norm(
            self.module.log_f,
            norm_type=norm_type,
            device=device,
            dtype=dtype,
        )
        actor_pre = self._compute_module_grad_norm(
            self.module.actor,
            norm_type=norm_type,
            device=device,
            dtype=dtype,
        )
        pre_norm = self._compute_grad_norm(params, norm_type=norm_type, device=device, dtype=dtype)
        self._apply_clip_algorithm(
            optimizer_ref=optimizer_ref,
            clip_val=clip_val,
            algorithm=algorithm,
        )
        post_norm = self._compute_grad_norm(params, norm_type=norm_type, device=device, dtype=dtype)
        clip_coef = self._compute_clip_coef(
            pre_norm=pre_norm,
            post_norm=post_norm,
            clip_val=clip_val,
            algorithm=algorithm,
        )
        flow_post = self._compute_module_grad_norm(self.module.log_f, norm_type=norm_type, device=device, dtype=dtype)
        actor_post = self._compute_module_grad_norm(
            self.module.actor, norm_type=norm_type, device=device, dtype=dtype
        )
        log_norm = torch.log(pre_norm.to(dtype=torch.float32) + float(self.module._grad_clip_log_eps))
        log_mean, log_std = self._update_grad_log_norm_ema(log_norm)
        return self._build_grad_metrics(
            pre_norm=pre_norm,
            post_norm=post_norm,
            clip_val=clip_val,
            clip_coef=clip_coef,
            flow_pre=flow_pre,
            flow_post=flow_post,
            actor_pre=actor_pre,
            actor_post=actor_post,
            log_mean=log_mean,
            log_std=log_std,
            device=device,
            dtype=dtype,
        )

    def _apply_clip_algorithm(
        self,
        *,
        optimizer_ref: torch.optim.Optimizer,
        clip_val: float,
        algorithm: str,
    ) -> None:
        if algorithm not in {"norm", "value"}:
            raise ValueError(
                "training_cfg.manual_gradient_clip_algorithm must be 'norm' or 'value', "
                f"got {algorithm!r}."
            )
        if clip_val <= float(_ZERO):
            return
        self.module.clip_gradients(
            optimizer_ref,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=algorithm,
        )

    def _compute_clip_coef(
        self,
        *,
        pre_norm: torch.Tensor,
        post_norm: torch.Tensor,
        clip_val: float,
        algorithm: str,
    ) -> float:
        denom = float(pre_norm.item()) + float(self.module._grad_clip_log_eps)
        if clip_val <= float(_ZERO):
            return float(_ONE)
        if algorithm == "value":
            return float(post_norm.item()) / denom
        return min(float(_ONE), clip_val / denom)

    def _resolve_clip_val(self) -> float:
        raw = self.module.training_cfg.get("manual_gradient_clip_val", _ZERO)
        return self.module.require_non_negative_float(raw, "training_cfg.manual_gradient_clip_val")

    def _resolve_clip_algorithm(self) -> str:
        raw = self.module.training_cfg.get("manual_gradient_clip_algorithm", self.module._grad_clip_algorithm_default)
        return str(raw)

    def _resolve_norm_type(self) -> float:
        raw = self.module.training_cfg.get("manual_gradient_clip_norm_type", self.module._grad_clip_norm_type_default)
        return self.module.require_positive_float(raw, "training_cfg.manual_gradient_clip_norm_type")

    @staticmethod
    def _unwrap_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        return getattr(optimizer, "optimizer", optimizer)

    @staticmethod
    def _collect_grad_params(
        optimizer_ref: torch.optim.Optimizer,
    ) -> tuple[list[torch.nn.Parameter], Optional[torch.device], Optional[torch.dtype]]:
        params = [
            param
            for group in optimizer_ref.param_groups
            for param in group.get("params", [])
            if param is not None and param.grad is not None
        ]
        if not params:
            return [], None, None
        return params, params[0].grad.device, params[0].grad.dtype

    def _get_param_name_map(self) -> Dict[int, str]:
        if self._param_name_map is None:
            self._param_name_map = {id(param): name for name, param in self.module.named_parameters()}
        return self._param_name_map

    def _assert_finite_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        if not params:
            return
        name_map = self._get_param_name_map()
        max_params = self.module.training_cfg.get(
            "grad_non_finite_max_params",
            self.module._grad_nonfinite_max_params_default,
        )
        if max_params is not None:
            max_params = self.module.require_positive_int(max_params, "training_cfg.grad_non_finite_max_params")
        entries: list[tuple[str, torch.Tensor]] = []
        total_non_finite = _ZERO
        for idx, param in enumerate(params):
            grad = param.grad
            if grad is None:
                continue
            if torch.isfinite(grad).all():
                continue
            total_non_finite += _ONE
            if max_params is None or len(entries) < max_params:
                name = name_map.get(id(param), f"param_{idx}")
                entries.append((name, grad))
        if total_non_finite == _ZERO:
            return
        lines = [
            "Non-finite gradients detected; aborting training.",
            f"non_finite_params={total_non_finite}",
            f"global_step={int(self.module.global_step)}",
        ]
        for name, grad in entries:
            lines.append(self._summarize_grad(name, grad))
        if max_params is not None and total_non_finite > max_params:
            lines.append(f"... truncated to {max_params} params (set training_cfg.grad_non_finite_max_params).")
        raise ValueError("\n".join(lines))

    @staticmethod
    def _summarize_grad(name: str, grad: torch.Tensor) -> str:
        finite = torch.isfinite(grad)
        non_finite = int((~finite).sum().item())
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        finite_vals = grad[finite]
        if finite_vals.numel() > _ZERO:
            calc = finite_vals.to(dtype=torch.float32)
            min_val = float(calc.min().item())
            max_val = float(calc.max().item())
            mean_val = float(calc.mean().item())
            abs_max = float(calc.abs().max().item())
        else:
            min_val = _NAN
            max_val = _NAN
            mean_val = _NAN
            abs_max = _NAN
        return (
            f"grad[{name}]: shape={tuple(grad.shape)}, dtype={grad.dtype}, "
            f"non_finite={non_finite} (nan={nan_count}, inf={inf_count}), "
            f"finite_min={min_val}, finite_max={max_val}, finite_mean={mean_val}, abs_max={abs_max}"
        )

    def _compute_grad_norm(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        norm_type: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not params:
            return torch.zeros((), device=device, dtype=dtype)
        if norm_type == float("inf"):
            norms = [param.grad.detach().abs().max().to(device=device, dtype=dtype) for param in params]
            return torch.stack(norms).max()
        norms = [param.grad.detach().norm(norm_type).to(device=device, dtype=dtype) for param in params]
        return torch.stack(norms).norm(norm_type)

    @staticmethod
    def _collect_module_grad_params(module: nn.Module) -> list[torch.nn.Parameter]:
        return [param for param in module.parameters() if param.grad is not None]

    def _compute_module_grad_norm(
        self,
        module: nn.Module,
        *,
        norm_type: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        params = self._collect_module_grad_params(module)
        if not params:
            return torch.zeros((), device=device, dtype=dtype)
        return self._compute_grad_norm(params, norm_type=norm_type, device=device, dtype=dtype)

    @staticmethod
    def _normal_icdf(prob: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((), device=prob.device, dtype=prob.dtype)
        one = torch.ones((), device=prob.device, dtype=prob.dtype)
        return torch.distributions.Normal(zero, one).icdf(prob)

    def _grad_log_norm_stats(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.module.grad_log_norm_ema.to(device=device, dtype=dtype)
        sq_mean = self.module.grad_log_norm_sq_ema.to(device=device, dtype=dtype)
        var = (sq_mean - mean * mean).clamp(min=float(_ZERO))
        std = torch.sqrt(var)
        return mean, std

    def _update_grad_log_norm_ema(self, log_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self._resolve_grad_clip_ema_beta()
        beta_t = torch.as_tensor(beta, device=log_norm.device, dtype=log_norm.dtype)
        one_t = torch.as_tensor(float(_ONE), device=log_norm.device, dtype=log_norm.dtype)
        if int(self.module.grad_ema_initialized.item()) == _ZERO:
            self.module.grad_log_norm_ema.copy_(log_norm)
            self.module.grad_log_norm_sq_ema.copy_(log_norm * log_norm)
            self.module.grad_ema_initialized.fill_(_ONE)
        else:
            self.module.grad_log_norm_ema.copy_(
                beta_t * self.module.grad_log_norm_ema + (one_t - beta_t) * log_norm
            )
            self.module.grad_log_norm_sq_ema.copy_(
                beta_t * self.module.grad_log_norm_sq_ema + (one_t - beta_t) * log_norm * log_norm
            )
        return self._grad_log_norm_stats(device=log_norm.device, dtype=log_norm.dtype)

    def _estimate_optimizer_steps_per_epoch(self) -> Optional[int]:
        if self.module.trainer is None:
            return None
        total = getattr(self.module.trainer, "num_training_batches", None)
        if total is None:
            return None
        accum = int(getattr(self.module.trainer, "accumulate_grad_batches", _ONE) or _ONE)
        steps = int(math.ceil(float(total) / float(accum)))
        return max(steps, _ONE)

    def _resolve_grad_clip_ema_beta(self) -> float:
        raw = self.module.training_cfg.get("grad_clip_ema_beta")
        if raw is not None:
            beta = self.module.require_float(raw, "training_cfg.grad_clip_ema_beta")
            if not (float(_ZERO) <= beta < float(_ONE)):
                raise ValueError(f"training_cfg.grad_clip_ema_beta must be in [0, 1), got {beta}.")
            return float(beta)
        steps = self._estimate_optimizer_steps_per_epoch()
        if steps is None:
            raise ValueError("training_cfg.grad_clip_ema_beta must be set when trainer steps are unavailable.")
        return float(_ONE) - float(_ONE) / float(steps)

    def _resolve_grad_clip_tail_prob(self) -> float:
        raw = self.module.training_cfg.get("grad_clip_tail_prob")
        if raw is not None:
            prob = self.module.require_probability_open(raw, "training_cfg.grad_clip_tail_prob")
        else:
            steps = self._estimate_optimizer_steps_per_epoch()
            if steps is None:
                raise ValueError("training_cfg.grad_clip_tail_prob must be set when trainer steps are unavailable.")
            prob = float(_ONE) / float(steps)
        eps = float(self.module._grad_clip_tail_prob_eps)
        prob = min(max(prob, eps), float(_ONE) - eps)
        return prob

    def _resolve_adaptive_clip_val(
        self,
        fallback: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        if int(self.module.grad_ema_initialized.item()) == _ZERO:
            return fallback
        mean, std = self._grad_log_norm_stats(device=device, dtype=dtype)
        tail_prob = self._resolve_grad_clip_tail_prob()
        tail = torch.as_tensor(float(_ONE) - tail_prob, device=device, dtype=dtype)
        z_value = self._normal_icdf(tail)
        clip_log = mean + z_value * std
        clip_val = torch.exp(clip_log).clamp(min=float(self.module._grad_clip_log_eps))
        return float(clip_val.item())

    def _build_grad_metrics(
        self,
        *,
        pre_norm: torch.Tensor,
        post_norm: torch.Tensor,
        clip_val: float,
        clip_coef: float,
        flow_pre: torch.Tensor,
        flow_post: torch.Tensor,
        actor_pre: torch.Tensor,
        actor_post: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        metrics = {
            "optim/grad_norm_pre": pre_norm.detach(),
            "optim/grad_norm_post": post_norm.detach(),
            "optim/grad_norm_flow_pre": flow_pre.detach(),
            "optim/grad_norm_flow_post": flow_post.detach(),
            "optim/grad_norm_actor_pre": actor_pre.detach(),
            "optim/grad_norm_actor_post": actor_post.detach(),
            "optim/grad_clip_val": torch.as_tensor(clip_val, device=device, dtype=dtype),
            "optim/grad_clip_coef": torch.as_tensor(clip_coef, device=device, dtype=dtype),
            "optim/grad_log_norm_ema": log_mean.detach(),
            "optim/grad_log_norm_std": log_std.detach(),
            "optim/grad_clip_adaptive": torch.as_tensor(float(self.module._adaptive_grad_clip), device=device, dtype=dtype),
        }
        if self.module._adaptive_grad_clip:
            tail_prob = self._resolve_grad_clip_tail_prob()
            metrics["optim/grad_clip_tail_prob"] = torch.as_tensor(tail_prob, device=device, dtype=dtype)
        return metrics


class GFlowNetTrainingLoop:
    def __init__(self, module: "LightningModule") -> None:
        self.module = module
        self.metric_logger = GFlowNetMetricLogger(module)
        self.grad_clipper = GFlowNetGradClipper(module)

    def init_metric_stores(self) -> None:
        self.metric_logger.init_metric_stores()

    def configure_optimizers(self):
        optimizer = setup_optimizer(self.module, self.module.optimizer_cfg)
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx: int):
        self._maybe_init_flow_bias(batch)
        optimizer = self.module.optimizers()
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        rollout_cfg = self.module.build_rollout_cfg(is_training=True)
        progress = self._resolve_training_progress()
        should_step = self._should_step_optimizer(batch_idx)
        sync_context = nullcontext()
        if not should_step:
            no_sync = getattr(self.module, "no_sync", None)
            if no_sync is not None:
                sync_context = no_sync()
        with sync_context:
            loss, metrics = self.module.engine.compute_batch_loss_streaming(
                batch=batch,
                device=self.module.device,
                rollout_cfg=rollout_cfg,
                backward_fn=self.module.manual_backward,
                progress=progress,
            )
        metrics = self._attach_loss(metrics, loss)
        control_metrics = self._update_success_controller(metrics)
        if control_metrics:
            metrics.update(control_metrics)
        if should_step:
            grad_metrics = self._step_optimizer(optimizer)
            if grad_metrics:
                metrics.update(grad_metrics)
        batch_size = int(batch.num_graphs)
        self.metric_logger.update_metrics(metrics, prefix="train", batch_size=batch_size)
        self.metric_logger.log_metric_store(prefix="train", batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        self._eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._eval_step(batch, batch_idx, prefix="test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _ = dataloader_idx
        return self._compute_rollout_records(batch=batch, batch_idx=batch_idx)

    def on_train_epoch_start(self) -> None:
        self._apply_potential_weight_schedule()
        self._apply_control_temperature()

    def on_train_epoch_end(self) -> None:
        if self._resolve_scheduler_interval() == _DEFAULT_SCHED_INTERVAL:
            self._step_scheduler()

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        _ = optimizer_idx
        return self.grad_clipper.clip_gradients(optimizer)

    def _eval_step(self, batch, batch_idx: int, *, prefix: str) -> None:
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = self._attach_loss(metrics, loss)
        extra_metrics = self._collect_eval_temperature_metrics(batch, batch_idx=batch_idx)
        if extra_metrics:
            metrics.update(extra_metrics)
        batch_size = int(batch.num_graphs)
        self.metric_logger.update_metrics(metrics, prefix=prefix, batch_size=batch_size)
        self.metric_logger.log_metric_store(prefix=prefix, batch_size=batch_size)

    def _collect_eval_temperature_metrics(
        self,
        batch,
        *,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        extras = self._resolve_eval_temperature_extras()
        if not extras:
            return {}
        metrics: Dict[str, torch.Tensor] = {}
        for temp in extras:
            extra_loss, extra_metrics = self._compute_batch_loss(
                batch,
                batch_idx=batch_idx,
            rollout_cfg=self.module.build_eval_rollout_cfg(temperature=temp),
            )
            extra_metrics = self._attach_loss(extra_metrics, extra_loss)
            suffix = self._format_temperature_suffix(temp)
            metrics.update(self._suffix_metrics(extra_metrics, suffix=suffix))
        return metrics

    @staticmethod
    def _format_temperature_suffix(temperature: float) -> str:
        text = f"{temperature:g}".replace(".", "p").replace("-", "m")
        return f"t{text}"

    @staticmethod
    def _suffix_metrics(metrics: Dict[str, torch.Tensor], *, suffix: str) -> Dict[str, torch.Tensor]:
        return {f"{name}_{suffix}": value for name, value in metrics.items()}

    def _resolve_eval_temperature_extras(self) -> list[float]:
        extras_cfg = self.module.evaluation_cfg.get("rollout_temperature_extra", self.module._eval_temp_extras_default)
        if extras_cfg is None:
            return []
        if isinstance(extras_cfg, Sequence) and not isinstance(extras_cfg, (str, bytes)):
            temps = [float(value) for value in extras_cfg]
        else:
            temps = [float(extras_cfg)]
        extras: list[float] = []
        for temp in temps:
            if temp < float(_ZERO):
                raise ValueError(
                    f"evaluation_cfg.rollout_temperature_extra must be >= {float(_ZERO)}, got {temp}."
                )
            extras.append(temp)
        base = float(getattr(self.module, "_eval_rollout_temperature", float("nan")))
        return [temp for temp in extras if temp != base]

    def _compute_batch_loss(
        self,
        batch,
        *,
        batch_idx: int | None = None,
        rollout_cfg=None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _ = batch_idx
        if rollout_cfg is None:
            rollout_cfg = self.module.build_rollout_cfg(is_training=self.module.training)
        return self.module.engine.compute_batch_loss(
            batch=batch,
            device=self.module.device,
            rollout_cfg=rollout_cfg,
        )

    def _compute_rollout_records(
        self,
        *,
        batch,
        batch_idx: int | None = None,
    ) -> list[Dict[str, Any]]:
        _ = batch_idx
        rollout_cfg = self.module.build_rollout_cfg(is_training=False)
        return self.module.engine.compute_rollout_records(
            batch=batch,
            device=self.module.device,
            rollout_cfg=rollout_cfg,
        )

    def _attach_loss(self, metrics: Dict[str, torch.Tensor], loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = dict(metrics)
        out["loss"] = loss.detach()
        return out

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
        scheduler_type = str(self.module.scheduler_cfg.get("type", "") or "").lower()
        if not scheduler_type:
            return None
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.module.scheduler_cfg.get("t_max", _DEFAULT_SCHED_T_MAX)),
                eta_min=float(self.module.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
            return self._pack_scheduler(scheduler)
        if scheduler_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts"}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.module.scheduler_cfg.get("t_0", _DEFAULT_SCHED_T0)),
                T_mult=int(self.module.scheduler_cfg.get("t_mult", _DEFAULT_SCHED_T_MULT)),
                eta_min=float(self.module.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
            return self._pack_scheduler(scheduler)
        return None

    def _pack_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict[str, Any]:
        return {
            "scheduler": scheduler,
            "interval": self.module.scheduler_cfg.get("interval", "epoch"),
            "monitor": self.module.scheduler_cfg.get("monitor", "val/loss"),
        }

    def _step_scheduler(self) -> None:
        sched = self.module.lr_schedulers()
        if sched is None:
            return
        schedulers = sched if isinstance(sched, list) else [sched]
        monitor = self.module.scheduler_cfg.get("monitor", None)
        metric = None
        if monitor and self.module.trainer is not None:
            metric = self.module.trainer.callback_metrics.get(monitor)
        for scheduler in schedulers:
            if monitor and metric is None:
                continue
            self.module.lr_scheduler_step(scheduler, metric)

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        grad_metrics = self.on_before_optimizer_step(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if self._resolve_scheduler_interval() == "step":
            self._step_scheduler()
        return grad_metrics or {}

    def _resolve_scheduler_interval(self) -> str:
        raw = self.module.scheduler_cfg.get("interval", _DEFAULT_SCHED_INTERVAL)
        return str(raw or _DEFAULT_SCHED_INTERVAL)

    def _resolve_training_progress(self) -> Optional[float]:
        trainer = self.module.trainer
        if trainer is None:
            return None
        total = getattr(trainer, "estimated_stepping_batches", None)
        if total is None or total <= _ZERO:
            total = getattr(trainer, "max_steps", None)
        if total is None or total <= _ZERO:
            return None
        step = float(getattr(trainer, "global_step", self.module.global_step))
        return min(max(step / float(total), float(_ZERO)), float(_ONE))

    def _accumulate_grad_batches(self) -> int:
        if self.module.trainer is None:
            return _ONE
        accum = int(getattr(self.module.trainer, "accumulate_grad_batches", _ONE) or _ONE)
        return max(accum, _ONE)

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        if self.module.trainer is None:
            return False
        total = getattr(self.module.trainer, "num_training_batches", None)
        if total is None:
            return False
        return (batch_idx + _ONE) >= int(total)

    def _should_zero_grad(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        return batch_idx % accum == _ZERO

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        if self._is_last_train_batch(batch_idx):
            return True
        return (batch_idx + _ONE) % accum == _ZERO

    def _maybe_init_flow_bias(self, batch: Any) -> None:
        if self.module._flow_bias_initialized:
            return
        strategy = self.module._flow_bias_init
        if strategy == "none":
            self.module._flow_bias_initialized = True
            return
        if strategy == "min_log_reward":
            bias = self._resolve_min_log_reward()
            self.module.log_f.set_output_bias(bias)
            self.module._flow_bias_initialized = True
            return
        if strategy == "batch_log_reward":
            bias = self._estimate_log_reward_bias(batch)
            self.module.log_f.set_output_bias(bias)
            self.module._flow_bias_initialized = True
            return
        if strategy == "value":
            bias = float(self.module._flow_bias_value)
            self.module.log_f.set_output_bias(bias)
            self.module._flow_bias_initialized = True
            return
        raise RuntimeError(f"Unhandled flow bias init strategy: {strategy!r}.")

    def _resolve_min_log_reward(self) -> float:
        min_log = getattr(self.module.reward_fn, "min_log_reward", None)
        if min_log is None:
            raise RuntimeError("reward_fn.min_log_reward is required for bias initialization.")
        return float(min_log)

    @torch.no_grad()
    def _estimate_log_reward_bias(self, batch: Any) -> float:
        device = self.module.device
        self.module.validate_batch_inputs(batch, is_training=True, require_rollout=True)
        inputs = self.module.batch_processor.prepare_rollout_inputs(batch, device)
        graph_cache = self.module.batch_processor.build_graph_cache(inputs, device=device)
        rollout = self.module.actor.rollout(
            graph=graph_cache,
            temperature=None,
            record_actions=False,
            record_visited=False,
        )
        reward_out = self.module.reward_fn(
            **self.module.engine.build_reward_kwargs(rollout, inputs=inputs)
        )
        log_reward = torch.where(
            inputs.dummy_mask,
            torch.zeros_like(reward_out.log_reward),
            reward_out.log_reward,
        )
        valid = torch.isfinite(log_reward)
        if not bool(valid.any().item()):
            raise RuntimeError("log_reward contains no finite values for flow bias init.")
        return float(log_reward[valid].mean().item())

    def _extract_scalar_metric(
        self,
        metrics: Dict[str, torch.Tensor],
        key: str,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = metrics.get(key)
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.detach().to(device=device)
        if value.numel() > _ONE:
            return value.float().mean()
        return value.float().view(())

    def _resolve_control_target(
        self,
        metrics: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.module._control_target_mode == "reachable_horizon_frac":
            reachable = metrics.get("control/reachable_horizon_frac")
            if reachable is None:
                return None
            if not torch.is_tensor(reachable):
                reachable = torch.as_tensor(reachable, device=device, dtype=dtype)
            else:
                reachable = reachable.to(device=device, dtype=dtype)
            target = reachable
        elif self.module._control_target_mode == "fixed":
            target = torch.as_tensor(self.module._control_target_min, device=device, dtype=dtype)
        else:
            raise ValueError("Unsupported control target mode.")
        min_target = torch.as_tensor(self.module._control_target_min, device=device, dtype=dtype)
        max_target = torch.as_tensor(self.module._control_target_max, device=device, dtype=dtype)
        return target.clamp(min=min_target, max=max_target)

    def _temperature_from_lambda(self, lambda_value: torch.Tensor) -> torch.Tensor:
        base = torch.as_tensor(self.module._control_temperature_base, device=lambda_value.device, dtype=lambda_value.dtype)
        min_temp = torch.as_tensor(self.module._control_temperature_min, device=lambda_value.device, dtype=lambda_value.dtype)
        max_temp = torch.as_tensor(self.module._control_temperature_max, device=lambda_value.device, dtype=lambda_value.dtype)
        temperature = base * torch.exp(lambda_value)
        return temperature.clamp(min=min_temp, max=max_temp)

    def _apply_control_temperature(self) -> None:
        if not self.module._control_enabled:
            return
        temperature = self._temperature_from_lambda(self.module.lambda_success)
        temp_value = float(temperature.item())
        self.module.actor.set_temperatures(policy_temperature=temp_value, backward_temperature=temp_value)

    def _update_success_controller(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.module._control_enabled:
            return {}
        device = self.module.lambda_success.device
        dtype = self.module.lambda_success.dtype
        success = self._extract_scalar_metric(metrics, "pass@1", device=device)
        if success is None:
            return {}
        target = self._resolve_control_target(metrics, device=device, dtype=dtype)
        if target is None:
            return {}
        gap = target - success
        dual_lr = torch.as_tensor(self.module._control_dual_lr, device=device, dtype=dtype)
        lambda_min = torch.as_tensor(self.module._control_lambda_min, device=device, dtype=dtype)
        lambda_max = torch.as_tensor(self.module._control_lambda_max, device=device, dtype=dtype)
        updated = (self.module.lambda_success + dual_lr * gap).clamp(min=lambda_min, max=lambda_max)
        self.module.lambda_success.copy_(updated)
        temperature = self._temperature_from_lambda(updated)
        temp_value = float(temperature.item())
        self.module.actor.set_temperatures(policy_temperature=temp_value, backward_temperature=temp_value)
        return {
            "control/success_rate": success.detach(),
            "control/target_success": target.detach(),
            "control/gap": gap.detach(),
            "control/lambda": updated.detach(),
            "control/temperature": temperature.detach(),
        }

    def _apply_potential_weight_schedule(self) -> None:
        if not hasattr(self.module.reward_fn, "potential_weight"):
            return
        schedule = self.module._potential_schedule
        if schedule == "none":
            return
        anneal_epochs = self._resolve_potential_anneal_epochs()
        if anneal_epochs <= _ZERO:
            self.module.reward_fn.potential_weight = float(_ZERO)
            return
        current_epoch = int(self.module.current_epoch)
        if current_epoch >= anneal_epochs:
            self.module.reward_fn.potential_weight = float(_ZERO)
            return
        ratio = float(current_epoch) / float(anneal_epochs)
        if schedule == "cosine":
            weight = self.module._potential_weight_init * (_HALF * (float(_ONE) + math.cos(math.pi * ratio)))
        else:
            weight = self.module._potential_weight_init + (
                (self.module._potential_weight_end - self.module._potential_weight_init) * ratio
            )
        self.module.reward_fn.potential_weight = float(weight)

    def _resolve_potential_anneal_epochs(self) -> int:
        if self.module._potential_anneal_epochs is not None:
            return int(self.module._potential_anneal_epochs)
        if self.module._potential_pure_phase_ratio is None:
            raise ValueError(
                "reward_fn must set potential_anneal_epochs or potential_pure_phase_ratio "
                "when potential scheduling is enabled."
            )
        if self.module.trainer is None:
            raise RuntimeError("Trainer is not initialized; cannot resolve potential anneal epochs.")
        total_epochs = getattr(self.module.trainer, "max_epochs", None)
        if total_epochs is None:
            raise RuntimeError("trainer.max_epochs is required for potential annealing.")
        total = int(total_epochs)
        if total <= _ZERO:
            raise ValueError("trainer.max_epochs must be positive for potential annealing.")
        ratio = float(_ONE) - float(self.module._potential_pure_phase_ratio)
        return int(math.floor(float(total) * ratio))


__all__ = ["GFlowNetTrainingLoop"]
