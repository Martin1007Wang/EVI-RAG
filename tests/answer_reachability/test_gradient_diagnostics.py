from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch
from omegaconf import open_dict

from src.models.gflownet import TrainingScheduleContext
from src.utils.entrypoint_utils import _strip_instantiate_metadata
from src.utils.fit_schedule import (
    apply_resolved_pass_fit_schedule,
    resolve_pass_fit_schedule,
)

_HYDRA_TEST_OVERRIDES = ["hydra/job_logging=stdout", "hydra/hydra_logging=none"]
_WEBQSP_DATA_ROOT = Path("/mnt/data/retrieval_dataset/webqsp")


@dataclass(frozen=True)
class GradientMassDiagnostics:
    success_count: int
    total_count: int
    success_step_grad_mass: float
    failure_step_grad_mass: float
    success_start_grad_mass: float
    failure_start_grad_mass: float

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return float(self.success_count) / float(self.total_count)

    @property
    def failure_count(self) -> int:
        return int(self.total_count - self.success_count)

    @property
    def step_failure_to_success_ratio(self) -> float:
        if self.success_step_grad_mass == 0.0:
            return float("inf")
        return self.failure_step_grad_mass / self.success_step_grad_mass

    @property
    def start_failure_to_success_ratio(self) -> float:
        if self.success_start_grad_mass == 0.0:
            return float("inf")
        return self.failure_start_grad_mass / self.success_start_grad_mass


def _build_real_train_batch_and_model():
    if not _WEBQSP_DATA_ROOT.exists():
        pytest.skip("requires local webqsp retrieval dataset under /mnt/data")

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=train_rankflow",
                "dataset=webqsp-sub",
                "logger=none",
                "extras.enforce_tags=false",
                "extras.print_config=false",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    with open_dict(cfg):
        cfg.data.num_workers = 0
        cfg.data.eval_num_workers = 0
        cfg.data.persistent_workers = False
        cfg.data.eval_persistent_workers = False
        cfg.data.prefetch_factor = None
        cfg.data.eval_prefetch_factor = None
        cfg.data.multiprocessing_context = None
        cfg.data.eval_multiprocessing_context = None

    datamodule = instantiate(_strip_instantiate_metadata(cfg.data))
    datamodule.setup("fit")
    resolved_schedule = resolve_pass_fit_schedule(
        fit_schedule_cfg=cfg.fit_schedule,
        trainer_cfg=cfg.trainer,
        train_size=len(datamodule.train_dataset),
        per_device_batch_size=int(cfg.data.batch_size),
    )
    apply_resolved_pass_fit_schedule(cfg, resolved_schedule)

    raw_batch = next(iter(datamodule.train_dataloader()))
    raw_batch = datamodule.on_before_batch_transfer(raw_batch, 0)
    batch = datamodule.on_after_batch_transfer(raw_batch, 0)

    torch.manual_seed(42)
    model = instantiate(_strip_instantiate_metadata(cfg.model))
    model.set_training_schedule_context(
        TrainingScheduleContext(
            estimated_stepping_batches=resolved_schedule.max_steps,
            trainer_max_steps=resolved_schedule.max_steps,
        )
    )
    model.train()
    return model, batch


def _measure_gradient_mass_diagnostics(*, model, batch) -> GradientMassDiagnostics:
    prepared_batch = model.policy.prepare_batch(batch)
    trajectory_batch = batch.without_raw_features()
    sample_batch = model.sampler.sample(
        batch=trajectory_batch,
        policy=model.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=int(model.cfg.training_cfg.rollouts_per_graph),
        temperature=model._resolve_sampling_temperature(global_step=0),
        action_prior_scale=model._resolve_action_prior_scale(global_step=0),
    )
    sample_batch.start_log_probs.retain_grad()
    sample_batch.log_pf_steps.retain_grad()
    loss = model.loss_fn.compute(sample_batch).loss
    model.zero_grad(set_to_none=True)
    loss.backward()

    success_mask = sample_batch.success_mask.to(dtype=torch.bool)
    move_mask = sample_batch.move_mask.to(dtype=torch.bool)
    failure_mask = ~success_mask

    log_pf_grad = sample_batch.log_pf_steps.grad.detach().abs()
    start_grad = sample_batch.start_log_probs.grad.detach().abs()
    success_step_mask = success_mask.unsqueeze(-1).expand_as(log_pf_grad) & move_mask
    failure_step_mask = failure_mask.unsqueeze(-1).expand_as(log_pf_grad) & move_mask

    return GradientMassDiagnostics(
        success_count=int(success_mask.sum().item()),
        total_count=int(success_mask.numel()),
        success_step_grad_mass=float(log_pf_grad[success_step_mask].sum().item()),
        failure_step_grad_mass=float(log_pf_grad[failure_step_mask].sum().item()),
        success_start_grad_mass=float(start_grad[success_mask].sum().item()),
        failure_start_grad_mass=float(start_grad[failure_mask].sum().item()),
    )


@pytest.mark.slow
def test_real_train_batch_failure_rollouts_dominate_target_gradient_mass() -> None:
    torch.manual_seed(42)
    model, batch = _build_real_train_batch_and_model()

    diagnostics = _measure_gradient_mass_diagnostics(model=model, batch=batch)

    # This is the core pathology behind the slow climb we observe in practice:
    # successful trajectories exist, but they are so rare that their target-policy
    # gradient mass is overwhelmed by the many failed rollouts in the same batch.
    assert diagnostics.success_rate < 0.05
    assert diagnostics.failure_count > diagnostics.success_count
    assert diagnostics.step_failure_to_success_ratio > 10.0
    assert diagnostics.start_failure_to_success_ratio > 10.0
