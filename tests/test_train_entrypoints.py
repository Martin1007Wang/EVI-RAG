from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
from omegaconf import OmegaConf

from src.runs.entrypoints import validate_train_entrypoint
from src.runs.lightning import resolve_instantiate_config

_HYDRA_TEST_OVERRIDES = ["hydra/job_logging=stdout", "hydra/hydra_logging=none"]


def _compose_train_experiment(experiment_name: str):
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={experiment_name}",
                "dataset=webqsp-sub",
                "logger=none",
                "extras.enforce_tags=false",
                "extras.print_config=false",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )


@pytest.mark.parametrize(
    (
        "experiment_name",
        "expected_run_name",
        "expected_rollouts",
        "expected_test_enabled",
        "expected_replay_enabled",
        "expected_tags",
        "expected_hit_bonus",
        "expected_frontier_bonus",
    ),
    [
        (
            "train_rankflow",
            "train_rankflow",
            4096,
            True,
            False,
            (),
            5.0,
            1.0,
        ),
        (
            "train_rankflow_fastiter",
            None,
            256,
            False,
            False,
            (),
            5.0,
            1.0,
        ),
        (
            "train_rankflow_guided",
            "train_rankflow_guided",
            4096,
            True,
            True,
            ("guided", "ablation"),
            5.0,
            1.0,
        ),
        (
            "train_rankflow_ablate_answer_utility",
            "train_rankflow_ablate_answer_utility",
            4096,
            True,
            False,
            ("ablation",),
            0.0,
            0.0,
        ),
    ],
)
def test_train_experiments_apply_intended_behavioral_overrides(
    experiment_name: str,
    expected_run_name: str | None,
    expected_rollouts: int,
    expected_test_enabled: bool,
    expected_replay_enabled: bool,
    expected_tags: tuple[str, ...],
    expected_hit_bonus: float,
    expected_frontier_bonus: float,
) -> None:
    cfg = _compose_train_experiment(experiment_name)
    training_cfg = instantiate(cfg.model.training_cfg)

    if expected_run_name is not None:
        assert cfg.run.name == expected_run_name
    assert cfg.run.test is expected_test_enabled
    assert cfg.model.eval_cfg.report_profile == "rank_only"
    assert cfg.model.eval_cfg.monte_carlo.rollouts == expected_rollouts
    assert cfg.model.eval_cfg.monte_carlo.batch_rollouts == 256
    assert cfg.callbacks.model_checkpoint.monitor == cfg.optimized_metric
    assert cfg.callbacks.early_stopping.monitor == cfg.optimized_metric
    assert training_cfg.auxiliary.replay.enabled is expected_replay_enabled
    assert training_cfg.answer_reward.hit_bonus == pytest.approx(expected_hit_bonus)
    assert training_cfg.answer_reward.frontier_bonus == pytest.approx(
        expected_frontier_bonus
    )
    for tag in expected_tags:
        assert tag in list(cfg.run.tags)

    if experiment_name == "train_rankflow":
        assert cfg.run.final_eval_experiment == "eval_rankflow"
        assert (
            cfg.fit_schedule.early_stopping_patience_passes
            > cfg.fit_schedule.val_every_passes
        )
        assert cfg.trainer.accumulate_grad_batches > 1
        assert list(training_cfg.auxiliary.proposal.keys()) == ["prior"]
        assert training_cfg.answer_reward.coverage_bonus == pytest.approx(0.2)
        assert training_cfg.answer_reward.component_penalty == pytest.approx(0.5)
    elif experiment_name == "train_rankflow_guided":
        assert training_cfg.auxiliary.replay.mix_alpha == pytest.approx(0.25)
        assert training_cfg.auxiliary.replay.guidance.add_shortest_path_guidance is True
        assert training_cfg.auxiliary.replay.buffer.replay_trajectories_per_step == 32
    else:
        assert "guided" not in list(cfg.run.tags)


@pytest.mark.parametrize(
    "experiment_name",
    [
        "train_rankflow",
        "train_rankflow_fastiter",
        "train_rankflow_guided",
        "train_rankflow_guided_fastiter",
        "train_rankflow_ablate_answer_utility",
    ],
)
def test_train_experiments_keep_model_training_config_instantiable(
    experiment_name: str,
) -> None:
    cfg = _compose_train_experiment(experiment_name)

    model_cfg = resolve_instantiate_config(cfg.model)

    assert model_cfg.training_cfg is not None

    model = instantiate(model_cfg)

    assert model.cfg.training_cfg is not None


def test_validate_train_entrypoint_rejects_eval_experiment() -> None:
    cfg = OmegaConf.create(
        {
            "run": {"name": "train_rankflow"},
            "dataset": {"name": "webqsp-sub"},
        }
    )

    with pytest.raises(ValueError, match="eval experiment"):
        validate_train_entrypoint(cfg, experiment_choice="eval_llm")


def test_validate_train_entrypoint_rejects_eval_run_on_train_entrypoint() -> None:
    cfg = OmegaConf.create({"run": {"name": "rankflow"}})

    with pytest.raises(ValueError, match="requires a train run config"):
        validate_train_entrypoint(cfg)


def test_validate_train_entrypoint_requires_dataset_for_train_run() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_rankflow"}})

    with pytest.raises(ValueError, match="requires `/dataset`"):
        validate_train_entrypoint(cfg)
