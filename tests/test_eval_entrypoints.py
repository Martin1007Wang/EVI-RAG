from __future__ import annotations

from pathlib import Path
from hydra import compose, initialize_config_dir
import pytest
from omegaconf import OmegaConf

from src.eval import _enforce_single_gpu_eval
from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.runs.entrypoints import validate_eval_entrypoint

_HYDRA_TEST_OVERRIDES = ["hydra/job_logging=stdout", "hydra/hydra_logging=none"]


def test_enforce_single_gpu_eval_rejects_non_gpu_accelerator() -> None:
    trainer_cfg = OmegaConf.create(
        {"accelerator": "cpu", "devices": 1, "strategy": "auto"}
    )

    with pytest.raises(ValueError, match="非 GPU accelerator"):
        _enforce_single_gpu_eval(trainer_cfg)


def test_enforce_single_gpu_eval_accepts_single_gpu_auto_strategy() -> None:
    trainer_cfg = OmegaConf.create(
        {"accelerator": "gpu", "devices": 1, "strategy": "auto"}
    )

    _enforce_single_gpu_eval(trainer_cfg)


def test_rankflow_eval_config_resolves_model_scheduler_without_trainer_max_steps() -> (
    None
):
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval_rankflow",
                "ckpt.gflownet=/tmp/mock.ckpt",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    assert "max_steps" in cfg.trainer
    assert cfg.trainer.max_steps is None
    assert (
        OmegaConf.to_container(cfg.model, resolve=True)["scheduler_cfg"]["t_max"]
        is None
    )


def test_eval_llm_run_inherits_eval_defaults() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "run=eval_llm",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    assert cfg.run.name == "eval_llm"
    assert cfg.run.execution_mode == "predict"
    assert cfg.run.dataset_variants is None
    assert cfg.run.ckpt_path is None
    assert list(cfg.run.tags) == ["eval", "llm"]


def test_eval_edge_retrieval_inherits_rankflow_eval_stack() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval_edge_retrieval",
                "ckpt.gflownet=/tmp/mock.ckpt",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    assert cfg.run.name == "rankflow"
    assert cfg.run.artifact_subdir == "eval_edge_retrieval"
    assert list(cfg.run.tags) == ["eval", "edge_retrieval"]
    assert "action_prior_cfg" not in cfg.model
    assert cfg.model.eval_cfg.task == "edge_retrieval"
    assert cfg.model.eval_cfg.report_profile == "rank_only"
    assert cfg.model.eval_cfg.monte_carlo.rollouts == 4096
    assert cfg.model.eval_cfg.monte_carlo.batch_rollouts == 256
    assert cfg.model.eval_cfg.monte_carlo.temperature == pytest.approx(1.0)


def test_normalize_search_eval_cfg_rejects_legacy_exact_keys() -> None:
    with pytest.raises(ValueError, match="Legacy exact answer-posterior config"):
        normalize_search_eval_cfg(
            OmegaConf.create(
                {
                    "report_profile": "full",
                    "answer_posterior_backend": "flow_frontier",
                    "flow_frontier": {"max_expansions": 1},
                }
            )
        )


def test_normalize_search_eval_cfg_rejects_removed_answer_mass_threshold() -> None:
    with pytest.raises(ValueError, match="Removed search-eval config"):
        normalize_search_eval_cfg(
            OmegaConf.create(
                {
                    "report_profile": "full",
                    "answer_mass_threshold": 0.9,
                }
            )
        )


def test_normalize_search_eval_cfg_populates_runtime_sampling_knobs() -> None:
    cfg = normalize_search_eval_cfg(
        OmegaConf.create(
            {
                "report_profile": "rank_only",
                "monte_carlo": {"rollouts": 128},
            }
        )
    )

    assert cfg["monte_carlo"]["rollouts"] == 128
    assert cfg["monte_carlo"]["batch_rollouts"] == 256
    assert cfg["monte_carlo"]["temperature"] == pytest.approx(1.0)
    assert cfg["monte_carlo"]["confidence"] == pytest.approx(0.95)
    assert cfg["monte_carlo"]["early_stop"] == {
        "enabled": True,
        "min_rollouts": 512,
        "stability_top_k": 1,
    }
    assert cfg["monte_carlo"]["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }


def test_validate_eval_entrypoint_accepts_eval_llm_without_dataset() -> None:
    cfg = OmegaConf.create(
        {"run": {"name": "eval_llm"}, "llm": {"providers": ["vllm"]}}
    )

    validate_eval_entrypoint(cfg)


def test_validate_eval_entrypoint_requires_llm_for_eval_llm() -> None:
    cfg = OmegaConf.create({"run": {"name": "eval_llm"}})

    with pytest.raises(ValueError, match="requires `/llm`"):
        validate_eval_entrypoint(cfg)


def test_validate_eval_entrypoint_rejects_train_run_on_eval_entrypoint() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_rankflow"}})

    with pytest.raises(ValueError, match="requires an eval run config"):
        validate_eval_entrypoint(cfg)


def test_validate_eval_entrypoint_requires_dataset_for_rankflow() -> None:
    cfg = OmegaConf.create({"run": {"name": "rankflow"}})

    with pytest.raises(ValueError, match="requires `/dataset`"):
        validate_eval_entrypoint(cfg)
