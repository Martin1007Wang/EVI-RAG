from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
from omegaconf import OmegaConf

from src.eval import (
    _configure_eval_split,
    _enforce_inprocess_eval_precision,
    _enforce_single_gpu_eval,
)
from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.train import (
    _build_final_eval_cfg,
    _maybe_load_model_weights,
    _run_final_eval_suite,
    _run_post_fit_evaluation,
)
from src.utils.entrypoint_contracts import (
    validate_eval_entry_contract,
    validate_train_entry_contract,
)
from src.utils.entrypoint_utils import _strip_instantiate_metadata

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


def test_enforce_inprocess_eval_precision_rejects_mismatch() -> None:
    cfg = OmegaConf.create({"trainer": {"precision": "32-true"}})

    with pytest.raises(ValueError, match="precision mismatch"):
        _enforce_inprocess_eval_precision(
            cfg,
            trainer=SimpleNamespace(precision="bf16-mixed"),
        )


def test_rankflow_eval_config_resolves_model_scheduler_without_trainer_max_steps() -> (
    None
):
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=rankflow",
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
    assert cfg.run.artifact_name == "eval_edge_retrieval"
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
        "per_node_top_k": 100,
        "per_state_top_k": 256,
    }


def test_configure_eval_split_updates_datamodule_when_supported() -> None:
    seen: dict[str, str] = {}

    class _DummyDataModule:
        def set_eval_split(self, split: str) -> None:
            seen["split"] = split

    split = _configure_eval_split(
        _DummyDataModule(),
        OmegaConf.create({"split": "validation"}),
    )

    assert split == "validation"
    assert seen == {"split": "validation"}


def test_configure_eval_split_defaults_to_test_when_missing() -> None:
    split = _configure_eval_split(object(), OmegaConf.create({}))

    assert split == "test"


def test_maybe_load_model_weights_uses_state_dict_payload(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _DummyModel:
        def load_state_dict(self, state_dict, strict):  # type: ignore[no-untyped-def]
            seen["state_dict"] = state_dict
            seen["strict"] = strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    monkeypatch.setattr(
        "src.train.torch.load",
        lambda *args, **kwargs: {"state_dict": {"layer.weight": 1}},
    )

    _maybe_load_model_weights(
        _DummyModel(),
        OmegaConf.create({"init_ckpt_path": "/tmp/init.ckpt"}),
    )

    assert seen == {"state_dict": {"layer.weight": 1}, "strict": False}


def test_train_rankflow_fastiter_experiment_disables_heavy_eval() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=train_rankflow_fastiter",
                "dataset=webqsp-sub",
                "logger=none",
                "extras.enforce_tags=false",
                "extras.print_config=false",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    training_cfg = instantiate(cfg.model.training_cfg)

    assert cfg.run.test is False
    assert cfg.data.eval_batch_size == 64
    assert cfg.data.num_workers == 8
    assert cfg.data.multiprocessing_context == "spawn"
    assert cfg.data.prefetch_factor == 2
    assert cfg.data.eval_num_workers == 8
    assert cfg.data.eval_prefetch_factor == 2
    assert cfg.data.eval_persistent_workers is True
    assert cfg.data.eval_multiprocessing_context == "spawn"
    assert cfg.data.train_feature_dtype == "auto"
    assert cfg.data.eval_feature_dtype == "auto"
    assert cfg.model.training_cfg.rollouts_per_graph == 32
    assert cfg.model.eval_cfg.report_profile == "rank_only"
    assert cfg.model.eval_cfg.monte_carlo.rollouts == 256
    assert cfg.model.eval_cfg.monte_carlo.batch_rollouts == 256
    assert cfg.model.eval_cfg.monte_carlo.temperature == pytest.approx(1.0)
    assert cfg.model.eval_cfg.monte_carlo.early_stop.enabled is True
    assert cfg.model.eval_cfg.monte_carlo.early_stop.min_rollouts == 512
    assert cfg.model.eval_cfg.monte_carlo.early_stop.stability_top_k == 1
    assert cfg.model.eval_cfg.monte_carlo.action_pruning.per_node_top_k == 100
    assert cfg.model.eval_cfg.monte_carlo.action_pruning.per_state_top_k == 256
    assert "action_prior_cfg" not in cfg.model
    assert training_cfg.proposal_bias_schedule.type == "cosine"
    assert training_cfg.proposal_bias_schedule.initial_scale == pytest.approx(0.8)
    assert training_cfg.proposal_bias_schedule.final_scale == pytest.approx(0.0)
    assert training_cfg.proposal_bias_schedule.hold_steps == 1000
    assert cfg.model.policy_cfg.state_mode == "subgraph"
    assert training_cfg.step_log_penalty == pytest.approx(0.0)
    assert training_cfg.potential_reward.answer_distance_weight == pytest.approx(0.0)
    assert (
        training_cfg.subgraph_proposal.oracle_answer_distance_weight
        == pytest.approx(0.5)
    )
    assert (
        training_cfg.subgraph_proposal.prior_question_similarity_weight
        == pytest.approx(0.75)
    )
    assert training_cfg.subgraph_proposal.prior_component_merge_weight == pytest.approx(
        1.0
    )
    assert training_cfg.success_replay.mix_alpha == pytest.approx(0.0)
    assert training_cfg.replay_mix_schedule.type == "cosine"
    assert training_cfg.replay_mix_schedule.final_alpha == pytest.approx(0.0)
    assert training_cfg.replay_mix_schedule.hold_steps == 2000
    assert cfg.callbacks.model_checkpoint.monitor == cfg.optimized_metric
    assert cfg.callbacks.early_stopping.monitor == cfg.optimized_metric
    assert cfg.fit_schedule.val_every_passes == pytest.approx(8.0)
    assert "guided" not in list(cfg.run.tags)


def test_train_rankflow_experiment_uses_canonical_monte_carlo_selector() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

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

    training_cfg = instantiate(cfg.model.training_cfg)

    assert cfg.data.batch_size == 64
    assert cfg.data.eval_batch_size == 32
    assert cfg.data.num_workers == 8
    assert cfg.data.multiprocessing_context == "spawn"
    assert cfg.data.prefetch_factor == 2
    assert cfg.data.eval_num_workers == 8
    assert cfg.data.eval_prefetch_factor == 2
    assert cfg.data.eval_persistent_workers is True
    assert cfg.data.eval_multiprocessing_context == "spawn"
    assert cfg.data.train_feature_dtype == "auto"
    assert cfg.data.eval_feature_dtype == "auto"
    assert cfg.data.train_num_samples is None
    assert "train_max_graphs_per_batch" not in cfg.data
    assert "train_max_nodes_per_batch" not in cfg.data
    assert cfg.fit_schedule.val_every_passes == pytest.approx(8.0)
    assert cfg.fit_schedule.early_stopping_patience_passes == pytest.approx(96.0)
    assert cfg.trainer.log_every_n_steps == 32
    assert cfg.model.training_cfg.rollouts_per_graph == 32
    assert cfg.model.eval_cfg.report_profile == "rank_only"
    assert cfg.model.eval_cfg.monte_carlo.rollouts == 4096
    assert cfg.model.eval_cfg.monte_carlo.batch_rollouts == 256
    assert cfg.model.eval_cfg.monte_carlo.temperature == pytest.approx(1.0)
    assert cfg.model.eval_cfg.monte_carlo.early_stop.enabled is True
    assert cfg.model.eval_cfg.monte_carlo.action_pruning.per_node_top_k == 100
    assert cfg.model.eval_cfg.monte_carlo.action_pruning.per_state_top_k == 256
    assert "action_prior_cfg" not in cfg.model
    assert training_cfg.proposal_bias_schedule.type == "cosine"
    assert training_cfg.proposal_bias_schedule.initial_scale == pytest.approx(0.8)
    assert training_cfg.proposal_bias_schedule.final_scale == pytest.approx(0.0)
    assert training_cfg.proposal_bias_schedule.hold_steps == 1000
    assert cfg.model.policy_cfg.state_mode == "subgraph"
    assert training_cfg.step_log_penalty == pytest.approx(0.0)
    assert training_cfg.potential_reward.answer_distance_weight == pytest.approx(0.0)
    assert (
        training_cfg.subgraph_proposal.oracle_answer_distance_weight
        == pytest.approx(0.5)
    )
    assert (
        training_cfg.subgraph_proposal.prior_question_similarity_weight
        == pytest.approx(0.75)
    )
    assert training_cfg.subgraph_proposal.prior_component_merge_weight == pytest.approx(
        1.0
    )
    assert training_cfg.subgraph_reward.beta_answer_bits == pytest.approx(0.2)
    assert training_cfg.subgraph_reward.beta_answer_full == pytest.approx(0.5)
    assert training_cfg.success_replay.mix_alpha == pytest.approx(0.2)
    assert training_cfg.replay_mix_schedule.type == "cosine"
    assert training_cfg.replay_mix_schedule.final_alpha == pytest.approx(0.0)
    assert training_cfg.replay_mix_schedule.hold_steps == 2000
    assert training_cfg.success_replay.min_buffer_size == 16
    assert training_cfg.success_replay.capacity == 128
    assert training_cfg.success_replay.replay_trajectories_per_step == 64
    assert training_cfg.success_replay.add_shortest_path_guidance is True
    assert training_cfg.success_replay.expand_imitation_weight == pytest.approx(1.0)
    assert (
        training_cfg.success_replay.expand_imitation_from_anchor_bonus
        == pytest.approx(2.0)
    )
    assert (
        training_cfg.success_replay.expand_imitation_answer_finish_bonus
        == pytest.approx(4.0)
    )
    assert training_cfg.success_replay.mask_stop_loss is True
    assert cfg.callbacks.model_checkpoint.monitor == cfg.optimized_metric
    assert cfg.callbacks.early_stopping.monitor == cfg.optimized_metric
    assert cfg.run.test is True
    assert "guided" not in list(cfg.run.tags)


@pytest.mark.parametrize(
    "experiment_name",
    ["train_rankflow"],
)
def test_train_experiments_keep_model_training_config_instantiable(
    experiment_name: str,
) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
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

    model_cfg = _strip_instantiate_metadata(cfg.model)

    assert model_cfg.training_cfg is not None

    model = instantiate(model_cfg)

    assert model.cfg.training_cfg is not None


def test_build_final_eval_cfg_uses_eval_template_and_preserves_model_shape(
    monkeypatch, tmp_path
) -> None:
    train_cfg = OmegaConf.create(
        {
            "seed": 7,
            "paths": {
                "output_dir": str(tmp_path / "train-run"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp-sub",
                "dataset_family": "webqsp",
                "dataset_scope": "sub",
            },
            "data": {"batch_size": 32, "num_workers": 4},
            "model": {
                "policy_cfg": {"backbone": {"hidden_dim": 512}},
                "eval_cfg": {
                    "report_profile": "rank_only",
                    "monte_carlo": {"rollouts": 4096, "batch_rollouts": 256},
                },
            },
            "run": {
                "final_eval_experiment": "rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
        }
    )
    eval_template = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "template"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp",
                "dataset_family": "webqsp",
                "dataset_scope": "full",
            },
            "data": {"batch_size": 64},
            "model": {
                "eval_cfg": {
                    "report_profile": "full",
                    "monte_carlo": {"rollouts": 4096, "batch_rollouts": 256},
                }
            },
            "trainer": {"accelerator": "gpu", "devices": 1},
            "run": {
                "name": "rankflow",
                "split": "test",
                "execution_mode": "predict",
                "dataset_variants": [
                    "${dataset.dataset_family}",
                    "${dataset.dataset_family}-sub",
                ],
                "ckpt_path": "${ckpt.gflownet}",
            },
        }
    )

    monkeypatch.setattr("src.train.compose_config", lambda **_: eval_template)

    final_eval_cfg = _build_final_eval_cfg(train_cfg, ckpt_path="/tmp/best.ckpt")

    assert final_eval_cfg.run.name == "rankflow"
    assert final_eval_cfg.run.split == "test"
    assert final_eval_cfg.ckpt_path == "/tmp/best.ckpt"
    assert final_eval_cfg.paths.output_dir.endswith("final_eval")
    assert final_eval_cfg.dataset.name == "webqsp-sub"
    assert final_eval_cfg.model.policy_cfg.backbone.hidden_dim == 512
    assert final_eval_cfg.model.eval_cfg.report_profile == "full"
    assert final_eval_cfg.model.eval_cfg.monte_carlo.rollouts == 4096
    assert final_eval_cfg.model.eval_cfg.monte_carlo.batch_rollouts == 256
    assert final_eval_cfg.model.eval_cfg.monte_carlo.temperature == pytest.approx(1.0)
    assert final_eval_cfg.model.eval_cfg.monte_carlo.early_stop.enabled is True
    assert (
        final_eval_cfg.model.eval_cfg.monte_carlo.action_pruning.per_node_top_k == 100
    )
    assert final_eval_cfg.trainer.devices == 1
    assert list(final_eval_cfg.callbacks.keys()) == []
    assert list(final_eval_cfg.logger.keys()) == []


def test_build_final_eval_cfg_rejects_monte_carlo_rollout_mismatch(
    monkeypatch, tmp_path
) -> None:
    train_cfg = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "train-run"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp-sub",
                "dataset_family": "webqsp",
                "dataset_scope": "sub",
            },
            "data": {"batch_size": 32, "num_workers": 4},
            "model": {
                "eval_cfg": {
                    "report_profile": "rank_only",
                    "monte_carlo": {"rollouts": 256, "batch_rollouts": 128},
                },
            },
            "run": {
                "final_eval_experiment": "rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
        }
    )
    eval_template = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "template"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp",
                "dataset_family": "webqsp",
                "dataset_scope": "full",
            },
            "data": {"batch_size": 64},
            "model": {
                "eval_cfg": {
                    "report_profile": "full",
                    "monte_carlo": {"rollouts": 4096, "batch_rollouts": 256},
                }
            },
            "trainer": {"accelerator": "gpu", "devices": 1},
            "run": {
                "name": "rankflow",
                "split": "test",
                "execution_mode": "predict",
                "dataset_variants": [
                    "${dataset.dataset_family}",
                    "${dataset.dataset_family}-sub",
                ],
                "ckpt_path": "${ckpt.gflownet}",
            },
        }
    )

    monkeypatch.setattr("src.train.compose_config", lambda **_: eval_template)

    with pytest.raises(ValueError, match="same answer-posterior estimator"):
        _build_final_eval_cfg(train_cfg, ckpt_path="/tmp/best.ckpt")


def test_build_final_eval_cfg_rejects_monte_carlo_batch_rollout_mismatch(
    monkeypatch, tmp_path
) -> None:
    train_cfg = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "train-run"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp-sub",
                "dataset_family": "webqsp",
                "dataset_scope": "sub",
            },
            "data": {"batch_size": 32, "num_workers": 4},
            "model": {
                "eval_cfg": {
                    "report_profile": "rank_only",
                    "monte_carlo": {"rollouts": 4096, "batch_rollouts": 128},
                },
            },
            "run": {
                "final_eval_experiment": "rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
        }
    )
    eval_template = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "template"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp",
                "dataset_family": "webqsp",
                "dataset_scope": "full",
            },
            "data": {"batch_size": 64},
            "model": {
                "eval_cfg": {
                    "report_profile": "full",
                    "monte_carlo": {"rollouts": 4096, "batch_rollouts": 256},
                }
            },
            "trainer": {"accelerator": "gpu", "devices": 1},
            "run": {
                "name": "rankflow",
                "split": "test",
                "execution_mode": "predict",
                "dataset_variants": [
                    "${dataset.dataset_family}",
                    "${dataset.dataset_family}-sub",
                ],
                "ckpt_path": "${ckpt.gflownet}",
            },
        }
    )

    monkeypatch.setattr("src.train.compose_config", lambda **_: eval_template)

    with pytest.raises(ValueError, match="same answer-posterior estimator"):
        _build_final_eval_cfg(train_cfg, ckpt_path="/tmp/best.ckpt")


def test_build_final_eval_cfg_rejects_monte_carlo_temperature_mismatch(
    monkeypatch, tmp_path
) -> None:
    train_cfg = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "train-run"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp-sub",
                "dataset_family": "webqsp",
                "dataset_scope": "sub",
            },
            "data": {"batch_size": 32, "num_workers": 4},
            "model": {
                "eval_cfg": {
                    "report_profile": "rank_only",
                    "monte_carlo": {
                        "rollouts": 4096,
                        "batch_rollouts": 256,
                        "temperature": 0.8,
                    },
                },
            },
            "run": {
                "final_eval_experiment": "rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
        }
    )
    eval_template = OmegaConf.create(
        {
            "paths": {
                "output_dir": str(tmp_path / "template"),
                "data_dir": "/mnt/data/retrieval_dataset",
            },
            "dataset": {
                "name": "webqsp",
                "dataset_family": "webqsp",
                "dataset_scope": "full",
            },
            "data": {"batch_size": 64},
            "model": {
                "eval_cfg": {
                    "report_profile": "full",
                    "monte_carlo": {
                        "rollouts": 4096,
                        "batch_rollouts": 256,
                        "temperature": 1.0,
                    },
                }
            },
            "trainer": {"accelerator": "gpu", "devices": 1},
            "run": {
                "name": "rankflow",
                "split": "test",
                "execution_mode": "predict",
                "dataset_variants": [
                    "${dataset.dataset_family}",
                    "${dataset.dataset_family}-sub",
                ],
                "ckpt_path": "${ckpt.gflownet}",
            },
        }
    )

    monkeypatch.setattr("src.train.compose_config", lambda **_: eval_template)

    with pytest.raises(ValueError, match="same answer-posterior estimator"):
        _build_final_eval_cfg(train_cfg, ckpt_path="/tmp/best.ckpt")


def test_run_post_fit_evaluation_uses_final_eval_suite(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "run": {"test": True, "final_eval_experiment": "rankflow"},
        }
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "src.train._resolve_post_fit_ckpt_path",
        lambda **_: "/tmp/best.ckpt",
    )

    def _build_eval_cfg(current_cfg, *, ckpt_path):  # type: ignore[no-untyped-def]
        seen["build"] = (current_cfg, ckpt_path)
        return OmegaConf.create({"run": {"split": "test"}})

    monkeypatch.setattr("src.train._build_final_eval_cfg", _build_eval_cfg)
    monkeypatch.setattr(
        "src.train._run_final_eval_suite",
        lambda eval_cfg, **kwargs: {
            "final_eval/webqsp-sub/test/answer/recall@10": 0.5,
            "seen_cfg": eval_cfg,
            "seen_kwargs": kwargs,
        },
    )

    metrics = _run_post_fit_evaluation(
        cfg=cfg,
        trainer=SimpleNamespace(
            checkpoint_callback=SimpleNamespace(best_model_path="/tmp/best.ckpt")
        ),
        model=SimpleNamespace(),
        datamodule=SimpleNamespace(),
    )

    assert seen["build"] == (cfg, "/tmp/best.ckpt")
    assert metrics["final_eval/webqsp-sub/test/answer/recall@10"] == 0.5


def test_run_final_eval_suite_prefers_inprocess_reuse_when_available(
    monkeypatch, tmp_path
) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "model": {"eval_cfg": {"report_profile": "full"}},
            "run": {"split": "test"},
        }
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr("src.train.resolve_dataset_variants", lambda _cfg: [])

    def _evaluate_model_inprocess(current_cfg, *, trainer, datamodule, model):  # type: ignore[no-untyped-def]
        seen["inprocess"] = (current_cfg, trainer, datamodule, model)
        return {}, {
            "model": SimpleNamespace(get_predict_metrics=lambda: {"answer/hit@1": 0.5})
        }

    monkeypatch.setattr("src.eval.evaluate_model_inprocess", _evaluate_model_inprocess)
    monkeypatch.setattr(
        "src.eval.evaluate_model",
        lambda current_cfg: (_ for _ in ()).throw(AssertionError(current_cfg)),
    )

    class _Reporter:
        def persist_outputs(self, *, cfg, callback_metrics, model, log):  # type: ignore[no-untyped-def]
            del cfg, callback_metrics, log
            return model.get_predict_metrics()

    monkeypatch.setattr("src.train.AnswerReachabilityEvalReporter", lambda: _Reporter())

    trainer = SimpleNamespace()
    datamodule = SimpleNamespace()
    model = SimpleNamespace()
    metrics = _run_final_eval_suite(
        cfg,
        trainer=trainer,
        model=model,
        datamodule=datamodule,
    )

    assert seen["inprocess"] == (cfg, trainer, datamodule, model)
    assert metrics["final_eval/webqsp-sub/test/answer/hit@1"] == 0.5


def test_run_final_eval_suite_releases_runtime_state_before_fresh_fallback(
    monkeypatch, tmp_path
) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "model": {"eval_cfg": {"report_profile": "full"}},
            "run": {"split": "test"},
        }
    )
    seen = {"reset": False, "teardown": False}

    monkeypatch.setattr("src.train.resolve_dataset_variants", lambda _cfg: [])
    monkeypatch.setattr(
        "src.eval.evaluate_model_inprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    def _evaluate_model(_current_cfg):  # type: ignore[no-untyped-def]
        assert seen == {"reset": True, "teardown": True}
        return {"test/answer/hit@1": 0.25}, {"model": SimpleNamespace()}

    monkeypatch.setattr("src.eval.evaluate_model", _evaluate_model)

    class _Reporter:
        def persist_outputs(self, *, cfg, callback_metrics, model, log):  # type: ignore[no-untyped-def]
            del cfg, model, log
            return callback_metrics

    monkeypatch.setattr("src.train.AnswerReachabilityEvalReporter", lambda: _Reporter())

    model = SimpleNamespace(
        reset_prediction_state=lambda: seen.__setitem__("reset", True)
    )
    datamodule = SimpleNamespace(teardown=lambda: seen.__setitem__("teardown", True))

    metrics = _run_final_eval_suite(
        cfg,
        trainer=SimpleNamespace(),
        model=model,
        datamodule=datamodule,
    )

    assert metrics["final_eval/webqsp-sub/test/test/answer/hit@1"] == 0.25


def test_run_post_fit_evaluation_falls_back_to_inprocess_test_when_ckpt_missing() -> (
    None
):
    trainer = SimpleNamespace(
        checkpoint_callback=SimpleNamespace(best_model_path=""),
        callback_metrics={"test/answer/recall@10": 0.3},
    )
    seen = {"called": False}

    def _test(**_: object) -> None:
        seen["called"] = True

    trainer.test = _test
    cfg = OmegaConf.create(
        {
            "run": {
                "test": True,
                "final_eval_experiment": "rankflow",
                "allow_test_without_checkpoint": True,
            }
        }
    )

    metrics = _run_post_fit_evaluation(
        cfg=cfg,
        trainer=trainer,
        model=SimpleNamespace(),
        datamodule=SimpleNamespace(),
    )

    assert seen["called"] is True
    assert metrics == {"test/answer/recall@10": 0.3}


def test_run_post_fit_evaluation_releases_runtime_state_after_inprocess_test(
    monkeypatch,
) -> None:
    seen = {"tested": False, "reset": False, "teardown": False}
    trainer = SimpleNamespace(callback_metrics={"test/answer/recall@10": 0.3})

    def _test(**_: object) -> None:
        seen["tested"] = True

    trainer.test = _test
    monkeypatch.setattr(
        "src.train._resolve_post_fit_ckpt_path",
        lambda **_: "/tmp/best.ckpt",
    )

    metrics = _run_post_fit_evaluation(
        cfg=OmegaConf.create({"run": {"test": True}}),
        trainer=trainer,
        model=SimpleNamespace(
            reset_prediction_state=lambda: seen.__setitem__("reset", True)
        ),
        datamodule=SimpleNamespace(teardown=lambda: seen.__setitem__("teardown", True)),
    )

    assert seen == {"tested": True, "reset": True, "teardown": True}
    assert metrics == {"test/answer/recall@10": 0.3}


def test_validate_train_entry_contract_rejects_eval_experiment() -> None:
    cfg = OmegaConf.create(
        {
            "run": {"name": "train_rankflow"},
            "dataset": {"name": "webqsp-sub"},
        }
    )

    with pytest.raises(ValueError, match="eval experiment"):
        validate_train_entry_contract(cfg, experiment_choice="eval_llm")


def test_validate_train_entry_contract_requires_dataset_for_train_run() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_rankflow"}})

    with pytest.raises(ValueError, match="requires `/dataset`"):
        validate_train_entry_contract(cfg)


def test_validate_eval_entry_contract_accepts_eval_llm_without_dataset() -> None:
    cfg = OmegaConf.create(
        {"run": {"name": "eval_llm"}, "llm": {"providers": ["vllm"]}}
    )

    validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_requires_llm_for_eval_llm() -> None:
    cfg = OmegaConf.create({"run": {"name": "eval_llm"}})

    with pytest.raises(ValueError, match="requires `/llm`"):
        validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_rejects_train_run_on_eval_entrypoint() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_rankflow"}})

    with pytest.raises(ValueError, match="requires an eval run config"):
        validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_requires_dataset_for_rankflow() -> None:
    cfg = OmegaConf.create({"run": {"name": "rankflow"}})

    with pytest.raises(ValueError, match="requires `/dataset`"):
        validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_uses_run_contract_metadata() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "name": "custom_eval",
                "contract": {
                    "entrypoint": "eval",
                    "required_groups": ["dataset"],
                    "recommended_experiment": "rankflow",
                },
            }
        }
    )

    with pytest.raises(ValueError, match="experiment=rankflow"):
        validate_eval_entry_contract(cfg)
