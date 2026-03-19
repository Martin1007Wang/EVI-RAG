from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval import _configure_eval_split, _enforce_single_gpu_eval
from src.train import (
    _align_validation_metrics_profile,
    _build_final_eval_cfg,
    _maybe_load_model_weights,
    _run_post_fit_evaluation,
)
from src.utils.entrypoint_contracts import (
    validate_eval_entry_contract,
    validate_train_entry_contract,
)


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


def test_align_validation_metrics_profile_uses_model_contract() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "contract": {"validation_metrics_profile": "rank_only"},
                "eval_cfg": {"metrics_profile": "full"},
            }
        }
    )

    _align_validation_metrics_profile(cfg)

    assert cfg.model.eval_cfg.metrics_profile == "rank_only"


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
                "contract": {"final_eval_metrics_profile": "full"},
                "policy_cfg": {"backbone": {"hidden_dim": 512}},
                "eval_cfg": {
                    "metrics_profile": "rank_only",
                    "max_expansions": 100000,
                    "max_frontier_size": 32768,
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
                    "metrics_profile": "full",
                    "max_expansions": 500000,
                    "max_frontier_size": 65536,
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
    assert final_eval_cfg.model.eval_cfg.metrics_profile == "full"
    assert final_eval_cfg.model.eval_cfg.max_expansions == 500000
    assert final_eval_cfg.model.eval_cfg.max_frontier_size == 65536
    assert final_eval_cfg.trainer.devices == 1
    assert list(final_eval_cfg.callbacks.keys()) == []
    assert list(final_eval_cfg.logger.keys()) == []


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
        lambda eval_cfg: {
            "final_eval/webqsp-sub/test/answer/recall@10": 0.5,
            "seen_cfg": eval_cfg,
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
