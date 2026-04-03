from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
import torch

from src.train import (
    _build_final_eval_cfg,
    _maybe_load_model_weights,
    _run_final_eval_suite,
    _run_post_fit_evaluation,
)


def _make_train_eval_cfg(
    tmp_path: Path, *, monte_carlo: dict[str, object] | None = None
):
    cfg = OmegaConf.create(
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
                "final_eval_experiment": "eval_rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
        }
    )
    if monte_carlo:
        cfg.model.eval_cfg.monte_carlo = OmegaConf.merge(
            cfg.model.eval_cfg.monte_carlo, monte_carlo
        )
    return cfg


def _make_final_eval_template(
    tmp_path: Path,
    *,
    monte_carlo: dict[str, object] | None = None,
):
    cfg = OmegaConf.create(
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
    if monte_carlo:
        cfg.model.eval_cfg.monte_carlo = OmegaConf.merge(
            cfg.model.eval_cfg.monte_carlo, monte_carlo
        )
    return cfg


def test_maybe_load_model_weights_loads_real_checkpoint_state_dict(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    expected_weight = torch.tensor([[1.5, -2.0]], dtype=model.weight.dtype)
    expected_bias = torch.tensor([0.75], dtype=model.bias.dtype)

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    ckpt_path = tmp_path / "init.ckpt"
    torch.save(
        {"state_dict": {"weight": expected_weight, "bias": expected_bias}},
        ckpt_path,
    )

    _maybe_load_model_weights(
        model,
        OmegaConf.create({"init_ckpt_path": str(ckpt_path)}),
    )

    assert torch.equal(model.weight.detach(), expected_weight)
    assert torch.equal(model.bias.detach(), expected_bias)


def test_maybe_load_model_weights_rejects_non_mapping_payload(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "init.ckpt"
    torch.save({"state_dict": ["not", "a", "mapping"]}, ckpt_path)

    with pytest.raises(TypeError, match="checkpoint containing a `state_dict`"):
        _maybe_load_model_weights(
            torch.nn.Linear(2, 1),
            OmegaConf.create({"init_ckpt_path": str(ckpt_path)}),
        )


def test_build_final_eval_cfg_uses_eval_template_and_preserves_model_shape(
    monkeypatch, tmp_path
) -> None:
    train_cfg = _make_train_eval_cfg(tmp_path)
    eval_template = _make_final_eval_template(tmp_path)

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
    assert final_eval_cfg.model.eval_cfg.monte_carlo.action_pruning.per_node_top_k == 0
    assert final_eval_cfg.trainer.devices == 1
    assert list(final_eval_cfg.callbacks.keys()) == []
    assert list(final_eval_cfg.logger.keys()) == []


@pytest.mark.parametrize(
    ("train_monte_carlo", "template_monte_carlo"),
    [
        ({"rollouts": 256}, None),
        ({"batch_rollouts": 128}, None),
        ({"temperature": 0.8}, {"temperature": 1.0}),
    ],
)
def test_build_final_eval_cfg_rejects_answer_posterior_mismatch(
    monkeypatch,
    tmp_path,
    train_monte_carlo,
    template_monte_carlo,
) -> None:
    train_cfg = _make_train_eval_cfg(tmp_path, monte_carlo=train_monte_carlo)
    eval_template = _make_final_eval_template(
        tmp_path,
        monte_carlo=template_monte_carlo,
    )

    monkeypatch.setattr("src.train.compose_config", lambda **_: eval_template)

    with pytest.raises(ValueError, match="same answer-posterior estimator"):
        _build_final_eval_cfg(train_cfg, ckpt_path="/tmp/best.ckpt")


def test_run_post_fit_evaluation_builds_final_eval_cfg_and_releases_state(
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "run": {"test": True, "final_eval_experiment": "eval_rankflow"},
        }
    )
    seen: dict[str, object] = {}
    release_counts = {"teardown": 0}
    final_eval_cfg = OmegaConf.create(
        {
            "run": {"split": "validation"},
            "dataset": {"name": "webqsp-sub"},
            "model": {"eval_cfg": {"report_profile": "rank_only"}},
        }
    )

    monkeypatch.setattr(
        "src.train._resolve_post_fit_ckpt_path",
        lambda **_: "/tmp/best.ckpt",
    )

    def _build_eval_cfg(current_cfg, *, ckpt_path):  # type: ignore[no-untyped-def]
        seen["build"] = (current_cfg, ckpt_path)
        return final_eval_cfg

    def _run_final_eval_suite(eval_cfg):  # type: ignore[no-untyped-def]
        seen["suite"] = eval_cfg
        return {
            "final_eval/webqsp-sub/validation/answer/recall@10": 0.5,
        }

    monkeypatch.setattr("src.train._build_final_eval_cfg", _build_eval_cfg)
    monkeypatch.setattr("src.train._run_final_eval_suite", _run_final_eval_suite)

    model = SimpleNamespace()
    datamodule = SimpleNamespace(
        teardown=lambda: release_counts.__setitem__(
            "teardown", release_counts["teardown"] + 1
        )
    )
    trainer = SimpleNamespace(
        checkpoint_callback=SimpleNamespace(best_model_path="/tmp/best.ckpt")
    )

    metrics = _run_post_fit_evaluation(
        cfg=cfg,
        trainer=trainer,
        model=model,
        datamodule=datamodule,
    )

    assert seen["build"] == (cfg, "/tmp/best.ckpt")
    assert seen["suite"] == final_eval_cfg
    assert metrics["final_eval/webqsp-sub/validation/answer/recall@10"] == 0.5
    assert release_counts == {"teardown": 1}


def test_run_final_eval_suite_uses_fresh_eval_stack(monkeypatch, tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "model": {"eval_cfg": {"report_profile": "full"}},
            "run": {"split": "test"},
        }
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr("src.runs.rankflow.resolve_dataset_variants", lambda _cfg: [])

    def _evaluate_model(current_cfg):  # type: ignore[no-untyped-def]
        seen["fresh"] = current_cfg
        return {}, {
            "model": SimpleNamespace(get_predict_metrics=lambda: {"answer/hit@1": 0.5})
        }

    monkeypatch.setattr("src.eval.evaluate_model", _evaluate_model)

    monkeypatch.setattr(
        "src.runs.rankflow.persist_outputs",
        lambda *, cfg, callback_metrics, model, log: model.get_predict_metrics(),
    )

    metrics = _run_final_eval_suite(cfg)

    assert seen["fresh"] == cfg
    assert metrics["final_eval/webqsp-sub/test/answer/hit@1"] == 0.5


def test_run_final_eval_suite_keeps_fresh_eval_metrics_namespaced(
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

    monkeypatch.setattr("src.runs.rankflow.resolve_dataset_variants", lambda _cfg: [])

    def _evaluate_model(_current_cfg):  # type: ignore[no-untyped-def]
        return {"test/answer/hit@1": 0.25}, {"model": SimpleNamespace()}

    monkeypatch.setattr("src.eval.evaluate_model", _evaluate_model)

    monkeypatch.setattr(
        "src.runs.rankflow.persist_outputs",
        lambda *, cfg, callback_metrics, model, log: callback_metrics,
    )

    metrics = _run_final_eval_suite(cfg)

    assert metrics["final_eval/webqsp-sub/test/test/answer/hit@1"] == 0.25


def test_run_post_fit_evaluation_requires_ckpt_for_final_eval(
    monkeypatch,
) -> None:
    trainer = SimpleNamespace(
        checkpoint_callback=SimpleNamespace(best_model_path=""),
        callback_metrics={"test/answer/recall@10": 0.3},
    )
    release_counts = {"teardown": 0}
    monkeypatch.setattr(
        "src.train._build_final_eval_cfg",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected final eval")
        ),
    )
    monkeypatch.setattr(
        "src.train._run_final_eval_suite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected final eval")
        ),
    )
    cfg = OmegaConf.create(
        {
            "run": {
                "test": True,
                "final_eval_experiment": "eval_rankflow",
                "allow_test_without_checkpoint": True,
            }
        }
    )
    model = SimpleNamespace()
    datamodule = SimpleNamespace(
        teardown=lambda: release_counts.__setitem__(
            "teardown", release_counts["teardown"] + 1
        )
    )

    with pytest.raises(RuntimeError, match="requires a resolved checkpoint path"):
        _run_post_fit_evaluation(
            cfg=cfg,
            trainer=trainer,
            model=model,
            datamodule=datamodule,
        )

    assert release_counts == {"teardown": 1}


def test_run_post_fit_evaluation_releases_runtime_state_after_inprocess_test(
    monkeypatch,
) -> None:
    seen = {"tested": False, "teardown": False}
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
        model=SimpleNamespace(),
        datamodule=SimpleNamespace(teardown=lambda: seen.__setitem__("teardown", True)),
    )

    assert seen == {"tested": True, "teardown": True}
    assert metrics == {"test/answer/recall@10": 0.3}
