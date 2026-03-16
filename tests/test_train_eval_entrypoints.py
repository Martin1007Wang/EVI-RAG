from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval import _enforce_single_gpu_eval
from src.train import _maybe_load_model_weights
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


def test_validate_train_entry_contract_rejects_eval_experiment() -> None:
    cfg = OmegaConf.create(
        {
            "run": {"name": "train_answer_reachability"},
            "dataset": {"name": "webqsp-sub"},
        }
    )

    with pytest.raises(ValueError, match="eval experiment"):
        validate_train_entry_contract(cfg, experiment_choice="eval_answer_reachability")


def test_validate_train_entry_contract_requires_dataset_for_train_run() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_answer_reachability"}})

    with pytest.raises(ValueError, match="requires `/dataset`"):
        validate_train_entry_contract(cfg)


def test_validate_eval_entry_contract_accepts_eval_llm_without_dataset() -> None:
    cfg = OmegaConf.create({"run": {"name": "eval_llm"}, "llm": {"provider": "vllm"}})

    validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_requires_llm_for_eval_llm() -> None:
    cfg = OmegaConf.create({"run": {"name": "eval_llm"}})

    with pytest.raises(ValueError, match="requires `/llm`"):
        validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_rejects_train_run_on_eval_entrypoint() -> None:
    cfg = OmegaConf.create({"run": {"name": "train_answer_reachability"}})

    with pytest.raises(ValueError, match="requires an eval run config"):
        validate_eval_entry_contract(cfg)


def test_validate_eval_entry_contract_requires_dataset_for_answer_reachability() -> (
    None
):
    cfg = OmegaConf.create({"run": {"name": "eval_answer_reachability"}})

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
                    "recommended_experiment": "eval_answer_reachability",
                },
            }
        }
    )

    with pytest.raises(ValueError, match="experiment=eval_answer_reachability"):
        validate_eval_entry_contract(cfg)
