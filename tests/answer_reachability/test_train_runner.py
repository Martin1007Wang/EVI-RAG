from __future__ import annotations

from omegaconf import OmegaConf

from src.runs.answer_reachability import AnswerReachabilityTrainRunner


def test_train_runner_delegates_to_train_model() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "src.models.gflownet_module.GFlowNetModule"},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {
                "name": "train_rankflow",
                "train": True,
                "test": True,
                "final_eval_experiment": "rankflow",
                "final_eval_split": "test",
                "final_eval_output_subdir": "final_eval",
            },
            "fit_schedule": {"max_passes": 1.0},
        }
    )
    runner = AnswerReachabilityTrainRunner()
    seen = {"called": False}

    def _train_model(current_cfg):  # type: ignore[no-untyped-def]
        seen["called"] = current_cfg is cfg
        return {"metric": 1.0}, {"model": object()}

    runner.validate(cfg)
    metric_dict, object_dict = runner.run(cfg=cfg, train_model=_train_model)

    assert seen["called"] is True
    assert metric_dict == {"metric": 1.0}
    assert "model" in object_dict
