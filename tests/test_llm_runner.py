from __future__ import annotations

from omegaconf import OmegaConf

from src.llm.runner import LlmEvalRunner


def test_llm_runner_replays_all_requested_splits(monkeypatch) -> None:
    from lightning_utilities.core.rank_zero import rank_zero_only

    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp", "artifact_dir": "/tmp"},
            "run": {
                "name": "eval_llm",
                "split": "test",
                "run_all_splits": True,
                "splits": ["validation", "test"],
            },
            "llm": {"provider": "openai"},
        }
    )
    runner = LlmEvalRunner(run_all_splits=True, splits=("validation", "test"))
    seen_splits: list[str] = []

    def _run_llm_eval(current_cfg):  # type: ignore[no-untyped-def]
        seen_splits.append(str(current_cfg.run.split))

    monkeypatch.setattr("src.llm.runner.run_llm_eval", _run_llm_eval)
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)

    runner.run(cfg=cfg, evaluate_model=lambda _cfg: ({}, {}))

    assert seen_splits == ["validation", "test"]
    assert cfg.run.split == "test"
