from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.runs.output_orchestrator import RunOutputOrchestrator
from src.utils.output_sinks import PredictionArtifactSettings


def test_run_output_orchestrator_persists_metrics_and_artifacts(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "run": {"split": "test"},
        }
    )
    captured: dict[str, object] = {}

    class _Model:
        def write_prediction_artifacts(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            output_dir = Path(str(kwargs["output_dir"]))
            return {"prompt_path": output_dir / "test.jsonl"}

    orchestrator = RunOutputOrchestrator(
        collect_metrics=lambda callback_metrics, model: callback_metrics,
        resolve_metrics_filename=lambda _cfg: "metrics.json",
        save_metrics=lambda cfg, metrics, filename: Path(cfg.paths.output_dir)
        / filename,
        build_artifact_settings=lambda _cfg: PredictionArtifactSettings(
            enabled=True,
            output_root=tmp_path,
            dataset_scope="sub",
        ),
    )

    result = orchestrator.persist(
        cfg=cfg,
        callback_metrics={"answer/hit@1": 0.75},
        model=_Model(),
        log=SimpleNamespace(
            info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None
        ),
    )

    assert result.metrics == {"answer/hit@1": 0.75}
    assert result.metrics_path == tmp_path / "metrics.json"
    assert result.artifact_paths == {
        "prompt_path": tmp_path / "eval_answer_reachability" / "sub" / "test.jsonl"
    }
    assert captured["split"] == "test"


def test_run_output_orchestrator_warns_when_metrics_missing(tmp_path: Path) -> None:
    messages: list[str] = []
    orchestrator = RunOutputOrchestrator(
        collect_metrics=lambda callback_metrics, model: {},
        resolve_metrics_filename=lambda _cfg: "metrics.json",
        save_metrics=lambda cfg, metrics, filename: Path(cfg.paths.output_dir)
        / filename,
        build_artifact_settings=lambda _cfg: PredictionArtifactSettings(enabled=False),
    )

    result = orchestrator.persist(
        cfg=OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}),
        callback_metrics={},
        model=object(),
        log=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda message, *args, **kwargs: messages.append(str(message)),
        ),
    )

    assert result.metrics == {}
    assert result.metrics_path is None
    assert messages == ["No metrics were produced; skipping metrics.json."]
