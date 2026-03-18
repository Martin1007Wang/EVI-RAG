from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.data.preprocess.config import build_embedding_cfg


def test_build_embedding_cfg_defaults_to_gpu_runtime_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.data.preprocess.config.torch.cuda.is_available", lambda: True
    )
    cfg = OmegaConf.create(
        {
            "encoder": "test-encoder",
            "device": "auto",
            "batch_size": None,
            "fp16": None,
            "progress_bar": False,
            "embeddings_out_dir": str(tmp_path),
            "question_ctx_max_tokens": 0,
        }
    )

    embedding_cfg = build_embedding_cfg(cfg)

    assert embedding_cfg is not None
    assert embedding_cfg.device == "cuda"
    assert embedding_cfg.batch_size == 256
    assert embedding_cfg.fp16 is True


def test_build_embedding_cfg_falls_back_to_cpu_defaults_when_gpu_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.data.preprocess.config.torch.cuda.is_available", lambda: False
    )
    cfg = OmegaConf.create(
        {
            "encoder": "test-encoder",
            "device": "auto",
            "batch_size": None,
            "fp16": None,
            "progress_bar": False,
            "embeddings_out_dir": str(tmp_path),
            "question_ctx_max_tokens": 0,
        }
    )

    embedding_cfg = build_embedding_cfg(cfg)

    assert embedding_cfg is not None
    assert embedding_cfg.device == "cpu"
    assert embedding_cfg.batch_size == 64
    assert embedding_cfg.fp16 is False
