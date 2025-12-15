import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule, LightningModule, Trainer, Callback

# 1. Setup Root (只做一次)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.g_agent_builder import GAgentSettings
from src.callbacks.g_agent_callback import GAgentGenerationCallback
from src.utils import instantiate_callbacks, instantiate_loggers, log_hyperparameters, task_wrapper
from src.models.retriever_module import RetrieverModule
from src.models.gflownet_module import GFlowNetModule
from src.models.llm_reasoner_module import LLMReasonerModule

logger = logging.getLogger(__name__)

# ==============================================================================
# Helper Functions (Utility Layer)
# ==============================================================================


def _load_checkpoint_strict(model: LightningModule, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()


def _extract_shared_resources(datamodule: LightningDataModule) -> Optional[Any]:
    if hasattr(datamodule, "shared_resources"):
        return getattr(datamodule, "shared_resources")
    if hasattr(datamodule, "_shared_resources"):
        return getattr(datamodule, "_shared_resources")
    return None


def _bootstrap_embedder(model: LightningModule, datamodule: LightningDataModule) -> None:
    resources = _extract_shared_resources(datamodule)
    if resources is None or not hasattr(model, "embedder"):
        return
    model.embedder.setup(resources, device=model.device)  # type: ignore[attr-defined]
    if hasattr(model.embedder, "_init_query_projection"):  # type: ignore[attr-defined]
        model.embedder._init_query_projection(question_dim=model.embedder.entity_dim)  # type: ignore[attr-defined]


def _get_split_loader(datamodule: LightningDataModule, split_name: str) -> Any:
    """
    根据 split 名称安全地从 DataModule 获取 DataLoader。
    """
    resolver = getattr(datamodule, "get_split_dataloader", None)
    if callable(resolver):
        return resolver(split_name)
    if split_name == "train":
        if hasattr(datamodule, "train_eval_dataloader"):
            return datamodule.train_eval_dataloader()  # type: ignore[attr-defined]
        # 退回到标准 train_dataloader，但可能包含 shuffle
        return datamodule.train_dataloader()
    if split_name in ["val", "validation"]:
        return datamodule.val_dataloader()
    if split_name == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unknown split name: {split_name}. Supported: train, val, test")


def _resolve_lmdb_path(cfg: DictConfig, split_name: str) -> Path:
    """
    Strict Path Resolution. No guessing based on private attributes.
    """
    # 1. Extract Root
    paths = cfg.get("dataset", {}).get("paths", {})
    if not paths:
        # Fallback to flattened structure if dataset is not nested
        paths = cfg.get("paths", {})

    emb_root_str = paths.get("embeddings")
    if not emb_root_str:
        raise ValueError(
            "Critical Config Error: Could not find 'embeddings' path in config. "
            "Ensure 'dataset.paths.embeddings' or 'paths.embeddings' is set."
        )

    emb_root = Path(emb_root_str)
    if not emb_root.exists():
        raise FileNotFoundError(f"Embeddings root directory does not exist: {emb_root}")

    # 2. Map Split to Filename (Deterministic)
    filename_map = {
        "train": "train.lmdb",
        "test": "test.lmdb",
        "validation": "validation.lmdb",
        "val": "validation.lmdb",
    }

    filename = filename_map.get(split_name, f"{split_name}.lmdb")
    target_path = emb_root / filename

    # 3. Fail Fast
    if not target_path.exists():
        # List available files for helpful error message
        available = [p.name for p in emb_root.glob("*.lmdb")]
        raise FileNotFoundError(
            f"LMDB file for split '{split_name}' not found at {target_path}.\n" f"Available LMDBs in {emb_root}: {available}"
        )

    return target_path


@contextmanager
def _temporary_callback(trainer: Trainer, callback: Callback):
    """
    Context Manager to safely inject and remove a callback.
    Guarantees cleanup even if predict fails.
    """
    trainer.callbacks.append(callback)
    try:
        yield
    finally:
        if callback in trainer.callbacks:
            trainer.callbacks.remove(callback)


def _run_standard_metrics(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    skip_metrics: bool,
) -> None:
    if skip_metrics:
        logger.info("Skipping standard metrics evaluation as requested.")
        return
    logger.info("Starting Standard Retrieval Metrics Evaluation...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=None)


def _run_g_agent_generation(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    cfg: DictConfig,
) -> None:
    g_agent_cfg = cfg.get("g_agent")
    if not g_agent_cfg or not g_agent_cfg.get("enabled"):
        return

    logger.info("Starting GAgent Trajectory Generation...")
    splits: List[str] = g_agent_cfg.get("splits", [])
    output_dir = Path(g_agent_cfg.get("output_dir", "g_agent_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预加载权重：evaluate 已经加载过 checkpoint；若跳过 metrics，权重同样已就绪。
    for split in splits:
        logger.info(f"\n[GAgent] Processing split: {split}")
        loader = _get_split_loader(datamodule, split)
        lmdb_path = _resolve_lmdb_path(cfg, split)

        anchor_top_k = int(g_agent_cfg.get("anchor_top_k", GAgentSettings.anchor_top_k))
        include_cfg = g_agent_cfg.get("force_include_gt", g_agent_cfg.get("include_gt", GAgentSettings.force_include_gt))
        if isinstance(include_cfg, (dict, DictConfig)):
            force_gt = bool(include_cfg.get(split, False))
        else:
            force_gt = bool(include_cfg)
        settings = GAgentSettings(
            enabled=True,
            anchor_top_k=anchor_top_k,
            output_path=output_dir / f"{split}_g_agent.pt",
            force_include_gt=force_gt,
        )

        g_callback = GAgentGenerationCallback(settings, lmdb_path)
        with _temporary_callback(trainer, g_callback):
            trainer.predict(model=model, dataloaders=loader, ckpt_path=None)


def _run_llm_reasoner_predict(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
) -> None:
    if not isinstance(model, LLMReasonerModule):
        return
    logger.info("Running LLM reasoner predict()...")
    # Force return_predictions=True to ensure outputs are available in on_predict_epoch_end.
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=None, return_predictions=True)


# ==============================================================================
# Main Task (Logic Layer)
# ==============================================================================


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict, dict]:
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger_obj = instantiate_loggers(cfg.get("logger"))
    base_callbacks = instantiate_callbacks(cfg.get("callbacks"))
    # 评估持久化配置：仅在 eval 场景下传递，训练不受影响
    persist_cfg = cfg.get("eval_persist", {})
    if isinstance(persist_cfg, DictConfig):
        persist_cfg = OmegaConf.to_container(persist_cfg, resolve=True)
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is not None:
        try:
            ds_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
        except Exception:
            ds_cfg = dataset_cfg
        for key in ("retriever", "gflownet"):
            if isinstance(persist_cfg.get(key), dict):
                persist_cfg[key].setdefault("dataset_cfg", ds_cfg)
    if isinstance(model, RetrieverModule):
        model.eval_persist_cfg = persist_cfg.get("retriever")
    elif isinstance(model, GFlowNetModule):
        model.eval_persist_cfg = persist_cfg.get("gflownet")

    logger.info("Setting up DataModule...")
    datamodule.setup()  # 显式 setup，确保 dataset 可用
    _bootstrap_embedder(model, datamodule)
    _load_checkpoint_strict(model, cfg.ckpt_path)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=base_callbacks, logger=logger_obj)

    if logger_obj:
        log_hyperparameters({"cfg": cfg, "model": model, "trainer": trainer})

    skip_metrics = cfg.get("g_agent", {}).get("skip_retriever_metrics", False)
    _run_standard_metrics(trainer, model, datamodule, skip_metrics)
    _run_g_agent_generation(trainer, model, datamodule, cfg)
    _run_llm_reasoner_predict(trainer, model, datamodule)

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model, "trainer": trainer}
    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
