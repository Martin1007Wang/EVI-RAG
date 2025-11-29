import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

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

logger = logging.getLogger(__name__)

# ==============================================================================
# Helper Functions (Utility Layer)
# ==============================================================================

def _get_split_loader(datamodule: LightningDataModule, split_name: str):
    """
    根据 split 名称安全地从 DataModule 获取 DataLoader。
    """
    if split_name == "train":
        # 警告：通常不建议在 Eval 模式下跑 Train 数据，除非是为了 Debug
        return datamodule._build_loader(datamodule.train_dataset, shuffle=False, drop_last=False)
    elif split_name in ["val", "validation"]:
        return datamodule.val_dataloader()
    elif split_name == "test":
        return datamodule.test_dataloader()
    else:
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
            f"LMDB file for split '{split_name}' not found at {target_path}.\n"
            f"Available LMDBs in {emb_root}: {available}"
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


# ==============================================================================
# Main Task (Logic Layer)
# ==============================================================================

@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict, dict]:
    # 1. 实例化核心组件
    # 注意：Recursive instantiation 可能导致 dataset 被初始化，但我们需要手动 setup
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Logger & Callbacks (Static)
    logger_obj = instantiate_loggers(cfg.get("logger"))
    base_callbacks = instantiate_callbacks(cfg.get("callbacks"))

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model}
    if logger_obj:
        log_hyperparameters(object_dict)

    # 2. 初始化环境
    logger.info("Setting up DataModule...")
    datamodule.setup()  # 显式 setup，确保 dataset 可用

    # 3. 初始化 Trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=base_callbacks, 
        logger=logger_obj
    )

    # ==========================
    # Phase A: Standard Metrics (Test Set)
    # ==========================
    skip_metrics = cfg.get("g_agent", {}).get("skip_retriever_metrics", False)
    
    if not skip_metrics:
        logger.info("Starting Standard Retrieval Metrics Evaluation...")
        # 这里使用 ckpt_path 加载权重
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        logger.info("Skipping standard metrics evaluation as requested.")

    # ==========================
    # Phase B: GAgent Generation (Trajectory Export)
    # ==========================
    g_agent_cfg = cfg.get("g_agent")
    if g_agent_cfg and g_agent_cfg.get("enabled"):
        logger.info("Starting GAgent Trajectory Generation...")
        
        splits = g_agent_cfg.get("splits", [])
        output_dir = Path(g_agent_cfg.get("output_dir", "g_agent_outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 预加载权重：如果 Phase A 跑过了，权重已经在内存里。
        # 如果 Phase A 跳过了，我们需要手动 load 一次，避免在循环里反复 load。
        if skip_metrics and cfg.ckpt_path:
             # 手动加载 checkpoint 权重到 model
             logger.info(f"Loading checkpoint from {cfg.ckpt_path}...")
             checkpoint = torch.load(cfg.ckpt_path, map_location=model.device)
             # 注意：Lightning checkpoint 有 'state_dict' wrapper
             state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
             model.load_state_dict(state_dict)
             model.eval() # 确保是 Eval 模式

        for split in splits:
            logger.info(f"\n[GAgent] Processing split: {split}")
            
            try:
                # 1. Resolve Resources
                loader = _get_split_loader(datamodule, split)
                lmdb_path = _resolve_lmdb_path(cfg, split)
                out_path = output_dir / f"{split}_g_agent.pt"

                # 2. Configure Settings
                include_gt_map = g_agent_cfg.get("include_gt", {})
                force_gt = include_gt_map.get(split, False) if isinstance(include_gt_map, dict) else False
                
                settings = GAgentSettings(
                    enabled=True,
                    beam_width_hop1=g_agent_cfg.beam_width_hop1,
                    final_k=g_agent_cfg.final_k,
                    output_path=out_path,
                    force_include_gt=force_gt,
                )
                
                # 3. Inject Callback & Predict
                # 使用 Context Manager 确保安全
                g_callback = GAgentGenerationCallback(settings, lmdb_path)
                
                with _temporary_callback(trainer, g_callback):
                    # ckpt_path=None 表示使用当前内存中的权重 (Efficient)
                    # 如果你不放心，可以改回 ckpt_path=cfg.ckpt_path
                    trainer.predict(
                        model=model, 
                        dataloaders=loader, 
                        ckpt_path=None 
                    )
                    
            except Exception as e:
                logger.error(f"Failed to generate trajectories for split '{split}': {e}")
                # 根据需求，这里可以选择 raise 终止，或者 continue 跑下一个 split
                raise e

    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()