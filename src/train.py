from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logging_utils import get_logger
from src.utils.run_name import set_run_name_in_config

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    # 1. 随机种子与环境初始化
    L.seed_everything(cfg.get("seed", 42), workers=True)
    run_name = set_run_name_in_config(cfg)
    log.info("Starting training run: %s", run_name)

    # 2. 实例化 DataModule 和 Model
    log.info("Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    # 3. 规范化实例化 Callbacks (解决数据结构降维 Bug)
    callbacks: list[L.Callback] = []
    if "callbacks" in cfg and cfg.callbacks:
        cb_dict = hydra.utils.instantiate(cfg.callbacks)
        # 如果 YAML 配置为字典，提取其 values
        if isinstance(cb_dict, dict):
            callbacks = list(cb_dict.values())
        elif isinstance(cb_dict, list):
            callbacks = cb_dict

    # 4. 实例化 Logger 与 Trainer
    experiment_logger = hydra.utils.instantiate(cfg.get("logger", None))
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=experiment_logger,
    )

    # 5. 执行训练
    # 注意：若是断点续训，由 cfg.ckpt_path 统一接管。微调权重的逻辑请移至 Model 的 __init__ 中。
    log.info("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
    )

    # 6. 执行测试
    test_metrics = {}
    if cfg.get("test", False):
        log.info("Starting testing...")
        # 依赖框架原生路由，直接使用 "best"
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
        test_metrics = dict(trainer.callback_metrics)

    # 7. 聚合指标与超参搜索返回
    metrics = {**dict(trainer.callback_metrics), **test_metrics}
    optimized_metric = cfg.get("optimized_metric")

    if optimized_metric and optimized_metric in metrics:
        value = metrics[optimized_metric]
        if torch.is_tensor(value):
            return value.detach().item()
        return float(value)

    return None


if __name__ == "__main__":
    main()
