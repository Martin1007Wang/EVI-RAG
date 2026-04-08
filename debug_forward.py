# debug_backward.py
import torch

torch.autograd.set_detect_anomaly(True)

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os

# 清理 Hydra 全局状态（多次运行时需要）
GlobalHydra.instance().clear()

# 用绝对路径指向 configs 目录
config_dir = os.path.abspath("configs")

with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
    cfg = compose(
        config_name="train.yaml",
        overrides=[
            "experiment=train_rankflow",
            "dataset=webqsp-sub",
        ],
    )

import hydra

model = hydra.utils.instantiate(cfg.model).cuda()

dm = hydra.utils.instantiate(cfg.data)
dm.setup("fit")

loader = dm.train_dataloader()
batch = next(iter(loader))
batch = batch.to("cuda")

# 手动跑，anomaly 输出完整打到终端
loss = model.training_step(batch, 0)
loss.backward()
