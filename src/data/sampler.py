# src/data/samplers.py
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class StepDrivenTrainSampler(Sampler[int]):
    """
    无限循环 + 固定步数的训练采样器。

    行为：
    - 每轮将全量索引 shuffle 一次，然后顺序吐出
    - 跨轮边界无缝衔接，不会因为 epoch 结束截断 batch
    - 恰好产出 num_samples 个索引后停止（对应 max_steps * batch_size）

    与 Lightning 配合：
    - Trainer(max_steps=N)  ← 控制训练步数
    - train_num_samples = N * batch_size  ← 控制本 sampler 的长度
    - Lightning 读到 sampler 时会用 len() 推算 epoch 长度，
      这里 len() 返回 num_samples，恰好让一个"伪 epoch" = 整个训练
    """

    def __init__(
        self,
        dataset_size: int,
        num_samples: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be > 0, got {dataset_size}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        emitted = 0
        while emitted < self.num_samples:
            # 每轮重新生成一个完整排列
            if self.shuffle:
                perm = torch.randperm(self.dataset_size, generator=rng).tolist()
            else:
                perm = list(range(self.dataset_size))

            for idx in perm:
                if emitted >= self.num_samples:
                    return
                yield idx
                emitted += 1
