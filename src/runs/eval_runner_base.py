from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig

from .common import resolve_dataset_variants, resolve_splits, temporary_cfg_overrides

EvaluateModelFn = Callable[[DictConfig], Tuple[Dict[str, Any], Dict[str, Any]]]


@dataclass
class BaseEvalRunner(ABC):
    name: str = ""
    task_name: str = "eval"
    tags: tuple[str, ...] = ()
    contract: dict[str, Any] | None = None
    split: str = "test"
    run_all_splits: bool = False
    splits: tuple[str, ...] = ("train", "validation", "test")
    ckpt_path: str | None = None
    dataset_variants: Any = None
    dataset_variant: str | None = None
    execution_mode: str = "predict"
    eval_mode: str | None = None

    def run(self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn) -> None:
        if self.dataset_variants:
            self._run_all_datasets(cfg=cfg, evaluate_model=evaluate_model)
            return
        self._run_split_mode(cfg=cfg, evaluate_model=evaluate_model)

    def _run_split_mode(
        self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn
    ) -> None:
        if not self.run_all_splits:
            self._run_once(cfg=cfg, evaluate_model=evaluate_model)
            return
        for split in resolve_splits(self.splits):
            self._logger().info("%s: split=%s", self._split_log_label(), split)
            with temporary_cfg_overrides(
                cfg,
                run_overrides=self._build_split_run_overrides(cfg=cfg, split=split),
            ):
                self._run_once(cfg=cfg, evaluate_model=evaluate_model)

    def _run_all_datasets(
        self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn
    ) -> None:
        variants = self._resolve_dataset_variants(cfg)
        if not variants:
            raise ValueError(
                "run.dataset_variants must be a non-empty list when evaluating multiple datasets."
            )
        for variant in variants:
            self._logger().info(
                "%s: dataset_variant=%s",
                self._dataset_log_label(),
                variant.label,
            )
            with temporary_cfg_overrides(
                cfg,
                dataset_cfg=variant.dataset_cfg,
                run_overrides={
                    **variant.run_overrides,
                    "dataset_variant": variant.label,
                },
            ):
                self._run_split_mode(cfg=cfg, evaluate_model=evaluate_model)

    def _resolve_dataset_variants(self, cfg: DictConfig):
        if not self._supports_dataset_variants():
            raise ValueError(f"{self.name} does not support run.dataset_variants.")
        return resolve_dataset_variants(cfg)

    def _supports_dataset_variants(self) -> bool:
        return False

    def _split_log_label(self) -> str:
        return "eval"

    def _dataset_log_label(self) -> str:
        return self._split_log_label()

    def _build_split_run_overrides(
        self, *, cfg: DictConfig, split: str
    ) -> dict[str, Any]:
        del cfg
        return {"split": split}

    @abstractmethod
    def _run_once(self, *, cfg: DictConfig, evaluate_model: EvaluateModelFn) -> None:
        raise NotImplementedError

    @abstractmethod
    def _logger(self) -> Any:
        raise NotImplementedError


__all__ = ["BaseEvalRunner", "EvaluateModelFn"]
