from __future__ import annotations

from typing import Optional


class DualFlowRuntimeMixin:
    def setup(self, stage: Optional[str] = None) -> None:
        _ = stage
        self._ensure_runtime_initialized()

    def _ensure_runtime_initialized(self) -> None:
        if self._cvt_mask is not None and self._relation_inverse_map is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("datamodule is required to initialize CVT assets.")
        resources = getattr(datamodule, "shared_resources", None)
        if resources is None:
            raise RuntimeError("datamodule.shared_resources is required to initialize CVT assets.")
        self._cvt_mask = resources.cvt_mask
        inverse_suffix = self._resolve_inverse_suffix()
        inverse_map, inverse_mask = resources.relation_inverse_assets(suffix=inverse_suffix)
        self._relation_inverse_map = inverse_map
        self._relation_inverse_mask = inverse_mask
        self._relation_vocab_size = int(inverse_map.numel())


__all__ = ["DualFlowRuntimeMixin"]
