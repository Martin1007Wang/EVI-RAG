from __future__ import annotations

import torch

from src.models.components import (
    CvtNodeInitializer,
    EmbeddingBackbone,
    LogZPredictor,
    QCBiANetwork,
    SinusoidalPositionalEncoding,
)

from .constants import _NEG_ONE, _ONE, _PB_MODES, _THREE, _TWO


class DualFlowInitMixin:
    def _init_backbone(
        self,
        *,
        emb_dim: int,
        finetune: bool,
        gnn_layers: int,
        gnn_dropout: float,
    ) -> None:
        self.backbone_fwd = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            adapter_cfg=self.embedding_adapter_cfg,
        )
        self.backbone_bwd = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            adapter_cfg=self.embedding_adapter_cfg,
        )

    def _init_cvt_init(self) -> None:
        enabled = bool(self.cvt_init_cfg.get("enabled", True))
        self._cvt_enabled = enabled
        self.cvt_init_fwd = CvtNodeInitializer()
        self.cvt_init_bwd = CvtNodeInitializer()

    def _init_actor(self) -> None:
        actor_cfg = self._resolve_actor_cfg()
        self.policy_fwd = QCBiANetwork(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=actor_cfg["edge_inter_dim"],
            dropout=actor_cfg["edge_dropout"],
            use_spherical=actor_cfg["use_spherical"],
            temperature_init=actor_cfg["temperature_init"],
            norm_eps=actor_cfg["norm_eps"],
            logit_scale_min=actor_cfg["logit_scale_min"],
            logit_scale_max=actor_cfg["logit_scale_max"],
        )
        self.policy_bwd = QCBiANetwork(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=actor_cfg["edge_inter_dim"],
            dropout=actor_cfg["edge_dropout"],
            use_spherical=actor_cfg["use_spherical"],
            temperature_init=actor_cfg["temperature_init"],
            norm_eps=actor_cfg["norm_eps"],
            logit_scale_min=actor_cfg["logit_scale_min"],
            logit_scale_max=actor_cfg["logit_scale_max"],
        )
        self.forward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _TWO)
        self.backward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _THREE)
        self.start_selector = self._build_start_selector()
        self.z_time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.z_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.hidden_dim)

    def _freeze_pb_modules(self) -> None:
        for module in (self.backbone_bwd, self.policy_bwd, self.backward_ctx_proj):
            for param in module.parameters():
                param.requires_grad = False

    def _resolve_pb_mode(self) -> str:
        cfg = self._resolve_db_cfg()
        mode = str(cfg["pb_mode"]).strip().lower()
        if mode not in _PB_MODES:
            raise ValueError(f"db_cfg.pb_mode must be one of {sorted(_PB_MODES)}, got {mode!r}.")
        return mode

    def _build_context_mlp(self, *, in_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _build_start_selector(self) -> torch.nn.Module:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _TWO, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, _ONE),
        )
        self._zero_init_linear(mlp[_NEG_ONE])
        return mlp

    @staticmethod
    def _zero_init_linear(layer: torch.nn.Linear) -> None:
        torch.nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


__all__ = ["DualFlowInitMixin"]
