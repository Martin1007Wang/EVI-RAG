from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_ZERO = 0
_ONE = 1
_TWO = 2
_DEFAULT_USE_SPHERICAL = False
_DEFAULT_TEMPERATURE_INIT = 0.07
_DEFAULT_NORM_EPS = 1.0e-6
_DEFAULT_LOGIT_SCALE_MIN = 1.0e-3
_DEFAULT_LOGIT_SCALE_MAX = 100.0


class QCBiANetwork(nn.Module):
    DEFAULT_USE_SPHERICAL = _DEFAULT_USE_SPHERICAL
    DEFAULT_TEMPERATURE_INIT = _DEFAULT_TEMPERATURE_INIT
    DEFAULT_NORM_EPS = _DEFAULT_NORM_EPS
    DEFAULT_LOGIT_SCALE_MIN = _DEFAULT_LOGIT_SCALE_MIN
    DEFAULT_LOGIT_SCALE_MAX = _DEFAULT_LOGIT_SCALE_MAX

    def __init__(
        self,
        *,
        d_plm: int,
        d_kg: int,
        d_inter: int = 256,
        dropout: float = 0.1,
        use_spherical: bool = _DEFAULT_USE_SPHERICAL,
        temperature_init: float = _DEFAULT_TEMPERATURE_INIT,
        norm_eps: float = _DEFAULT_NORM_EPS,
        logit_scale_min: float = _DEFAULT_LOGIT_SCALE_MIN,
        logit_scale_max: float = _DEFAULT_LOGIT_SCALE_MAX,
    ) -> None:
        super().__init__()
        self.d_plm = int(d_plm)
        self.d_kg = int(d_kg)
        if self.d_plm <= _ZERO or self.d_kg <= _ZERO:
            raise ValueError("d_plm and d_kg must be > 0.")
        self.d_inter = int(d_inter)
        if self.d_inter <= _ZERO:
            raise ValueError("d_inter must be > 0.")
        self.dropout_rate = float(dropout)
        if self.dropout_rate < float(_ZERO):
            raise ValueError("dropout must be >= 0.")
        self.use_spherical = bool(use_spherical)
        self.norm_eps = float(norm_eps)
        if self.norm_eps <= float(_ZERO):
            raise ValueError("norm_eps must be > 0.")
        self.logit_scale_min = float(logit_scale_min)
        self.logit_scale_max = float(logit_scale_max)
        if self.logit_scale_min <= float(_ZERO):
            raise ValueError("logit_scale_min must be > 0.")
        if self.logit_scale_max <= self.logit_scale_min:
            raise ValueError("logit_scale_max must be > logit_scale_min.")
        if self.use_spherical:
            temperature_init = float(temperature_init)
            if temperature_init <= float(_ZERO):
                raise ValueError("temperature_init must be > 0 when use_spherical is true.")
            init_logit_scale = math.log(float(_ONE) / temperature_init)
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        else:
            self.logit_scale = None

        self.W_q = nn.Linear(self.d_plm, self.d_kg)
        self.W_u = nn.Linear(self.d_plm, self.d_kg)
        self.W_r = nn.Linear(self.d_kg, self.d_kg, bias=False)
        self.W_v = nn.Linear(self.d_kg, self.d_kg)
        self.ln_q = nn.LayerNorm(self.d_kg)
        self.ln_u = nn.LayerNorm(self.d_kg)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.W_inter = nn.Linear(self.d_kg, self.d_inter)
        self.W_gate = nn.Linear(self.d_inter, self.d_kg * _TWO)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        self._scale = float(math.sqrt(self.d_kg))

    @staticmethod
    def _l2_normalize(x: torch.Tensor, *, eps: float) -> torch.Tensor:
        denom = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        return x / denom

    def _resolve_logit_scale(self) -> torch.Tensor:
        if self.logit_scale is None:
            raise RuntimeError("logit_scale is unavailable when use_spherical is false.")
        scale = self.logit_scale.exp()
        return scale.clamp(min=self.logit_scale_min, max=self.logit_scale_max)

    def forward(
        self,
        h_q: torch.Tensor,
        h_u: torch.Tensor,
        e_r_neighbors: torch.Tensor,
        e_v_neighbors: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if e_v_neighbors.dim() == _TWO:
            return self._forward_edges(
                h_q=h_q,
                h_u=h_u,
                e_r_neighbors=e_r_neighbors,
                e_v_neighbors=e_v_neighbors,
                mask=mask,
            )
        if e_v_neighbors.dim() != 3:
            raise ValueError("e_v_neighbors must be 2D (edges) or 3D (B,K).")
        return self._forward_grouped(
            h_q=h_q,
            h_u=h_u,
            e_r_neighbors=e_r_neighbors,
            e_v_neighbors=e_v_neighbors,
            mask=mask,
        )

    def _forward_edges(
        self,
        *,
        h_q: torch.Tensor,
        h_u: torch.Tensor,
        e_r_neighbors: torch.Tensor,
        e_v_neighbors: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if h_q.dim() != _TWO or h_u.dim() != _TWO:
            raise ValueError("h_q and h_u must be [E, d].")
        if e_r_neighbors.dim() != _TWO or e_v_neighbors.dim() != _TWO:
            raise ValueError("edge relation/entity inputs must be [E, d].")
        if (
            h_q.size(0) != h_u.size(0)
            or h_u.size(0) != e_r_neighbors.size(0)
            or h_u.size(0) != e_v_neighbors.size(0)
        ):
            raise ValueError("edge inputs must share the same first dimension.")

        z_q = self.dropout(self.ln_q(self.W_q(h_q)))
        z_u = self.dropout(self.ln_u(self.W_u(h_u)))
        z_r = self.W_r(e_r_neighbors)
        z_v = self.W_v(e_v_neighbors)

        inter = F.gelu(self.W_inter(z_q))
        gate_params = self.W_gate(inter)
        gamma, beta = gate_params.chunk(_TWO, dim=-1)
        z_r_morphed = (float(_ONE) + gamma) * z_r + beta

        query_vec = z_u * z_r_morphed
        if self.use_spherical:
            query_vec = self._l2_normalize(query_vec, eps=self.norm_eps)
            z_v = self._l2_normalize(z_v, eps=self.norm_eps)
            logits = (query_vec * z_v).sum(dim=-1) * self._resolve_logit_scale()
        else:
            logits = (query_vec * z_v).sum(dim=-1) / self._scale
        if mask is not None:
            if mask.dim() != _ONE:
                raise ValueError("edge mask must be 1D.")
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        return logits

    def _forward_grouped(
        self,
        *,
        h_q: torch.Tensor,
        h_u: torch.Tensor,
        e_r_neighbors: torch.Tensor,
        e_v_neighbors: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if h_q.dim() != _TWO or h_u.dim() != _TWO:
            raise ValueError("h_q and h_u must be [B, d].")
        if e_r_neighbors.dim() != 3 or e_v_neighbors.dim() != 3:
            raise ValueError("e_r_neighbors/e_v_neighbors must be [B, K, d].")
        if h_q.size(0) != h_u.size(0) or h_q.size(0) != e_v_neighbors.size(0):
            raise ValueError("grouped inputs must share the same batch size.")

        z_q = self.dropout(self.ln_q(self.W_q(h_q)))
        z_u = self.dropout(self.ln_u(self.W_u(h_u)))
        z_r = self.W_r(e_r_neighbors)
        z_v = self.W_v(e_v_neighbors)

        inter = F.gelu(self.W_inter(z_q))
        gate_params = self.W_gate(inter)
        gamma, beta = gate_params.chunk(_TWO, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        z_r_morphed = (float(_ONE) + gamma) * z_r + beta
        query_vec = z_u.unsqueeze(1) * z_r_morphed
        if self.use_spherical:
            query_vec = self._l2_normalize(query_vec, eps=self.norm_eps)
            z_v = self._l2_normalize(z_v, eps=self.norm_eps)
            logits = (query_vec * z_v).sum(dim=-1) * self._resolve_logit_scale()
        else:
            logits = (query_vec * z_v).sum(dim=-1) / self._scale
        if mask is not None:
            if mask.dim() != _TWO:
                raise ValueError("grouped mask must be [B, K].")
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        return logits


__all__ = ["QCBiANetwork"]
