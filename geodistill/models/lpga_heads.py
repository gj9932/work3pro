"""LPGA prediction heads + composed LPGA wrapper (paper §3.3, §3.6).

Per paper §3.3 input isolation:

    DepthHead  : reads g_mono only             (predicts companded ζ)
    ReliHead   : reads g_mono only             (predicts q_hat)
    RayHead    : reads g_mono only             (predicts δ_hat)
    RiskHead   : reads [g_mono, e_ego]         (the only head that sees ego state)

The factorized allocation (§3.6) is

    a_hat = r_hat * [η + (1 - η) * q_hat]                          (no extra a-regression head)
    d_hat = F_inv(ζ_hat, StopGrad(a_hat), D_max, β)

So ``LPGA`` composes the four heads + the analytic decode + the StopGrad. Both
properties are guarded by tests in ``test/geodistill/lpga_test.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from geodistill.core.r2ac import r2ac_inverse
from .embeddings import EgoStateEmbedding


__all__ = ["DepthHead", "RiskHead", "ReliHead", "RayHead", "LPGAConfig", "LPGA"]


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class DepthHead(nn.Module):
    """Predicts ζ_hat ∈ [0, 1] (companded depth). Reads g_mono only."""

    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.net = _mlp(d_in, d_hidden, 1)

    def forward(self, g_mono: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(g_mono).squeeze(-1))


class ReliHead(nn.Module):
    """Predicts q_hat ∈ [0, 1] (image-side reliability). Reads g_mono only."""

    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.net = _mlp(d_in, d_hidden, 1)

    def forward(self, g_mono: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(g_mono).squeeze(-1))


class RayHead(nn.Module):
    """Predicts sub-token ray offset δ_hat ∈ [-1, 1]^2. Reads g_mono only."""

    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.net = _mlp(d_in, d_hidden, 2)

    def forward(self, g_mono: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(g_mono))


class RiskHead(nn.Module):
    """Predicts r_hat ∈ [0, 1]. The ONLY head that consumes e_ego."""

    def __init__(self, d_in: int, d_e_ego: int, d_hidden: int = 256):
        super().__init__()
        self.d_e_ego = d_e_ego
        self.net = _mlp(d_in + d_e_ego, d_hidden, 1)

    def forward(self, g_mono: torch.Tensor, e_ego: torch.Tensor) -> torch.Tensor:
        if e_ego.shape[-1] != self.d_e_ego:
            raise ValueError(f"RiskHead expects e_ego dim={self.d_e_ego}, got {e_ego.shape[-1]}")
        if e_ego.shape[0] == 1 and g_mono.shape[0] != 1:
            e_ego = e_ego.expand(g_mono.shape[0], -1)
        x = torch.cat([g_mono, e_ego], dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


@dataclass(frozen=True)
class LPGAConfig:
    d_mono: int = 512
    d_e_ego: int = 32
    d_hidden: int = 256
    D_max: float = 80.0
    beta: float = float(torch.log(torch.tensor(16.0)))
    eta: float = 0.5                       # minimum risk allocation η
    a_eps: float = 1e-3
    use_yaw_rate: bool = False


class LPGA(nn.Module):
    """LPGA = TokenEncoder/GeoEncoder consumers + 4 heads + analytic decode.

    The token / geo encoders live in :mod:`geodistill.models.lpga_token_encoder`;
    this module is purposely scope-limited to the 4 heads, the analytic
    composition of ``a_hat``, and the StopGrad-on-decode. That makes input
    isolation testable in one place.
    """

    def __init__(self, cfg: LPGAConfig):
        super().__init__()
        self.cfg = cfg
        self.depth_head = DepthHead(cfg.d_mono, cfg.d_hidden)
        self.reli_head = ReliHead(cfg.d_mono, cfg.d_hidden)
        self.ray_head = RayHead(cfg.d_mono, cfg.d_hidden)
        self.risk_head = RiskHead(cfg.d_mono, cfg.d_e_ego, cfg.d_hidden)
        self.ego_embed = EgoStateEmbedding(cfg.d_e_ego, use_yaw_rate=cfg.use_yaw_rate)

        # Defensive: reject any module/parameter named like an a-regression head.
        for n in ("a_head", "a_proj", "alloc_head"):
            if hasattr(self, n):
                raise AssertionError(f"LPGA must not own an a-regression head ({n}); main method is analytic.")

    # --------------------------------------------------------------- compose
    def compose_a(self, r_hat: torch.Tensor, q_hat: torch.Tensor) -> torch.Tensor:
        eta = self.cfg.eta
        return r_hat * (eta + (1.0 - eta) * q_hat)

    def decode_depth(self, zeta_hat: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # StopGrad on a_hat: depth gradient must NOT flow back through F_inv into
        # the allocation, per paper §3.6.
        return r2ac_inverse(zeta_hat, a_hat.detach(), self.cfg.D_max, self.cfg.beta, self.cfg.a_eps)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        g_mono: torch.Tensor,                                  # (N, d_mono)
        v_ego: torch.Tensor | float,
        yaw_rate: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        N = g_mono.shape[0]
        v_ego_t = torch.as_tensor(v_ego, dtype=torch.float32, device=g_mono.device).reshape(-1)
        yaw_t = (torch.as_tensor(yaw_rate, dtype=torch.float32, device=g_mono.device).reshape(-1)
                 if (self.cfg.use_yaw_rate and yaw_rate is not None) else None)
        e_ego = self.ego_embed(v_ego_t, yaw_t, broadcast_to=N)

        zeta_hat = self.depth_head(g_mono)
        q_hat = self.reli_head(g_mono)
        delta_hat = self.ray_head(g_mono)
        r_hat = self.risk_head(g_mono, e_ego)
        a_hat = self.compose_a(r_hat, q_hat)
        d_hat = self.decode_depth(zeta_hat, a_hat)
        return {
            "zeta_hat": zeta_hat,
            "q_hat": q_hat,
            "delta_hat": delta_hat,
            "r_hat": r_hat,
            "a_hat": a_hat,
            "d_hat": d_hat,
        }

    # ---------------------------------------------------------------- audits
    def assert_input_isolation(self, d_mono: int, calib_dim: int = 16) -> None:
        """Sanity: the heads must not have parameters that consume calibration.

        This is a lightweight structural check. The strict gradient-based check
        lives in ``test/geodistill/lpga_test.py``.
        """
        for name, head in [("depth", self.depth_head), ("reli", self.reli_head), ("ray", self.ray_head)]:
            first = next(head.parameters())
            if first.shape[-1] != d_mono:
                raise AssertionError(
                    f"{name}_head first-layer in_features={first.shape[-1]} != d_mono={d_mono}; "
                    f"calibration may have leaked in"
                )
        first = next(self.risk_head.parameters())
        if first.shape[-1] != d_mono + self.cfg.d_e_ego:
            raise AssertionError(
                f"risk_head first-layer in_features={first.shape[-1]} != d_mono + d_e_ego "
                f"({d_mono + self.cfg.d_e_ego}); ego state may be missing or calibration leaked in"
            )
