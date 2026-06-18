"""LPGA token encoder ``A_token`` and geometry encoder ``A_geo`` (paper §3.3).

Hierarchy (input isolation is the contract):

    h_p (3584)
       │
       ▼
    LN(h_p) + e_cam(p) + e_uv(p)   ──► A_token  ──► g_p^mono ∈ R^{N×512}
                                                        │
                                                        ├──► (DepthHead, ReliHead, RayHead)  – read g_mono only
                                                        │
                                                        └──► concat e_calib(p) ──► A_geo ──► g_p^geo ∈ R^{N×512}
                                                                                            │
                                                                                            └──► relation edge encoders
    e_ego ──► RiskHead([g_mono, e_ego])

Calibration NEVER enters g_mono (so depth/reli/ray cannot exploit dataset
camera priors). Ego state ONLY enters RiskHead. Both are enforced at
``LPGA._assert_isolated()`` (defined in :mod:`geodistill.models.lpga_heads`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .embeddings import CameraIdEmbedding, UVEmbedding, CalibrationEmbedding


__all__ = ["TokenEncoderConfig", "TokenEncoder", "GeoEncoder"]


@dataclass(frozen=True)
class TokenEncoderConfig:
    d_in: int = 3584                # Qwen merger output (override on synthetic)
    d_bottleneck: int = 512
    d_e_cam: int = 32
    d_e_uv: int = 64
    d_e_calib: int = 64
    layers: int = 2
    kind: str = "mlp"               # "mlp" | "transformer"
    transformer_heads: int = 8
    dropout: float = 0.0
    num_cameras: int = 6


def _make_mlp(d_in: int, d_out: int, layers: int, dropout: float = 0.0) -> nn.Module:
    if layers <= 1:
        return nn.Linear(d_in, d_out)
    mods: list[nn.Module] = [nn.Linear(d_in, d_out), nn.GELU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(d_out, d_out), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
    mods.append(nn.Linear(d_out, d_out))
    return nn.Sequential(*mods)


class TokenEncoder(nn.Module):
    """A_token: produces g_p^mono from frozen Qwen tokens + (e_cam, e_uv)."""

    def __init__(self, cfg: TokenEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_in)
        self.e_cam = CameraIdEmbedding(cfg.num_cameras, cfg.d_e_cam)
        self.e_uv = UVEmbedding(cfg.d_e_uv)

        d_concat = cfg.d_in + cfg.d_e_cam + cfg.d_e_uv
        if cfg.kind == "mlp":
            self.body = _make_mlp(d_concat, cfg.d_bottleneck, cfg.layers, cfg.dropout)
        elif cfg.kind == "transformer":
            self.in_proj = nn.Linear(d_concat, cfg.d_bottleneck)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_bottleneck,
                nhead=cfg.transformer_heads,
                dim_feedforward=4 * cfg.d_bottleneck,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.body = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
        else:
            raise ValueError(f"unknown TokenEncoder kind: {cfg.kind!r}")

    def forward(
        self,
        h_img: torch.Tensor,                       # (N, d_in)
        token_camera_id: torch.Tensor,             # (N,)
        token_uv: torch.Tensor,                    # (N, 2)
        image_hw: torch.Tensor,                    # (Ncam, 2)
    ) -> torch.Tensor:
        if h_img.shape[-1] != self.cfg.d_in:
            raise ValueError(f"TokenEncoder expects d_in={self.cfg.d_in}, got {h_img.shape[-1]}")
        x = torch.cat([
            self.norm(h_img),
            self.e_cam(token_camera_id),
            self.e_uv(token_uv, image_hw, token_camera_id),
        ], dim=-1)
        if self.cfg.kind == "mlp":
            return self.body(x)                                                # (N, d_bottleneck)
        # transformer treats N as a single sequence
        x = self.in_proj(x).unsqueeze(0)                                      # (1, N, D)
        return self.body(x).squeeze(0)


class GeoEncoder(nn.Module):
    """A_geo: produces g_p^geo from g_p^mono concatenated with calibration.

    Calibration enters here ONLY. Down-stream consumers (relation edge encoders,
    cross-view back-projection) read ``g_geo``; depth/reli/ray heads read
    ``g_mono`` directly.
    """

    def __init__(self, d_in: int, d_e_calib: int, d_out: int, layers: int = 2,
                 dropout: float = 0.0, num_cameras: int = 6):
        super().__init__()
        self.d_in = d_in
        self.d_e_calib = d_e_calib
        self.d_out = d_out
        self.calib = CalibrationEmbedding(d_e_calib)
        self.body = _make_mlp(d_in + d_e_calib, d_out, layers=layers, dropout=dropout)

    def forward(
        self,
        g_mono: torch.Tensor,                      # (N, d_in)
        K: torch.Tensor,                           # (Ncam, 3, 3)
        cam_to_ego: torch.Tensor,                  # (Ncam, 4, 4)
        image_hw: torch.Tensor,                    # (Ncam, 2)
        token_camera_id: torch.Tensor,             # (N,)
    ) -> torch.Tensor:
        e_calib = self.calib(K, cam_to_ego, image_hw, token_camera_id)        # (N, d_e_calib)
        x = torch.cat([g_mono, e_calib], dim=-1)
        return self.body(x)                                                    # (N, d_out)
