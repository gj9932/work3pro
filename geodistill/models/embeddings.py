"""Per-token feature embeddings for LPGA (paper §3.3).

The paper feeds every LPGA component a fixed mix of token features + small
discrete / continuous embeddings. To make input isolation enforceable at the
module level, each embedding kind lives in its own factory here:

    e_cam(p)   nn.Embedding over camera index               (discrete id)
    e_uv(p)    sinusoidal of normalized token center        (continuous 2D)
    e_ego      sinusoidal of (v_ego, yaw_rate)              (continuous 2D, scene-level)
    e_calib(p) tiny MLP over [K_norm, R6, t] per camera     (continuous 16D)

Higher-level modules (TokenEncoder / GeoEncoder / RiskHead) decide which
embeddings to actually consume.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "sinusoidal_1d",
    "CameraIdEmbedding",
    "UVEmbedding",
    "EgoStateEmbedding",
    "CalibrationEmbedding",
]


def sinusoidal_1d(x: torch.Tensor, dim: int, tau: float = 10000.0) -> torch.Tensor:
    """Map a scalar feature to a (..., dim) sin/cos embedding.

    Args:
        x: any shape; the embedding is broadcast over the last axis.
        dim: output dim (must be even; an odd dim is right-padded with 0).
        tau: frequency base (paper default 10000 for the 3D PE; reused here).
    """
    if x.dim() == 0:
        x = x.unsqueeze(0)
    half = max(1, dim // 2)
    device = x.device
    freqs = tau ** (-torch.arange(half, dtype=torch.float32, device=device) / float(half))
    args = x.unsqueeze(-1) * freqs                       # (..., half)
    out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if out.shape[-1] < dim:
        pad = torch.zeros(*out.shape[:-1], dim - out.shape[-1], device=device, dtype=out.dtype)
        out = torch.cat([out, pad], dim=-1)
    return out[..., :dim]


class CameraIdEmbedding(nn.Module):
    """Discrete per-camera embedding e_cam(p)."""

    def __init__(self, num_cameras: int, dim: int):
        super().__init__()
        self.num_cameras = num_cameras
        self.embed = nn.Embedding(num_cameras, dim)

    def forward(self, camera_id: torch.Tensor) -> torch.Tensor:
        return self.embed(camera_id.long())


class UVEmbedding(nn.Module):
    """Sinusoidal embedding of normalized token center (u/W, v/H).

    Per-axis sinusoidal with ``dim // 2`` channels each, then concatenated.
    """

    def __init__(self, dim: int, tau: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("UVEmbedding dim must be even")
        self.dim = dim
        self.tau = tau

    def forward(self, token_uv: torch.Tensor, image_hw: torch.Tensor, token_camera_id: torch.Tensor) -> torch.Tensor:
        """``token_uv``: (N,2) pixels; ``image_hw``: (Ncam,2) (H,W)."""
        H = image_hw[token_camera_id.long(), 0].float().clamp_min(1.0)
        W = image_hw[token_camera_id.long(), 1].float().clamp_min(1.0)
        u_norm = token_uv[:, 0] / W                       # (N,)
        v_norm = token_uv[:, 1] / H
        half = self.dim // 2
        eu = sinusoidal_1d(u_norm, half, self.tau)
        ev = sinusoidal_1d(v_norm, half, self.tau)
        return torch.cat([eu, ev], dim=-1)                # (N, dim)


class EgoStateEmbedding(nn.Module):
    """Sinusoidal embedding of ego state (v_ego, yaw_rate).

    Used only by the risk head (paper §3.3 input isolation rule).
    """

    def __init__(self, dim: int, tau: float = 10000.0, use_yaw_rate: bool = False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("EgoStateEmbedding dim must be even")
        self.dim = dim
        self.tau = tau
        self.use_yaw_rate = use_yaw_rate

    def forward(self, v_ego: torch.Tensor, yaw_rate: torch.Tensor | None = None,
                broadcast_to: int | None = None) -> torch.Tensor:
        """``v_ego``: scalar tensor; broadcast to N tokens if requested."""
        v = torch.as_tensor(v_ego, dtype=torch.float32).reshape(-1)
        if self.use_yaw_rate and yaw_rate is not None:
            half = self.dim // 2
            ev = sinusoidal_1d(v, half, self.tau)
            ey = sinusoidal_1d(torch.as_tensor(yaw_rate, dtype=torch.float32).reshape(-1), half, self.tau)
            e = torch.cat([ev, ey], dim=-1)
        else:
            e = sinusoidal_1d(v, self.dim, self.tau)
        if broadcast_to is not None:
            if e.shape[0] == 1:
                e = e.expand(broadcast_to, -1)
            elif e.shape[0] != broadcast_to:
                raise ValueError(f"ego embedding {tuple(e.shape)} cannot broadcast to N={broadcast_to}")
        return e


class CalibrationEmbedding(nn.Module):
    """Per-token calibration embedding e_calib(p).

    Encodes intrinsic K (fx,fy,cx,cy normalized by the image diagonal) and the
    rotation+translation of T_cam->ego (rotation as flattened 3x3 = 9, translation = 3).
    A tiny MLP projects the 16-dim raw vector to the configured embedding dim.

    NOTE: this output is consumed ONLY by ``GeoEncoder`` (§3.3) and the
    relation/back-projection pipeline. It MUST NOT enter the depth/risk/reli/ray
    heads — that constraint is enforced at the LPGA wrapper, not here.
    """

    RAW_DIM = 4 + 9 + 3   # K(4) + R(9) + t(3)

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(self.RAW_DIM, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def raw_features(K: torch.Tensor, cam_to_ego: torch.Tensor, image_hw: torch.Tensor) -> torch.Tensor:
        """Build (Ncam, RAW_DIM) raw calibration vector per camera."""
        K = K.float()
        cam_to_ego = cam_to_ego.float()
        image_hw = image_hw.float()
        diag = (image_hw[:, 0] ** 2 + image_hw[:, 1] ** 2).sqrt().clamp_min(1.0)
        fx = K[:, 0, 0] / diag
        fy = K[:, 1, 1] / diag
        cx = K[:, 0, 2] / diag
        cy = K[:, 1, 2] / diag
        R = cam_to_ego[:, :3, :3].reshape(-1, 9)
        t = cam_to_ego[:, :3, 3]
        return torch.cat([torch.stack([fx, fy, cx, cy], dim=-1), R, t], dim=-1)

    def forward(self, K: torch.Tensor, cam_to_ego: torch.Tensor, image_hw: torch.Tensor,
                token_camera_id: torch.Tensor) -> torch.Tensor:
        raw = CalibrationEmbedding.raw_features(K, cam_to_ego, image_hw)   # (Ncam, RAW)
        e_cam = self.proj(raw)                                              # (Ncam, dim)
        return e_cam[token_camera_id.long()]                                # (N, dim)
