"""Semantic-preserving geometry injection (paper §3.11).

    H_geo = H_img + α_pe · Φ(c_hat) + α_rel · W_up(z_rel)

where ``α_pe = α_pe_max · tanh(β_pe)`` and ``α_rel = α_rel_max · tanh(β_rel)``,
with ``β_pe = β_rel = 0`` at initialization, so the very first forward pass
satisfies ``H_geo == H_img`` *bit-exactly*. ``W_up`` is initialized to a small
non-zero Normal so the gates can receive non-zero gradient at step 1 (a strict
zero W_up would lock the relation branch out for ever).

The companion ``SemanticPreservingLoss`` exposes:

    L_keep = 1 - cos(Pool(H_geo), StopGrad(Pool(H_img)))      (global pool)
    L_sem  = 1 - linear_CKA(H_geo, StopGrad(H_img))           (per camera then mean)

Both consume ``cam_offsets`` (a (Ncam+1,) cumulative split) so multi-camera
batches are pooled the way the paper specifies (§3.13).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


__all__ = [
    "GeometryInjectorConfig",
    "GeometryInjector",
    "SemanticPreservingLoss",
    "linear_cka",
    "pool_by_camera",
]


@dataclass(frozen=True)
class GeometryInjectorConfig:
    hidden_size: int = 3584
    d_rel: int = 512
    alpha_pe_max: float = 1.0
    alpha_rel_max: float = 1.0
    sigma_up: float = 1e-4         # paper §3.11: small non-zero so gate has gradient


class GeometryInjector(nn.Module):
    def __init__(self, cfg: GeometryInjectorConfig):
        super().__init__()
        self.cfg = cfg
        # gate scalars (β are leaf params; α = α_max * tanh(β))
        self.beta_pe = nn.Parameter(torch.zeros(()))
        self.beta_rel = nn.Parameter(torch.zeros(()))
        self.W_up = nn.Linear(cfg.d_rel, cfg.hidden_size, bias=False)
        nn.init.normal_(self.W_up.weight, mean=0.0, std=cfg.sigma_up)

    @property
    def alpha_pe(self) -> torch.Tensor:
        return self.cfg.alpha_pe_max * torch.tanh(self.beta_pe)

    @property
    def alpha_rel(self) -> torch.Tensor:
        return self.cfg.alpha_rel_max * torch.tanh(self.beta_rel)

    def forward(
        self,
        h_img: torch.Tensor,                    # (N, hidden_size)
        pe_3d: torch.Tensor,                    # (N, hidden_size)
        z_rel: torch.Tensor,                    # (N, d_rel)
    ) -> torch.Tensor:
        if pe_3d.shape != h_img.shape:
            raise ValueError(f"pe_3d shape {tuple(pe_3d.shape)} != h_img {tuple(h_img.shape)}")
        if z_rel.shape[-1] != self.cfg.d_rel:
            raise ValueError(f"z_rel last dim {z_rel.shape[-1]} != d_rel {self.cfg.d_rel}")
        return h_img + self.alpha_pe * pe_3d + self.alpha_rel * self.W_up(z_rel)


# ----------------------------------------------------------------------- pool / CKA helpers
def pool_by_camera(features: torch.Tensor, cam_offsets: torch.Tensor) -> torch.Tensor:
    """Pool tokens of each camera (mean), then mean across cameras (paper §3.13).

    Returns (D,) global vector.
    """
    cam_offsets = cam_offsets.long()
    Ncam = int(cam_offsets.numel() - 1)
    cams = []
    for c in range(Ncam):
        lo = int(cam_offsets[c])
        hi = int(cam_offsets[c + 1])
        if hi > lo:
            cams.append(features[lo:hi].mean(dim=0))
    if not cams:
        return features.new_zeros(features.shape[-1])
    return torch.stack(cams, dim=0).mean(dim=0)


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Linear CKA between two (N, D) matrices (paper §3.11).

    CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c = X - mean_row(X).
    """
    if X.shape != Y.shape:
        raise ValueError(f"linear_cka shapes mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    num = (Xc.T @ Yc).pow(2).sum()
    den = (Xc.T @ Xc).pow(2).sum().sqrt() * (Yc.T @ Yc).pow(2).sum().sqrt()
    return num / (den + eps)


class SemanticPreservingLoss(nn.Module):
    """L_keep + L_sem with StopGrad on the H_img side."""

    def __init__(self):
        super().__init__()

    def forward(self, h_geo: torch.Tensor, h_img: torch.Tensor, cam_offsets: torch.Tensor) -> dict[str, torch.Tensor]:
        h_img = h_img.detach()

        pooled_geo = pool_by_camera(h_geo, cam_offsets)
        pooled_img = pool_by_camera(h_img, cam_offsets)
        cos = torch.nn.functional.cosine_similarity(pooled_geo.unsqueeze(0), pooled_img.unsqueeze(0), dim=-1)
        L_keep = 1.0 - cos.squeeze(0)

        Ncam = int(cam_offsets.numel() - 1)
        ckas = []
        for c in range(Ncam):
            lo = int(cam_offsets[c]); hi = int(cam_offsets[c + 1])
            if hi - lo >= 2:
                ckas.append(linear_cka(h_geo[lo:hi], h_img[lo:hi]))
        if ckas:
            cka = torch.stack(ckas).mean()
        else:
            cka = h_geo.new_tensor(1.0)
        L_sem = 1.0 - cka

        return {"L_keep": L_keep, "L_sem": L_sem, "cka": cka}
