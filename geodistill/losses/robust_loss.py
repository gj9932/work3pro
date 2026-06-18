"""Robustness distillation loss (paper §3.13 stage C).

    L_robust = || Pool(H_geo^deg) - StopGrad(Pool(H_geo^full)) ||_2

The pooling matches the paper convention used by ``L_keep``: per-camera mean,
then mean across cameras (see :func:`geodistill.models.geometry_injection.pool_by_camera`).
"""

from __future__ import annotations

import torch

from geodistill.models.geometry_injection import pool_by_camera


__all__ = ["L_robust"]


def L_robust(
    h_geo_deg: torch.Tensor,
    h_geo_full: torch.Tensor,
    cam_offsets_deg: torch.Tensor,
    cam_offsets_full: torch.Tensor,
) -> torch.Tensor:
    pooled_deg = pool_by_camera(h_geo_deg, cam_offsets_deg)
    pooled_full = pool_by_camera(h_geo_full, cam_offsets_full).detach()
    return (pooled_deg - pooled_full).norm(p=2)
