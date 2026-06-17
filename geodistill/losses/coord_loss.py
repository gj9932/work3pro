"""L_coord and L_ray (paper §3.7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = ["L_coord", "L_ray"]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    weights = weights.float()
    denom = weights.sum().clamp_min(eps)
    return (values * weights).sum() / denom


def L_coord(c_hat: torch.Tensor, c_T: torch.Tensor, w_p: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Huber on per-token (N,3) ego coordinates. ``c_T`` is detached by the caller."""
    elem = F.huber_loss(c_hat, c_T, reduction="none", delta=beta).sum(dim=-1)
    return _weighted_mean(elem, w_p)


def L_ray(delta_hat: torch.Tensor, delta_T: torch.Tensor, w_p: torch.Tensor) -> torch.Tensor:
    elem = F.smooth_l1_loss(delta_hat, delta_T, reduction="none").sum(dim=-1)
    return _weighted_mean(elem, w_p)
