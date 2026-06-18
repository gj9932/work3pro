"""R²AC: Risk-Reliability Adaptive Companding depth transform.

Paper §3.6 (``paper/work3pro_cvpr2027_draft5_zh_qwen(1).md``).

Continuous, analytically-invertible, strictly-monotone companding of metric
depth ``d`` controlled by a single per-token sensitivity scalar ``a`` ∈ [0, 1].

    mu(a)        = exp(beta * a) - 1
    F(d; a)      = log(1 + mu(a) * d / D_max) / log(1 + mu(a))
    F_inv(z; a)  = D_max * (exp(beta * a * z) - 1) / (exp(beta * a) - 1)

Limits (a -> 0, L'Hôpital):  F(d; 0) = d / D_max,  F_inv(z; 0) = D_max * z.
Near/far sensitivity ratio:  (dF/dd|0) / (dF/dd|D_max) = 1 + mu(a) = exp(beta * a).

This module is pure-math and dependency-light (torch only). It carries NO learned
parameters; the head that predicts ``a`` lives in ``geodistill/models/depth_heads.py``.
"""

from __future__ import annotations

import torch

__all__ = ["companding_mu", "r2ac_forward", "r2ac_inverse", "r2ac_dForward_dd", "sensitivity_ratio"]


def companding_mu(a: torch.Tensor, beta: float) -> torch.Tensor:
    """mu(a) = expm1(beta * a). Uses expm1 for small-arg numerical stability."""
    return torch.expm1(beta * a)


def r2ac_forward(d: torch.Tensor, a: torch.Tensor, D_max: float, beta: float, a_eps: float = 1e-3) -> torch.Tensor:
    """Map metric depth ``d`` -> companded scalar ``z`` ∈ [0, 1] (for d ∈ [0, D_max]).

    Args:
        d:      metric depth, any broadcastable shape.
        a:      sensitivity ∈ [0, 1], broadcastable with ``d``.
        D_max:  max depth (companding range).
        beta:   log of max sensitivity ratio (default paper: log(16)).
        a_eps:  below |a| this, use the linear limit F = d / D_max.
    """
    d = torch.as_tensor(d, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.float32)
    x = d / D_max
    mu = companding_mu(a, beta)
    # log1p / safe denom; linear branch where a is ~0.
    num = torch.log1p(mu * x)
    den = torch.log1p(mu)
    companded = num / torch.clamp(den.abs(), min=1e-12)
    linear = x
    small = a.abs() < a_eps
    return torch.where(small, linear, companded)


def r2ac_inverse(z: torch.Tensor, a: torch.Tensor, D_max: float, beta: float, a_eps: float = 1e-3) -> torch.Tensor:
    """Map companded scalar ``z`` ∈ [0, 1] -> metric depth ``d`` ∈ [0, D_max]."""
    z = torch.as_tensor(z, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.float32)
    ba = beta * a
    num = torch.expm1(ba * z)
    den = torch.expm1(ba)
    decoded = D_max * num / torch.where(den.abs() < 1e-12, torch.ones_like(den), den)
    linear = D_max * z
    small = a.abs() < a_eps
    return torch.where(small, linear, decoded)


def r2ac_dForward_dd(d: torch.Tensor, a: torch.Tensor, D_max: float, beta: float, a_eps: float = 1e-3) -> torch.Tensor:
    """Analytic derivative dF/dd (>0 everywhere) — used to verify strict monotonicity."""
    d = torch.as_tensor(d, dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.float32)
    mu = companding_mu(a, beta)
    den = torch.log1p(mu) * (1.0 + mu * d / D_max) * D_max
    grad = mu / torch.clamp(den.abs(), min=1e-12)
    linear = torch.full_like(d, 1.0 / D_max)
    small = a.abs() < a_eps
    return torch.where(small, linear, grad)


def sensitivity_ratio(a: torch.Tensor, beta: float) -> torch.Tensor:
    """(dF/dd at d=0) / (dF/dd at d=D_max) = exp(beta * a)."""
    a = torch.as_tensor(a, dtype=torch.float32)
    return torch.exp(beta * a)
