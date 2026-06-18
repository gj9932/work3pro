"""R²AC + factorized-allocation losses (paper §3.6).

    w_p     = m_p^D · (q_p^teacher + ε_q)
    ζ*_p    = F(d_T, a*_p, D_max, β)        ← caller MUST pass oracle a*
    L_comp  = Σ w_p SmoothL1(ζ_hat_p, ζ*_p) / Σ w_p
    L_rank  on metric domain, with d_hat_p = F_inv(ζ_hat_p, StopGrad(a_hat_p))
    L_r     = Σ m SmoothL1(r_hat, r_p) / Σ m
    L_q     = Σ (m · s_p) SmoothL1(q_hat, q_geom) / Σ (m · s_p)
    L_alloc = λ_r · L_r + λ_q · L_q          (assert λ_r + λ_q == 1)
    L_a_aux = Σ m SmoothL1(a_hat, a*) / Σ m  (ablation only — main method does not use)

The caller is responsible for computing ``a*`` from
:func:`geodistill.core.risk_field.oracle_allocation` so that the target ζ* is
constructed with the oracle and never with the student's prediction. Putting
that responsibility outside the loss module keeps the main-method oracle path
explicit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = ["compute_w_p", "L_comp", "L_rank", "L_r", "L_q", "L_alloc", "L_a_aux"]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    weights = weights.float()
    denom = weights.sum().clamp_min(eps)
    return (values * weights).sum() / denom


def compute_w_p(m_D: torch.Tensor, q_teacher: torch.Tensor, eps_q: float) -> torch.Tensor:
    """w_p = m_p^D · (q_p^teacher + ε_q)."""
    return m_D.float() * (q_teacher.float() + eps_q)


def L_comp(
    zeta_hat: torch.Tensor,        # (N,)
    zeta_star: torch.Tensor,       # (N,) ← built from F(d_T, a*) by the caller
    w_p: torch.Tensor,             # (N,)
) -> torch.Tensor:
    elem = F.smooth_l1_loss(zeta_hat, zeta_star, reduction="none")
    return _weighted_mean(elem, w_p)


def L_rank(
    d_hat: torch.Tensor,           # (N,) — must already be d_hat = F_inv(z_hat, StopGrad(a_hat))
    d_T: torch.Tensor,             # (N,) — teacher metric depth
    pair_index: torch.Tensor,      # (E, 2) long
    pair_weight: torch.Tensor,     # (E,) — w_pq = m_pq * sqrt(w_p * w_q)
    tau_d: float = 1.0,
) -> torch.Tensor:
    if pair_index.numel() == 0:
        return d_hat.new_zeros(())
    p = pair_index[:, 0].long()
    q = pair_index[:, 1].long()
    s = torch.sign(d_T[q] - d_T[p])
    margin = -s * (d_hat[q] - d_hat[p]) / max(tau_d, 1e-6)
    elem = torch.log1p(torch.exp(margin))
    return _weighted_mean(elem, pair_weight)


def L_r(r_hat: torch.Tensor, r_p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(F.smooth_l1_loss(r_hat, r_p, reduction="none"), m.float())


def L_q(q_hat: torch.Tensor, q_geom: torch.Tensor, s_p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """L_q uses sampling-support s_p so single-point tokens cannot dominate."""
    w = (m.float() * s_p.float())
    return _weighted_mean(F.smooth_l1_loss(q_hat, q_geom, reduction="none"), w)


def L_alloc(L_r_value: torch.Tensor, L_q_value: torch.Tensor,
            lambda_r: float, lambda_q: float, atol: float = 1e-6) -> torch.Tensor:
    if abs((lambda_r + lambda_q) - 1.0) > atol:
        raise AssertionError(
            f"paper §3.6 requires λ_r + λ_q == 1, got {lambda_r} + {lambda_q} = {lambda_r + lambda_q}"
        )
    return lambda_r * L_r_value + lambda_q * L_q_value


def L_a_aux(a_hat: torch.Tensor, a_star: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Auxiliary direct regression of a_hat to a* — ABLATION ONLY.

    Paper §3.6: the main method must NOT add this loss (a_hat is analytically
    composed from r̂, q̂, so its independent regression target would create a
    redundant degree of freedom).
    """
    return _weighted_mean(F.smooth_l1_loss(a_hat, a_star, reduction="none"), m.float())
