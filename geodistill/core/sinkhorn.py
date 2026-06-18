"""Sinkhorn solvers for entropic / partial optimal transport (paper §3.5).

Two entry points used by the teacher constructions:

- ``sinkhorn_balanced``: entropic OT with fixed marginals (mu^V, mu^L).
- ``sinkhorn_partial``: relaxed-marginal (<=) partial OT with a transported-mass
  budget ``m_tr`` (paper's default uses partial OT on local candidates).

Both return the coupling ``pi`` (N, M) and the iteration count actually run.
The cost matrix is the assembled ``M_{p,i}`` (+ optional FGW structure term folded
in by the caller). Everything is dense and tiny here (local candidate sets), which
is exactly the regime the paper restricts the solver to.
"""

from __future__ import annotations

import torch

__all__ = ["sinkhorn_balanced", "sinkhorn_partial"]


def sinkhorn_balanced(
    cost: torch.Tensor,
    mu_V: torch.Tensor,
    mu_L: torch.Tensor,
    epsilon: float,
    n_iters: int = 200,
    tol: float = 1e-6,
):
    """Entropic OT with hard marginals via log-domain Sinkhorn.

    Returns (pi, iters_run).
    """
    cost = torch.as_tensor(cost, dtype=torch.float32)
    N, M = cost.shape
    log_a = torch.log(mu_V.clamp_min(1e-12))
    log_b = torch.log(mu_L.clamp_min(1e-12))
    K = -cost / max(epsilon, 1e-8)                       # log-kernel
    f = torch.zeros(N)
    g = torch.zeros(M)
    iters_run = n_iters
    for it in range(n_iters):
        f_prev = f
        # f_i = log_a_i - logsumexp_j (K_ij + g_j)
        f = log_a - torch.logsumexp(K + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(K + f.unsqueeze(1), dim=0)
        if (f - f_prev).abs().max() < tol:
            iters_run = it + 1
            break
    pi = torch.exp(f.unsqueeze(1) + K + g.unsqueeze(0))
    return pi, iters_run


def sinkhorn_partial(
    cost: torch.Tensor,
    mu_V: torch.Tensor,
    mu_L: torch.Tensor,
    epsilon: float,
    m_tr: float,
    n_iters: int = 200,
    tol: float = 1e-6,
):
    """Partial OT with inequality marginals (pi 1 <= mu_V, pi^T 1 <= mu_L) and a
    total transported-mass budget ``m_tr``.

    Implemented by augmenting with a dummy row/column absorbing the un-transported
    mass (standard partial-OT reduction), then running balanced Sinkhorn on the
    augmented problem. The dummy entries carry zero cost so slack is free; the
    budget is enforced through the augmented marginals.

    Returns (pi, iters_run) with ``pi`` the (N, M) sub-coupling (dummy stripped).
    """
    cost = torch.as_tensor(cost, dtype=torch.float32)
    N, M = cost.shape
    total_V = float(mu_V.sum())
    total_L = float(mu_L.sum())
    m_tr = float(min(m_tr, total_V, total_L))
    # Augmented marginals: extra column soaks (total_V - m_tr) from rows,
    # extra row soaks (total_L - m_tr) from cols, corner soaks the rest.
    slack_V = total_V - m_tr
    slack_L = total_L - m_tr
    a_aug = torch.cat([mu_V, torch.tensor([slack_L])])           # N+1
    b_aug = torch.cat([mu_L, torch.tensor([slack_V])])           # M+1
    big = cost.max().item() * 1e3 + 1e3
    cost_aug = torch.full((N + 1, M + 1), 0.0)
    cost_aug[:N, :M] = cost
    cost_aug[N, M] = big                                          # forbid corner mass when possible
    pi_aug, iters = sinkhorn_balanced(cost_aug, a_aug, b_aug, epsilon, n_iters, tol)
    return pi_aug[:N, :M], iters
