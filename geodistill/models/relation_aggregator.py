"""Relation aggregator (paper §3.11).

Produces the per-token relation message ``z_p^rel`` from sparse edge messages
``r_pq``.

Modes:
- ``confidence_attention``  ω_pq = softmax(A_conf(r_pq) + log(e_conf+ε))   (main)
- ``mean``                   ω_pq = 1 / |N(p)|                              (ablation)
- ``learned_attention``      ω_pq = softmax(A_conf(r_pq))                   (ablation, no e_conf)

The aggregated message is then projected by ``V_rel`` (a Linear layer kept in
the aggregator so the d_rel contract is owned in one place):

    z_p^rel = Σ_q ω_pq · V_rel(r_pq)                                         (R^d_rel)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


__all__ = ["RelationAggregatorConfig", "RelationAggregator"]


AGG_MODES = ("confidence_attention", "mean", "learned_attention")


@dataclass(frozen=True)
class RelationAggregatorConfig:
    d_edge: int = 256
    d_rel: int = 512
    mode: str = "confidence_attention"


class RelationAggregator(nn.Module):
    def __init__(self, cfg: RelationAggregatorConfig):
        super().__init__()
        if cfg.mode not in AGG_MODES:
            raise ValueError(f"unknown aggregator mode {cfg.mode!r}")
        self.cfg = cfg
        self.V_rel = nn.Linear(cfg.d_edge, cfg.d_rel, bias=False)

    def forward(
        self,
        r_pq: torch.Tensor,                      # (E, d_edge)
        edge_index: torch.Tensor,                # (E, 2) long
        N: int,                                  # number of tokens
        conf_logit: torch.Tensor | None = None,  # (E,)
        e_conf: torch.Tensor | None = None,      # (E,) edge reliability ∈ [0, 1]
    ) -> torch.Tensor:
        E = r_pq.shape[0]
        d_rel = self.cfg.d_rel
        device = r_pq.device
        if E == 0:
            return torch.zeros((N, d_rel), device=device, dtype=r_pq.dtype)

        v = self.V_rel(r_pq)                                                # (E, d_rel)
        p_idx = edge_index[:, 0].long()

        if self.cfg.mode == "mean":
            weights = torch.ones(E, device=device, dtype=v.dtype)
            return _scatter_softmax_aggregate(p_idx, weights, v, N)

        # logit-based modes
        if self.cfg.mode == "confidence_attention":
            if conf_logit is None or e_conf is None:
                raise ValueError("confidence_attention requires conf_logit and e_conf")
            base_logit = conf_logit + torch.log(e_conf.clamp_min(1e-6))
        else:  # learned_attention (no e_conf)
            if conf_logit is None:
                raise ValueError("learned_attention requires conf_logit")
            base_logit = conf_logit

        # softmax per source token p
        return _scatter_softmax_aggregate(p_idx, base_logit, v, N, softmax=True)


def _scatter_softmax_aggregate(
    src_idx: torch.Tensor,                        # (E,) long
    weight_logit_or_uniform: torch.Tensor,        # (E,)
    values: torch.Tensor,                         # (E, d)
    N: int,
    softmax: bool = False,
) -> torch.Tensor:
    """Per-source softmax (or normalized mean) + weighted sum.

    ``softmax=False`` treats ``weight_logit_or_uniform`` as raw weights to be
    normalized by sum. ``softmax=True`` applies softmax over edges grouped by
    source.
    """
    device = values.device
    d = values.shape[-1]
    out = torch.zeros((N, d), device=device, dtype=values.dtype)
    if values.shape[0] == 0:
        return out

    if softmax:
        # group-wise softmax via the standard subtract-max trick.
        max_per_src = torch.full((N,), float("-inf"), device=device, dtype=values.dtype)
        max_per_src = max_per_src.scatter_reduce(0, src_idx, weight_logit_or_uniform, reduce="amax",
                                                 include_self=True)
        m = max_per_src[src_idx]
        exp_w = torch.exp(weight_logit_or_uniform - m)
        denom = torch.zeros(N, device=device, dtype=values.dtype).index_add(0, src_idx, exp_w)
        denom = denom[src_idx].clamp_min(1e-12)
        omega = exp_w / denom
    else:
        denom = torch.zeros(N, device=device, dtype=values.dtype).index_add(0, src_idx, weight_logit_or_uniform)
        denom = denom[src_idx].clamp_min(1e-12)
        omega = weight_logit_or_uniform / denom

    weighted = values * omega.unsqueeze(-1)
    out = out.index_add(0, src_idx, weighted)
    return out
