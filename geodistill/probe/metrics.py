"""Depth metrics + Spearman (paper §4.3).

AbsRel, RMSE, delta1 (threshold 1.25) — standard monocular depth metrics — plus a
binned variant over 0-10 / 10-30 / 30m+, and a dependency-free Spearman rank
correlation.
"""

from __future__ import annotations

import torch


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, delta_thresh: float = 1.25) -> dict:
    pred = torch.as_tensor(pred, dtype=torch.float32)
    gt = torch.as_tensor(gt, dtype=torch.float32)
    m = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 1e-3)
    pred, gt = pred[m], gt[m]
    if pred.numel() == 0:
        return {"AbsRel": float("nan"), "RMSE": float("nan"), "delta1": float("nan"), "n": 0}
    absrel = ((pred - gt).abs() / gt).mean()
    rmse = torch.sqrt(((pred - gt) ** 2).mean())
    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = (ratio < delta_thresh).float().mean()
    return {"AbsRel": float(absrel), "RMSE": float(rmse), "delta1": float(delta1), "n": int(pred.numel())}


def depth_metrics_binned(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    bins = {"0_10m": (0.0, 10.0), "10_30m": (10.0, 30.0), "30m_plus": (30.0, float("inf"))}
    out = {}
    for name, (lo, hi) in bins.items():
        sel = (gt >= lo) & (gt < hi)
        out[name] = depth_metrics(pred[sel], gt[sel])
    return out


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    m = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[m], y[m]
    if x.numel() < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp_min(1e-12)
    return float((rx * ry).sum() / denom)


def _rankdata(v: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties shared), 1-based not required for correlation."""
    n = v.numel()
    order = torch.argsort(v)
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    # resolve ties by averaging
    sorted_v, _ = torch.sort(v)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = torch.arange(i, j + 1, dtype=torch.float32).mean()
            idx = order[i:j + 1]
            ranks[idx] = avg
        i = j + 1
    return ranks
