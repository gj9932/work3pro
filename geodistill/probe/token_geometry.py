"""Token geometry probe (paper §4.3 "Token 几何探针", Table 2).

Freeze token features (Qwen merger output, or synthetic), train a same-capacity
linear / 2-layer probe to regress token depth from features, and report:
  - AbsRel, RMSE, delta1   (overall and on 0-10 / 10-30 / 30m+ bins)
  - Spearman correlation between q^OT (teacher concentration*mass) and |depth error|

The probe target is the *teacher* depth (so it measures how recoverable the
teacher signal is from frozen tokens); evaluation metrics are computed against the
held-out GT depth so they are honest. Train/val split is by token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from geodistill.data.contract import TokenScene
from geodistill.teacher.base import TeacherOutput
from .metrics import depth_metrics, depth_metrics_binned, spearman


class LinearProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class ProbeConfig:
    hidden: int = 0            # 0 -> linear probe; >0 -> 2-layer probe
    epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 1e-4
    val_frac: float = 0.3
    seed: int = 0
    D_max: float = 80.0


def _assemble(scenes, teacher_builder):
    """Run teacher per scene; collect (features, teacher_depth, gt_depth, q_ot)."""
    feats, td, gt, qot, cam = [], [], [], [], []
    for s in scenes:
        t = teacher_builder(s)
        m = t.m_T & torch.isfinite(t.d_teacher) & s.gt_valid & torch.isfinite(s.gt_depth)
        if int(m.sum()) == 0:
            continue
        feats.append(s.features[m])
        td.append(t.d_teacher[m])
        gt.append(s.gt_depth[m])
        qot.append(t.q_conc[m] * t.q_mass[m])
        cam.append(s.token_camera_id[m])
    if not feats:
        return None
    return (torch.cat(feats), torch.cat(td), torch.cat(gt), torch.cat(qot), torch.cat(cam))


def run_token_geometry_probe(scenes, teacher_builder, cfg: ProbeConfig = ProbeConfig()) -> dict:
    """Train probe on teacher depth, evaluate against GT depth. Returns metric dict."""
    data = _assemble(scenes, teacher_builder)
    if data is None:
        return {"error": "no_valid_tokens"}
    feats, td, gt, qot, _cam = data
    n = feats.shape[0]
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(cfg.val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # standardize features on train
    mu = feats[tr_idx].mean(0, keepdim=True)
    sd = feats[tr_idx].std(0, keepdim=True).clamp_min(1e-6)
    X = (feats - mu) / sd
    # regress normalized depth target for conditioning
    y = (td / cfg.D_max)

    probe = LinearProbe(feats.shape[1], cfg.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.SmoothL1Loss()
    probe.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        pred = probe(X[tr_idx])
        loss = lossf(pred, y[tr_idx])
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred_val = probe(X[val_idx]).clamp(0.0, 1.0) * cfg.D_max
    gt_val = gt[val_idx]
    qot_val = qot[val_idx]

    metrics = depth_metrics(pred_val, gt_val)
    binned = depth_metrics_binned(pred_val, gt_val)
    abs_err = (pred_val - gt_val).abs()
    rho = spearman(qot_val, abs_err)

    return {
        "teacher": teacher_builder.__name__ if hasattr(teacher_builder, "__name__") else "teacher",
        "n_tokens": int(n),
        "n_train": int(tr_idx.numel()),
        "n_val": int(val_idx.numel()),
        "probe_hidden": cfg.hidden,
        "overall": metrics,
        "binned": binned,
        "spearman_qOT_vs_abserr": rho,
    }
