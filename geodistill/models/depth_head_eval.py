"""Train + evaluate depth-representation heads on frozen tokens (paper §4.4, Table 2).

For each head variant we:
  1. assemble (features, teacher_depth, gt_depth) across scenes using a chosen teacher;
  2. train the head on train tokens (target = teacher depth, in the head's own domain);
  3. evaluate decoded metric depth against held-out GT depth (AbsRel/RMSE/delta1 + bins).

R²AC uses the companded-domain SmoothL1 with ORACLE sensitivity a* (paper §3.6:
z* = F(d_teacher; a*)), and decodes with predicted a (StopGrad). a* needs risk +
reliability; we compute it from the scene + teacher.

The external-depth surrogate ("Qwen + UniDepth + 3D PE") is NOT trained on tokens:
it reads a simulated frozen monocular estimator. On synthetic data the estimator =
GT depth with calibrated multiplicative bias + range-growing noise (worse far),
which mimics a real monocular network's far-field degradation so the Pareto slot is
populated without faking a learned result. On real data this is replaced by true
UniDepthV2 outputs (left TBD).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from geodistill.data.contract import TokenScene
from geodistill.core.r2ac import r2ac_forward
from geodistill.core.risk_field import RiskFieldConfig, soft_risk, oracle_allocation
from geodistill.core.camera import invert_se3
from .depth_heads import HEAD_REGISTRY, R2ACHead, HeadTrainConfig
from geodistill.probe.metrics import depth_metrics, depth_metrics_binned


def _ego_xy(scene: TokenScene, c_teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward (x) and lateral (y) ego distance of the teacher representative."""
    x = c_teacher[:, 0]
    y = c_teacher[:, 1]
    return x, y


def assemble_head_data(scenes, teacher_builder, risk_cfg: RiskFieldConfig, eta: float = 0.5):
    feats, td, gt, astar, qot = [], [], [], [], []
    for s in scenes:
        t = teacher_builder(s)
        m = t.m_T & torch.isfinite(t.d_teacher) & s.gt_valid & torch.isfinite(s.gt_depth)
        if int(m.sum()) == 0:
            continue
        x_ego, y_ego = _ego_xy(s, t.c_teacher)
        r = soft_risk(x_ego[m], y_ego[m], torch.tensor(s.v_ego), risk_cfg)
        a = oracle_allocation(r, t.q_teacher[m], eta)
        feats.append(s.features[m]); td.append(t.d_teacher[m]); gt.append(s.gt_depth[m])
        astar.append(a); qot.append(t.q_conc[m] * t.q_mass[m])
    if not feats:
        return None
    return (torch.cat(feats), torch.cat(td), torch.cat(gt), torch.cat(astar), torch.cat(qot))


def _split(n, val_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nv = max(1, int(val_frac * n))
    return perm[nv:], perm[:nv]


def train_eval_head(head_name: str, scenes, teacher_builder,
                    cfg: HeadTrainConfig = HeadTrainConfig(),
                    risk_cfg: RiskFieldConfig = RiskFieldConfig(),
                    val_frac: float = 0.3, seed: int = 0) -> dict:
    if head_name == "unidepth_pe":
        return _external_depth_surrogate(scenes, teacher_builder, cfg, seed)

    data = assemble_head_data(scenes, teacher_builder, risk_cfg, risk_cfg.eta)
    if data is None:
        return {"head": head_name, "error": "no_valid_tokens"}
    feats, td, gt, astar, qot = data
    n, dim = feats.shape
    tr, va = _split(n, val_frac, seed)

    mu = feats[tr].mean(0, keepdim=True); sd = feats[tr].std(0, keepdim=True).clamp_min(1e-6)
    X = (feats - mu) / sd

    head = HEAD_REGISTRY[head_name](dim, D_max=cfg.D_max)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sl1 = nn.SmoothL1Loss()

    head.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        if isinstance(head, R2ACHead):
            z, a = head.z_and_a(X[tr])
            z_star = r2ac_forward(td[tr], astar[tr], cfg.D_max, cfg.beta)  # oracle a*
            loss = sl1(z, z_star)
            # also fit predicted a to oracle a* so decode-time a is calibrated
            loss = loss + sl1(a, astar[tr])
        else:
            pred = head.depth(X[tr])
            loss = sl1(pred / cfg.D_max, (td[tr] / cfg.D_max).clamp(0, 1))
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred_val = head.depth(X[va]).clamp(0, cfg.D_max)
    gt_val = gt[va]
    return {
        "head": head_name,
        "n_tokens": int(n), "n_val": int(va.numel()),
        "trainable_params": int(sum(p.numel() for p in head.parameters())),
        "overall": depth_metrics(pred_val, gt_val),
        "binned": depth_metrics_binned(pred_val, gt_val),
        "external_depth": False,
    }


def _external_depth_surrogate(scenes, teacher_builder, cfg: HeadTrainConfig, seed: int) -> dict:
    """Stand-in for 'Qwen + UniDepth + 3D PE': frozen monocular estimator, not trained.

    Synthetic estimator(d) = d * (1 + bias) + noise, noise std growing with range so
    the far field degrades like a real monocular net. Clearly labelled surrogate.
    """
    g = torch.Generator().manual_seed(seed)
    preds, gts = [], []
    for s in scenes:
        m = s.gt_valid & torch.isfinite(s.gt_depth)
        if int(m.sum()) == 0:
            continue
        d = s.gt_depth[m]
        bias = 0.02
        noise_std = 0.5 + 0.06 * d  # grows with depth
        est = d * (1 + bias) + torch.randn(d.shape, generator=g) * noise_std
        preds.append(est.clamp(0, cfg.D_max)); gts.append(d)
    if not preds:
        return {"head": "unidepth_pe", "error": "no_valid_tokens"}
    pred = torch.cat(preds); gt = torch.cat(gts)
    return {
        "head": "unidepth_pe",
        "n_tokens": int(pred.numel()), "n_val": int(pred.numel()),
        "trainable_params": 0,
        "overall": depth_metrics(pred, gt),
        "binned": depth_metrics_binned(pred, gt),
        "external_depth": True,
        "note": "SURROGATE external monocular depth (synthetic). Real run wires UniDepthV2 (TBD).",
    }


MODEL_VARIANTS = ["raw", "log", "power_warp", "r2ac", "unidepth_pe", "lpga"]
