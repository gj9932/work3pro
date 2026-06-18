"""Lightweight relation probe for P0 module screening.

This probe asks whether frozen token pairs contain recoverable relation signals
on edges sampled from a teacher: relative depth order and cross-view geometric
correspondence. It is not the full relation residual module from the paper; it is
the minimal screening metric needed by ``judge_modules`` before spending full
downstream QA/planning budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from geodistill.teacher.edges import EdgeBudget, sample_edges


@dataclass(frozen=True)
class RelationProbeConfig:
    hidden: int = 64
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 1e-4
    val_frac: float = 0.3
    seed: int = 0
    depth_margin: float = 0.5
    cross_match_thresh: float = 1.0


class PairProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        in_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hp: torch.Tensor, hq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hp, hq, (hp - hq).abs(), hp * hq], dim=1)
        return self.net(x).squeeze(-1)


def _finite_xyz(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x).all(dim=1)


def _assemble_relation_data(scenes, teacher_builder, cfg: RelationProbeConfig):
    hp_order, hq_order, y_order = [], [], []
    hp_cross, hq_cross, y_cross = [], [], []
    edge_counts = []

    for scene in scenes:
        teacher = teacher_builder(scene)
        edges, epp = sample_edges(scene, teacher, EdgeBudget())
        edge_counts.append(epp)
        if not edges or scene.gt_depth is None or scene.gt_valid is None or scene.gt_xyz_ego is None:
            continue

        valid_xyz = _finite_xyz(scene.gt_xyz_ego)
        for p, q in edges:
            if not bool(scene.gt_valid[p] and scene.gt_valid[q] and valid_xyz[p] and valid_xyz[q]):
                continue
            dp = scene.gt_depth[p]
            dq = scene.gt_depth[q]
            if torch.isfinite(dp) and torch.isfinite(dq) and float((dq - dp).abs()) > cfg.depth_margin:
                hp_order.append(scene.features[p])
                hq_order.append(scene.features[q])
                y_order.append(float(dq > dp))

            if int(scene.token_camera_id[p]) != int(scene.token_camera_id[q]):
                dist = (scene.gt_xyz_ego[p] - scene.gt_xyz_ego[q]).norm()
                hp_cross.append(scene.features[p])
                hq_cross.append(scene.features[q])
                y_cross.append(float(dist <= cfg.cross_match_thresh))

    def pack(hp, hq, y):
        if not hp:
            return None
        return torch.stack(hp), torch.stack(hq), torch.tensor(y, dtype=torch.float32)

    mean_edges = float(sum(edge_counts) / max(1, len(edge_counts)))
    return pack(hp_order, hq_order, y_order), pack(hp_cross, hq_cross, y_cross), mean_edges


def _split(n: int, val_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    return perm[n_val:], perm[:n_val]


def _majority_accuracy(y: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    p = float(y.mean())
    return max(p, 1.0 - p)


def _train_eval_binary(data, cfg: RelationProbeConfig) -> dict:
    if data is None:
        return {"n": 0, "accuracy": float("nan"), "majority_accuracy": float("nan"), "gain": float("nan")}

    hp, hq, y = data
    n = int(y.numel())
    if n < 4 or float(y.min()) == float(y.max()):
        return {
            "n": n,
            "positive_rate": float(y.mean()) if n else float("nan"),
            "accuracy": float("nan"),
            "majority_accuracy": _majority_accuracy(y),
            "gain": float("nan"),
            "note": "insufficient_or_single_class",
        }

    tr, va = _split(n, cfg.val_frac, cfg.seed)
    dim = int(hp.shape[1])
    mu = torch.cat([hp[tr], hq[tr]], dim=0).mean(0, keepdim=True)
    sd = torch.cat([hp[tr], hq[tr]], dim=0).std(0, keepdim=True).clamp_min(1e-6)
    hp = (hp - mu) / sd
    hq = (hq - mu) / sd

    probe = PairProbe(dim, cfg.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos = y[tr].sum().clamp_min(1.0)
    neg = (1.0 - y[tr]).sum().clamp_min(1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).clamp(0.1, 10.0))

    probe.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        logits = probe(hp[tr], hq[tr])
        loss = lossf(logits, y[tr])
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(probe(hp[va], hq[va])) >= 0.5).float()
    acc = float((pred == y[va]).float().mean())
    maj = _majority_accuracy(y[va])
    return {
        "n": n,
        "n_val": int(va.numel()),
        "positive_rate": float(y.mean()),
        "accuracy": acc,
        "majority_accuracy": maj,
        "gain": acc - maj,
    }


def run_relation_probe(scenes, teacher_builder, cfg: RelationProbeConfig = RelationProbeConfig()) -> dict:
    order_data, cross_data, mean_edges = _assemble_relation_data(scenes, teacher_builder, cfg)
    return {
        "order": _train_eval_binary(order_data, cfg),
        "cross_view": _train_eval_binary(cross_data, cfg),
        "mean_edges_per_token": mean_edges,
    }
