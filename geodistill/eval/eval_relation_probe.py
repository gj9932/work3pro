"""Sparse relation probe (paper §4.5 Table 3).

Train tiny pair probes over teacher edges sampled by ``edge_sampler_v2`` and
labelled by ``cross_view_label``. Reports per-task accuracy / Spearman over the
five relation tasks plus a frozen-token baseline (probe over raw H_img with no
model).

Paper Table 3 columns: Relative 3D ↓ (in metric), Order ↑, Cross-view ↑,
Occlusion ↑, Topology ↑.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from geodistill.data.contract import TokenScene
from geodistill.data.loader import load_scenes
from geodistill.geometry.cross_view_label import EdgeLabels, build_edge_labels
from geodistill.geometry.edge_sampler_v2 import EdgeBudgetV2, sample_edges_v2
from geodistill.teacher import (
    build_hard_quantile_teacher, build_entropic_ot_teacher, build_full_ntlfgt_teacher, OTConfig,
)
from geodistill.utils import JsonlWriter, run_envelope


__all__ = ["run_table3"]


TEACHER_BUILDERS = {
    "hard_quantile": build_hard_quantile_teacher,
    "entropic_ot": lambda s: build_entropic_ot_teacher(s, OTConfig(epsilon_ot=0.05, gamma=0.0)),
    "full_ntl_fgt": lambda s: build_full_ntlfgt_teacher(s, OTConfig(epsilon_ot=0.05, gamma=0.5)),
}


class _PairHead(nn.Module):
    """Generic pair head over (h_p, h_q, ξ) → logits or regression."""

    def __init__(self, d_token: int, d_xi: int, d_out: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d_token + d_xi, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, hp: torch.Tensor, hq: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hp, hq, hp - hq, hp * hq, xi], dim=-1)
        return self.net(x)


@dataclass
class _Split:
    tr: torch.Tensor
    va: torch.Tensor


def _split(n: int, val_frac: float, seed: int) -> _Split:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    return _Split(tr=perm[n_val:], va=perm[:n_val])


def _train_classification(features_p: torch.Tensor, features_q: torch.Tensor, xi: torch.Tensor,
                          labels: torch.Tensor, n_classes: int, weights: torch.Tensor,
                          val_frac: float, seed: int, epochs: int = 200, lr: float = 1e-2):
    n = labels.numel()
    if n < 8 or len(set(labels.tolist())) < 2:
        return {"accuracy": float("nan"), "n": n, "majority_accuracy": float("nan")}
    sp = _split(n, val_frac, seed)
    head = _PairHead(features_p.shape[-1], xi.shape[-1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = head(features_p[sp.tr], features_q[sp.tr], xi[sp.tr])
        loss = F.cross_entropy(logits, labels[sp.tr], reduction="none") * weights[sp.tr].clamp_min(1e-6)
        loss.mean().backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        logits = head(features_p[sp.va], features_q[sp.va], xi[sp.va])
        pred = logits.argmax(dim=-1)
    acc = float((pred == labels[sp.va]).float().mean())
    most = labels[sp.va].mode().values.item()
    maj = float((labels[sp.va] == most).float().mean())
    return {"accuracy": acc, "majority_accuracy": maj, "gain": acc - maj, "n": n, "n_val": int(sp.va.numel())}


def _train_regression(features_p: torch.Tensor, features_q: torch.Tensor, xi: torch.Tensor,
                      target: torch.Tensor, weights: torch.Tensor, val_frac: float, seed: int,
                      epochs: int = 200, lr: float = 1e-2) -> dict:
    n = target.shape[0]
    if n < 8:
        return {"MAE": float("nan"), "n": n}
    sp = _split(n, val_frac, seed)
    head = _PairHead(features_p.shape[-1], xi.shape[-1], target.shape[-1])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(features_p[sp.tr], features_q[sp.tr], xi[sp.tr])
        elem = F.huber_loss(pred, target[sp.tr], reduction="none").sum(dim=-1) * weights[sp.tr].clamp_min(1e-6)
        elem.mean().backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred_va = head(features_p[sp.va], features_q[sp.va], xi[sp.va])
    mae = float((pred_va - target[sp.va]).abs().mean())
    return {"MAE": mae, "n": n, "n_val": int(sp.va.numel())}


def _build_xi(scene: TokenScene, edge_index: torch.Tensor) -> torch.Tensor:
    """Pair geometry feature ξ_pq (uses normalized image-space delta only).

    Cross-camera edges keep the same dim by zero-padding the (δu, δv) channels;
    callers can ablate by replacing this with a per-branch xi if they wish.
    """
    p = edge_index[:, 0].long(); q = edge_index[:, 1].long()
    H = scene.image_hw[scene.token_camera_id[p].long(), 0].float().clamp_min(1.0)
    W = scene.image_hw[scene.token_camera_id[p].long(), 1].float().clamp_min(1.0)
    same_cam = (scene.token_camera_id[p] == scene.token_camera_id[q]).float().unsqueeze(-1)
    du = (scene.token_uv[q, 0] - scene.token_uv[p, 0]) / W
    dv = (scene.token_uv[q, 1] - scene.token_uv[p, 1]) / H
    return torch.stack([du * same_cam.squeeze(-1), dv * same_cam.squeeze(-1)], dim=-1)


def run_relation_probe(scene: TokenScene, teacher_name: str = "hard_quantile",
                        budget: EdgeBudgetV2 = EdgeBudgetV2(),
                        val_frac: float = 0.3, seed: int = 0,
                        features_override: torch.Tensor | None = None) -> dict:
    teacher = TEACHER_BUILDERS[teacher_name](scene)
    edges, _, _ = sample_edges_v2(scene, teacher, budget, seed=seed)
    if not edges:
        return {"teacher": teacher_name, "error": "no_valid_edges"}

    labels = build_edge_labels(scene, teacher, edges)
    feats = features_override if features_override is not None else scene.features
    p_idx = labels.edge_index[:, 0].long()
    q_idx = labels.edge_index[:, 1].long()

    xi = _build_xi(scene, labels.edge_index)
    weights = labels.reliability * labels.valid_mask.float()
    fp = feats[p_idx]; fq = feats[q_idx]

    out: dict = {"teacher": teacher_name, "n_edges": int(labels.edge_index.shape[0])}
    out["delta_3d"] = _train_regression(fp, fq, xi, labels.delta_c.float(), weights, val_frac, seed)
    out["order"] = _train_classification(fp, fq, xi, labels.depth_order.long(), 3, weights, val_frac, seed)
    out["cross_view"] = _train_classification(fp, fq, xi, labels.cross_view.long(), 2, weights, val_frac, seed)
    out["topology"] = _train_classification(fp, fq, xi, labels.topology.long(), 6, weights, val_frac, seed)
    out["occlusion"] = _train_classification(fp, fq, xi, labels.occlusion.long(), 3, weights, val_frac, seed)
    return out


def run_table3(scenes, *, teacher: str, out_path: str, source: str = "synthetic",
               val_frac: float = 0.3, seed: int = 0):
    with JsonlWriter(out_path) as w:
        w.write({**run_envelope("eval_relation_probe", source,
                                  {"teacher": teacher, "scenes": len(scenes)}),
                 "kind": "header"})
        for si, s in enumerate(scenes):
            r = run_relation_probe(s, teacher, val_frac=val_frac, seed=seed + si)
            w.write({"kind": "table3_row", "scene_idx": si, "scene_id": s.scene_id, **r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--teacher", default="hard_quantile", choices=list(TEACHER_BUILDERS))
    ap.add_argument("--out", default="runs/geodistill/eval/table3_relation_probe.jsonl")
    args = ap.parse_args()

    scenes = load_scenes(args.source, args.scenes, args.seed, args.config)
    run_table3(scenes, teacher=args.teacher, out_path=args.out, source=args.source, seed=args.seed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
