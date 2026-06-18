"""Sparse edge sampler v2 (paper §3.10).

Adds ``P_hard`` (appearance-similar but 3D-far) on top of the v1 categories
already present in :mod:`geodistill.teacher.edges`. v1 is left untouched so
``judge_modules`` keeps reproducible numbers; the main relation pipeline
consumes v2.

Edge categories:
- ``P_local`` : same-camera nearest in image space            (image-near)
- ``P_ray``   : same-camera nearest in ego-ray angle           (ray-near)
- ``P_cross`` : different camera, nearest in predicted 3D       (cross-view positive)
- ``P_hard``  : appearance similar (cosine top) but 3D far       (hard negative)
- ``P_far``   : random far valid token                          (regularization)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geodistill.data.contract import TokenScene
from geodistill.teacher.base import TeacherOutput


__all__ = ["EdgeBudgetV2", "sample_edges_v2"]


@dataclass(frozen=True)
class EdgeBudgetV2:
    P_local: int = 6
    P_ray: int = 4
    P_cross: int = 4
    P_hard: int = 1
    P_far: int = 1

    @property
    def total(self) -> int:
        return self.P_local + self.P_ray + self.P_cross + self.P_hard + self.P_far


def _topk_idx(scores: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    if scores.numel() == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    k = min(k, scores.numel())
    return torch.topk(scores, k, largest=largest).indices


def sample_edges_v2(
    scene: TokenScene,
    teacher: TeacherOutput,
    budget: EdgeBudgetV2 = EdgeBudgetV2(),
    seed: int = 0,
):
    """Returns (edges_list, edge_types, mean_per_token).

    ``edges_list`` is a list of (p, q) int pairs;
    ``edge_types`` is a list[str] aligned with ``edges_list`` for diagnostic logging.
    """
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    vidx = valid.nonzero(as_tuple=True)[0]
    if vidx.numel() < 2:
        return [], [], 0.0

    uv = scene.token_uv.float()
    cam = scene.token_camera_id.long()
    c3d = teacher.c_teacher.float()
    feats = scene.features.float()
    feats_n = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed)

    edges: list[tuple[int, int]] = []
    types: list[str] = []
    counts = []

    for p in vidx.tolist():
        cnt = 0
        same_cam_mask = (cam == cam[p]) & valid
        same_cam_mask[p] = False
        sc_idx = same_cam_mask.nonzero(as_tuple=True)[0]
        cross_mask = (cam != cam[p]) & valid
        cr_idx = cross_mask.nonzero(as_tuple=True)[0]

        if sc_idx.numel():
            d_img = (uv[sc_idx] - uv[p]).norm(dim=1)
            for q in sc_idx[_topk_idx(-d_img, budget.P_local)].tolist():
                edges.append((p, int(q))); types.append("local"); cnt += 1
            ray_p = c3d[p] / c3d[p].norm().clamp_min(1e-6)
            ray_q = c3d[sc_idx] / c3d[sc_idx].norm(dim=1, keepdim=True).clamp_min(1e-6)
            ang = 1.0 - ray_q @ ray_p
            for q in sc_idx[_topk_idx(-ang, budget.P_ray)].tolist():
                edges.append((p, int(q))); types.append("ray"); cnt += 1

        if cr_idx.numel():
            d3 = (c3d[cr_idx] - c3d[p]).norm(dim=1)
            for q in cr_idx[_topk_idx(-d3, budget.P_cross)].tolist():
                edges.append((p, int(q))); types.append("cross"); cnt += 1

        # P_hard: appearance similar (cosine top), but 3D far. Pick within all valid \ {p}.
        if budget.P_hard > 0 and vidx.numel() > 1:
            others = vidx[vidx != p]
            sim = feats_n[others] @ feats_n[p]
            d3_all = (c3d[others] - c3d[p]).norm(dim=1)
            # composite: sim - lambda * 3D-near; we pick high sim AND large d3
            score = sim - (1.0 / (d3_all + 1.0))
            for q in others[_topk_idx(score, budget.P_hard)].tolist():
                edges.append((p, int(q))); types.append("hard"); cnt += 1

        # P_far: random far valid
        if budget.P_far > 0 and vidx.numel() > 1:
            far_pool = vidx[vidx != p]
            sel = far_pool[torch.randint(0, far_pool.numel(),
                                         (min(budget.P_far, far_pool.numel()),),
                                         generator=g)]
            for q in sel.tolist():
                edges.append((p, int(q))); types.append("far"); cnt += 1

        counts.append(cnt)

    mean_epp = float(sum(counts) / max(1, len(counts)))
    return edges, types, mean_epp
