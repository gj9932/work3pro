"""Minimal sparse relation edge sampler — for the 'sparse edges / token' statistic
and as the substrate the relation adapter would consume (paper §3.10).

Edge budget per token P split into:
  P_local : same-camera spatial neighbours (token-grid adjacency)
  P_ray   : same/adjacent ray bin (ego-ray angle nearest)
  P_cross : cross-camera neighbours close in predicted 3D (cross-view candidates)
  P_far   : random far valid tokens

This is intentionally lightweight (counts + index lists), enough to report
``sparse_edges_per_token`` honestly and to be reused by a later relation adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geodistill.data.contract import TokenScene
from geodistill.teacher.base import TeacherOutput


@dataclass(frozen=True)
class EdgeBudget:
    P_local: int = 6
    P_ray: int = 4
    P_cross: int = 4
    P_far: int = 2

    @property
    def total(self) -> int:
        return self.P_local + self.P_ray + self.P_cross + self.P_far


def sample_edges(scene: TokenScene, teacher: TeacherOutput, budget: EdgeBudget = EdgeBudget()):
    """Return (edges, mean_edges_per_token). ``edges`` is a list of (p, q) int pairs.

    Only valid-teacher tokens participate (they have a 3D position).
    """
    valid = teacher.m_T & torch.isfinite(teacher.d_teacher)
    vidx = valid.nonzero(as_tuple=True)[0]
    if vidx.numel() < 2:
        return [], 0.0

    uv = scene.token_uv
    cam = scene.token_camera_id
    c3d = teacher.c_teacher
    edges = []
    per_token = []

    for p in vidx.tolist():
        cnt = 0
        same_cam = (cam == cam[p]) & valid
        same_cam[p] = False
        sc_idx = same_cam.nonzero(as_tuple=True)[0]
        # P_local: nearest in image space
        if sc_idx.numel():
            d_img = (uv[sc_idx] - uv[p]).norm(dim=1)
            loc = sc_idx[torch.topk(d_img, min(budget.P_local, sc_idx.numel()), largest=False).indices]
            for q in loc.tolist():
                edges.append((p, int(q))); cnt += 1
            # P_ray: nearest by 3D direction within same camera
            rd_p = c3d[p] / c3d[p].norm().clamp_min(1e-6)
            rd = c3d[sc_idx] / c3d[sc_idx].norm(dim=1, keepdim=True).clamp_min(1e-6)
            ang = 1.0 - rd @ rd_p
            ray = sc_idx[torch.topk(ang, min(budget.P_ray, sc_idx.numel()), largest=False).indices]
            for q in ray.tolist():
                edges.append((p, int(q))); cnt += 1
        # P_cross: cross-camera nearest in predicted 3D
        cross = (cam != cam[p]) & valid
        cr_idx = cross.nonzero(as_tuple=True)[0]
        if cr_idx.numel():
            d3 = (c3d[cr_idx] - c3d[p]).norm(dim=1)
            crs = cr_idx[torch.topk(d3, min(budget.P_cross, cr_idx.numel()), largest=False).indices]
            for q in crs.tolist():
                edges.append((p, int(q))); cnt += 1
        # P_far: random far valid tokens
        if budget.P_far > 0 and vidx.numel() > 1:
            far_pool = vidx[vidx != p]
            sel = far_pool[torch.randint(0, far_pool.numel(), (min(budget.P_far, far_pool.numel()),))]
            for q in sel.tolist():
                edges.append((p, int(q))); cnt += 1
        per_token.append(cnt)

    mean_epp = float(sum(per_token) / max(1, len(per_token)))
    return edges, mean_epp
