"""Teacher 1: Hard projection + 10% quantile (paper §3.4, baseline teacher).

For each token, aggregate the LiDAR points falling in its region, take the
``q``-quantile depth (default 0.1) as the hard depth label, select the foreground
subset within ``tau_fg`` of that quantile, and use its confidence-weighted pixel
centroid as the ray target. Reliability is ``q_label = s_p * q_geom``.

This is the cheapest teacher and the explicit baseline the OT teachers must beat.
``q_conc`` / ``q_mass`` are defined as 1 for any token with >=1 point (hard
assignment is maximally "concentrated" by construction) so the contract matches.
"""

from __future__ import annotations

import time

import torch

from geodistill.data.contract import TokenScene
from geodistill.core.camera import unproject_cam_to_ego
from .base import (
    TeacherOutput,
    CandidateSets,
    build_candidate_sets,
    empty_teacher,
    reliability_geom,
    support_score,
    sub_token_offset,
)


def build_hard_quantile_teacher(
    scene: TokenScene,
    q: float = 0.1,
    tau_fg: float = 1.0,
    n0: float = 3.0,
    tau_q: float = 0.1,
    cands: CandidateSets | None = None,
) -> TeacherOutput:
    t0 = time.perf_counter()
    N = scene.num_tokens
    cands = cands or build_candidate_sets(scene, dilate_px=0.0)
    out = empty_teacher(N, name="hard_quantile")

    n_used = 0
    for p in range(N):
        depths = cands.cand_depth[p]
        if depths.numel() == 0:
            continue
        c = int(scene.token_camera_id[p])
        dq = torch.quantile(depths, q)
        fg = depths <= dq + tau_fg
        uv_fg = cands.cand_uv[p][fg]
        if uv_fg.numel() == 0:
            uv_fg = cands.cand_uv[p]
        uv_centroid = uv_fg.mean(dim=0)
        # representative ego coord: unproject token-region centroid at quantile depth
        c_ego = unproject_cam_to_ego(uv_centroid, dq, scene.K[c], scene.cam_to_ego[c])

        out.d_teacher[p] = dq
        out.c_teacher[p] = c_ego
        out.delta_T[p] = sub_token_offset(uv_centroid, scene.token_uv[p], scene.token_box[p])
        out.m_T[p] = bool(dq >= 0)
        s_p = support_score(int(depths.numel()), n0)
        q_geom = reliability_geom(depths, tau_q)
        out.q_conc[p] = 1.0           # hard assignment: single mode by construction
        out.q_mass[p] = 1.0           # all in-region mass used
        out.q_teacher[p] = s_p * q_geom
        n_used += 1

    out.diag = {
        "teacher": "hard_quantile",
        "build_time_s": time.perf_counter() - t0,
        "tokens_with_label": n_used,
        "sinkhorn_iters": 0,
        "mean_candidates_per_labeled_token": float(
            cands.n_cand[cands.n_cand > 0].float().mean()
        ) if int((cands.n_cand > 0).sum()) > 0 else 0.0,
    }
    return out
