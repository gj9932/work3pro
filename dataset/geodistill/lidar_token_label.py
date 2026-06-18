"""Token-level LiDAR label builder (paper §3.4).

For each Qwen merged token region ``R_p``:
    Q_p   = projected LiDAR points falling inside R_p
    D_p   = camera-z depths of Q_p
    d_p   = Quantile(D_p, q)                    (default q=0.1, a.k.a. foreground depth)
    S_p   = {j ∈ Q_p | d_j ≤ d_p + τ_fg}        (foreground subset)
    (u_p, v_p) = weighted centroid of S_p over (u_j, v_j)
    δ_p^* = sub-token offset of (u_p, v_p) inside R_p
    s_p   = 1 − exp(−|D_p| / n_0)               (sampling support)
    q_geom = exp(−MAD(D_p) / (τ_q · Median(D_p) + ε))
    q_label = s_p · q_geom
    c_p^lidar = T_cam→ego · d_p · K^{-1} [u_p, v_p, 1]

These quantities populate the same fields as :class:`geodistill.teacher.TeacherOutput`
so trainers / probes consume them identically. The OT teacher
(``geodistill.teacher.ot_common``) is the alternative path that *replaces* d_p
with the soft barycenter; the hard-quantile path here is the §3.4 baseline /
warm-up teacher and the FGT fallback when transport fails.

This module is the production wrapper that composes:
- :func:`geodistill.core.camera.project_ego_to_cam`       (LiDAR → camera)
- :func:`geodistill.core.camera.unproject_cam_to_ego`     (camera → ego, c_p^lidar)
- :func:`geodistill.core.camera.assign_points_to_tokens`  (point → token)
- :class:`geodistill.teacher.TeacherOutput`                (output contract)

It is meaningfully thicker than ``hard_quantile.py`` because it adds (a) ego-motion
multi-sweep aggregation and (b) dynamic-box exclusion of moving objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geodistill.core.camera import project_ego_to_cam, unproject_cam_to_ego, assign_points_to_tokens
from geodistill.data.contract import TokenScene
from geodistill.teacher.base import (
    TeacherOutput, build_candidate_sets, empty_teacher,
    reliability_geom, support_score, sub_token_offset,
)
from geodistill.teacher.hard_quantile import build_hard_quantile_teacher


__all__ = ["TokenLabelConfig", "build_token_labels"]


@dataclass(frozen=True)
class TokenLabelConfig:
    foreground_quantile: float = 0.1
    tau_fg: float = 1.0                           # foreground band, meters
    n_0: float = 3.0                              # support saturation
    tau_q: float = 0.1                            # reliability scale
    use_min_depth_baseline: bool = False          # mirror SpaceDrive ablation


def build_token_labels(
    scene: TokenScene,
    cfg: TokenLabelConfig | None = None,
    cands=None,
) -> TeacherOutput:
    """Production hard-quantile token-level LiDAR label.

    This is a thin wrapper around the P0 ``build_hard_quantile_teacher`` so the
    GPU-side trainer has a documented entry point with the same configuration
    object that the yaml config exposes. Multi-sweep accumulation lives in
    :mod:`dataset.geodistill.multi_sweep_lidar`; ``scene.points_ego`` is
    expected to already be the (optionally accumulated) center-frame point set.
    """
    cfg = cfg or TokenLabelConfig()
    return build_hard_quantile_teacher(
        scene,
        q=cfg.foreground_quantile,
        tau_fg=cfg.tau_fg,
        n0=cfg.n_0,
        tau_q=cfg.tau_q,
        cands=cands,
    )
