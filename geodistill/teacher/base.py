"""Teacher output contract + per-token candidate assembly (paper §3.4–3.5).

All three teacher constructions (hard-quantile, entropic-OT, full NTL-FGT) emit a
``TeacherOutput`` with the same fields so downstream stats/probe/heads are agnostic
to which teacher produced them:

    d_teacher   (N,)   per-token metric (camera-z) depth teacher
    c_teacher   (N,3)  per-token ego-frame 3D representative
    delta_T     (N,2)  sub-token ray offset target in [-1,1]^2
    m_T         (N,)   bool valid-teacher mask
    q_conc      (N,)   coupling concentration  (1 - normalized entropy); 1 for hard
    q_mass      (N,)   transported mass ratio;  1 for hard when token has points
    q_teacher   (N,)   fused reliability used to weight geometry losses
    diag        dict   solver diagnostics (sinkhorn iters, candidate counts, ...)

``build_candidate_sets`` does the shared work: project LiDAR into each camera,
assign to token regions, and gather a local candidate index list per token. The
hard teacher uses these directly; the OT teachers run a solver over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from geodistill.data.contract import TokenScene
from geodistill.core.camera import project_ego_to_cam, assign_points_to_tokens, ray_dirs_ego


@dataclass
class TeacherOutput:
    d_teacher: torch.Tensor
    c_teacher: torch.Tensor
    delta_T: torch.Tensor
    m_T: torch.Tensor
    q_conc: torch.Tensor
    q_mass: torch.Tensor
    q_teacher: torch.Tensor
    name: str = "teacher"
    diag: dict = field(default_factory=dict)


@dataclass
class CandidateSets:
    """Per-token local LiDAR candidate sets (in the token's own camera frame)."""
    # ragged, stored as lists indexed by global token id
    cand_point_idx: list          # list[LongTensor] global LiDAR indices per token
    cand_uv: list                 # list[(k,2)] projected pixels (token's camera)
    cand_depth: list              # list[(k,)] camera-z depth
    cand_xyz_ego: list            # list[(k,3)] ego coords (== points_ego[idx])
    token_uv: torch.Tensor        # (N,2)
    token_box: torch.Tensor       # (N,4)
    token_camera_id: torch.Tensor # (N,)
    n_cand: torch.Tensor          # (N,) candidate count per token


def build_candidate_sets(scene: TokenScene, dilate_px: float = 0.0) -> CandidateSets:
    """Assemble per-token candidate LiDAR points.

    A point is a candidate for token p if it projects (in p's camera, positive
    depth, in-image) into p's region box (optionally dilated by ``dilate_px`` to
    let neighbouring-surface points participate in OT, which hard-projection cannot).
    """
    N = scene.num_tokens
    cand_idx = [torch.empty(0, dtype=torch.long) for _ in range(N)]
    cand_uv = [torch.empty(0, 2) for _ in range(N)]
    cand_depth = [torch.empty(0) for _ in range(N)]
    cand_xyz = [torch.empty(0, 3) for _ in range(N)]

    box = scene.token_box.clone()
    if dilate_px > 0:
        box[:, 0] -= dilate_px
        box[:, 1] -= dilate_px
        box[:, 2] += dilate_px
        box[:, 3] += dilate_px

    for c in range(scene.num_cameras):
        proj = project_ego_to_cam(scene.points_ego, scene.K[c], scene.cam_to_ego[c])
        uv, depth, valid = proj["uv"], proj["depth"], proj["valid"]
        gidx = valid.nonzero(as_tuple=True)[0]
        if gidx.numel() == 0:
            continue
        uv_v = uv[valid]
        depth_v = depth[valid]
        sl = scene.tokens_of_camera(c)
        boxes_c = box[sl]
        tok_local = assign_points_to_tokens(uv_v, boxes_c)
        base = int(scene.cam_offsets[c])
        for local_t in range(boxes_c.shape[0]):
            sel = tok_local == local_t
            if int(sel.sum()) == 0:
                continue
            g = local_t + base
            pts_sel = gidx[sel]
            cand_idx[g] = pts_sel
            cand_uv[g] = uv_v[sel]
            cand_depth[g] = depth_v[sel]
            cand_xyz[g] = scene.points_ego[pts_sel]

    n_cand = torch.tensor([c.numel() for c in cand_idx], dtype=torch.long)
    return CandidateSets(
        cand_point_idx=cand_idx, cand_uv=cand_uv, cand_depth=cand_depth,
        cand_xyz_ego=cand_xyz, token_uv=scene.token_uv, token_box=scene.token_box,
        token_camera_id=scene.token_camera_id, n_cand=n_cand,
    )


def empty_teacher(N: int, name: str) -> TeacherOutput:
    return TeacherOutput(
        d_teacher=torch.full((N,), float("nan")),
        c_teacher=torch.full((N, 3), float("nan")),
        delta_T=torch.zeros((N, 2)),
        m_T=torch.zeros(N, dtype=torch.bool),
        q_conc=torch.zeros(N),
        q_mass=torch.zeros(N),
        q_teacher=torch.zeros(N),
        name=name,
    )


def reliability_geom(depths: torch.Tensor, tau_q: float) -> float:
    """q_geom = exp(-MAD(D)/(tau_q*median(D)+eps)) — single-surface-ness of a token."""
    if depths.numel() == 0:
        return 0.0
    med = depths.median()
    mad = (depths - med).abs().median()
    return float(torch.exp(-mad / (tau_q * med + 1e-6)))


def support_score(n: int, n0: float) -> float:
    """s_p = 1 - exp(-n/n0) — sampling support of hard label."""
    return float(1.0 - torch.exp(torch.tensor(-n / max(n0, 1e-6))))


def sub_token_offset(uv_target: torch.Tensor, token_uv_p: torch.Tensor, box_p: torch.Tensor) -> torch.Tensor:
    """delta = clip([2(u*-u)/W_R, 2(v*-v)/H_R], -1, 1)."""
    w = (box_p[2] - box_p[0]).clamp_min(1e-6)
    h = (box_p[3] - box_p[1]).clamp_min(1e-6)
    du = 2.0 * (uv_target[0] - token_uv_p[0]) / w
    dv = 2.0 * (uv_target[1] - token_uv_p[1]) / h
    return torch.stack([du, dv]).clamp(-1.0, 1.0)
