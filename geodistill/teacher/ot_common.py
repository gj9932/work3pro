"""Shared OT machinery for the entropic-OT and full-NTL-FGT teachers (paper §3.5).

This builds the two discrete measures (token side / LiDAR side), assembles the
feature-level transport cost ``M_{p,i}``, the FGW fixed-structure costs
``C^{V,fix}`` / ``C^{L,fix}``, runs an entropic (Fused) Gromov-Wasserstein solver,
and turns the coupling into a concentration-aware per-token teacher.

Tractability (faithful to the paper): LiDAR is voxel-aggregated to a bounded
anchor set, and only tokens carrying candidates participate. The FGW structure
term is solved with the standard entropic mirror-descent / proximal scheme
(Peyré et al.): outer iterations linearize the quadratic GW term, each inner
solve is a Sinkhorn. ``gamma=0`` collapses to plain entropic OT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from geodistill.data.contract import TokenScene
from geodistill.core.camera import (
    project_ego_to_cam,
    unproject_cam_to_ego,
    ray_dirs_ego,
    invert_se3,
)
from geodistill.core.sinkhorn import sinkhorn_balanced, sinkhorn_partial
from .base import (
    TeacherOutput,
    CandidateSets,
    build_candidate_sets,
    empty_teacher,
    reliability_geom,
    support_score,
    sub_token_offset,
)


@dataclass(frozen=True)
class OTConfig:
    epsilon_ot: float = 0.05
    gamma: float = 0.0              # FGW feature/structure trade-off
    m_tr_frac: float = 0.9          # partial transported-mass budget (fraction of min marginal)
    n_iters_sinkhorn: int = 150
    n_outer_fgw: int = 5            # FGW outer linearizations (1 effective when gamma=0)
    voxel_size: float = 0.5         # LiDAR voxel aggregation (m)
    max_anchors: int = 400
    lambda_uv: float = 1.0          # feature cost weights
    lambda_ray: float = 2.0
    lambda_cam: float = 5.0
    lambda_d: float = 0.0           # paper main method fixes this to 0
    tau_conc: float = 0.5           # barycenter vs medoid switch
    topk_medoid: int = 5
    eta_q: float = 0.5              # reliability fusion
    n0: float = 3.0
    tau_q: float = 0.1
    cand_dilate_px: float = 14.0    # dilate token boxes so OT sees neighbour surfaces


def voxel_aggregate(points_ego: torch.Tensor, voxel: float, max_anchors: int):
    """Aggregate LiDAR points into voxel-center anchors. Returns (anchors, counts, member_idx)."""
    if points_ego.shape[0] == 0:
        return points_ego.new_zeros((0, 3)), points_ego.new_zeros((0,), dtype=torch.long), []
    keys = torch.floor(points_ego / voxel).to(torch.long)
    uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
    A = uniq.shape[0]
    anchors = torch.zeros((A, 3))
    counts = torch.zeros(A, dtype=torch.long)
    anchors.index_add_(0, inv, points_ego)
    counts.index_add_(0, inv, torch.ones(points_ego.shape[0], dtype=torch.long))
    anchors = anchors / counts.clamp_min(1).unsqueeze(1).float()
    if A > max_anchors:
        keep = torch.topk(counts, max_anchors).indices
        anchors, counts = anchors[keep], counts[keep]
        keepset = set(keep.tolist())
        remap = {int(k): j for j, k in enumerate(keep.tolist())}
        member = [(inv == k).nonzero(as_tuple=True)[0] for k in keep.tolist()]
    else:
        member = [(inv == a).nonzero(as_tuple=True)[0] for a in range(A)]
    return anchors, counts, member


def _feature_cost(scene: TokenScene, active: torch.Tensor, anchors: torch.Tensor, cfg: OTConfig):
    """Assemble M_{p,i} for active tokens x anchors. Also returns per-(p) visibility mask."""
    Na = active.shape[0]
    A = anchors.shape[0]
    cost = torch.full((Na, A), 0.0)
    cam_ids = scene.token_camera_id[active]
    # precompute anchor projections + ray dirs per camera
    proj_by_cam = {}
    for c in range(scene.num_cameras):
        proj = project_ego_to_cam(anchors, scene.K[c], scene.cam_to_ego[c])
        # anchor ray dir in ego for visible anchors
        rd = torch.zeros((A, 3))
        vis = proj["valid"]
        if int(vis.sum()) > 0:
            rd[vis] = ray_dirs_ego(proj["uv"][vis], scene.K[c], scene.cam_to_ego[c])
        proj_by_cam[c] = {"uv": proj["uv"], "depth": proj["depth"], "valid": vis, "ray": rd}

    # token ray dirs in ego
    tok_ray = torch.zeros((Na, 3))
    for j in range(Na):
        c = int(cam_ids[j])
        tok_ray[j] = ray_dirs_ego(scene.token_uv[active[j]].unsqueeze(0), scene.K[c], scene.cam_to_ego[c])[0]

    img_diag = float((scene.image_hw[0].float() ** 2).sum().sqrt())
    for j in range(Na):
        c = int(cam_ids[j])
        pj = proj_by_cam[c]
        vis = pj["valid"]
        uv_p = scene.token_uv[active[j]]
        # uv/ray terms are only meaningful for visible anchors; invisible anchors
        # get garbage projected pixels (near-zero depth division), so gate them and
        # apply only the bounded visibility penalty. (Paper: visibility = hard mask,
        # lambda_cam = soft penalty on the rest.)
        d_uv = ((pj["uv"] - uv_p).norm(dim=1) / img_diag).clamp(max=2.0)
        ray_align = (1.0 - (pj["ray"] * tok_ray[j].unsqueeze(0)).sum(dim=1)).clamp(0.0, 2.0)
        feat = cfg.lambda_uv * d_uv ** 2 + cfg.lambda_ray * ray_align
        not_vis = (~vis)
        c_row = torch.where(not_vis, torch.full_like(feat, cfg.lambda_cam), feat)
        cost[j] = c_row
    return cost, proj_by_cam, tok_ray


def _fixed_structure_token(scene: TokenScene, active: torch.Tensor, tok_ray: torch.Tensor) -> torch.Tensor:
    """C^{V,fix}: pairwise token structure (image-pos dist + ego-ray angle + cam sep). (Na,Na) scalarized."""
    Na = active.shape[0]
    uv = scene.token_uv[active]
    img_diag = float((scene.image_hw[0].float() ** 2).sum().sqrt())
    d_uv = torch.cdist(uv, uv) / img_diag
    ray_ang = 1.0 - tok_ray @ tok_ray.T
    cam = scene.token_camera_id[active]
    cam_sep = (cam.unsqueeze(0) != cam.unsqueeze(1)).float()
    return (d_uv + ray_ang + 0.5 * cam_sep)


def _fixed_structure_lidar(anchors: torch.Tensor) -> torch.Tensor:
    """C^{L,fix}: pairwise anchor structure (3D dist + ego-ray angle). (A,A) scalarized."""
    A = anchors.shape[0]
    if A == 0:
        return anchors.new_zeros((0, 0))
    d3 = torch.cdist(anchors, anchors)
    d3 = d3 / d3.max().clamp_min(1e-6)
    rd = anchors / anchors.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ray_ang = 1.0 - rd @ rd.T
    return (d3 + ray_ang)


def solve_coupling(cost: torch.Tensor, C_V: torch.Tensor, C_L: torch.Tensor,
                   mu_V: torch.Tensor, mu_L: torch.Tensor, cfg: OTConfig):
    """Entropic (Fused) GW. gamma=0 -> single partial Sinkhorn. Returns (pi, total_iters)."""
    m_tr = cfg.m_tr_frac * float(min(mu_V.sum(), mu_L.sum()))
    if cfg.gamma <= 0 or C_V.numel() == 0 or C_L.numel() == 0:
        pi, iters = sinkhorn_partial(cost, mu_V, mu_L, cfg.epsilon_ot, m_tr, cfg.n_iters_sinkhorn)
        return pi, iters
    # FGW mirror-descent: linearize quadratic GW term each outer step.
    # GW gradient for inner-product structures: grad = -2 * C_V @ pi @ C_L^T (constant terms drop).
    pi, total = sinkhorn_partial(cost, mu_V, mu_L, cfg.epsilon_ot, m_tr, cfg.n_iters_sinkhorn)
    cost_scale = cost.abs().mean().clamp_min(1e-9)
    for _ in range(max(1, cfg.n_outer_fgw)):
        gw_grad = -2.0 * (C_V @ pi @ C_L.T)
        # Normalize structure gradient to the feature-cost scale so gamma is a true
        # trade-off, not swamped by the O(Na*A) magnitude of the quadratic term.
        gw_grad = gw_grad * (cost_scale / gw_grad.abs().mean().clamp_min(1e-9))
        eff_cost = (1.0 - cfg.gamma) * cost + cfg.gamma * gw_grad
        eff_cost = eff_cost - eff_cost.min()
        pi, it = sinkhorn_partial(eff_cost, mu_V, mu_L, cfg.epsilon_ot, m_tr, cfg.n_iters_sinkhorn)
        total += it
    return pi, total


def coupling_to_teacher(scene: TokenScene, active: torch.Tensor, anchors: torch.Tensor,
                        pi: torch.Tensor, mu_V: torch.Tensor, cands: CandidateSets,
                        cfg: OTConfig, name: str, diag: dict) -> TeacherOutput:
    """Concentration-aware representative -> per-token teacher (paper §3.5)."""
    N = scene.num_tokens
    out = empty_teacher(N, name=name)
    eps = 1e-9
    for j in range(active.shape[0]):
        p = int(active[j])
        row = pi[j]
        mass = float(row.sum())
        if mass <= eps:
            continue
        tilde = row / (row.sum() + eps)
        support = (tilde > 1e-6)
        k_supp = int(support.sum())
        if k_supp == 0:
            continue
        # concentration = 1 - normalized entropy
        if k_supp == 1:
            q_conc = 1.0
        else:
            H = -(tilde[support] * (tilde[support] + eps).log()).sum()
            q_conc = float(1.0 - H / torch.log(torch.tensor(float(k_supp))))
        q_mass = float(min(mass / (float(mu_V[j]) + eps), 1.0))

        bary = (tilde.unsqueeze(1) * anchors).sum(dim=0)
        if q_conc >= cfg.tau_conc:
            c_T = bary
        else:
            # medoid over top-k support anchors (real anchor, never cross-surface mean)
            topk = torch.topk(tilde, min(cfg.topk_medoid, k_supp)).indices
            sub = anchors[topk]
            w = tilde[topk]
            dmat = torch.cdist(sub, sub)
            med = torch.argmin((w.unsqueeze(0) * dmat).sum(dim=1))
            c_T = sub[med]

        c = int(scene.token_camera_id[p])
        ego_to_cam = invert_se3(scene.cam_to_ego[c])
        c_cam = (torch.cat([c_T, c_T.new_ones(1)]) @ ego_to_cam.T)[:3]
        d_T = float(c_cam[2])
        if d_T < 0:
            continue
        # project c_T back to token camera for ray target
        proj = project_ego_to_cam(c_T.unsqueeze(0), scene.K[c], scene.cam_to_ego[c])
        uv_T = proj["uv"][0]

        # reliability fusion (uses hard region stats when available)
        depths = cands.cand_depth[p]
        s_p = support_score(int(depths.numel()), cfg.n0)
        q_geom = reliability_geom(depths, cfg.tau_q)
        eta_qp = cfg.eta_q * s_p
        g = eta_qp * q_geom + (1.0 - eta_qp) * q_mass
        q_teacher = q_conc * g

        out.d_teacher[p] = d_T
        out.c_teacher[p] = c_T
        out.delta_T[p] = sub_token_offset(uv_T, scene.token_uv[p], scene.token_box[p])
        out.m_T[p] = q_mass > 0.0
        out.q_conc[p] = q_conc
        out.q_mass[p] = q_mass
        out.q_teacher[p] = q_teacher
    out.diag = diag
    return out


def build_ot_teacher(scene: TokenScene, cfg: OTConfig, name: str,
                     cands: CandidateSets | None = None,
                     structure_shuffle_seed: int | None = None) -> TeacherOutput:
    """Full pipeline: measures -> cost -> coupling -> concentration-aware teacher.

    ``structure_shuffle_seed``: when not None, the LiDAR-side fixed-structure
    matrix ``C_L`` is row-permuted with this seed. This is the §4.7 "transport
    structure shuffle" hook; default ``None`` reproduces the original behavior
    bit-exactly.
    """
    t0 = time.perf_counter()
    N = scene.num_tokens
    cands = cands or build_candidate_sets(scene, dilate_px=cfg.cand_dilate_px)
    active = (cands.n_cand > 0).nonzero(as_tuple=True)[0]
    if active.numel() == 0 or scene.points_ego.shape[0] == 0:
        out = empty_teacher(N, name=name)
        out.diag = {"teacher": name, "build_time_s": time.perf_counter() - t0,
                    "tokens_with_label": 0, "sinkhorn_iters": 0,
                    "num_anchors": 0, "mean_candidates_per_labeled_token": 0.0}
        return out

    anchors, counts, _members = voxel_aggregate(scene.points_ego, cfg.voxel_size, cfg.max_anchors)
    cost, _proj, tok_ray = _feature_cost(scene, active, anchors, cfg)

    # measures: token mass uniform over active; anchor mass ∝ voxel point count
    mu_V = torch.ones(active.shape[0])
    mu_L = counts.float()
    mu_L = mu_L / mu_L.sum() * mu_V.sum()  # match total mass for balanced-ish partial OT

    if cfg.gamma > 0:
        C_V = _fixed_structure_token(scene, active, tok_ray)
        C_L = _fixed_structure_lidar(anchors)
        if structure_shuffle_seed is not None and C_L.numel() > 0:
            g = torch.Generator().manual_seed(int(structure_shuffle_seed))
            perm = torch.randperm(C_L.shape[0], generator=g)
            C_L = C_L[perm][:, perm].contiguous()
    else:
        C_V = anchors.new_zeros((0, 0))
        C_L = anchors.new_zeros((0, 0))

    pi, iters = solve_coupling(cost, C_V, C_L, mu_V, mu_L, cfg)

    diag = {
        "teacher": name,
        "build_time_s": time.perf_counter() - t0,
        "tokens_with_label": int((pi.sum(dim=1) > 1e-9).sum()),
        "sinkhorn_iters": int(iters),
        "num_anchors": int(anchors.shape[0]),
        "mean_candidates_per_labeled_token": float(cands.n_cand[active].float().mean()),
        "gamma": cfg.gamma,
        "epsilon_ot": cfg.epsilon_ot,
        "transported_mass": float(pi.sum()),
        "structure_shuffle_seed": structure_shuffle_seed,
    }
    return coupling_to_teacher(scene, active, anchors, pi, mu_V, cands, cfg, name, diag)
