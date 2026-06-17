"""Synthetic, geometrically self-consistent TokenScene generator (P0, CPU-only).

Why this exists: locally there is no GPU / nuScenes / Qwen weights, so we cannot
produce real Table 1/2/6 numbers. This fixture lets the *entire* teacher + probe +
depth-head loop run and be verified end-to-end on CPU. It is NOT a result source;
it is a correctness harness. All "results" produced from it are clearly labelled
``source=synthetic`` and never written into paper cells.

Construction (so the recovered geometry has a real target):
1. Place ``n_surfaces`` fronto-parallel-ish planar patches at random ego positions
   within sensor range, each with a real depth.
2. For each camera, build a per-token depth buffer by z-projecting the patches and
   keeping the nearest (occlusion is therefore real, multi-surface tokens occur at
   patch boundaries).
3. Sample LiDAR points ON the visible surfaces (+ noise + a few stray background
   returns) and store them in ego frame -> these are the privileged teacher points.
4. Token GT depth = depth-buffer value at the token center; GT xyz = unprojected.
5. Token features = a deterministic function of (gt_depth, ray-dir, surface-id) +
   Gaussian noise, so a probe *can* recover depth from features above chance but
   not perfectly — mimicking "frozen semantic tokens carry partial metric signal."

This deliberately gives the OT/structure teacher something to beat hard-projection
on: boundary tokens see two surfaces, and stray background points create the
multi-modal candidate sets the paper's medoid/concentration logic targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contract import TokenScene
from geodistill.core.camera import (
    invert_se3,
    project_ego_to_cam,
    unproject_cam_to_ego,
    ray_dirs_ego,
)


@dataclass(frozen=True)
class SyntheticConfig:
    num_cameras: int = 2
    camera_yaw_span: float = 0.25   # half-fan of the rig (rad); < half-FOV so views overlap
    grid_h: int = 8                 # merged-token grid rows per camera
    grid_w: int = 12                # merged-token grid cols per camera
    token_px: float = 28.0          # pixel size per merged token edge (patch14 * merge2)
    feature_dim: int = 32           # tiny stand-in for Qwen 3584 (kept small for CPU)
    num_surfaces: int = 9
    patch_radius: float = 4.5       # surface patch radius (m); larger -> more valid tokens
    depth_min: float = 3.0
    depth_max: float = 75.0
    lidar_per_surface: int = 220
    lidar_depth_noise: float = 0.15
    stray_bg_points: int = 120      # background returns -> multi-modal token candidates
    feature_noise: float = 0.35     # std of the noise floor on every feature dim
    signal_rank: int = 7            # number of leading dims carrying the depth code
    signal_gain: float = 2.0        # amplitude of the depth code (controls probe SNR)
    fx: float = 360.0
    fy: float = 360.0


def _intrinsics(cfg: SyntheticConfig) -> torch.Tensor:
    W = cfg.grid_w * cfg.token_px
    H = cfg.grid_h * cfg.token_px
    K = torch.tensor([[cfg.fx, 0.0, W / 2.0], [0.0, cfg.fy, H / 2.0], [0.0, 0.0, 1.0]])
    return K.unsqueeze(0).repeat(cfg.num_cameras, 1, 1)


def _cam_to_ego(cfg: SyntheticConfig) -> torch.Tensor:
    """Cameras fanned out around ego +x, looking outward (toy multi-view rig)."""
    mats = []
    span = cfg.camera_yaw_span
    yaws = torch.linspace(-span, span, cfg.num_cameras) if cfg.num_cameras > 1 else torch.tensor([0.0])
    for yaw in yaws:
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        # camera z-axis (optical) points along ego forward rotated by yaw.
        R = torch.tensor([
            [-sy, 0.0, cy],
            [-cy, 0.0, -sy],
            [0.0, -1.0, 0.0],
        ])  # maps cam (x right, y down, z fwd) into ego (x fwd, y left, z up)-ish
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([1.4, 0.0, 1.5])  # mounted ahead of and above ego origin
        mats.append(T)
    return torch.stack(mats, dim=0)


def make_synthetic_scene(cfg: SyntheticConfig | None = None, seed: int = 0) -> TokenScene:
    cfg = cfg or SyntheticConfig()
    g = torch.Generator().manual_seed(seed)

    K = _intrinsics(cfg)
    cam_to_ego = _cam_to_ego(cfg)
    Ncam = cfg.num_cameras
    H = int(cfg.grid_h * cfg.token_px)
    W = int(cfg.grid_w * cfg.token_px)
    image_hw = torch.tensor([[H, W]] * Ncam, dtype=torch.long)

    # --- surfaces: random ego points with a depth, each a small fronto patch ---
    surf_depth = torch.empty(cfg.num_surfaces).uniform_(cfg.depth_min, cfg.depth_max, generator=g)
    # place each surface in front of a random camera direction
    surf_yaw = torch.empty(cfg.num_surfaces).uniform_(-0.7, 0.7, generator=g)
    surf_lat = torch.empty(cfg.num_surfaces).uniform_(-0.4, 0.4, generator=g)

    # token grid (per camera) -> boxes, centers
    token_box_list, token_uv_list, cam_id_list = [], [], []
    cam_offsets = [0]
    for c in range(Ncam):
        ys = torch.arange(cfg.grid_h, dtype=torch.float32)
        xs = torch.arange(cfg.grid_w, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u0 = (gx * cfg.token_px).reshape(-1)
        v0 = (gy * cfg.token_px).reshape(-1)
        u1 = u0 + cfg.token_px
        v1 = v0 + cfg.token_px
        token_box_list.append(torch.stack([u0, v0, u1, v1], dim=1))
        token_uv_list.append(torch.stack([(u0 + u1) / 2, (v0 + v1) / 2], dim=1))
        n_tok = cfg.grid_h * cfg.grid_w
        cam_id_list.append(torch.full((n_tok,), c, dtype=torch.long))
        cam_offsets.append(cam_offsets[-1] + n_tok)

    token_box = torch.cat(token_box_list, 0)
    token_uv = torch.cat(token_uv_list, 0)
    token_camera_id = torch.cat(cam_id_list, 0)
    N = token_uv.shape[0]

    # --- build per-token GT depth via a depth buffer over surfaces ---
    gt_depth = torch.full((N,), float("nan"))
    gt_surface = torch.full((N,), -1, dtype=torch.long)
    gt_xyz = torch.full((N, 3), float("nan"))

    # surface ego centers
    surf_center = torch.stack([
        surf_depth * torch.cos(surf_yaw),         # x forward
        surf_depth * torch.sin(surf_yaw) + surf_lat,  # y left
        torch.zeros_like(surf_depth) + 0.0,       # z ~ ground-ish
    ], dim=1)

    for c in range(Ncam):
        sl = slice(cam_offsets[c], cam_offsets[c + 1])
        uv_c = token_uv[sl]
        ray_c = ray_dirs_ego(uv_c, K[c], cam_to_ego[c])     # (n,3) unit in ego
        cam_origin = cam_to_ego[c][:3, 3]
        # intersect each ray with each surface (treat surface as a sphere shell of
        # radius surf_depth around its center -> nearest positive t along ray).
        best_depth = torch.full((uv_c.shape[0],), float("inf"))
        best_surf = torch.full((uv_c.shape[0],), -1, dtype=torch.long)
        best_xyz = torch.full((uv_c.shape[0], 3), float("nan"))
        for s in range(cfg.num_surfaces):
            to_center = surf_center[s] - cam_origin
            t = (ray_c * to_center).sum(dim=1)               # projection length along ray
            hit = ray_c * t.unsqueeze(1) + cam_origin        # closest point on ray to center
            dist = (hit - surf_center[s]).norm(dim=1)
            on_patch = (dist < cfg.patch_radius) & (t > cfg.depth_min)    # patch radius gate
            cam_z = t  # since ray is unit and ~aligned, t approximates camera-z proxy
            better = on_patch & (cam_z < best_depth)
            best_depth = torch.where(better, cam_z, best_depth)
            best_surf = torch.where(better, torch.full_like(best_surf, s), best_surf)
            best_xyz = torch.where(better.unsqueeze(1), hit, best_xyz)
        valid = torch.isfinite(best_depth)
        gt_depth[sl] = torch.where(valid, best_depth, torch.full_like(best_depth, float("nan")))
        gt_surface[sl] = best_surf
        gt_xyz[sl] = best_xyz

    gt_valid = torch.isfinite(gt_depth)

    # --- LiDAR points: sample on visible surface hits + stray background ---
    pts = []
    hit_xyz = gt_xyz[gt_valid]
    if hit_xyz.shape[0] > 0:
        reps = max(1, cfg.lidar_per_surface * cfg.num_surfaces // max(1, hit_xyz.shape[0]))
        base = hit_xyz.repeat(reps, 1)
        noise = torch.randn(base.shape, generator=g) * cfg.lidar_depth_noise
        pts.append(base + noise)
    # stray background points: far, scattered -> create multi-modal candidate sets
    if cfg.stray_bg_points > 0:
        bx = torch.empty(cfg.stray_bg_points).uniform_(cfg.depth_max * 0.6, cfg.depth_max, generator=g)
        by = torch.empty(cfg.stray_bg_points).uniform_(-15.0, 15.0, generator=g)
        bz = torch.empty(cfg.stray_bg_points).uniform_(-1.0, 3.0, generator=g)
        pts.append(torch.stack([bx, by, bz], dim=1))
    points_ego = torch.cat(pts, dim=0) if pts else torch.zeros((0, 3))

    # --- token features: controllable-SNR encoding of depth + ray + surface ---
    # First ``signal_rank`` dims carry a clean, monotone-in-depth code at a chosen
    # SNR so a linear probe recovers depth at a *controlled* accuracy (the fixture
    # is a correctness harness, not a hard benchmark). Remaining dims are pure noise.
    feats = torch.randn(N, cfg.feature_dim, generator=g) * cfg.feature_noise
    d_norm = torch.nan_to_num(gt_depth / cfg.depth_max, nan=0.5).unsqueeze(1)
    sid = torch.nan_to_num(gt_surface.float(), nan=-1).unsqueeze(1) / max(1, cfg.num_surfaces)
    signal = torch.cat([
        d_norm, d_norm ** 2, torch.sqrt(d_norm.clamp_min(0)),
        torch.sin(3.0 * d_norm), torch.cos(2.0 * d_norm), sid,
        token_uv / max(H, W),
    ], dim=1)
    k = min(signal.shape[1], cfg.signal_rank, cfg.feature_dim)
    feats[:, :k] = feats[:, :k] + cfg.signal_gain * signal[:, :k]

    return TokenScene(
        features=feats,
        token_uv=token_uv,
        token_box=token_box,
        token_camera_id=token_camera_id,
        cam_offsets=torch.tensor(cam_offsets, dtype=torch.long),
        K=K,
        cam_to_ego=cam_to_ego,
        image_hw=image_hw,
        points_ego=points_ego,
        v_ego=float(torch.empty(1).uniform_(2.0, 12.0, generator=g)),
        yaw_rate=0.0,
        gt_depth=gt_depth,
        gt_valid=gt_valid,
        gt_xyz_ego=gt_xyz,
        scene_id=f"synthetic_seed{seed}",
        meta={"surface_depth": surf_depth.tolist(), "config": cfg.__dict__},
    )


def make_synthetic_dataset(n_scenes: int, cfg: SyntheticConfig | None = None, seed0: int = 0) -> list[TokenScene]:
    return [make_synthetic_scene(cfg, seed=seed0 + i) for i in range(n_scenes)]
