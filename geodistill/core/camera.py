"""Camera geometry helpers for token-level teacher construction (paper §3.4–3.5).

Conventions (all torch float32):
- Intrinsics K: (3, 3) per camera.
- Extrinsics T_ego->cam stored as cam->ego (4, 4) homogeneous (matches dataset).
- Points in ego frame: (M, 3).

Provides projection ego->pixel, unprojection pixel+depth->ego, ray directions,
and a token-region membership test used to assemble per-token LiDAR sets ``Q_p``.
"""

from __future__ import annotations

import torch

__all__ = [
    "invert_se3",
    "project_ego_to_cam",
    "unproject_cam_to_ego",
    "ray_dirs_ego",
    "assign_points_to_tokens",
]


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert a (..., 4, 4) rigid transform efficiently."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def project_ego_to_cam(points_ego: torch.Tensor, K: torch.Tensor, cam_to_ego: torch.Tensor, min_depth: float = 1e-3):
    """Project ego points into one camera.

    Returns dict: uv (M,2), depth (M,), valid (M,) where depth>min_depth.
    """
    points_ego = torch.as_tensor(points_ego, dtype=torch.float32)
    ego_to_cam = invert_se3(torch.as_tensor(cam_to_ego, dtype=torch.float32))
    M = points_ego.shape[0]
    ones = torch.ones((M, 1), dtype=torch.float32)
    p_h = torch.cat([points_ego, ones], dim=1)             # (M,4)
    p_cam = (p_h @ ego_to_cam.T)[:, :3]                    # (M,3)
    depth = p_cam[:, 2]
    pix_h = p_cam @ K.to(torch.float32).T                  # (M,3)
    denom = depth.clamp_min(min_depth).unsqueeze(-1)
    uv = pix_h[:, :2] / denom
    valid = depth > min_depth
    return {"uv": uv, "depth": depth, "valid": valid, "points_cam": p_cam}


def unproject_cam_to_ego(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, cam_to_ego: torch.Tensor) -> torch.Tensor:
    """Back-project pixel(s) + metric depth to ego coordinates. uv:(...,2), depth:(...,)."""
    uv = torch.as_tensor(uv, dtype=torch.float32)
    depth = torch.as_tensor(depth, dtype=torch.float32)
    Kinv = torch.inverse(K.to(torch.float32))
    ones = torch.ones(uv.shape[:-1] + (1,), dtype=torch.float32)
    pix_h = torch.cat([uv, ones], dim=-1)                  # (...,3)
    dirs = pix_h @ Kinv.T                                  # (...,3) camera ray (z=1 plane)
    p_cam = dirs * depth.unsqueeze(-1)
    p_cam_h = torch.cat([p_cam, torch.ones(p_cam.shape[:-1] + (1,))], dim=-1)
    p_ego = (p_cam_h @ cam_to_ego.to(torch.float32).T)[..., :3]
    return p_ego


def ray_dirs_ego(uv: torch.Tensor, K: torch.Tensor, cam_to_ego: torch.Tensor) -> torch.Tensor:
    """Unit ray directions in ego frame for pixels ``uv`` (...,2)."""
    uv = torch.as_tensor(uv, dtype=torch.float32)
    Kinv = torch.inverse(K.to(torch.float32))
    ones = torch.ones(uv.shape[:-1] + (1,), dtype=torch.float32)
    pix_h = torch.cat([uv, ones], dim=-1)
    dirs_cam = pix_h @ Kinv.T
    R = cam_to_ego[:3, :3].to(torch.float32)
    dirs_ego = dirs_cam @ R.T
    return dirs_ego / dirs_ego.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def assign_points_to_tokens(uv: torch.Tensor, token_box: torch.Tensor) -> torch.Tensor:
    """Assign each projected point to a token region.

    Args:
        uv:        (M, 2) projected pixels.
        token_box: (N, 4) (u0, v0, u1, v1) per token.
    Returns:
        token_idx (M,) long, -1 when a point falls outside every token box.
    """
    uv = torch.as_tensor(uv, dtype=torch.float32)
    token_box = torch.as_tensor(token_box, dtype=torch.float32)
    u = uv[:, 0].unsqueeze(1)   # (M,1)
    v = uv[:, 1].unsqueeze(1)
    u0, v0, u1, v1 = token_box[:, 0], token_box[:, 1], token_box[:, 2], token_box[:, 3]
    inside = (u >= u0) & (u < u1) & (v >= v0) & (v < v1)   # (M,N)
    has = inside.any(dim=1)
    idx = torch.full((uv.shape[0],), -1, dtype=torch.long)
    first = inside.float().argmax(dim=1)
    idx[has] = first[has]
    return idx
