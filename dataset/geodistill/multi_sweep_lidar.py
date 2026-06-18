"""Multi-sweep LiDAR aggregation (paper §3.4).

Static points: accumulate adjacent LiDAR sweeps after ego-motion compensation
to densify far-distance coverage.
Dynamic points: only the center frame contributes (otherwise moving objects
would smear into ghost trails).

All public APIs accept torch tensors; use them on the GPU box where nuScenes
is available. On the CPU box this module is import-clean (no real LiDAR data).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


__all__ = ["MultiSweepConfig", "accumulate_static_sweeps", "filter_dynamic_to_center"]


@dataclass(frozen=True)
class MultiSweepConfig:
    enable: bool = True
    static_sweeps: int = 4
    dynamic_center_only: bool = True


def accumulate_static_sweeps(
    sweeps_ego: list[torch.Tensor],          # each (M_t, 3) in its own ego frame
    poses_to_center: list[torch.Tensor],     # each (4, 4) sweep ego -> center ego
) -> torch.Tensor:
    """Concatenate sweeps in the center-ego frame.

    Each sweep ``t`` is transformed by ``poses_to_center[t] @ [x; 1]`` to align
    onto the center frame. Returns (M_total, 3).
    """
    if len(sweeps_ego) != len(poses_to_center):
        raise ValueError("sweeps_ego and poses_to_center must align")
    out = []
    for pts, T in zip(sweeps_ego, poses_to_center):
        if pts.numel() == 0:
            continue
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)
        ph = torch.cat([pts.float(), ones], dim=1)
        moved = (ph @ T.float().T)[:, :3]
        out.append(moved)
    if not out:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.cat(out, dim=0)


def filter_dynamic_to_center(
    accumulated: torch.Tensor,                # (M, 3) in center-ego frame
    boxes_3d: torch.Tensor | None,            # (B, 7) — (cx,cy,cz,dx,dy,dz,yaw)
    is_dynamic: torch.Tensor | None,          # (B,) bool — which boxes are moving
    center_points: torch.Tensor,              # (M_c, 3) only-center-frame points
) -> torch.Tensor:
    """Drop accumulated points that fall inside any dynamic 3D box; keep center-only points."""
    if boxes_3d is None or is_dynamic is None:
        return accumulated if accumulated.numel() else center_points

    moving = boxes_3d[is_dynamic.bool()]
    if moving.shape[0] == 0:
        return torch.cat([accumulated, center_points], dim=0)

    keep = torch.ones(accumulated.shape[0], dtype=torch.bool, device=accumulated.device)
    for i in range(moving.shape[0]):
        cx, cy, cz, dx, dy, dz, yaw = moving[i].tolist()
        cos_y, sin_y = float(torch.cos(torch.tensor(yaw))), float(torch.sin(torch.tensor(yaw)))
        # transform points into the box's local frame
        x = accumulated[:, 0] - cx
        y = accumulated[:, 1] - cy
        xb = x * cos_y + y * sin_y
        yb = -x * sin_y + y * cos_y
        zb = accumulated[:, 2] - cz
        inside = (xb.abs() <= dx / 2.0) & (yb.abs() <= dy / 2.0) & (zb.abs() <= dz / 2.0)
        keep &= ~inside

    return torch.cat([accumulated[keep], center_points], dim=0)
