"""Multi-sweep LiDAR ego-motion accumulation for nuScenes (paper §3.4).

This wraps :mod:`dataset.geodistill.multi_sweep_lidar` with the nuScenes-specific
loading: per-sweep LiDAR file → ego frame at center, dynamic-box exclusion via
the center-frame 3D boxes.

Local CPU has no LiDAR data; the accumulator works on whatever tensors you
feed it, so this module is import-clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dataset.geodistill.multi_sweep_lidar import (
    accumulate_static_sweeps, filter_dynamic_to_center,
)
from geotoken.geometry.lidar_rasterizer import (
    load_lidar_points, transform_lidar_points_to_ego,
)


__all__ = ["MultiSweepLoader", "accumulate"]


@dataclass(frozen=True)
class MultiSweepLoader:
    static_sweeps: int = 4
    dynamic_center_only: bool = True
    z_min: float = -3.0
    z_max: float = 3.0


def accumulate(
    center_lidar_path: str,
    extra_sweep_paths: list[str],
    poses_to_center: list[torch.Tensor],
    boxes_3d: torch.Tensor | None,
    is_dynamic: torch.Tensor | None,
    cfg: MultiSweepLoader = MultiSweepLoader(),
) -> torch.Tensor:
    """Return (M, 3) center-ego LiDAR points after multi-sweep accumulation.

    ``extra_sweep_paths`` contains the K previous sweeps (in chronological
    order) and ``poses_to_center[k]`` is the rigid transform from sweep k's
    ego frame into the center frame. Center-frame points are appended last
    so the dynamic-box filter only drops *accumulated* points that fall
    inside moving boxes.
    """
    center_pts = load_lidar_points(center_lidar_path)[:, :3]
    center_pts = transform_lidar_points_to_ego(center_pts)
    extras = []
    for p in extra_sweep_paths[: cfg.static_sweeps]:
        pts = load_lidar_points(p)[:, :3]
        pts = transform_lidar_points_to_ego(pts)
        extras.append(pts)
    accumulated = accumulate_static_sweeps(extras, poses_to_center[: len(extras)])
    if cfg.dynamic_center_only:
        return filter_dynamic_to_center(accumulated, boxes_3d, is_dynamic, center_pts)
    return torch.cat([accumulated, center_pts], dim=0)
