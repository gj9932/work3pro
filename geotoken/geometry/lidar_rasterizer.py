from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

try:
    import numpy as np
except ImportError:  # keep package importable before dependencies are installed
    np = None

try:
    import torch
except ImportError:  # keep package importable before dependencies are installed
    torch = None

from .bev_grid import BEVAnchorGrid, OCCUPANCY_FREE, OCCUPANCY_OCCUPIED, OCCUPANCY_UNKNOWN


@dataclass(frozen=True)
class LidarRasterizerConfig:
    z_range: tuple[float, float] = (-3.0, 3.0)
    occupied_min_points: int = 1
    mark_free_rays: bool = True


class LidarRasterizer:
    def __init__(self, grid: BEVAnchorGrid, config: LidarRasterizerConfig | None = None):
        _require_torch()
        self.grid = grid
        self.config = config or LidarRasterizerConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any], grid: BEVAnchorGrid | None = None):
        cfg = config.get("lidar_rasterizer", {})
        return cls(
            grid or BEVAnchorGrid.from_config(config),
            LidarRasterizerConfig(
                z_range=tuple(cfg.get("z_range", (-3.0, 3.0))),
                occupied_min_points=int(cfg.get("occupied_min_points", 1)),
                mark_free_rays=bool(cfg.get("mark_free_rays", True)),
            ),
        )

    def rasterize_file(
        self,
        lidar_path: str | Path,
        lidar_calibrated_sensor: Any = None,
        lidar_ego_pose: Any = None,
        target_ego_pose: Any = None,
    ) -> dict[str, torch.Tensor]:
        points = load_lidar_points(lidar_path)
        points_ego = transform_lidar_points_to_ego(points, lidar_calibrated_sensor, lidar_ego_pose, target_ego_pose)
        return self.rasterize(points_ego)

    def rasterize(self, points_ego: torch.Tensor) -> dict[str, torch.Tensor]:
        points_ego = torch.as_tensor(points_ego, dtype=torch.float32)
        if points_ego.ndim != 2 or points_ego.shape[1] < 3:
            raise ValueError(f"points_ego must have shape [N, >=3], got {tuple(points_ego.shape)}")

        ix, iy, valid = self._cell_indices(points_ego[:, :3])
        nx, ny = self.grid.grid_shape
        linear = ix[valid] * ny + iy[valid]
        counts = torch.bincount(linear, minlength=self.grid.num_anchors).to(torch.long)
        occupied = counts >= self.config.occupied_min_points

        labels = torch.full((self.grid.num_anchors,), OCCUPANCY_UNKNOWN, dtype=torch.long)
        free = self._free_ray_mask(occupied.reshape(nx, ny)) if self.config.mark_free_rays else torch.zeros((nx, ny), dtype=torch.bool)
        labels[free.reshape(-1)] = OCCUPANCY_FREE
        labels[occupied] = OCCUPANCY_OCCUPIED

        occupancy_map = labels.reshape(nx, ny)
        return {
            "points_ego": points_ego,
            "valid_point_mask": valid,
            "point_counts": counts,
            "occupancy_labels": labels,
            "occupancy_map": occupancy_map,
            "free_space_boundary_mask": _free_space_boundary(occupancy_map),
        }

    def _cell_indices(self, points_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.grid.config
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        ix = torch.floor((x - cfg.x_range[0]) / cfg.cell_size[0]).to(torch.long)
        iy = torch.floor((y - cfg.y_range[0]) / cfg.cell_size[1]).to(torch.long)
        nx, ny = self.grid.grid_shape
        valid = (
            (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
            & (z >= self.config.z_range[0])
            & (z <= self.config.z_range[1])
        )
        return ix.clamp(0, nx - 1), iy.clamp(0, ny - 1), valid

    def _free_ray_mask(self, occupied_map: torch.Tensor) -> torch.Tensor:
        nx, ny = self.grid.grid_shape
        origin_ix = int(math.floor((0.0 - self.grid.config.x_range[0]) / self.grid.config.cell_size[0]))
        origin_iy = int(math.floor((0.0 - self.grid.config.y_range[0]) / self.grid.config.cell_size[1]))
        origin_ix = min(max(origin_ix, 0), nx - 1)
        origin_iy = min(max(origin_iy, 0), ny - 1)

        free = torch.zeros((nx, ny), dtype=torch.bool)
        occupied_cells = torch.nonzero(occupied_map, as_tuple=False).tolist()
        for ix, iy in occupied_cells:
            cells = _bresenham(origin_ix, origin_iy, int(ix), int(iy))[:-1]
            for cx, cy in cells:
                if not occupied_map[cx, cy]:
                    free[cx, cy] = True
        return free


def load_lidar_points(lidar_path: str | Path) -> torch.Tensor:
    _require_torch()
    path = Path(lidar_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if np is None:
        raise ImportError("numpy is required to load LiDAR point files")
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 5 == 0:
            array = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            array = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            array = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer point dimension for LiDAR file: {path}")
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"LiDAR points must have shape [N, >=3], got {array.shape}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def transform_lidar_points_to_ego(
    points: torch.Tensor,
    lidar_calibrated_sensor: Any = None,
    lidar_ego_pose: Any = None,
    target_ego_pose: Any = None,
) -> torch.Tensor:
    _require_torch()
    points = torch.as_tensor(points, dtype=torch.float32)
    sensor_to_ego = transform_matrix(lidar_calibrated_sensor)
    if lidar_ego_pose is not None and target_ego_pose is not None:
        source_ego_to_global = transform_matrix(lidar_ego_pose)
        global_to_target_ego = torch.inverse(transform_matrix(target_ego_pose))
        transform = global_to_target_ego @ source_ego_to_global @ sensor_to_ego
    else:
        transform = sensor_to_ego
    xyz = apply_transform(points[:, :3], transform)
    if points.shape[1] > 3:
        return torch.cat([xyz, points[:, 3:]], dim=1)
    return xyz


def transform_matrix(record: Any = None) -> torch.Tensor:
    _require_torch()
    if record is None:
        return torch.eye(4, dtype=torch.float32)
    if isinstance(record, torch.Tensor):
        matrix = record.detach().clone().to(dtype=torch.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform tensor must be [4, 4], got {tuple(matrix.shape)}")
        return matrix
    if hasattr(record, "shape") and tuple(record.shape) == (4, 4):
        return torch.as_tensor(record, dtype=torch.float32)
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = quaternion_to_rotation(record["rotation"])
    matrix[:3, 3] = torch.as_tensor(record["translation"], dtype=torch.float32)
    return matrix


def quaternion_to_rotation(quaternion: list[float] | tuple[float, ...]) -> torch.Tensor:
    _require_torch()
    w, x, y, z = [float(v) for v in quaternion]
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        return torch.eye(3, dtype=torch.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def apply_transform(points_xyz: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype, device=points_xyz.device)
    points_h = torch.cat([points_xyz, ones], dim=1)
    return (points_h @ transform.to(points_xyz.device, points_xyz.dtype).T)[:, :3]


def _free_space_boundary(occupancy_map: torch.Tensor) -> torch.Tensor:
    nx, ny = occupancy_map.shape
    boundary = torch.zeros((nx, ny), dtype=torch.bool)
    free = occupancy_map == OCCUPANCY_FREE
    for ix in range(nx):
        for iy in range(ny):
            if not free[ix, iy]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and occupancy_map[jx, jy] != OCCUPANCY_FREE:
                    boundary[ix, iy] = True
                    break
    return boundary.reshape(-1)


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for LiDAR rasterization")
