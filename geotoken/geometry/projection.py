from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # keep package importable before dependencies are installed
    torch = None

from .bev_grid import BEVAnchorGrid
from .lidar_rasterizer import transform_matrix


@dataclass(frozen=True)
class CameraProjectionConfig:
    image_size: tuple[int, int] = (256, 704)
    anchor_height: float = 0.0
    min_depth: float = 1e-3


class CameraProjector:
    def __init__(self, config: CameraProjectionConfig | None = None):
        _require_torch()
        self.config = config or CameraProjectionConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        dataset_cfg = config.get("dataset", {})
        projection_cfg = config.get("projection", {})
        image_size = projection_cfg.get("image_size", dataset_cfg.get("image_size", (256, 704)))
        return cls(
            CameraProjectionConfig(
                image_size=tuple(image_size),
                anchor_height=float(projection_cfg.get("anchor_height", 0.0)),
                min_depth=float(projection_cfg.get("min_depth", 1e-3)),
            )
        )

    def project_anchors(
        self,
        anchors_xy: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        return project_ego_points_to_cameras(
            anchors_xy,
            camera_intrinsics,
            camera_extrinsics,
            image_size=image_size or self.config.image_size,
            anchor_height=self.config.anchor_height,
            min_depth=self.config.min_depth,
        )

    def project_grid(
        self,
        grid: BEVAnchorGrid,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.project_anchors(grid.centers, camera_intrinsics, camera_extrinsics, image_size=image_size)


def project_ego_points_to_cameras(
    points_ego: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    image_size: tuple[int, int] | list[int],
    anchor_height: float = 0.0,
    min_depth: float = 1e-3,
) -> dict[str, torch.Tensor]:
    _require_torch()
    points_ego = torch.as_tensor(points_ego, dtype=torch.float32)
    intrinsics = torch.as_tensor(camera_intrinsics, dtype=torch.float32)
    extrinsics = torch.stack([transform_matrix(m) for m in camera_extrinsics]) if not isinstance(camera_extrinsics, torch.Tensor) else camera_extrinsics.to(dtype=torch.float32)

    if points_ego.ndim != 2 or points_ego.shape[1] not in {2, 3}:
        raise ValueError(f"points_ego must have shape [L, 2] or [L, 3], got {tuple(points_ego.shape)}")
    if points_ego.shape[1] == 2:
        z = torch.full((points_ego.shape[0], 1), float(anchor_height), dtype=points_ego.dtype, device=points_ego.device)
        points_xyz = torch.cat([points_ego, z], dim=1)
    else:
        points_xyz = points_ego

    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"camera_intrinsics must have shape [N_cam, 3, 3], got {tuple(intrinsics.shape)}")
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"camera_extrinsics must have shape [N_cam, 4, 4], got {tuple(extrinsics.shape)}")
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError("camera_intrinsics and camera_extrinsics must have the same camera count")

    height, width = int(image_size[0]), int(image_size[1])
    device = intrinsics.device
    dtype = intrinsics.dtype
    points_xyz = points_xyz.to(device=device, dtype=dtype)
    ones = torch.ones((points_xyz.shape[0], 1), dtype=dtype, device=device)
    points_h = torch.cat([points_xyz, ones], dim=1)

    ego_to_camera = torch.inverse(extrinsics.to(device=device, dtype=dtype))
    points_cam_h = torch.einsum("nxy,ly->nlx", ego_to_camera, points_h)
    points_cam = points_cam_h[..., :3]
    depth = points_cam[..., 2]
    pixels_h = torch.einsum("nij,nlj->nli", intrinsics.to(device=device, dtype=dtype), points_cam)
    denom = depth.clamp_min(min_depth).unsqueeze(-1)
    uv = pixels_h[..., :2] / denom

    valid_depth = depth > min_depth
    in_image = (uv[..., 0] >= 0.0) & (uv[..., 0] < width) & (uv[..., 1] >= 0.0) & (uv[..., 1] < height)
    valid = valid_depth & in_image
    return {
        "uv": uv,
        "depth": depth,
        "valid_mask": valid,
        "camera_visible": valid.any(dim=0),
        "points_camera": points_cam,
    }


def projection_from_sample(sample: dict[str, Any], anchors_xy: torch.Tensor, image_size: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
    intrinsics = sample["camera_intrinsics"]
    extrinsics = sample["camera_extrinsics"]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if image_size is None:
        images = sample.get("images")
        if images is not None:
            image_size = (int(images.shape[-2]), int(images.shape[-1]))
        else:
            image_size = (256, 704)
    return project_ego_points_to_cameras(anchors_xy, intrinsics, extrinsics, image_size=image_size)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for camera projection")
