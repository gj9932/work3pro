from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keep module importable before dependencies are installed
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

try:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:  # keep module importable before dependencies are installed
    Image = None

from .corruptions import apply_camera_corruption, build_camera_corruption

DEFAULT_CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


class NuScenesClipDataset(Dataset):
    """nuScenes multi-camera clip dataset for GeoToken Step 2."""

    def __init__(
        self,
        root: str | Path,
        info_path: str | Path,
        camera_names: list[str] | tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        clip_length: int = 1,
        center_frame: bool = True,
        image_size: tuple[int, int] | list[int] | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        corruption: Any = None,
        load_lidar: bool = True,
        load_boxes_3d: bool = False,
        load_map_labels: bool = False,
    ):
        if clip_length != 1:
            raise ValueError("Step 2 minimal NuScenesClipDataset only supports clip_length=1")
        if not center_frame:
            raise ValueError("Step 2 minimal NuScenesClipDataset requires center_frame=True")

        self.root = Path(root).expanduser()
        self.info_path = Path(info_path).expanduser()
        self.camera_names = tuple(camera_names)
        self.clip_length = clip_length
        self.center_frame = center_frame
        self.image_size = tuple(image_size) if image_size else None
        self.transform = transform
        self.corruption = build_camera_corruption(corruption)
        self.load_lidar = load_lidar
        self.load_boxes_3d = load_boxes_3d
        self.load_map_labels = load_map_labels
        self.infos = self._load_infos(self.info_path)

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides):
        dataset_cfg = dict(config.get("dataset", config))
        dataset_cfg.update(overrides)
        return cls(
            root=dataset_cfg["root"],
            info_path=dataset_cfg["info_path"],
            camera_names=dataset_cfg.get("camera_names", DEFAULT_CAMERA_NAMES),
            clip_length=dataset_cfg.get("clip_length", 1),
            center_frame=dataset_cfg.get("center_frame", True),
            image_size=dataset_cfg.get("image_size"),
            corruption=dataset_cfg.get("corruption"),
            load_lidar=dataset_cfg.get("load_lidar", True),
            load_boxes_3d=dataset_cfg.get("load_boxes_3d", False),
            load_map_labels=dataset_cfg.get("load_map_labels", False),
        )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.infos[index]
        frame = self._load_frame(info)
        sample = {
            "images": frame["images"].unsqueeze(0),
            "camera_intrinsics": frame["camera_intrinsics"].unsqueeze(0),
            "camera_extrinsics": frame["camera_extrinsics"].unsqueeze(0),
            "ego_poses": frame["ego_poses"].unsqueeze(0),
            "sample_tokens": [info.get("sample_token")],
            "scene_token": info.get("scene_token"),
            "timestamp": info.get("timestamp"),
        }
        if self.load_lidar:
            sample["lidar_path"] = self._get_lidar_path(info)
        if self.load_boxes_3d:
            sample["boxes_3d"] = self._get_optional(info, "boxes_3d", "gt_boxes", default=[])
        if self.load_map_labels:
            sample["map_labels"] = self._get_optional(info, "map_labels", default={})
        return apply_camera_corruption(sample, self.corruption)

    def _load_frame(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = []
        intrinsics = []
        extrinsics = []
        ego_poses = []
        cam_infos = info["cam_infos"]

        for cam_name in self.camera_names:
            cam_info = cam_infos[cam_name]
            images.append(self._load_image(cam_info["filename"]))
            calibrated = cam_info["calibrated_sensor"]
            intrinsics.append(_camera_intrinsic(calibrated))
            extrinsics.append(_transform_matrix(calibrated))
            ego_poses.append(_transform_matrix(cam_info["ego_pose"]))

        return {
            "images": torch.stack(images, dim=0),
            "camera_intrinsics": torch.stack(intrinsics, dim=0),
            "camera_extrinsics": torch.stack(extrinsics, dim=0),
            "ego_poses": torch.stack(ego_poses, dim=0),
        }

    def _load_image(self, filename: str) -> torch.Tensor:
        image_path = self.root / filename
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if Image is None or torch is None:
            raise ImportError("Pillow and torch are required to load nuScenes images")
        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            height, width = self.image_size
            image = image.resize((width, height), Image.BILINEAR)
        if self.transform:
            return self.transform(image)
        byte_data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        height, width = image.size[1], image.size[0]
        return byte_data.view(height, width, 3).permute(2, 0, 1).float().div(255.0).contiguous()

    def _get_lidar_path(self, info: dict[str, Any]) -> str | None:
        lidar_infos = info.get("lidar_infos", {})
        lidar_info = lidar_infos.get("LIDAR_TOP")
        if not lidar_info:
            return None
        return str(self.root / lidar_info["filename"])

    @staticmethod
    def _load_infos(info_path: Path) -> list[dict[str, Any]]:
        with info_path.open("rb") as f:
            infos = pickle.load(f)
        if not isinstance(infos, list):
            raise ValueError(f"nuScenes info file must contain a list: {info_path}")
        return infos

    @staticmethod
    def _get_optional(info: dict[str, Any], *keys: str, default=None):
        for key in keys:
            if key in info:
                return info[key]
        return default


def _camera_intrinsic(calibrated_sensor: dict[str, Any]) -> torch.Tensor:
    intrinsic = calibrated_sensor.get("camera_intrinsic")
    if intrinsic is None:
        return torch.eye(3, dtype=torch.float32)
    return torch.as_tensor(intrinsic, dtype=torch.float32)


def _transform_matrix(record: dict[str, Any]) -> torch.Tensor:
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = _quaternion_to_rotation(record["rotation"])
    matrix[:3, 3] = torch.as_tensor(record["translation"], dtype=torch.float32)
    return matrix


def _quaternion_to_rotation(quaternion: list[float] | tuple[float, ...]) -> torch.Tensor:
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
