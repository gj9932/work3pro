"""nuScenes dataset wrapper for GeoDistill-VLM (frozen Qwen2.5-VL native path).

This wrapper composes :class:`dataset.geotoken.NuScenesClipDataset` (legacy GeoToken
clip schema) with the Qwen2.5-VL image processor.  It does **not** modify the
legacy dataset; instead it disables the legacy resize pipeline and feeds the raw
PIL images straight into ``AutoProcessor`` so Qwen's dynamic-grid logic owns
``image_grid_thw``.

Output schema (single-frame, six-camera default):

    pixel_values    Tensor                # processor output (varies by version)
    image_grid_thw  LongTensor (N_cam, 3) # per-camera (t, h, w)
    images_pil      list[list[PIL]]       # T x N_cam original images (debug only)
    image_size_hw   LongTensor (N_cam, 2) # original (H, W) per camera
    camera_intrinsics   FloatTensor (T, N_cam, 3, 3)
    camera_extrinsics   FloatTensor (T, N_cam, 4, 4)
    ego_poses           FloatTensor (T, N_cam, 4, 4)
    ego_state           FloatTensor (2,)  # (v_ego, yaw_rate); zero when missing
    sample_tokens       list[str]
    scene_token         str
    timestamp           int
    lidar_path          str | None  (when load_lidar=True)
    boxes_3d            list  (when load_boxes_3d=True)

References:
- ``paper/work3pro_cvpr2027_draft5_zh_qwen.md`` §3.1, §4.2.
- ``plan/Task_work3_1pro.md`` M1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keep importable in legacy env
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

try:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    Image = None

from dataset.geotoken.nuscenes_clip_dataset import (
    DEFAULT_CAMERA_NAMES,
    NuScenesClipDataset,
)


class NuScenesQwenDataset(Dataset):
    """nuScenes 6-cam clip dataset wired to Qwen2.5-VL native processing.

    The legacy :class:`NuScenesClipDataset` is reused for info parsing and
    calibration, but image loading is overridden so Qwen's processor owns the
    pixel pipeline (no upstream resize, no normalization).
    """

    def __init__(
        self,
        root: str | Path,
        info_path: str | Path,
        processor: Any,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        load_lidar: bool = True,
        load_boxes_3d: bool = True,
        load_map_labels: bool = False,
        load_ego_state: bool = True,
        return_pil: bool = False,
    ) -> None:
        if torch is None or Image is None:
            raise ImportError("torch and Pillow are required for NuScenesQwenDataset")
        # 复用 NuScenesClipDataset 的 info / calibration 解析,但禁掉 image_size 与 transform
        # —— 由 Qwen processor 全权负责像素流(动态分辨率 + 归一化)。
        self._inner = NuScenesClipDataset(
            root=root,
            info_path=info_path,
            camera_names=tuple(camera_names),
            clip_length=1,
            center_frame=True,
            image_size=None,
            transform=None,
            corruption=None,
            load_lidar=load_lidar,
            load_boxes_3d=load_boxes_3d,
            load_map_labels=load_map_labels,
        )
        self.processor = processor
        self.camera_names = tuple(camera_names)
        self.load_ego_state = load_ego_state
        self.return_pil = return_pil

    # ----------------------------------------------------------------- API
    @property
    def root(self) -> Path:
        return self._inner.root

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self._inner.infos[index]
        cam_infos = info["cam_infos"]

        pil_images = []
        intrinsics = []
        extrinsics = []
        ego_poses = []
        sizes_hw = []
        for cam_name in self.camera_names:
            cam_info = cam_infos[cam_name]
            pil = _load_pil(self._inner.root / cam_info["filename"])
            pil_images.append(pil)
            sizes_hw.append((pil.height, pil.width))
            calibrated = cam_info["calibrated_sensor"]
            intrinsics.append(_camera_intrinsic(calibrated))
            extrinsics.append(_transform_matrix(calibrated))
            ego_poses.append(_transform_matrix(cam_info["ego_pose"]))

        # 喂给 Qwen 的处理器:images=list[PIL]。返回的 pixel_values 与 image_grid_thw
        # 由 processor 决定。我们不在 dataset 内 resize / normalize。
        proc_out = self.processor(images=pil_images, text=None, return_tensors="pt")
        pixel_values = proc_out["pixel_values"]
        image_grid_thw = proc_out["image_grid_thw"]

        sample: dict[str, Any] = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_size_hw": torch.as_tensor(sizes_hw, dtype=torch.long),
            "camera_intrinsics": torch.stack(intrinsics, dim=0).unsqueeze(0),
            "camera_extrinsics": torch.stack(extrinsics, dim=0).unsqueeze(0),
            "ego_poses": torch.stack(ego_poses, dim=0).unsqueeze(0),
            "sample_tokens": [info.get("sample_token")],
            "scene_token": info.get("scene_token"),
            "timestamp": info.get("timestamp"),
        }
        if self.load_ego_state:
            sample["ego_state"] = _ego_state(info)
        if self._inner.load_lidar:
            sample["lidar_path"] = self._inner._get_lidar_path(info)
        if self._inner.load_boxes_3d:
            sample["boxes_3d"] = self._inner._get_optional(info, "boxes_3d", "gt_boxes", default=[])
        if self._inner.load_map_labels:
            sample["map_labels"] = self._inner._get_optional(info, "map_labels", default={})
        if self.return_pil:
            sample["images_pil"] = [pil_images]
        return sample


# -------------------------------------------------------------------- helpers
def _load_pil(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _camera_intrinsic(calibrated_sensor: dict[str, Any]):
    intrinsic = calibrated_sensor.get("camera_intrinsic")
    if intrinsic is None:
        return torch.eye(3, dtype=torch.float32)
    return torch.as_tensor(intrinsic, dtype=torch.float32)


def _transform_matrix(record: dict[str, Any]):
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = _quaternion_to_rotation(record["rotation"])
    matrix[:3, 3] = torch.as_tensor(record["translation"], dtype=torch.float32)
    return matrix


def _quaternion_to_rotation(quaternion):
    w, x, y, z = (float(v) for v in quaternion)
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


def _ego_state(info: dict[str, Any]):
    """Extract (v_ego, yaw_rate) from nuScenes info; default zeros when missing.

    Different info pickles store ego state under different keys; we fall back to
    a zero tensor instead of failing so M1 unblocks even on partial dumps.
    """
    candidates = (
        ("ego_state", ("v_ego", "yaw_rate")),
        ("ego_motion", ("v_ego", "yaw_rate")),
    )
    for parent_key, fields in candidates:
        parent = info.get(parent_key)
        if isinstance(parent, dict) and all(f in parent for f in fields):
            return torch.as_tensor([float(parent[fields[0]]), float(parent[fields[1]])], dtype=torch.float32)
    if "v_ego" in info:
        v = float(info["v_ego"])
        yaw = float(info.get("yaw_rate", 0.0))
        return torch.as_tensor([v, yaw], dtype=torch.float32)
    return torch.zeros(2, dtype=torch.float32)
