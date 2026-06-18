from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable

try:
    import torch
except ImportError:  # keep module importable before dependencies are installed
    torch = None


def apply_camera_corruption(sample: dict, corruption: Any = None) -> dict:
    """Apply an optional camera corruption hook without changing sample schema."""
    if corruption is None:
        return sample
    if isinstance(corruption, CameraCorruption):
        return corruption(sample)
    if callable(corruption):
        return corruption(sample)
    raise ValueError(f"Unsupported camera corruption hook: {corruption!r}")


class CameraCorruption:
    """Base interface for robustness corruptions added in later milestones."""

    def __call__(self, sample: dict) -> dict:
        return sample


class IdentityCameraCorruption(CameraCorruption):
    pass


class CalibrationNoiseCorruption(CameraCorruption):
    """Perturb camera intrinsics / extrinsics for calibration robustness eval.

    GeoDistill-VLM §4.3 / Task M12 require calibration noise as a corruption type.
    Schema unchanged: only numerical values in `camera_intrinsics` /
    `camera_extrinsics` are jittered.
    """

    def __init__(
        self,
        intrinsic_pixel_std: float = 1.0,
        extrinsic_translation_std: float = 0.02,
        extrinsic_rotation_deg_std: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.intrinsic_pixel_std = float(intrinsic_pixel_std)
        self.extrinsic_translation_std = float(extrinsic_translation_std)
        self.extrinsic_rotation_deg_std = float(extrinsic_rotation_deg_std)
        self._rng = random.Random(seed)

    def __call__(self, sample: dict) -> dict:
        if torch is None:
            return sample
        if "camera_intrinsics" in sample:
            sample["camera_intrinsics"] = self._jitter_intrinsics(sample["camera_intrinsics"])
        if "camera_extrinsics" in sample:
            sample["camera_extrinsics"] = self._jitter_extrinsics(sample["camera_extrinsics"])
        return sample

    def _jitter_intrinsics(self, intrinsics):
        K = intrinsics.clone()
        # Jitter (cx, cy) only; fx/fy 不动以避免训练标签深度尺度变化。
        flat = K.reshape(-1, 3, 3)
        for i in range(flat.shape[0]):
            flat[i, 0, 2] += self._gauss() * self.intrinsic_pixel_std
            flat[i, 1, 2] += self._gauss() * self.intrinsic_pixel_std
        return K

    def _jitter_extrinsics(self, extrinsics):
        T = extrinsics.clone()
        flat = T.reshape(-1, 4, 4)
        for i in range(flat.shape[0]):
            flat[i, :3, 3] = flat[i, :3, 3] + self._random_translation(flat.dtype, flat.device)
            flat[i, :3, :3] = self._small_rotation(flat.dtype, flat.device) @ flat[i, :3, :3]
        return T

    def _gauss(self) -> float:
        return self._rng.gauss(0.0, 1.0)

    def _random_translation(self, dtype, device):
        v = [self._gauss() * self.extrinsic_translation_std for _ in range(3)]
        return torch.as_tensor(v, dtype=dtype, device=device)

    def _small_rotation(self, dtype, device):
        rad = math.radians(self.extrinsic_rotation_deg_std)
        rx, ry, rz = (self._gauss() * rad for _ in range(3))
        cos = math.cos
        sin = math.sin
        Rx = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos(rx), -sin(rx)], [0.0, sin(rx), cos(rx)]],
            dtype=dtype,
            device=device,
        )
        Ry = torch.tensor(
            [[cos(ry), 0.0, sin(ry)], [0.0, 1.0, 0.0], [-sin(ry), 0.0, cos(ry)]],
            dtype=dtype,
            device=device,
        )
        Rz = torch.tensor(
            [[cos(rz), -sin(rz), 0.0], [sin(rz), cos(rz), 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )
        return Rz @ Ry @ Rx


class LowLightCorruption(CameraCorruption):
    """Multiply image brightness by a small factor to mimic night driving.

    Operates in-place on `sample["images"]` whose values are in [0, 1] floats.
    Sample schema unchanged.
    """

    def __init__(self, gain: float = 0.25, gamma: float = 1.6) -> None:
        if gain <= 0.0 or gain > 1.0:
            raise ValueError(f"gain must be in (0, 1]; got {gain}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive; got {gamma}")
        self.gain = float(gain)
        self.gamma = float(gamma)

    def __call__(self, sample: dict) -> dict:
        if torch is None or "images" not in sample:
            return sample
        images = sample["images"]
        sample["images"] = (images.clamp(0.0, 1.0) * self.gain).pow(self.gamma).clamp(0.0, 1.0)
        return sample


def build_camera_corruption(config: Any = None) -> Callable[[dict], dict] | None:
    if config is None or config == "none":
        return None
    if config == "identity" or config == {}:
        return IdentityCameraCorruption()
    if isinstance(config, str):
        return _build_named(config, {})
    if isinstance(config, dict):
        kind = config.get("type") or config.get("name")
        if not kind:
            raise ValueError(f"Camera corruption config must have a 'type' field: {config!r}")
        params = {k: v for k, v in config.items() if k not in ("type", "name")}
        return _build_named(kind, params)
    raise ValueError(f"Unknown camera corruption config: {config!r}")


def _build_named(kind: str, params: dict) -> CameraCorruption:
    if kind == "identity":
        return IdentityCameraCorruption()
    if kind == "calibration_noise":
        return CalibrationNoiseCorruption(**params)
    if kind == "low_light":
        return LowLightCorruption(**params)
    raise ValueError(f"Unknown camera corruption kind: {kind!r}")
