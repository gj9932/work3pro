from __future__ import annotations

from typing import Any, Callable


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


def build_camera_corruption(config: Any = None) -> Callable[[dict], dict] | None:
    if config is None or config == "none":
        return None
    if config == "identity" or config == {}:
        return IdentityCameraCorruption()
    raise ValueError(f"Unknown camera corruption config: {config!r}")
