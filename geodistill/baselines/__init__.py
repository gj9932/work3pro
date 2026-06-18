"""GeoDistill-VLM baselines (paper §4.4)."""

from .profile import ProfileSpec, PROFILES, apply_profile_to_model, get_profile_spec

__all__ = ["ProfileSpec", "PROFILES", "apply_profile_to_model", "get_profile_spec",
           "SpaceDriveStyleAdapter", "SpaceDriveCalibratedAdapter", "LLaVA15Adapter"]


def __getattr__(name: str):
    """Lazy import for third-party wrappers (raise only when actually needed)."""
    if name == "SpaceDriveStyleAdapter":
        from .spacedrive_style import SpaceDriveStyleAdapter
        return SpaceDriveStyleAdapter
    if name == "SpaceDriveCalibratedAdapter":
        from .spacedrive_calibrated import SpaceDriveCalibratedAdapter
        return SpaceDriveCalibratedAdapter
    if name == "LLaVA15Adapter":
        from .llava15 import LLaVA15Adapter
        return LLaVA15Adapter
    raise AttributeError(name)
