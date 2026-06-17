"""GeoDistill-VLM evaluation suite (paper §4.5, §4.7)."""

from .shortcut_audit import (
    SHUFFLE_KINDS, ShuffleConfig, apply_shuffle, shortcut_audit_step,
)

__all__ = ["SHUFFLE_KINDS", "ShuffleConfig", "apply_shuffle", "shortcut_audit_step"]
