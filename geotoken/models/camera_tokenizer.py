from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraTokenizerConfig:
    embed_dim: int = 256
    backbone: str = "light_cnn"
    active_anchor_count: int = 400
