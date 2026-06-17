"""Shared data contract for P0 experiments.

Every data source (synthetic fixture OR real nuScenes+Qwen) produces a
``TokenScene``. Teacher construction and the probe consume only this contract,
so they are agnostic to whether features came from a real frozen Qwen merger or
a synthetic generator.

Frames/conventions match ``geodistill.core.camera``:
- ``points_ego`` (M, 3) in the center-ego frame.
- ``K`` (Ncam, 3, 3); ``cam_to_ego`` (Ncam, 4, 4).
- token arrays length N (concatenated across cameras).
- ``gt_depth`` (N,) is the per-token ground-truth camera-z depth used ONLY by the
  probe for evaluation. On synthetic data it is exact. On real nuScenes it is the
  dense-LiDAR-aggregated depth (documented: probe "GT" is itself LiDAR-derived).
  ``gt_valid`` marks tokens with a defined GT depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class TokenScene:
    # --- token side (frozen-merger output, or synthetic) ---
    features: torch.Tensor          # (N, D)
    token_uv: torch.Tensor          # (N, 2) pixel center
    token_box: torch.Tensor         # (N, 4) (u0,v0,u1,v1)
    token_camera_id: torch.Tensor   # (N,) long
    cam_offsets: torch.Tensor       # (Ncam+1,) cumulative token split per camera

    # --- calibration ---
    K: torch.Tensor                 # (Ncam, 3, 3)
    cam_to_ego: torch.Tensor        # (Ncam, 4, 4)
    image_hw: torch.Tensor          # (Ncam, 2) (H, W) per camera

    # --- LiDAR side (training-only privileged) ---
    points_ego: torch.Tensor        # (M, 3)

    # --- ego ---
    v_ego: float = 0.0
    yaw_rate: float = 0.0

    # --- evaluation-only ground truth (probe target) ---
    gt_depth: Optional[torch.Tensor] = None     # (N,)
    gt_valid: Optional[torch.Tensor] = None      # (N,) bool
    gt_xyz_ego: Optional[torch.Tensor] = None    # (N, 3) optional

    # --- bookkeeping ---
    scene_id: str = "synthetic"
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_cameras(self) -> int:
        return int(self.K.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def tokens_of_camera(self, cam_idx: int) -> slice:
        lo = int(self.cam_offsets[cam_idx])
        hi = int(self.cam_offsets[cam_idx + 1])
        return slice(lo, hi)
