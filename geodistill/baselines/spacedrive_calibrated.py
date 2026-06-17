"""Baseline 4 — LiDAR-calibrated SpaceDrive (paper §4.4).

UniDepthV2 + a small calibration head trained with the SAME LiDAR depth + coord
losses as LPGA. This is the strongest fair baseline for the precision-efficiency
Pareto: it shares the LiDAR label budget with us but keeps the external depth
backbone.

The calibration head bottleneck is fixed at 512 (same as LPGA) so any
parameter-count difference comes only from UniDepthV2 itself.

Like ``spacedrive_style.py`` this is a lazy wrapper; the actual training loop
lives on the GPU box.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spacedrive_style import SpaceDriveStyleAdapter, SPACEDRIVE_HINT


__all__ = ["SpaceDriveCalibratedAdapter", "CalibrationHeadConfig"]


@dataclass(frozen=True)
class CalibrationHeadConfig:
    bottleneck: int = 512                   # Equal to LPGA bottleneck (paper §4.4)
    lambda_comp: float = 1.0
    lambda_coord: float = 1.0


class SpaceDriveCalibratedAdapter(SpaceDriveStyleAdapter):
    def __init__(self, cfg=None, head_cfg: CalibrationHeadConfig | None = None):
        super().__init__(cfg=cfg)
        self.head_cfg = head_cfg or CalibrationHeadConfig()

    def train_calibration(self, batches) -> dict:
        """GPU-side: train a 512-bottleneck calibration head over UniDepthV2 outputs
        with LPGA-equivalent ``L_comp + L_coord`` losses."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.train_calibration requires the GPU box. {SPACEDRIVE_HINT}"
        )
