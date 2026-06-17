"""Baseline 3 — SpaceDrive-style (frozen UniDepthV2 + 3D PE, no LiDAR calibration).

This is a thin lazy wrapper around ``third_party/SpaceDrive``. On the local CPU
box (no submodule, no transformers, no UniDepth) every public entry point
raises a clear, actionable ImportError so callers can swap to the synthetic
fixture / open the GPU box.

Wire-up (GPU side):
    1. ``scripts/geodistill/init_third_party.sh`` clones SpaceDrive (+ nested
       UniDepth) and pins the commit in ``THIRD_PARTY_LICENSES.md``.
    2. ``SpaceDriveStyleAdapter`` imports their config + UniDepthV2 forward
       pass and projects metric depth through our own
       :class:`geodistill.geometry.pe_3d.Universal3DPE` so the §4.6 ablation
       rows ("固定 PE scale / learnable zero-init gate / 无 L_keep") stay
       controllable from this side.
    3. The adapter writes per-token ``c_hat`` into the same TokenScene
       contract so all downstream eval / tables run unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["SpaceDriveStyleAdapter", "SPACEDRIVE_HINT"]


SPACEDRIVE_HINT = (
    "third_party/SpaceDrive is not initialized. On the GPU box run "
    "scripts/geodistill/init_third_party.sh; locally use --source synthetic "
    "or any of profiles 1-2, 5-17."
)


@dataclass(frozen=True)
class SpaceDriveStyleConfig:
    spacedrive_config: str = "third_party/SpaceDrive/projects/configs/spacedrive/spacedrive_qwen.py"
    unidepth_ckpt: str = "third_party/SpaceDrive/unidepth/checkpoints/unidepthv2-l.pth"
    image_resolution: tuple[int, int] = (640, 640)


class SpaceDriveStyleAdapter:
    """Lazy wrapper. Instantiate only on the GPU box; raises locally."""

    def __init__(self, cfg: SpaceDriveStyleConfig | None = None):
        self.cfg = cfg or SpaceDriveStyleConfig()
        self._import_or_raise()

    def _import_or_raise(self) -> None:
        try:
            import importlib
            self._sd = importlib.import_module("third_party.SpaceDrive")
            self._unidepth = importlib.import_module("third_party.SpaceDrive.unidepth")
        except Exception as exc:                                                       # noqa: BLE001
            raise ImportError(SPACEDRIVE_HINT) from exc

    def encode(self, sample) -> dict:
        """GPU-side: run UniDepthV2 + 3D PE on a multi-camera sample.

        Returns a dict matching the TokenScene contract so downstream eval /
        tables flow unchanged. The implementation lives on the GPU box.
        """
        raise NotImplementedError(
            "SpaceDriveStyleAdapter.encode requires a real third_party/SpaceDrive "
            "checkpoint and UniDepthV2 weights. Run on the GPU box."
        )
