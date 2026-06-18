"""Bench2Drive closed-loop dataset (paper §4.5).

The actual simulator + scenarios live in ``third_party/SpaceDrive/Bench2Drive``;
this module is a lazy wrapper that surfaces the same TokenScene-style contract
expected by ``eval_bench2drive``. On CPU it raises immediately so callers do
not silently fall back to a synthetic stand-in.
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["Bench2DriveDataset", "BENCH2DRIVE_HINT"]


BENCH2DRIVE_HINT = (
    "Bench2Drive requires the simulator + SpaceDrive submodule. "
    "Run scripts/geodistill/init_third_party.sh on the GPU box."
)


class Bench2DriveDataset:
    def __init__(self, root: str | Path, allow_b2d_vl: bool = False) -> None:
        self.root = Path(root)
        self.allow_b2d_vl = allow_b2d_vl
        try:
            import importlib
            self._sim = importlib.import_module("third_party.SpaceDrive.bench2drive")
        except Exception as exc:                                                       # noqa: BLE001
            raise ImportError(BENCH2DRIVE_HINT) from exc

    def __iter__(self):
        raise NotImplementedError(BENCH2DRIVE_HINT)
