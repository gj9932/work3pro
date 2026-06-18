"""Baseline 18 — LLaVA-1.5-7B + CLIP ViT-L/14 (paper §4.4).

Vision Encoder selection ablation: same training schedule as ``full``, but the
backbone is LLaVA-1.5-7B with CLIP ViT-L/14 instead of Qwen2.5-VL-7B.

We re-use SpaceDrive's LLaVA wrapper (pinned commit) so the comparison stays
faithful to the published baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spacedrive_style import SPACEDRIVE_HINT


__all__ = ["LLaVA15Adapter"]


@dataclass(frozen=True)
class LLaVA15Config:
    spacedrive_llava_config: str = "third_party/SpaceDrive/projects/configs/spacedrive/spacedrive_llava.py"


class LLaVA15Adapter:
    def __init__(self, cfg: LLaVA15Config | None = None):
        self.cfg = cfg or LLaVA15Config()
        try:
            import importlib
            self._sd_llava = importlib.import_module("third_party.SpaceDrive.projects.spacedrive.spacedrive_llava")
        except Exception as exc:                                                       # noqa: BLE001
            raise ImportError(SPACEDRIVE_HINT) from exc

    def encode(self, sample) -> dict:
        raise NotImplementedError(
            f"LLaVA15Adapter.encode requires the GPU box. {SPACEDRIVE_HINT}"
        )
