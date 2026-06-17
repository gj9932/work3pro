"""Teacher 2: Entropic OT with gamma=0 (feature-only coupling, paper §3.5 ablation row).

Plain entropic optimal transport between token measure and LiDAR anchor measure,
using only the feature-level cost M_{p,i} (projection / ray / visibility). No
Gromov-Wasserstein structure term. This isolates "does soft transport beat hard
projection?" before the structure term is added in full NTL-FGT.
"""

from __future__ import annotations

from dataclasses import replace

from geodistill.data.contract import TokenScene
from .base import CandidateSets
from .ot_common import OTConfig, build_ot_teacher, TeacherOutput


def build_entropic_ot_teacher(
    scene: TokenScene,
    cfg: OTConfig | None = None,
    cands: CandidateSets | None = None,
) -> TeacherOutput:
    cfg = cfg or OTConfig()
    cfg = replace(cfg, gamma=0.0)   # enforce feature-only
    return build_ot_teacher(scene, cfg, name="entropic_ot_gamma0", cands=cands)
