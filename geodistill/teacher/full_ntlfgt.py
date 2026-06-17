"""Teacher 3: Full NTL-FGT (FGW-inspired feature + fixed-structure transport, paper §3.5).

Adds the FGW fixed-structure term (gamma>0) on top of the feature cost:
token-token structure (image position, ego-ray angle, camera separation) aligned
to LiDAR-LiDAR structure (3D distance, ego-ray angle). The fixed structure
dominates the coupling, disambiguating repeated-texture / cross-view / sparse-far
candidate sets that feature-only OT cannot resolve.

The learned typed-relation predictor term (kappa_pred) and the F0/F1/F2
stop-gradient schedule from the paper are training-time refinements that need the
learned relation heads; in this P0 teacher-construction harness we evaluate the
fixed-structure-dominant coupling (the paper's main coupling driver). The learned
term is left as a documented extension hook, not silently faked.
"""

from __future__ import annotations

from dataclasses import replace

from geodistill.data.contract import TokenScene
from .base import CandidateSets
from .ot_common import OTConfig, build_ot_teacher, TeacherOutput


def build_full_ntlfgt_teacher(
    scene: TokenScene,
    cfg: OTConfig | None = None,
    cands: CandidateSets | None = None,
) -> TeacherOutput:
    cfg = cfg or OTConfig(gamma=0.5)
    if cfg.gamma <= 0:
        cfg = replace(cfg, gamma=0.5)   # full FGT must use structure term
    return build_ot_teacher(scene, cfg, name="full_ntl_fgt", cands=cands)
