"""Teacher constructions for GeoDistill-VLM P0 (paper §3.4-3.5).

Three faithful variants with a shared ``TeacherOutput`` contract:
  - hard_quantile   : hard projection + 10% quantile (baseline)
  - entropic_ot     : entropic OT, gamma=0 (feature-only soft coupling)
  - full_ntl_fgt    : FGW-inspired feature + fixed-structure coupling
"""

from .base import TeacherOutput, CandidateSets, build_candidate_sets, empty_teacher
from .hard_quantile import build_hard_quantile_teacher
from .entropic_ot import build_entropic_ot_teacher
from .full_ntlfgt import build_full_ntlfgt_teacher
from .ot_common import OTConfig, build_ot_teacher

# registry for CLIs: name -> builder(scene, **kwargs)
TEACHER_BUILDERS = {
    "hard_quantile": build_hard_quantile_teacher,
    "entropic_ot": build_entropic_ot_teacher,
    "full_ntl_fgt": build_full_ntlfgt_teacher,
}

__all__ = [
    "TeacherOutput",
    "CandidateSets",
    "build_candidate_sets",
    "empty_teacher",
    "OTConfig",
    "build_ot_teacher",
    "build_hard_quantile_teacher",
    "build_entropic_ot_teacher",
    "build_full_ntlfgt_teacher",
    "TEACHER_BUILDERS",
]
