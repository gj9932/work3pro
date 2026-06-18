"""GeoDistill loss functions (paper §3.5–3.13).

Public surface (one function/class per file, importable directly):

    L_comp / L_rank / L_r / L_q / L_alloc          (r2ac_loss.py)
    L_coord / L_ray                                (coord_loss.py)
    L_delta / L_order / L_cross / L_topo / L_occ   (relation_loss.py)
    SemanticPreservingLoss (L_keep, L_sem)         (re-exported from models.geometry_injection)
    L_robust                                       (robust_loss.py)
"""

from .r2ac_loss import (
    compute_w_p, L_comp, L_rank, L_r, L_q, L_alloc, L_a_aux,
)
from .coord_loss import L_coord, L_ray
from .relation_loss import RelationLossWeights, RelationLossOutput, relation_loss
from .robust_loss import L_robust
from geodistill.models.geometry_injection import SemanticPreservingLoss

__all__ = [
    "compute_w_p", "L_comp", "L_rank", "L_r", "L_q", "L_alloc", "L_a_aux",
    "L_coord", "L_ray",
    "RelationLossWeights", "RelationLossOutput", "relation_loss",
    "L_robust",
    "SemanticPreservingLoss",
]
