"""GeoDistill-VLM P0 core math: R²AC, risk field, camera geometry, Sinkhorn OT."""

from .r2ac import (
    companding_mu,
    r2ac_forward,
    r2ac_inverse,
    r2ac_dForward_dd,
    sensitivity_ratio,
)
from .risk_field import RiskFieldConfig, stopping_distance, soft_risk, oracle_allocation
from .camera import (
    invert_se3,
    project_ego_to_cam,
    unproject_cam_to_ego,
    ray_dirs_ego,
    assign_points_to_tokens,
)
from .sinkhorn import sinkhorn_balanced, sinkhorn_partial

__all__ = [
    "companding_mu",
    "r2ac_forward",
    "r2ac_inverse",
    "r2ac_dForward_dd",
    "sensitivity_ratio",
    "RiskFieldConfig",
    "stopping_distance",
    "soft_risk",
    "oracle_allocation",
    "invert_se3",
    "project_ego_to_cam",
    "unproject_cam_to_ego",
    "ray_dirs_ego",
    "assign_points_to_tokens",
    "sinkhorn_balanced",
    "sinkhorn_partial",
]
