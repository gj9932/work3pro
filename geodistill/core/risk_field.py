"""Risk field + factorized allocation (paper §3.6).

    d_safe = v_ego * t_react + v_ego^2 / (2 * a_brake) + d_margin
    r_p    = sigmoid((d_safe - x_ego)/tau_r) * sigmoid(x_ego/tau_front) * exp(-y_ego^2/(2 w^2))
    a*_p   = r_p * [eta + (1 - eta) * q_teacher]

``x_ego`` is forward distance, ``y_ego`` lateral distance of the token's teacher
representative in ego frame. All quantities are per-token tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["RiskFieldConfig", "stopping_distance", "soft_risk", "oracle_allocation"]


@dataclass(frozen=True)
class RiskFieldConfig:
    t_react: float = 1.0       # s
    a_brake: float = 4.0       # m/s^2
    d_margin: float = 2.0      # m
    tau_r: float = 2.0         # m, softness near d_safe
    tau_front: float = 1.0     # m, softness near ego front
    w_corridor: float = 2.0    # m, corridor lateral half-width
    eta: float = 0.5           # minimum risk allocation


def stopping_distance(v_ego: torch.Tensor, cfg: RiskFieldConfig) -> torch.Tensor:
    v = torch.as_tensor(v_ego, dtype=torch.float32).clamp_min(0.0)
    return v * cfg.t_react + v * v / (2.0 * cfg.a_brake) + cfg.d_margin


def soft_risk(x_ego: torch.Tensor, y_ego: torch.Tensor, v_ego: torch.Tensor, cfg: RiskFieldConfig) -> torch.Tensor:
    """Per-token soft driving risk r_p ∈ [0, 1]."""
    x_ego = torch.as_tensor(x_ego, dtype=torch.float32)
    y_ego = torch.as_tensor(y_ego, dtype=torch.float32)
    d_safe = stopping_distance(v_ego, cfg)
    within = torch.sigmoid((d_safe - x_ego) / cfg.tau_r)
    front = torch.sigmoid(x_ego / cfg.tau_front)
    corridor = torch.exp(-(y_ego ** 2) / (2.0 * cfg.w_corridor ** 2))
    return within * front * corridor


def oracle_allocation(r_p: torch.Tensor, q_teacher: torch.Tensor, eta: float) -> torch.Tensor:
    """a*_p = r_p * [eta + (1 - eta) * q_teacher], in [0, 1]."""
    r_p = torch.as_tensor(r_p, dtype=torch.float32)
    q_teacher = torch.as_tensor(q_teacher, dtype=torch.float32)
    return r_p * (eta + (1.0 - eta) * q_teacher)
