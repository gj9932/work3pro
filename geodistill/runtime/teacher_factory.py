"""Build a teacher builder from yaml config.

Encapsulates the small switch over teacher kinds so the trainers don't repeat
the OTConfig / hard-quantile dispatch.
"""

from __future__ import annotations

from typing import Callable

from geodistill.data.contract import TokenScene
from geodistill.teacher import (
    OTConfig, TeacherOutput,
    build_hard_quantile_teacher, build_entropic_ot_teacher, build_full_ntlfgt_teacher,
)


__all__ = ["build_teacher_fn", "TEACHER_KINDS"]


TEACHER_KINDS = ("hard_quantile", "entropic_ot", "full_ntl_fgt")


def build_teacher_fn(yaml_cfg: dict) -> Callable[[TokenScene], TeacherOutput]:
    teacher_cfg = yaml_cfg.get("teacher", {})
    kind = str(teacher_cfg.get("kind", "hard_quantile"))
    if kind == "hard_quantile":
        q = float(yaml_cfg.get("lidar_label", {}).get("foreground_quantile", 0.1))
        tau_fg = float(yaml_cfg.get("lidar_label", {}).get("tau_fg", 1.0))
        n0 = float(yaml_cfg.get("lidar_label", {}).get("n_0", 3.0))
        tau_q = float(yaml_cfg.get("lidar_label", {}).get("tau_q", 0.1))
        return lambda scene: build_hard_quantile_teacher(scene, q=q, tau_fg=tau_fg, n0=n0, tau_q=tau_q)
    if kind == "entropic_ot":
        ot = OTConfig(
            epsilon_ot=float(teacher_cfg.get("epsilon_ot", 0.05)),
            gamma=0.0,
            m_tr_frac=float(teacher_cfg.get("m_tr_frac", 0.9)),
            n_iters_sinkhorn=int(teacher_cfg.get("n_iters_sinkhorn", 150)),
            voxel_size=float(teacher_cfg.get("voxel_size", 0.5)),
            max_anchors=int(teacher_cfg.get("max_anchors", 400)),
        )
        return lambda scene: build_entropic_ot_teacher(scene, ot)
    if kind == "full_ntl_fgt":
        ot = OTConfig(
            epsilon_ot=float(teacher_cfg.get("epsilon_ot", 0.05)),
            gamma=float(teacher_cfg.get("gamma", 0.5)),
            m_tr_frac=float(teacher_cfg.get("m_tr_frac", 0.9)),
            n_iters_sinkhorn=int(teacher_cfg.get("n_iters_sinkhorn", 150)),
            n_outer_fgw=int(teacher_cfg.get("n_outer_fgw", 5)),
            voxel_size=float(teacher_cfg.get("voxel_size", 0.5)),
            max_anchors=int(teacher_cfg.get("max_anchors", 400)),
        )
        return lambda scene: build_full_ntlfgt_teacher(scene, ot)
    raise ValueError(f"unknown teacher kind {kind!r}; choose from {TEACHER_KINDS}")
