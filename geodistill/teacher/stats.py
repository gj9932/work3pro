"""Per-teacher statistics (paper item 2 / Table 7 + Table 8 diagnostics).

Given a ``TeacherOutput`` (and the scene that produced it), compute every statistic
the user asked for:
  - valid token ratio
  - depth distribution (min/mean/median/quantiles/max)
  - 0-10m / 10-30m / 30m+ token counts
  - q_conc / q_mass / q_teacher distributions
  - teacher construction time / batch        (from diag)
  - extra memory                              (measured around build, passed in)
  - candidate anchors / token                 (from diag)
  - Sinkhorn iterations                        (from diag)
  - sparse edges / token                       (relation edge sampler, item built here)

All distributions are summarized as (mean, std, p10, p50, p90, min, max) so JSONL
rows stay flat and comparable across teachers.
"""

from __future__ import annotations

import torch

from geodistill.data.contract import TokenScene
from geodistill.teacher.base import TeacherOutput


def _dist(x: torch.Tensor) -> dict:
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {k: float("nan") for k in ["mean", "std", "p10", "p50", "p90", "min", "max", "n"]} | {"n": 0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p10": float(torch.quantile(x, 0.10)),
        "p50": float(torch.quantile(x, 0.50)),
        "p90": float(torch.quantile(x, 0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(x.numel()),
    }


def depth_bins(depth: torch.Tensor, valid: torch.Tensor) -> dict:
    d = depth[valid & torch.isfinite(depth)]
    return {
        "count_0_10m": int(((d >= 0) & (d < 10)).sum()),
        "count_10_30m": int(((d >= 10) & (d < 30)).sum()),
        "count_30m_plus": int((d >= 30).sum()),
    }


def teacher_statistics(
    scene: TokenScene,
    teacher: TeacherOutput,
    extra_memory_bytes: int | None = None,
    edges_per_token: float | None = None,
) -> dict:
    N = scene.num_tokens
    m = teacher.m_T
    valid_ratio = float(m.float().mean())

    stats = {
        "teacher": teacher.name,
        "num_tokens": N,
        "valid_token_ratio": valid_ratio,
        "tokens_valid": int(m.sum()),
        "depth_dist": _dist(teacher.d_teacher[m]),
        "depth_bins": depth_bins(teacher.d_teacher, m),
        "q_conc_dist": _dist(teacher.q_conc[m]),
        "q_mass_dist": _dist(teacher.q_mass[m]),
        "q_teacher_dist": _dist(teacher.q_teacher[m]),
        # from solver diagnostics
        "build_time_s": teacher.diag.get("build_time_s", float("nan")),
        "candidate_anchors_per_token": teacher.diag.get("mean_candidates_per_labeled_token", float("nan")),
        "num_anchors": teacher.diag.get("num_anchors", 0),
        "sinkhorn_iterations": teacher.diag.get("sinkhorn_iters", 0),
        "transported_mass": teacher.diag.get("transported_mass", float("nan")),
    }
    stats["extra_memory_mb"] = (extra_memory_bytes / 1e6) if extra_memory_bytes is not None else float("nan")
    stats["sparse_edges_per_token"] = edges_per_token if edges_per_token is not None else float("nan")
    return stats
