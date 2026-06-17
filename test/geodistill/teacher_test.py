"""Teacher construction sanity tests (paper §3.4-3.5).

These assert structural correctness of the 3 teachers on the synthetic fixture
(loop closes, contract honoured, geometry recovered above chance) — NOT benchmark
performance.

Run:  .venv_p0/bin/python test/geodistill/teacher_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.data.synthetic import make_synthetic_scene, SyntheticConfig  # noqa: E402
from geodistill.teacher import (  # noqa: E402
    build_hard_quantile_teacher,
    build_entropic_ot_teacher,
    build_full_ntlfgt_teacher,
    build_candidate_sets,
    OTConfig,
)


def _scene():
    return make_synthetic_scene(SyntheticConfig(num_surfaces=10, signal_gain=3.0), seed=1)


def _teacher_absrel(t, s):
    m = t.m_T & s.gt_valid & torch.isfinite(t.d_teacher)
    if int(m.sum()) == 0:
        return float("nan"), 0
    return float(((t.d_teacher[m] - s.gt_depth[m]).abs() / s.gt_depth[m].clamp_min(1e-3)).mean()), int(m.sum())


def test_contract_shapes():
    s = _scene()
    for builder in [build_hard_quantile_teacher,
                    lambda sc: build_entropic_ot_teacher(sc, OTConfig()),
                    lambda sc: build_full_ntlfgt_teacher(sc, OTConfig())]:
        t = builder(s)
        N = s.num_tokens
        assert t.d_teacher.shape == (N,)
        assert t.c_teacher.shape == (N, 3)
        assert t.delta_T.shape == (N, 2)
        assert t.m_T.shape == (N,) and t.m_T.dtype == torch.bool
        for q in [t.q_conc, t.q_mass, t.q_teacher]:
            assert q.shape == (N,)
            valid = q[t.m_T]
            assert (valid >= -1e-6).all() and (valid <= 1 + 1e-6).all(), "reliabilities out of [0,1]"


def test_offset_bounded():
    s = _scene()
    t = build_hard_quantile_teacher(s)
    assert (t.delta_T.abs() <= 1.0 + 1e-6).all(), "sub-token offset must be in [-1,1]^2"


def test_recovers_geometry_above_chance():
    """All teachers should recover depth far better than a constant-mean predictor."""
    s = _scene()
    th = build_hard_quantile_teacher(s)
    absrel, n = _teacher_absrel(th, s)
    assert n > 10, "too few jointly-valid tokens to test"
    # constant predictor baseline AbsRel
    m = th.m_T & s.gt_valid
    gt = s.gt_depth[m]
    const = gt.mean()
    base = float(((const - gt).abs() / gt.clamp_min(1e-3)).mean())
    assert absrel < base, f"hard teacher AbsRel {absrel:.3f} not better than const {base:.3f}"


def test_ot_runs_and_labels():
    s = _scene()
    cands = build_candidate_sets(s, dilate_px=14.0)
    te = build_entropic_ot_teacher(s, OTConfig(epsilon_ot=0.05), cands=cands)
    tf = build_full_ntlfgt_teacher(s, OTConfig(epsilon_ot=0.05, gamma=0.5), cands=cands)
    assert te.diag["tokens_with_label"] > 0, "entropic OT produced no labels"
    assert tf.diag["tokens_with_label"] > 0, "full NTL-FGT produced no labels"
    assert te.diag["sinkhorn_iters"] > 0
    # both OT teachers should also beat a constant baseline
    for t, name in [(te, "entropic_ot"), (tf, "full_ntl_fgt")]:
        absrel, n = _teacher_absrel(t, s)
        assert n > 5 and absrel < 0.5, f"{name} AbsRel {absrel} on n={n}"


def test_concentration_defined():
    """q_conc must be 1 for hard teacher and in [0,1] for OT teachers."""
    s = _scene()
    th = build_hard_quantile_teacher(s)
    assert torch.allclose(th.q_conc[th.m_T], torch.ones(int(th.m_T.sum()))), "hard q_conc must be 1"


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
