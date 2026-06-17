"""Unit tests for R²AC core math (paper §3.6).

Run:  .venv_p0/bin/python -m pytest test/geodistill/r2ac_test.py -q
  or  .venv_p0/bin/python test/geodistill/r2ac_test.py   (falls back to manual runner)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.core.r2ac import (  # noqa: E402
    r2ac_forward,
    r2ac_inverse,
    r2ac_dForward_dd,
    sensitivity_ratio,
)

D_MAX = 80.0
BETA = torch.log(torch.tensor(16.0)).item()  # paper default: max ratio 16


def test_roundtrip_inverse():
    """F_inv(F(d; a); a) == d across a grid of a and d."""
    d = torch.linspace(0.0, D_MAX, 50)
    for a_val in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        a = torch.full_like(d, a_val)
        z = r2ac_forward(d, a, D_MAX, BETA)
        d_rec = r2ac_inverse(z, a, D_MAX, BETA)
        assert torch.allclose(d, d_rec, atol=1e-3), f"roundtrip failed a={a_val}, max err {(d-d_rec).abs().max()}"


def test_limit_continuity():
    """As a -> 0, F -> d/D_max and F_inv -> D_max*z (continuous, no NaN)."""
    d = torch.linspace(0.0, D_MAX, 50)
    z_lin = d / D_MAX
    for a_val in [1e-9, 1e-6, 1e-4, 1e-3]:
        a = torch.full_like(d, a_val)
        z = r2ac_forward(d, a, D_MAX, BETA)
        assert torch.isfinite(z).all(), f"non-finite F at a={a_val}"
        assert torch.allclose(z, z_lin, atol=1e-2), f"limit mismatch a={a_val}"


def test_strict_monotonic():
    """F is strictly increasing in d (analytic dF/dd > 0, and sampled diffs > 0)."""
    d = torch.linspace(0.0, D_MAX, 200)
    for a_val in [0.0, 0.3, 0.7, 1.0]:
        a = torch.full_like(d, a_val)
        grad = r2ac_dForward_dd(d, a, D_MAX, BETA)
        assert (grad > 0).all(), f"non-positive derivative a={a_val}"
        z = r2ac_forward(d, a, D_MAX, BETA)
        assert (z[1:] - z[:-1] > 0).all(), f"non-monotone sampled F a={a_val}"


def test_sensitivity_ratio():
    """(dF/dd|0)/(dF/dd|Dmax) == exp(beta*a) == 1+mu(a), matching closed form."""
    for a_val in [0.0, 0.25, 0.5, 1.0]:
        a = torch.tensor(a_val)
        d0 = torch.tensor(0.0)
        dN = torch.tensor(D_MAX)
        g0 = r2ac_dForward_dd(d0, a, D_MAX, BETA)
        gN = r2ac_dForward_dd(dN, a, D_MAX, BETA)
        ratio = g0 / gN
        expected = sensitivity_ratio(a, BETA)
        assert torch.allclose(ratio, expected, rtol=1e-3), f"ratio {ratio} != {expected} a={a_val}"
    # a=1 must give exactly the configured max ratio (16).
    assert torch.allclose(sensitivity_ratio(torch.tensor(1.0), BETA), torch.tensor(16.0), rtol=1e-4)


def test_endpoints():
    """F(0)=0 and F(D_max)=1 for all a."""
    for a_val in [0.0, 0.5, 1.0]:
        a = torch.tensor(a_val)
        assert torch.allclose(r2ac_forward(torch.tensor(0.0), a, D_MAX, BETA), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(r2ac_forward(torch.tensor(D_MAX), a, D_MAX, BETA), torch.tensor(1.0), atol=1e-5)


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
