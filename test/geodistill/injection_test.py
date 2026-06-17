"""Tests for 3D PE + GeometryInjector + L_keep / L_sem (paper §3.8, §3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.geometry.pe_3d import Universal3DPE                                  # noqa: E402
from geodistill.models.geometry_injection import (                                   # noqa: E402
    GeometryInjector, GeometryInjectorConfig, SemanticPreservingLoss,
    linear_cka, pool_by_camera,
)


def test_pe_dim_sum_matches_qwen_hidden():
    pe = Universal3DPE(d_x=1194, d_y=1194, d_z=1196)
    assert pe.out_dim == 3584, f"expected 3584 (Qwen2.5-VL hidden), got {pe.out_dim}"
    c = torch.randn(7, 3)
    out = pe(c)
    assert out.shape == (7, 3584)


def test_pe_finite_for_extreme_coords():
    pe = Universal3DPE(d_x=64, d_y=64, d_z=64)
    c = torch.tensor([[0.0, 0.0, 0.0], [80.0, -40.0, 5.0], [-1e3, 1e3, 1e2]])
    out = pe(c)
    assert torch.isfinite(out).all()
    # at c=0 sin parts are 0, cos parts are 1 → out has the constant pattern repeated
    assert torch.allclose(out[0, 0:32], torch.zeros(32))      # sin block (first half)
    assert torch.allclose(out[0, 32:64], torch.ones(32))      # cos block


def test_step0_identity_bit_exact():
    """β_pe = β_rel = 0 -> α = 0 -> H_geo == H_img exactly."""
    cfg = GeometryInjectorConfig(hidden_size=64, d_rel=32)
    inj = GeometryInjector(cfg)
    h_img = torch.randn(20, 64)
    pe = torch.randn(20, 64)
    z = torch.randn(20, 32)
    h_geo = inj(h_img, pe, z)
    assert torch.equal(h_geo, h_img), "step-0 must yield H_geo == H_img bit-exactly"


def test_gates_have_grad_at_step1():
    """W_up has small non-zero std => one backward gives both β params non-zero gradient."""
    cfg = GeometryInjectorConfig(hidden_size=16, d_rel=8, sigma_up=1e-4)
    inj = GeometryInjector(cfg)
    h_img = torch.randn(12, 16)
    pe = torch.randn(12, 16)
    z = torch.randn(12, 8)
    h_geo = inj(h_img, pe, z)
    h_geo.pow(2).sum().backward()
    assert inj.beta_pe.grad is not None and inj.beta_pe.grad.abs() > 0
    assert inj.beta_rel.grad is not None and inj.beta_rel.grad.abs() > 0


def test_w_up_zero_kills_relation_gate():
    """If W_up were exactly zero, β_rel.grad would be zero — confirms the design choice."""
    cfg = GeometryInjectorConfig(hidden_size=16, d_rel=8, sigma_up=0.0)
    inj = GeometryInjector(cfg)
    with torch.no_grad():
        inj.W_up.weight.zero_()
    h_img = torch.randn(12, 16)
    pe = torch.randn(12, 16)
    z = torch.randn(12, 8)
    h_geo = inj(h_img, pe, z)
    h_geo.pow(2).sum().backward()
    assert inj.beta_rel.grad is None or float(inj.beta_rel.grad.abs()) == 0.0


def test_cka_self_is_one():
    X = torch.randn(40, 16)
    cka = linear_cka(X, X)
    assert torch.allclose(cka, torch.tensor(1.0), atol=1e-5)


def test_cka_in_unit_interval():
    X = torch.randn(40, 16)
    Y = torch.randn(40, 16)
    cka = float(linear_cka(X, Y))
    assert 0.0 - 1e-5 <= cka <= 1.0 + 1e-5


def test_pool_uses_cam_offsets():
    feats = torch.tensor([
        [1.0, 1.0],
        [3.0, 3.0],
        [10.0, 10.0],
    ])
    cam_offsets = torch.tensor([0, 2, 3])
    pooled = pool_by_camera(feats, cam_offsets)
    # cam0 mean = (2,2), cam1 mean = (10,10), then mean across cams = (6,6)
    assert torch.allclose(pooled, torch.tensor([6.0, 6.0]))


def test_keep_zero_at_step0():
    cfg = GeometryInjectorConfig(hidden_size=8, d_rel=4)
    inj = GeometryInjector(cfg)
    h_img = torch.randn(6, 8)
    pe = torch.randn(6, 8)
    z = torch.randn(6, 4)
    h_geo = inj(h_img, pe, z)
    cam_offsets = torch.tensor([0, 3, 6])
    loss = SemanticPreservingLoss()(h_geo, h_img, cam_offsets)
    assert torch.allclose(loss["L_keep"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(loss["L_sem"], torch.tensor(0.0), atol=1e-5)


def _run_all():
    failures = 0
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
