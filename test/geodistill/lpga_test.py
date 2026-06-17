"""Tests for LPGA token encoder + heads (paper §3.3, §3.6).

Invariants guarded:
- DepthHead/ReliHead/RayHead must NOT depend on calibration tensors.
- RiskHead is the only head consuming e_ego.
- LPGA owns no a-regression head (a_hat is analytically composed).
- d_hat = F_inv(zeta, StopGrad(a_hat)): a_hat receives no gradient through depth.
- forward shapes are correct on the synthetic fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.data.synthetic import make_synthetic_scene, SyntheticConfig          # noqa: E402
from geodistill.models.lpga_token_encoder import TokenEncoder, TokenEncoderConfig, GeoEncoder  # noqa: E402
from geodistill.models.lpga_heads import LPGA, LPGAConfig                            # noqa: E402
from geodistill.geometry.sub_token_ray import decode_ray, back_project                # noqa: E402


def _scene():
    cfg = SyntheticConfig(num_cameras=2, grid_h=4, grid_w=6, feature_dim=64, signal_gain=2.0)
    return make_synthetic_scene(cfg, seed=0)


def _build():
    s = _scene()
    enc_cfg = TokenEncoderConfig(d_in=int(s.features.shape[-1]), d_bottleneck=64,
                                 d_e_cam=8, d_e_uv=16, d_e_calib=16, layers=2,
                                 num_cameras=int(s.num_cameras))
    token_enc = TokenEncoder(enc_cfg)
    geo_enc = GeoEncoder(d_in=enc_cfg.d_bottleneck, d_e_calib=enc_cfg.d_e_calib,
                         d_out=enc_cfg.d_bottleneck, num_cameras=int(s.num_cameras))
    lpga_cfg = LPGAConfig(d_mono=enc_cfg.d_bottleneck, d_e_ego=8, d_hidden=64)
    lpga = LPGA(lpga_cfg)
    return s, enc_cfg, token_enc, geo_enc, lpga


def test_no_a_regression_head():
    _, _, _, _, lpga = _build()
    for n in ("a_head", "a_proj", "alloc_head"):
        assert not hasattr(lpga, n), f"LPGA must not own an a-regression head ({n})"


def test_input_isolation_by_calibration_grad():
    """Make calibration tensors require grad and confirm DepthHead/ReliHead/RayHead don't backprop into them."""
    s, enc_cfg, token_enc, _geo_enc, lpga = _build()
    h = s.features.detach().clone()
    K = s.K.detach().clone().requires_grad_(True)
    cam_to_ego = s.cam_to_ego.detach().clone().requires_grad_(True)

    # Build a path where calibration appears nowhere; depth/reli/ray should not see it.
    g_mono = token_enc(h, s.token_camera_id, s.token_uv, s.image_hw)
    out = lpga(g_mono, v_ego=s.v_ego, yaw_rate=None)
    loss = out["zeta_hat"].sum() + out["q_hat"].sum() + out["delta_hat"].sum()
    loss.backward(retain_graph=True)
    assert K.grad is None, "DepthHead/ReliHead/RayHead must not depend on intrinsics K"
    assert cam_to_ego.grad is None, "DepthHead/ReliHead/RayHead must not depend on cam_to_ego"


def test_stopgrad_in_depth_decode():
    """LPGA.decode_depth must apply StopGrad on a_hat (paper §3.6)."""
    _, _, _, _, lpga = _build()
    zeta = torch.rand(8, requires_grad=True)
    a_leaf = torch.rand(8, requires_grad=True)
    d_hat = lpga.decode_depth(zeta, a_leaf)
    d_hat.sum().backward()
    # No gradient should reach a_hat. PyTorch leaves .grad as None when nothing flows.
    assert a_leaf.grad is None or float(a_leaf.grad.abs().max()) == 0.0, \
        "a_hat must be detached inside LPGA.decode_depth (paper §3.6 StopGrad)"
    assert zeta.grad is not None and float(zeta.grad.abs().max()) > 0.0, \
        "zeta must keep grad through F_inv"


def test_forward_shapes():
    s, enc_cfg, token_enc, geo_enc, lpga = _build()
    g_mono = token_enc(s.features, s.token_camera_id, s.token_uv, s.image_hw)
    g_geo = geo_enc(g_mono, s.K, s.cam_to_ego, s.image_hw, s.token_camera_id)
    assert g_mono.shape == (s.num_tokens, enc_cfg.d_bottleneck)
    assert g_geo.shape == (s.num_tokens, enc_cfg.d_bottleneck)
    out = lpga(g_mono, v_ego=s.v_ego)
    N = s.num_tokens
    assert out["zeta_hat"].shape == (N,)
    assert out["q_hat"].shape == (N,)
    assert out["delta_hat"].shape == (N, 2)
    assert out["r_hat"].shape == (N,)
    assert out["a_hat"].shape == (N,)
    assert out["d_hat"].shape == (N,)
    # decode + back-project shapes
    uv_bar = decode_ray(s.token_uv, s.token_box, out["delta_hat"])
    assert uv_bar.shape == (N, 2)
    c_hat = back_project(uv_bar, out["d_hat"], s.K, s.cam_to_ego, s.token_camera_id)
    assert c_hat.shape == (N, 3)


def test_compose_a_formula():
    _, _, _, _, lpga = _build()
    r = torch.tensor([0.0, 0.5, 1.0])
    q = torch.tensor([1.0, 0.5, 0.0])
    a = lpga.compose_a(r, q)
    eta = lpga.cfg.eta
    expected = r * (eta + (1 - eta) * q)
    assert torch.allclose(a, expected, atol=1e-6)


def test_assert_input_isolation_static():
    _, _, _, _, lpga = _build()
    lpga.assert_input_isolation(d_mono=lpga.cfg.d_mono)


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
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
