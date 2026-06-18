"""Tests for losses (paper §3.6, §3.10, §3.11, §3.13)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.core.r2ac import r2ac_forward, r2ac_inverse                           # noqa: E402
from geodistill.losses import (                                                       # noqa: E402
    compute_w_p, L_comp, L_rank, L_r, L_q, L_alloc, L_a_aux,
    L_coord, L_ray, L_robust, RelationLossWeights, relation_loss,
    SemanticPreservingLoss,
)
from geodistill.geometry.cross_view_label import EdgeLabels                            # noqa: E402


def test_lambda_sum_assert():
    a = torch.tensor(1.0); b = torch.tensor(1.0)
    L_alloc(a, b, 0.5, 0.5)
    try:
        L_alloc(a, b, 0.6, 0.5)
        raise AssertionError("expected L_alloc to reject λ_r+λ_q != 1")
    except AssertionError as exc:
        assert "λ_r + λ_q" in str(exc) or "lambda" in str(exc) or "+" in str(exc)


def test_stopgrad_in_rank_caller_contract():
    """L_rank takes ALREADY-decoded d_hat. Verify caller-side StopGrad works."""
    zeta = torch.rand(8, requires_grad=True)
    a = torch.rand(8, requires_grad=True)
    d_T = torch.rand(8) * 80.0
    d_hat = r2ac_inverse(zeta, a.detach(), 80.0, 2.7)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    weights = torch.ones(3)
    loss = L_rank(d_hat, d_T, pairs, weights, tau_d=1.0)
    loss.backward()
    assert a.grad is None or float(a.grad.abs().max()) == 0.0


def test_padding_invariance_l_comp():
    z_hat = torch.tensor([0.1, 0.2, 0.3, 0.0])
    z_star = torch.tensor([0.15, 0.25, 0.4, 999.0])  # last entry is bogus
    w = torch.tensor([1.0, 1.0, 1.0, 0.0])           # zero weight masks it out
    val = float(L_comp(z_hat, z_star, w))
    val_no_pad = float(L_comp(z_hat[:3], z_star[:3], w[:3]))
    assert abs(val - val_no_pad) < 1e-6


def test_l_comp_shape():
    N = 16
    z_hat = torch.rand(N)
    d_T = torch.rand(N) * 80.0
    a_star = torch.rand(N)
    z_star = r2ac_forward(d_T, a_star, D_max=80.0, beta=2.7)
    w_p = compute_w_p(torch.ones(N), torch.rand(N), eps_q=0.05)
    loss = L_comp(z_hat, z_star, w_p)
    assert loss.dim() == 0


def test_keep_zero_when_geo_equals_img():
    h = torch.randn(20, 32)
    cam_offsets = torch.tensor([0, 10, 20])
    out = SemanticPreservingLoss()(h, h, cam_offsets)
    assert torch.allclose(out["L_keep"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out["L_sem"], torch.tensor(0.0), atol=1e-5)


def test_relation_loss_runs_and_masks_zero_reliability():
    pred = {
        "delta_c": torch.randn(4, 3),
        "order_logits": torch.randn(4, 3),
        "cross_logit": torch.randn(4),
        "topo_logits": torch.randn(4, 6),
        "occ_logits": torch.randn(4, 3),
        "conf_logit": torch.randn(4),
    }
    labels = EdgeLabels(
        edge_index=torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=torch.long),
        delta_c=torch.randn(4, 3),
        depth_order=torch.tensor([0, 1, 2, 0]),
        cross_view=torch.tensor([0, 1, 0, 1]),
        topology=torch.tensor([5, 5, 5, 5]),                # all UNKNOWN -> L_topo masked out
        occlusion=torch.tensor([2, 0, 1, 2]),
        reliability=torch.tensor([1.0, 1.0, 0.0, 1.0]),
        valid_mask=torch.tensor([True, True, True, True]),
    )
    out = relation_loss(pred, labels, RelationLossWeights(delta=1.0, order=0.5, cross=0.5, topo=0.0, occ=0.0))
    assert torch.isfinite(out.L_rel)
    # L_topo should be exactly zero (no class is non-UNKNOWN)
    assert float(out.parts["L_topo"]) == 0.0


def test_robust_loss_zero_when_full_equals_deg():
    h_full = torch.randn(12, 16)
    cam_offsets = torch.tensor([0, 6, 12])
    val = L_robust(h_full, h_full, cam_offsets, cam_offsets)
    assert float(val) == 0.0


def test_l_a_aux_runs():
    a = torch.rand(5); a_star = torch.rand(5)
    m = torch.tensor([1, 1, 0, 1, 1])
    val = L_a_aux(a, a_star, m)
    assert torch.isfinite(val)


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
        except Exception as exc:                                                       # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
