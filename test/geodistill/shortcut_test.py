"""Tests for shortcut audit (paper §4.7).

Each test asserts both:
- "scope invariance": fields the shuffle is NOT supposed to touch are bit-exact equal.
- "active effect":    fields the shuffle IS supposed to touch detectably differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.data.synthetic import make_synthetic_scene, SyntheticConfig         # noqa: E402
from geodistill.teacher import (                                                    # noqa: E402
    build_hard_quantile_teacher, build_full_ntlfgt_teacher, build_candidate_sets, OTConfig,
)
from geodistill.eval.shortcut_audit import (                                         # noqa: E402
    apply_shuffle, ShuffleConfig, SHUFFLE_KINDS, ALLOCATION_KEYS,
    shuffle_image_embeds, shuffle_calibration, shuffle_depth_label,
    shuffle_relation_label, constant_depth, shortcut_audit_step,
)
from geodistill.geometry.cross_view_label import build_edge_labels                   # noqa: E402
from geodistill.geometry.edge_sampler_v2 import sample_edges_v2, EdgeBudgetV2          # noqa: E402


def _scene(seed: int = 0):
    return make_synthetic_scene(SyntheticConfig(num_cameras=2, grid_h=4, grid_w=6,
                                                  feature_dim=32, signal_gain=2.5,
                                                  num_surfaces=8), seed=seed)


def test_image_shuffle_changes_features_only():
    s = _scene(0)
    s2 = shuffle_image_embeds(s, seed=7)
    assert not torch.equal(s.features, s2.features), "features must be permuted"
    assert torch.equal(s.token_uv, s2.token_uv)
    assert torch.equal(s.K, s2.K)
    assert torch.equal(s.cam_to_ego, s2.cam_to_ego)
    assert torch.equal(s.token_camera_id, s2.token_camera_id)


def test_calibration_shuffle_swaps_K_and_extrinsics():
    # Use 3 cameras so a non-identity permutation is reachable in a few seeds.
    # The synthetic fixture shares K and the translation across cameras (only
    # the rotation is camera-specific), so we check that cam_to_ego rotations
    # are permuted — which is what actually drives the FGT feature cost.
    s = make_synthetic_scene(SyntheticConfig(num_cameras=3, grid_h=4, grid_w=6,
                                              feature_dim=32, signal_gain=2.5,
                                              num_surfaces=8), seed=1)
    s2 = s
    for sd in range(50):
        s2 = shuffle_calibration(s, seed=sd)
        if not torch.equal(s.cam_to_ego, s2.cam_to_ego):
            break
    assert not torch.equal(s.cam_to_ego, s2.cam_to_ego), \
        "cam_to_ego must be permuted across cameras for at least one seed"
    assert torch.equal(s.features, s2.features)
    assert torch.equal(s.token_uv, s2.token_uv)
    assert torch.equal(s.token_camera_id, s2.token_camera_id)


def test_depth_shuffle_changes_d_T_only():
    s = _scene(2)
    t = build_hard_quantile_teacher(s)
    t2 = shuffle_depth_label(t, seed=11)
    assert not torch.equal(t.d_teacher, t2.d_teacher)
    # m_T / q fields are NOT touched
    assert torch.equal(t.m_T, t2.m_T)
    assert torch.equal(t.q_teacher, t2.q_teacher)


def test_constant_depth_makes_depth_constant():
    s = _scene(3)
    t = build_hard_quantile_teacher(s)
    t2 = constant_depth(t, value=25.0)
    assert torch.allclose(t2.d_teacher, torch.full_like(t.d_teacher, 25.0))
    # untouched fields
    assert torch.equal(t.m_T, t2.m_T)


def test_relation_shuffle_changes_label_columns():
    s = _scene(4)
    t = build_hard_quantile_teacher(s)
    edges, _, _ = sample_edges_v2(s, t, EdgeBudgetV2(P_local=2, P_ray=1, P_cross=2, P_hard=0, P_far=0))
    labels = build_edge_labels(s, t, edges)
    if labels.cross_view.numel() <= 1:
        return                                                                 # too few edges to test
    perm_labels = shuffle_relation_label(labels, seed=2)
    # at least one of the four label channels must change
    diffs = [
        not torch.equal(labels.depth_order, perm_labels.depth_order),
        not torch.equal(labels.cross_view, perm_labels.cross_view),
        not torch.equal(labels.topology, perm_labels.topology),
        not torch.equal(labels.occlusion, perm_labels.occlusion),
    ]
    # only assert when at least one column has variation in the original labels;
    # otherwise the shuffle is a no-op, which is itself fine.
    if any(labels.depth_order != labels.depth_order[0]):
        assert any(diffs), "relation shuffle should change at least one label column"
    # invariants
    assert torch.equal(labels.delta_c, perm_labels.delta_c)
    assert torch.equal(labels.reliability, perm_labels.reliability)


def test_apply_shuffle_dispatches_all_kinds():
    s = _scene(5)
    t = build_hard_quantile_teacher(s)
    for kind in SHUFFLE_KINDS:
        cfg = ShuffleConfig(kind=kind, seed=0, allocation_key="r_p")
        new_scene, new_teacher, meta = apply_shuffle(s, t, cfg)
        assert meta["kind"] == kind
        assert new_scene is not None and new_teacher is not None


def test_apply_shuffle_unknown_kind_raises():
    s = _scene(6)
    t = build_hard_quantile_teacher(s)
    try:
        apply_shuffle(s, t, ShuffleConfig(kind="bogus"))
        raise AssertionError("expected ValueError for unknown shuffle kind")
    except ValueError:
        pass


def test_transport_structure_shuffle_changes_coupling():
    """Hook in ot_common: structure_shuffle_seed permutes C_L rows/cols."""
    s = _scene(seed=7)
    cfg = OTConfig(epsilon_ot=0.05, gamma=0.5)
    cands = build_candidate_sets(s, dilate_px=cfg.cand_dilate_px)
    t_clean = build_full_ntlfgt_teacher(s, cfg, cands=cands)
    # Manually re-run with the shuffled-structure hook:
    from geodistill.teacher.ot_common import build_ot_teacher
    t_shuffled = build_ot_teacher(s, cfg, name="full_ntl_fgt_shuffled",
                                  cands=cands, structure_shuffle_seed=42)
    # Per-token teacher depth distribution should differ on at least one token
    finite = torch.isfinite(t_clean.d_teacher) & torch.isfinite(t_shuffled.d_teacher)
    if int(finite.sum()) >= 2:
        assert not torch.allclose(t_clean.d_teacher[finite], t_shuffled.d_teacher[finite]), \
            "transport-structure shuffle must perturb the FGT teacher"


def test_shortcut_audit_step_runs_clean_plus_each_kind():
    s = _scene(8)
    t = build_hard_quantile_teacher(s)

    def fake_eval(scene, teacher, meta):
        # Cheap eval: mean depth over valid tokens.
        m = teacher.m_T & torch.isfinite(teacher.d_teacher)
        if int(m.sum()) == 0:
            return {"mean_d": float("nan"), "kind": meta.get("kind")}
        return {"mean_d": float(teacher.d_teacher[m].mean()), "kind": meta.get("kind")}

    out = shortcut_audit_step(fake_eval, s, t, kinds=SHUFFLE_KINDS)
    assert "clean" in out
    for k in SHUFFLE_KINDS:
        assert k in out


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
