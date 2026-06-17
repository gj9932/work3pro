"""Tests for sparse relation adapter (paper §3.10, §3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.data.synthetic import make_synthetic_scene, SyntheticConfig          # noqa: E402
from geodistill.teacher import build_hard_quantile_teacher                           # noqa: E402
from geodistill.geometry.edge_sampler_v2 import sample_edges_v2, EdgeBudgetV2        # noqa: E402
from geodistill.geometry.cross_view_label import build_edge_labels                    # noqa: E402
from geodistill.models.relation_edge import (                                         # noqa: E402
    RelationEdgeConfig, SameCamEdgeEncoder, CrossCamEdgeEncoder, RelationEdgeEncoders,
    RelationHeads,
)
from geodistill.models.relation_aggregator import RelationAggregator, RelationAggregatorConfig  # noqa: E402
from geodistill.core.camera import ray_dirs_ego                                        # noqa: E402


def _scene():
    cfg = SyntheticConfig(num_cameras=2, grid_h=4, grid_w=6, feature_dim=64, signal_gain=2.5)
    return make_synthetic_scene(cfg, seed=2)


def test_same_cross_params_unshared():
    cfg = RelationEdgeConfig(d_token=16, d_hidden=32, d_out=32, num_cameras=2)
    same = SameCamEdgeEncoder(cfg)
    cross = CrossCamEdgeEncoder(cfg)
    same_ids = {id(p) for p in same.parameters()}
    cross_ids = {id(p) for p in cross.parameters()}
    assert same_ids.isdisjoint(cross_ids), "same/cross-cam encoders must not share parameters"


def test_cross_branch_rejects_uv():
    cfg = RelationEdgeConfig(d_token=16, d_hidden=32, d_out=32, num_cameras=2)
    cross = CrossCamEdgeEncoder(cfg)
    g_p = torch.randn(3, 16); g_q = torch.randn(3, 16)
    bad_xi = torch.randn(3, 5)   # δu/δv pretending to be a cross feature
    cam_pair = torch.zeros(3, dtype=torch.long)
    try:
        cross(g_p, g_q, bad_xi, cam_pair)
        raise AssertionError("expected CrossCamEdgeEncoder to reject ξ_dim=5")
    except ValueError as exc:
        assert "δu" in str(exc) or "9" in str(exc)


def test_same_branch_rejects_cross_xi():
    cfg = RelationEdgeConfig(d_token=16, d_hidden=32, d_out=32, num_cameras=2)
    same = SameCamEdgeEncoder(cfg)
    g_p = torch.randn(3, 16); g_q = torch.randn(3, 16)
    bad_xi = torch.randn(3, 9)   # 9 = cross dim
    try:
        same(g_p, g_q, bad_xi)
        raise AssertionError("expected SameCamEdgeEncoder to reject ξ_dim=9")
    except ValueError:
        pass


def test_relation_router_forward():
    s = _scene()
    cfg = RelationEdgeConfig(d_token=16, d_hidden=32, d_out=32, num_cameras=int(s.num_cameras))
    enc = RelationEdgeEncoders(cfg)
    g = torch.randn(s.num_tokens, cfg.d_token)
    edges = []
    for c in range(s.num_cameras):
        sl = s.tokens_of_camera(c)
        ids = list(range(sl.start, sl.stop))
        if len(ids) >= 2:
            edges.append((ids[0], ids[1]))                            # same-cam
        if c + 1 < s.num_cameras:
            sl2 = s.tokens_of_camera(c + 1)
            if sl2.stop > sl2.start:
                edges.append((ids[0], sl2.start))                      # cross-cam
    edge_index = torch.tensor(edges, dtype=torch.long)
    # build per-token ego rays
    rays = []
    for c in range(s.num_cameras):
        sl = s.tokens_of_camera(c)
        rays.append(ray_dirs_ego(s.token_uv[sl], s.K[c], s.cam_to_ego[c]))
    ego_rays = torch.cat(rays, dim=0)
    r = enc(g, edge_index, s.token_camera_id, ego_rays, s.token_uv, s.image_hw)
    assert r.shape == (edge_index.shape[0], cfg.d_out)


def test_aggregator_normalized_per_source():
    N = 6
    d_edge = 8
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 3], [2, 4]], dtype=torch.long)
    r_pq = torch.randn(edge_index.shape[0], d_edge)
    cfg = RelationAggregatorConfig(d_edge=d_edge, d_rel=8, mode="confidence_attention")
    agg = RelationAggregator(cfg)
    conf = torch.randn(edge_index.shape[0])
    e_conf = torch.rand(edge_index.shape[0]).clamp_min(0.01)

    # Build the omega weights manually using the same softmax-by-source path.
    z = agg(r_pq, edge_index, N, conf_logit=conf, e_conf=e_conf)
    assert z.shape == (N, cfg.d_rel)
    # Per-source omega normalization invariance: re-run with a constant offset on conf;
    # the output should be unchanged (softmax is shift-invariant).
    z2 = agg(r_pq, edge_index, N, conf_logit=conf + 7.5, e_conf=e_conf)
    assert torch.allclose(z, z2, atol=1e-5)


def test_edge_label_cross_view_consistency():
    s = _scene()
    teacher = build_hard_quantile_teacher(s)
    edges, types, _ = sample_edges_v2(s, teacher, EdgeBudgetV2(P_local=2, P_ray=1, P_cross=2, P_hard=0, P_far=0))
    labels = build_edge_labels(s, teacher, edges, voxel_size=0.5, tau_depth=1.0, tau_eq_depth=0.5)
    assert labels.edge_index.shape[0] == len(edges)
    # cross_view label is restricted to cross-camera pairs.
    for i, (p, q) in enumerate(edges):
        if int(s.token_camera_id[p]) == int(s.token_camera_id[q]):
            assert int(labels.cross_view[i]) == 0


def test_relation_heads_shapes():
    cfg = RelationEdgeConfig(d_token=16, d_hidden=32, d_out=32, num_cameras=2)
    heads = RelationHeads(cfg)
    r_pq = torch.randn(7, cfg.d_out)
    out = heads(r_pq)
    assert out["delta_c"].shape == (7, 3)
    assert out["order_logits"].shape == (7, cfg.n_order_classes)
    assert out["cross_logit"].shape == (7,)
    assert out["topo_logits"].shape == (7, cfg.n_topo_classes)
    assert out["occ_logits"].shape == (7, cfg.n_occ_classes)
    assert out["conf_logit"].shape == (7,)


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
