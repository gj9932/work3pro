"""Tests for baseline profiles (paper §4.4).

Synthetic-only:
- Profiles 1, 2, 5–17 must `apply_profile_to_model` cleanly on a built model.
- Profiles 3 / 4 / 18 must raise ImportError with the SpaceDrive hint locally.
- Each baseline yaml must load and reference a known profile name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geodistill.baselines import PROFILES, get_profile_spec, apply_profile_to_model    # noqa: E402
from geodistill.baselines.profile import ProfileSpec                                    # noqa: E402
from geodistill.data.synthetic import make_synthetic_scene, SyntheticConfig             # noqa: E402
from geodistill.trainers import GeoDistillConfig, GeoDistillVLM                          # noqa: E402
from geodistill.utils import read_jsonl  # noqa: F401  (sanity import)
from geotoken.config import load_config                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CFG_DIR = ROOT / "configs" / "geodistill" / "baselines"


def _model_for_profile():
    s = make_synthetic_scene(SyntheticConfig(num_cameras=2, grid_h=4, grid_w=6, feature_dim=64,
                                              signal_gain=2.0))
    cfg = GeoDistillConfig(hidden_size=64, d_bottleneck=64, d_e_cam=8, d_e_uv=16,
                            d_e_calib=16, d_e_ego=8, num_cameras=2,
                            pe_d_xyz=(20, 20, 24), rel_d_edge=32, rel_d_hidden=32)
    return s, GeoDistillVLM(cfg)


def test_all_18_yamls_present():
    yamls = sorted(BASELINE_CFG_DIR.glob("baseline_*.yaml"))
    assert len(yamls) == 18, f"expected 18 baseline yamls, got {len(yamls)}"


def test_yamls_reference_known_profiles():
    for p in sorted(BASELINE_CFG_DIR.glob("baseline_*.yaml")):
        cfg = load_config(p)
        prof = cfg.get("baseline", {}).get("profile")
        assert prof in PROFILES, f"{p.name}: profile {prof!r} unknown"


def test_local_profiles_apply_cleanly():
    """Profiles that don't need third_party must apply on the synthetic model."""
    s, model = _model_for_profile()
    local_profiles = [name for name, spec in PROFILES.items() if spec.third_party is None]
    assert len(local_profiles) >= 15, "expected at least 15 local profiles"
    for name in local_profiles:
        spec = get_profile_spec(name)
        # Build a fresh model per profile so previous freezes don't leak.
        _, m = _model_for_profile()
        apply_profile_to_model(m, spec)


def test_third_party_profiles_raise_locally():
    """Profiles 3 / 4 / 18 must fail fast with the SpaceDrive hint."""
    from geodistill.baselines.spacedrive_style import SpaceDriveStyleAdapter, SPACEDRIVE_HINT
    from geodistill.baselines.spacedrive_calibrated import SpaceDriveCalibratedAdapter
    from geodistill.baselines.llava15 import LLaVA15Adapter

    for cls in (SpaceDriveStyleAdapter, SpaceDriveCalibratedAdapter, LLaVA15Adapter):
        try:
            cls()
            raise AssertionError(f"{cls.__name__} must raise ImportError locally (no third_party/SpaceDrive)")
        except ImportError as exc:
            assert "third_party/SpaceDrive" in str(exc), \
                f"{cls.__name__} import error must reference third_party/SpaceDrive"


def test_apply_to_model_raises_on_third_party():
    s, model = _model_for_profile()
    spec = get_profile_spec("spacedrive_style")
    try:
        apply_profile_to_model(model, spec)
        raise AssertionError("apply_profile_to_model must refuse third_party profiles")
    except RuntimeError as exc:
        assert "wrapper" in str(exc) or "third_party" in str(exc).lower()


def test_disable_pe_freezes_beta_pe():
    s, model = _model_for_profile()
    spec = get_profile_spec("relation_no_pe")
    apply_profile_to_model(model, spec)
    assert not model.injector.beta_pe.requires_grad
    assert float(model.injector.beta_pe) == 0.0


def test_disable_relation_freezes_w_up():
    s, model = _model_for_profile()
    spec = get_profile_spec("r2ac_pe_no_relation")
    apply_profile_to_model(model, spec)
    assert not model.injector.beta_rel.requires_grad
    for p in model.injector.W_up.parameters():
        assert not p.requires_grad


def test_qwen_raw_freezes_everything_lpga():
    s, model = _model_for_profile()
    spec = get_profile_spec("qwen_raw")
    apply_profile_to_model(model, spec)
    for sub in (model.token_encoder, model.lpga, model.relation_encoders, model.injector):
        for p in sub.parameters():
            assert not p.requires_grad, "qwen_raw must freeze all LPGA parts"


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
