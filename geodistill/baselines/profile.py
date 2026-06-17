"""Baseline profiles for paper §4.4 (18 rows in tables 1-6).

Most baselines are flag-driven on top of the same ``GeoDistillVLM`` trainer; only
3 / 4 / 18 require ``third_party/SpaceDrive`` (lazy imports in
:mod:`geodistill.baselines.spacedrive_style`,
:mod:`geodistill.baselines.spacedrive_calibrated`,
:mod:`geodistill.baselines.llava15`).

Profile names (mirroring paper §4.4 row order):

     1: qwen_raw                    no training, eval Qwen2.5-VL only
     2: qwen_lora                   LoRA only (no LPGA / PE / relation)
     3: spacedrive_style            3rd_party — frozen UniDepthV2 + 3D PE
     4: spacedrive_calibrated       3rd_party — UniDepthV2 + LiDAR L_comp + L_coord
     5: raw_depth                   raw-depth adapter, no R²AC
     6: log_depth                   log-depth adapter
     7: power_warp                  fixed power-warp adapter
     8: log_companding              fixed log-companding adapter
     9: r2ac_a_const                R²AC, a = a_const (sweep grid)
    10: r2ac_risk_only              q_hat ≡ 1, L_q = 0
    11: r2ac_reli_only              r_hat ≡ 1, L_r = 0
    12: r2ac_oracle_decode          oracle a* both at target and decode
    13: relation_only               no PE, no R²AC depth
    14: r2ac_pe_no_relation         α_rel = 0 (relation residual disabled)
    15: relation_no_pe              α_pe = 0 (3D PE disabled)
    16: full                        full GeoDistill-VLM
    17: full_unfreeze_visual        full + unfreeze Qwen Vision Encoder
    18: llava15                     3rd_party — LLaVA-1.5-7B + CLIP ViT-L/14

The profile is applied at three stages:
- yaml time            (which yaml the user picks under configs/geodistill/baselines/)
- model build          (apply_profile_to_model)
- loss combination     (apply_profile_to_loss_weights)

This file owns the model-build half. Loss-side switches consume
``profile.spec()`` returned by :func:`get_profile_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geodistill.trainers.build_lpga import GeoDistillVLM


__all__ = ["ProfileSpec", "PROFILES", "apply_profile_to_model", "get_profile_spec"]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    enable_lpga: bool = True
    enable_relation: bool = True
    enable_pe: bool = True
    enable_lora: bool = True
    train_visual: bool = False
    depth_kind: str = "r2ac"          # "r2ac" | "raw" | "log" | "power_warp" | "log_companding" | "external"
    a_mode: str = "predicted"          # "predicted" | "constant" | "risk_only" | "reli_only" | "oracle"
    a_const: float = 0.5
    use_lambda_q: bool = True
    use_lambda_r: bool = True
    third_party: str | None = None     # "spacedrive_style" | "spacedrive_calibrated" | "llava15"
    notes: str = ""


PROFILES: dict[str, ProfileSpec] = {
    "qwen_raw":                ProfileSpec("qwen_raw", enable_lpga=False, enable_relation=False, enable_pe=False, enable_lora=False),
    "qwen_lora":               ProfileSpec("qwen_lora", enable_lpga=False, enable_relation=False, enable_pe=False),
    "spacedrive_style":        ProfileSpec("spacedrive_style", enable_lpga=False, enable_relation=False, enable_pe=True, third_party="spacedrive_style"),
    "spacedrive_calibrated":   ProfileSpec("spacedrive_calibrated", enable_lpga=False, enable_relation=False, enable_pe=True, third_party="spacedrive_calibrated"),
    "raw_depth":               ProfileSpec("raw_depth", depth_kind="raw", enable_relation=False, enable_pe=True),
    "log_depth":               ProfileSpec("log_depth", depth_kind="log", enable_relation=False, enable_pe=True),
    "power_warp":              ProfileSpec("power_warp", depth_kind="power_warp", enable_relation=False, enable_pe=True),
    "log_companding":          ProfileSpec("log_companding", depth_kind="log_companding", enable_relation=False, enable_pe=True),
    "r2ac_a_const":            ProfileSpec("r2ac_a_const", a_mode="constant"),
    "r2ac_risk_only":          ProfileSpec("r2ac_risk_only", a_mode="risk_only", use_lambda_q=False),
    "r2ac_reli_only":          ProfileSpec("r2ac_reli_only", a_mode="reli_only", use_lambda_r=False),
    "r2ac_oracle_decode":      ProfileSpec("r2ac_oracle_decode", a_mode="oracle"),
    "relation_only":           ProfileSpec("relation_only", depth_kind="r2ac", enable_pe=False),
    "r2ac_pe_no_relation":     ProfileSpec("r2ac_pe_no_relation", enable_relation=False),
    "relation_no_pe":          ProfileSpec("relation_no_pe", enable_pe=False),
    "full":                    ProfileSpec("full"),
    "full_unfreeze_visual":    ProfileSpec("full_unfreeze_visual", train_visual=True),
    "llava15":                 ProfileSpec("llava15", third_party="llava15"),
}


def get_profile_spec(name: str) -> ProfileSpec:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def apply_profile_to_model(model: GeoDistillVLM, spec: ProfileSpec) -> GeoDistillVLM:
    """Mutate ``model`` to honour the profile's enable flags.

    Profiles 3 / 4 / 18 (``third_party`` set) are not applied here — the
    caller routes them to the corresponding wrapper module.
    """
    if spec.third_party is not None:
        raise RuntimeError(
            f"profile {spec.name!r} requires the {spec.third_party!r} wrapper "
            f"under geodistill/baselines/. Don't apply it directly to GeoDistillVLM."
        )

    # Disable PE: clamp α_pe to 0 by freezing β_pe
    if not spec.enable_pe:
        with torch.no_grad():
            model.injector.beta_pe.zero_()
        model.injector.beta_pe.requires_grad_(False)

    # Disable relation residual
    if not spec.enable_relation:
        with torch.no_grad():
            model.injector.beta_rel.zero_()
        model.injector.beta_rel.requires_grad_(False)
        for p in model.injector.W_up.parameters():
            p.requires_grad_(False)

    # Disable LPGA entirely (qwen_raw / qwen_lora) — freeze every LPGA submodule.
    if not spec.enable_lpga:
        for sub in (model.token_encoder, model.geo_encoder, model.lpga,
                    model.relation_encoders, model.relation_heads, model.relation_aggregator,
                    model.injector):
            for p in sub.parameters():
                p.requires_grad_(False)

    return model
