"""Stage C trainer: robustness distillation (paper §3.13, optional).

Pairs a clean teacher view with a corrupted student view and minimizes

    L_robust = || Pool(H_geo^deg) - StopGrad(Pool(H_geo^full)) ||_2

Corruption types match the paper §4.5 robustness eval:
    - single_camera_drop
    - multi_camera_drop
    - front_only
    - image_occlusion
    - calibration_noise
    - low_light

Locally we run on the synthetic fixture so corruptions are applied at the
TokenScene level (drop tokens of a camera; perturb K / cam_to_ego). The GPU
trainer will instead apply them at pixel level before the Qwen forward — both
yield the same TokenScene contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from geodistill.data.contract import TokenScene
from geodistill.losses import L_robust
from geodistill.trainers.build_lpga import GeoDistillVLM


__all__ = ["StageCConfig", "stage_c_step", "apply_corruption"]


@dataclass
class StageCConfig:
    corruption: str = "single_camera_drop"
    drop_camera: int = 0
    occlusion_ratio: float = 0.3
    intrinsic_pixel_std: float = 1.0
    extrinsic_translation_std: float = 0.02
    seed: int = 0


def apply_corruption(scene: TokenScene, cfg: StageCConfig) -> TokenScene:
    """Return a degraded copy of ``scene`` (TokenScene-level corruption).

    The returned scene retains the contract — ``cam_offsets`` is recomputed if
    cameras are dropped, ``image_hw / K / cam_to_ego`` are masked accordingly.
    """
    new = copy.copy(scene)
    g = torch.Generator().manual_seed(cfg.seed)

    if cfg.corruption == "single_camera_drop":
        keep = [c for c in range(scene.num_cameras) if c != cfg.drop_camera]
        return _filter_cameras(scene, keep)
    if cfg.corruption == "multi_camera_drop":
        keep = [c for c in range(scene.num_cameras) if c % 2 == 0]
        if not keep:
            keep = [0]
        return _filter_cameras(scene, keep)
    if cfg.corruption == "front_only":
        return _filter_cameras(scene, [0])
    if cfg.corruption == "image_occlusion":
        # Zero a fraction of token features to simulate occluded regions
        feats = scene.features.clone()
        N = feats.shape[0]
        n_drop = int(N * cfg.occlusion_ratio)
        if n_drop > 0:
            drop = torch.randperm(N, generator=g)[:n_drop]
            feats[drop] = 0.0
        new.features = feats
        return new
    if cfg.corruption == "calibration_noise":
        K = scene.K.clone()
        cam_to_ego = scene.cam_to_ego.clone()
        K[:, 0, 2] += torch.randn(scene.num_cameras, generator=g) * cfg.intrinsic_pixel_std
        K[:, 1, 2] += torch.randn(scene.num_cameras, generator=g) * cfg.intrinsic_pixel_std
        cam_to_ego[:, :3, 3] += torch.randn(scene.num_cameras, 3, generator=g) * cfg.extrinsic_translation_std
        new.K = K
        new.cam_to_ego = cam_to_ego
        return new
    if cfg.corruption == "low_light":
        new.features = scene.features * 0.5 + torch.randn_like(scene.features) * 0.1
        return new
    raise ValueError(f"unknown corruption: {cfg.corruption!r}")


def _filter_cameras(scene: TokenScene, keep: list[int]) -> TokenScene:
    sel = torch.zeros(scene.num_tokens, dtype=torch.bool)
    new_offsets = [0]
    for c in keep:
        sl = scene.tokens_of_camera(c)
        sel[sl] = True
        new_offsets.append(new_offsets[-1] + (sl.stop - sl.start))

    new = copy.copy(scene)
    new.features = scene.features[sel]
    new.token_uv = scene.token_uv[sel]
    new.token_box = scene.token_box[sel]
    # remap camera ids to a contiguous [0, len(keep))
    remap = {c: i for i, c in enumerate(keep)}
    cams = scene.token_camera_id[sel].tolist()
    new.token_camera_id = torch.tensor([remap[c] for c in cams], dtype=torch.long)
    new.cam_offsets = torch.tensor(new_offsets, dtype=torch.long)
    new.K = scene.K[keep]
    new.cam_to_ego = scene.cam_to_ego[keep]
    new.image_hw = scene.image_hw[keep]
    if scene.gt_depth is not None:
        new.gt_depth = scene.gt_depth[sel]
    if scene.gt_valid is not None:
        new.gt_valid = scene.gt_valid[sel]
    if scene.gt_xyz_ego is not None:
        new.gt_xyz_ego = scene.gt_xyz_ego[sel]
    return new


def stage_c_step(
    model: GeoDistillVLM,
    scene_full: TokenScene,
    scene_deg: TokenScene,
    z_rel_full: torch.Tensor | None = None,
    z_rel_deg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute L_robust between full and degraded H_geo on a TokenScene pair."""
    # Full: teacher (no grad)
    with torch.no_grad():
        out_f = model.forward_lpga(
            h_img=scene_full.features,
            token_uv=scene_full.token_uv,
            token_box=scene_full.token_box,
            token_camera_id=scene_full.token_camera_id,
            K=scene_full.K, cam_to_ego=scene_full.cam_to_ego, image_hw=scene_full.image_hw,
            v_ego=scene_full.v_ego,
        )
        if z_rel_full is None:
            z_rel_full = out_f["g_mono"].new_zeros((scene_full.num_tokens, model.cfg.d_bottleneck))
        h_full = model.forward_inject(scene_full.features, out_f["c_hat"], z_rel_full)
    # Degraded: student with grad
    out_d = model.forward_lpga(
        h_img=scene_deg.features,
        token_uv=scene_deg.token_uv,
        token_box=scene_deg.token_box,
        token_camera_id=scene_deg.token_camera_id,
        K=scene_deg.K, cam_to_ego=scene_deg.cam_to_ego, image_hw=scene_deg.image_hw,
        v_ego=scene_deg.v_ego,
    )
    if z_rel_deg is None:
        z_rel_deg = out_d["g_mono"].new_zeros((scene_deg.num_tokens, model.cfg.d_bottleneck))
    h_deg = model.forward_inject(scene_deg.features, out_d["c_hat"], z_rel_deg)
    return L_robust(h_deg, h_full, scene_deg.cam_offsets, scene_full.cam_offsets)


# ====================================================================== CLI
def main():
    """Stage C trainer: cycle through corruptions, minimize L_robust.

    Inherits the trained model from stage B. Each step:
      1. fetch a clean (scene, teacher) pair
      2. pick a corruption (round-robin or random)
      3. apply_corruption(scene) → scene_deg
      4. L = L_robust(model, scene_full, scene_deg)

    The teacher is unused beyond the iterator contract.
    """
    import argparse
    import random
    from geodistill.runtime import (
        load_config_with_extends, train_loop, RunnerConfig, iter_scenes,
    )
    from geodistill.runtime.runner import StepOutput
    from geodistill.trainers.build_lpga import build_from_config
    from geodistill.trainers.common import OptimSpec, build_optimizer, synthetic_hidden_size

    DEFAULT_CORRUPTIONS = [
        "single_camera_drop", "multi_camera_drop", "front_only",
        "image_occlusion", "calibration_noise", "low_light",
    ]

    ap = argparse.ArgumentParser(description="GeoDistill stage C robustness distillation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs/geodistill/stage_c")
    ap.add_argument("--source", default=None,
                     choices=[None, "synthetic", "cached_nuscenes", "live_nuscenes"])
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None,
                     help="Resume from a stage B checkpoint. Defaults to runs/geodistill/stage_b/last.pt.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS,
                     help="Corruptions to cycle through (paper §4.5).")
    args = ap.parse_args()

    yaml_cfg = load_config_with_extends(args.config)
    sched = yaml_cfg.get("schedule", {})
    runner_cfg = RunnerConfig(
        out_dir=args.out_dir,
        max_steps=args.max_steps or int(sched.get("stage_c_steps", 20000)),
        grad_accum_steps=int(sched.get("grad_accum_steps", 1)),
        grad_clip=float(sched.get("grad_clip", 1.0)),
        log_every=args.log_every, ckpt_every=args.ckpt_every,
        amp=(args.amp if args.device == "cuda" else "off"),
        device=args.device,
        seed=args.seed if args.seed is not None else int(yaml_cfg.get("experiment", {}).get("seed", 42)),
        resume=args.resume or "runs/geodistill/stage_b/last.pt",
    )
    model = build_from_config(yaml_cfg, hidden_size_override=synthetic_hidden_size(yaml_cfg, args.source))
    spec = OptimSpec(
        lr=float(sched.get("lr_lpga", 2e-4)),
        weight_decay=float(sched.get("weight_decay", 0.01)),
        grad_clip=runner_cfg.grad_clip,
    )
    optim = build_optimizer(model.parameters(), spec)

    source = args.source or yaml_cfg.get("dataset", {}).get("source", "live_nuscenes")
    batch_iter = iter_scenes(source, yaml_cfg, scenes=args.scenes, seed=runner_cfg.seed,
                              infinite=True)
    rng = random.Random(runner_cfg.seed)

    def _loss_fn(model, batch, step):
        scene_full, _teacher = batch
        corr = rng.choice(args.corruptions)
        cfg = StageCConfig(corruption=corr, drop_camera=rng.randrange(scene_full.num_cameras),
                            seed=runner_cfg.seed + step)
        scene_deg = apply_corruption(scene_full, cfg)
        L = stage_c_step(model, scene_full, scene_deg)
        return StepOutput(loss=L, parts={"L_robust": L},
                            diag={"corruption": corr, "n_full_tokens": int(scene_full.num_tokens),
                                  "n_deg_tokens": int(scene_deg.num_tokens)})

    train_loop(
        model=model, optim=optim, batch_iter=batch_iter,
        loss_fn=_loss_fn, cfg=runner_cfg,
        extra_envelope={"stage": "C", "source": source,
                          "corruptions": args.corruptions,
                          "config": str(yaml_cfg.get("config_path", ""))},
    )


if __name__ == "__main__":
    main()
